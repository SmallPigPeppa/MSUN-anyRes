import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import resnet50
import lightning
from lightning.pytorch import cli
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from lightning_datamodulev3 import ImageNetDataModule
import torchmetrics
import random
from typing import List, Tuple
import copy
import wandb
from torchmetrics import MetricCollection, Accuracy


class MultiScaleResNet(lightning.LightningModule):
    """Multi-scale ResNet50 with SIR and explicit CE/SIR thresholds."""

    def __init__(
            self,
            num_classes: int = 1000,
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-4,
            max_epochs: int = 100,
            alpha: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Prepare resolution groups
        res_lists = [list(range(32, 65, 16)),
                     list(range(80, 145, 16)),
                     list(range(160, 209, 16)),
                     [224]]

        base = resnet50(pretrained=False, num_classes=self.hparams.num_classes)
        self._build_msun(res_lists, base)

        # losses
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.acc = Accuracy(task="multiclass", num_classes=self.hparams.num_classes)


        # test_resolutions list
        self.test_resolutions = list(range(32, 225, 16))
        self.test_resolutions = [32]
        # one Accuracy per (subnet_idx, resolution)
        self.test_accs = nn.ModuleDict({
            f"acc_{i}_{r}": Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
            for i in range(len(self.subnets))
            for r in self.test_resolutions
        })

    def _build_msun(self, res_lists: List[List[int]], base: nn.Module):
        """Build stem, unified head, and per-resolution subnets."""
        # build unified head
        u = copy.deepcopy(base)
        u.conv1 = nn.Identity()
        u.bn1 = nn.Identity()
        u.relu = nn.Identity()
        u.maxpool = nn.Identity()
        u.layer1 = nn.Identity()
        self.unified_net = u

        # build subnes
        configs = [
            {'k': 3, 's': 1, 'p': 2, 'pool': False, 'r': res_lists[0]},
            {'k': 5, 's': 1, 'p': 2, 'pool': True, 'r': res_lists[1]},
            {'k': 7, 's': 2, 'p': 3, 'pool': True, 'r': res_lists[2]},
            {'k': 7, 's': 2, 'p': 3, 'pool': True, 'r': res_lists[3]}
        ]
        self.res_lists = [c['r'] for c in configs]
        self.subnets = nn.ModuleList()
        for c in configs:
            layers = [nn.Conv2d(3, 64, c['k'], c['s'], c['p'], bias=False),
                      nn.BatchNorm2d(64), nn.ReLU(inplace=True)]
            if c['pool']:
                layers.append(nn.MaxPool2d(3, 2, 1))
            layers.append(copy.deepcopy(base.layer1))
            self.subnets.append(nn.Sequential(*layers))

        # Automatically determine the unified spatial size from the last subnet
        with torch.no_grad():
            max_res = max(self.res_lists[-1])
            dummy = torch.zeros(1, 3, max_res, max_res, device=self.device)
            self.z_size = self.subnets[-1](dummy).shape[-1]

    def forward_random(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        zs, ys = [], []
        for net, r_list in zip(self.subnets, self.res_lists):
            r = random.choice(r_list)
            z = net(F.interpolate(x, size=(r, r), mode='bilinear', align_corners=False))
            zs.append(z)
            ys.append(self.unified_net(F.interpolate(z, size=self.z_size,
                                                     mode='bilinear', align_corners=False)))
        return zs, ys

    def forward_by_res(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = x.shape[2]
        for net, r_list in zip(self.subnets, self.res_lists):
            if h in r_list:
                z = net(x)
                y = self.unified_net(F.interpolate(z, size=self.z_size,
                                                   mode='bilinear', align_corners=False))
                return z, y

    def forward_by_idx(self, x: torch.Tensor, idx: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.subnets[idx](x)
        y = self.unified_net(F.interpolate(z, size=self.z_size,
                                           mode='bilinear', align_corners=False))
        return z, y

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        zs, ys = self.forward_random(imgs)

        # CE losses with explicit thresholds
        ce_thr = [0., 0., 0., 0.]
        ce_losses = [self.ce_loss(y, labels) for y in ys]
        masked_ce = [l if l >= t else torch.zeros_like(l) for l, t in zip(ce_losses, ce_thr)]
        total_ce = sum(masked_ce)

        # SIR losses against last subnet with explicit thresholds
        ref = zs[-1]
        sir_thr = [0., 0., 0.]
        sir_losses = [self.mse_loss(
            F.interpolate(z, self.z_size, mode='bilinear', align_corners=False),
            F.interpolate(ref, self.z_size, mode='bilinear', align_corners=False)
        ) for z in zs[:-1]]
        masked_sir = [l if l >= t else torch.zeros_like(l) for l, t in zip(sir_losses, sir_thr)]
        total_sir = sum(masked_sir)

        # Combined loss
        loss = total_ce + self.hparams.alpha * total_sir
        # log metrics
        logs = {}
        for i, (ce, sir, y) in enumerate(zip(masked_ce, masked_sir + [None], ys), start=1):
            logs[f'ce{i}'] = ce
            logs[f'sir{i}'] = sir if i < len(masked_ce) else torch.tensor(0.0, device=self.device)
            logs[f'acc{i}'] = self.acc(y, labels)
        logs['ce_tot'] = total_ce
        logs['sir_tot'] = total_sir
        self.log_dict({f'train/{k}': v for k, v in logs.items()}, prog_bar=['train/acc1'])
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        fixed = [32, 48, 96, 128, 176, 224]
        accs, sir_vals = {}, {}
        for r in fixed:
            _, y = self.forward_by_res(F.interpolate(imgs, (r, r), mode='bilinear', align_corners=False))
            accs[r] = self.acc(y, labels)
        ref_z, _ = self.forward_by_res(F.interpolate(imgs, (224, 224), mode='bilinear', align_corners=False))
        for r in fixed[:-1]:
            z, _ = self.forward_by_res(F.interpolate(imgs, (r, r), mode='bilinear', align_corners=False))
            sir_vals[r] = self.mse_loss(
                F.interpolate(z, self.z_size, mode='bilinear', align_corners=False),
                F.interpolate(ref_z, self.z_size, mode='bilinear', align_corners=False)
            )
        logs = {f'acc{r}': v for r, v in accs.items()}
        logs.update({f'sir{r}': v for r, v in sir_vals.items()})
        loss = self.ce_loss(
            self.unified_net(
                F.interpolate(ref_z, size=self.z_size, mode='bilinear', align_corners=False)
            ), labels
        )
        logs['loss'] = loss
        self.log_dict({f'val/{k}': v for k, v in logs.items()}, prog_bar=['val/acc224'])
        return loss

    def test_step(self, batch, batch_idx):
        imgs, labels = batch

        # for each subnet and each resolution
        for i, subnet in enumerate(self.subnets):
            for r in self.test_resolutions:
                # resize → subnet → unified head → predict
                x_r = F.interpolate(imgs, size=(r, r), mode='bilinear', align_corners=False)
                _, y = self.forward_by_idx(x_r, i)
                preds = y.argmax(dim=1)

                # update the right Accuracy instance
                key = f"acc_{i}_{r}"
                self.test_accs[key](preds, labels)

    def on_test_epoch_end(self):
        # prepare columns and rows
        cols = ["subnet"] + [str(r) for r in self.test_resolutions]
        rows = []
        for i in range(len(self.subnets)):
            # compute acc for each resolution
            accs = [
                self.test_accs[f"acc_{i}_{r}"].compute().item()
                for r in self.test_resolutions
            ]
            rows.append([f"subnet{i + 1}", *accs])

        # reset all metrics in one go
        for m in self.test_accs.values():
            m.reset()

        # log to W&B
        table = wandb.Table(data=rows, columns=cols)
        # wandb.log({"test/accuracy_table": table})
        self.log_table("test/accuracy_table", table, logger=True)

class CLI(cli.LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments(
            "trainer.max_epochs", "model.max_epochs",
        )
        parser.add_lightning_class_args(ModelCheckpoint, 'model_checkpoint')
        parser.add_lightning_class_args(LearningRateMonitor, 'lr_monitor')


if __name__ == '__main__':
    CLI(
        MultiScaleResNet,
        ImageNetDataModule,
        save_config_callback=None,
        seed_everything_default=42
    )
