import random
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
from torchvision.models import densenet121
import lightning
from lightning.pytorch import cli
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from lightning_datamodule import ImageNetDataModule


class MultiScaleDenseNet(lightning.LightningModule):
    """Multi-scale DenseNet121 with SIR and explicit CE/SIR thresholds."""

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

        # Define resolution groups
        res_lists = [list(range(32, 65, 16)),
                     list(range(80, 145, 16)),
                     list(range(160, 209, 16)),
                     [224]]

        # Base DenseNet for shared head
        base = densenet121(pretrained=False, num_classes=self.hparams.num_classes)
        self._build_msun(res_lists, base)

        # Losses and metrics
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_classes)

    def _build_msun(self, res_lists, base: nn.Module):
        # Unified head: remove initial conv/pool and first denseblock
        u = copy.deepcopy(base)
        u.features.conv0 = nn.Identity()
        u.features.norm0 = nn.Identity()
        u.features.relu0 = nn.Identity()
        u.features.pool0 = nn.Identity()
        u.features.denseblock1 = nn.Identity()
        self.unified_net = u

        # Per-scale subnets: initial conv layers + first denseblock
        configs = [
            {'k': 3, 's': 1, 'p': 1, 'pool': False, 'r': res_lists[0]},
            {'k': 5, 's': 1, 'p': 2, 'pool': True, 'r': res_lists[1]},
            {'k': 7, 's': 2, 'p': 3, 'pool': True, 'r': res_lists[2]},
            {'k': 7, 's': 2, 'p': 3, 'pool': True, 'r': res_lists[3]},
        ]
        self.res_lists = [c['r'] for c in configs]
        self.subnets = nn.ModuleList()
        for c in configs:
            layers = [
                nn.Conv2d(3, 64, kernel_size=c['k'], stride=c['s'], padding=c['p'], bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ]
            if c['pool']:
                layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            # attach DenseNet first block
            layers.append(base.features.denseblock1)
            self.subnets.append(nn.Sequential(*layers))

        # Determine spatial size for unified head
        with torch.no_grad():
            max_r = max(self.res_lists[-1])
            dummy = torch.zeros(1, 3, max_r, max_r, device=self.device)
            zs = self.subnets[-1](
                F.interpolate(dummy, size=(max_r, max_r), mode='bilinear', align_corners=False)
            )
            self.z_size = zs.shape[-1]

    def forward_random(self, x: torch.Tensor):
        zs, ys = [], []
        for net, r_list in zip(self.subnets, self.res_lists):
            r = random.choice(r_list)
            z = net(F.interpolate(x, size=(r, r), mode='bilinear', align_corners=False))
            y = self.unified_net(
                F.interpolate(z, size=self.z_size, mode='bilinear', align_corners=False)
            )
            zs.append(z)
            ys.append(y)
        return zs, ys

    def forward_by_res(self, x: torch.Tensor):
        h = x.shape[2]
        for net, r_list in zip(self.subnets, self.res_lists):
            if h in r_list:
                z = net(x)
                y = self.unified_net(
                    F.interpolate(z, size=self.z_size, mode='bilinear', align_corners=False)
                )
                return z, y

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        zs, ys = self.forward_random(imgs)

        # Cross-entropy losses per scale
        ce_thr = [0., 0., 0., 0.]
        ce_losses = [self.ce_loss(y, labels) for y in ys]
        masked_ce = [l if l >= t else torch.zeros_like(l) for l, t in zip(ce_losses, ce_thr)]
        total_ce = sum(masked_ce)

        # SIR losses vs largest scale
        ref = zs[-1]
        sir_thr = [0., 0., 0.]
        sir_losses = [
            self.mse_loss(
                F.interpolate(z, self.z_size, mode='bilinear', align_corners=False),
                F.interpolate(ref, self.z_size, mode='bilinear', align_corners=False)
            ) for z in zs[:-1]
        ]
        masked_sir = [l if l >= t else torch.zeros_like(l) for l, t in zip(sir_losses, sir_thr)]
        total_sir = sum(masked_sir)

        # combined loss
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

    def configure_optimizers(self):
        opt = optim.AdamW(
            self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay
        )
        sched = LinearWarmupCosineAnnealingLR(
            opt,
            warmup_epochs=int(self.hparams.max_epochs * 0.05),
            max_epochs=self.hparams.max_epochs,
            warmup_start_lr=0.01 * self.hparams.learning_rate,
            eta_min=0.01 * self.hparams.learning_rate
        )
        return {'optimizer': opt, 'lr_scheduler': sched}


class CLI(cli.LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("trainer.max_epochs", "model.max_epochs")
        parser.add_lightning_class_args(ModelCheckpoint, 'model_checkpoint')
        parser.add_lightning_class_args(LearningRateMonitor, 'lr_monitor')


if __name__ == '__main__':
    CLI(
        MultiScaleDenseNet,
        ImageNetDataModule,
        save_config_callback=None,
        seed_everything_default=42
    )
