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
from lightning_datamodule import ImageNetDataModule
# from parquet.lightning_datamodule_parquet import ImageNetParquetDataModule as ImageNetDataModule
import torchmetrics
import random
from typing import List, Tuple


class MultiScaleResNet(lightning.LightningModule):
    """Multi-scale ResNet50 with SIR and explicit CE/SIR thresholds."""
    def __init__(
        self,
        num_classes: int = 1000,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 100,
        unified_res: int = 56,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Prepare resolution groups
        res_lists = [list(range(32, 81, 16)), list(range(96, 145, 16)),
                     list(range(160, 209, 16)), [224]]
        base = resnet50(pretrained=False)
        self.setup_msun(res_lists, unified_res, base)

        # Classifier and losses
        feat_dim = base.fc.in_features
        self.classifier = nn.Linear(feat_dim, num_classes)
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_classes)

    def setup_msun(self, res_lists: List[List[int]], unified_size: int, base: nn.Module):
        """Build stem, unified head, and per-resolution subnets."""
        self.unified_net = nn.Sequential(base.layer2, base.layer3,
                                         base.layer4, nn.AdaptiveAvgPool2d(1))
        self.unified_size = unified_size
        configs = [
            {'k':3,'s':1,'p':2,'pool':False,'r':res_lists[0]},
            {'k':5,'s':1,'p':2,'pool':True ,'r':res_lists[1]},
            {'k':7,'s':2,'p':3,'pool':True ,'r':res_lists[2]},
            {'k':7,'s':2,'p':3,'pool':True ,'r':res_lists[3]}
        ]
        self.res_lists = [c['r'] for c in configs]
        self.subnets = nn.ModuleList()
        for c in configs:
            layers = [nn.Conv2d(3,64,c['k'],c['s'],c['p'],bias=False),
                      nn.BatchNorm2d(64), nn.ReLU(inplace=True)]
            if c['pool']:
                layers.append(nn.MaxPool2d(3,2,1))
            layers.append(base.layer1)
            self.subnets.append(nn.Sequential(*layers))

    def encode_random(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        zs, ys = [], []
        for net, r_list in zip(self.subnets, self.res_lists):
            r = random.choice(r_list)
            z = net(F.interpolate(x, size=(r,r), mode='bilinear', align_corners=False))
            zs.append(z)
            ys.append(self.unified_net(F.interpolate(z, size=self.unified_size,
                                                    mode='bilinear', align_corners=False)).flatten(1))
        return zs, ys

    def encode_by_res(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = x.shape[2]
        for net, r_list in zip(self.subnets, self.res_lists):
            if h in r_list:
                z = net(x)
                y = self.unified_net(F.interpolate(z, size=self.unified_size,
                                                    mode='bilinear', align_corners=False))
                return z, y.flatten(1)
        z = self.subnets[-1](x)
        y = self.unified_net(F.interpolate(z, size=self.unified_size,
                                            mode='bilinear', align_corners=False))
        return z, y.flatten(1)

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        zs, feats = self.encode_random(imgs)

        # CE losses with explicit thresholds
        ce_thr = [0., 0., 0., 0.]
        ce_losses = [self.ce_loss(self.classifier(y), labels) for y in feats]
        masked_ce = [l if l >= t else torch.zeros_like(l) for l, t in zip(ce_losses, ce_thr)]
        total_ce = sum(masked_ce)

        # SIR losses against last subnet with explicit thresholds
        ref = zs[-1]
        sir_thr = [0., 0., 0.]
        sir_losses = [self.mse_loss(
            F.interpolate(z, self.unified_size, mode='bilinear', align_corners=False),
            F.interpolate(ref, self.unified_size, mode='bilinear', align_corners=False)
        ) for z in zs[:-1]]
        masked_sir = [l if l >= t else torch.zeros_like(l) for l, t in zip(sir_losses, sir_thr)]
        total_sir = sum(masked_sir)

        # Combined loss
        loss = total_ce + self.hparams.alpha * total_sir

        # Log succinctly including per-subnet accuracy
        logs = {}
        for i, (ce, sir, y) in enumerate(zip(masked_ce, masked_sir, feats), start=1):
            logs[f'ce{i}'] = ce
            logs[f'sir{i}'] = sir if i < len(masked_ce) else torch.tensor(0.0, device=self.device)
            acc_i = self.acc(self.classifier(y), labels)
            logs[f'acc{i}'] = acc_i
        logs['ce_tot'] = total_ce
        logs['sir_tot'] = total_sir

        self.log_dict({f'train/{k}': v for k, v in logs.items()}, prog_bar=['train/acc1'])
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        fixed = [32, 56, 96, 128, 224]
        accs, sir_vals = {}, {}
        for r in fixed:
            _, y = self.encode_by_res(F.interpolate(imgs, (r, r), mode='bilinear', align_corners=False))
            accs[r] = self.acc(self.classifier(y), labels)
        ref_z, _ = self.encode_by_res(F.interpolate(imgs, (224, 224), mode='bilinear', align_corners=False))
        for r in fixed[:-1]:
            z, _ = self.encode_by_res(F.interpolate(imgs, (r, r), mode='bilinear', align_corners=False))
            sir_vals[r] = self.mse_loss(
                F.interpolate(z, self.unified_size, mode='bilinear', align_corners=False),
                F.interpolate(ref_z, self.unified_size, mode='bilinear', align_corners=False)
            )
        logs = {f'acc{r}': v for r, v in accs.items()}
        logs.update({f'sir{r}': v for r, v in sir_vals.items()})
        loss = self.ce_loss(
            self.classifier(
                self.unified_net(
                    F.interpolate(ref_z, size=self.unified_size, mode='bilinear', align_corners=False)
                ).flatten(1)
            ), labels
        )
        logs['loss'] = loss
        self.log_dict({f'val/{k}': v for k, v in logs.items()}, prog_bar=['val/acc224'])
        return loss

    def configure_optimizers(self):
        opt = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate,
                          weight_decay=self.hparams.weight_decay)
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
        parser.link_arguments(
            "trainer.max_epochs","model.max_epochs",
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
