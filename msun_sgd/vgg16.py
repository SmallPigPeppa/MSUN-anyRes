import random
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torchvision.models import vgg16_bn
import lightning
from lightning.pytorch import cli
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from lightning_datamodulev3 import ImageNetDataModule
LAYERS = 24


class MultiScaleVGG(lightning.LightningModule):
    """Multi-scale VGG16 with SIR and explicit CE/SIR thresholds."""

    def __init__(
            self,
            num_classes: int = 1000,
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-4,
            max_epochs: int = 100,
            alpha: float = 1.0,
            pretrained: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Define resolution groups
        res_lists = [list(range(32, 65, 16)),
                     list(range(80, 145, 16)),
                     list(range(160, 209, 16)),
                     [224]]

        # Base VGG16 backbone
        base = vgg16_bn(pretrained=self.hparams.pretrained, num_classes=self.hparams.num_classes)

        # Build MSUN: unified head and per-scale sub-nets
        self._build_msun(res_lists, base)

        # Losses & metrics
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_classes)

    def _build_msun(self, res_lists, base: nn.Module):
        # Store resolution lists
        self.res_lists = res_lists

        # Unified head: remove first LAYERS convolutional layers
        unified = copy.deepcopy(base)
        for i in range(LAYERS):
            unified.features[i] = nn.Identity()
        self.unified_net = unified

        # Per-scale subnets with custom pooling modifications
        self.subnets = nn.ModuleList()
        for idx, _ in enumerate(res_lists):
            v = copy.deepcopy(base)
            layers = []
            for i in range(LAYERS):
                layer = v.features[i]
                # First subnet: replace MaxPool at layers 6,13,23 with stride-1 pooling
                if idx == 0 and i in [6, 13, 23]:
                    layer = nn.MaxPool2d(kernel_size=2, stride=1)
                # Second subnet: replace MaxPool at layer 23 with stride-1 pooling
                elif idx == 1 and i == 23:
                    layer = nn.MaxPool2d(kernel_size=2, stride=1)
                layers.append(layer)
            self.subnets.append(nn.Sequential(*layers))

        # Determine feature-map spatial size after subnets
        with torch.no_grad():
            max_r = max(res_lists[-1])
            dummy = torch.zeros(1, 3, max_r, max_r, device=self.device)
            out_feat = self.subnets[-1](F.interpolate(dummy, size=(max_r, max_r), mode='bilinear', align_corners=False))
            self.z_size = out_feat.shape[-1]

    def forward_random(self, x: torch.Tensor):
        zs, ys = [], []
        for net, r_list in zip(self.subnets, self.res_lists):
            r = random.choice(r_list)
            z = net(F.interpolate(x, size=(r, r), mode='bilinear', align_corners=False))
            y = self.unified_net(
                F.interpolate(z, size=(self.z_size, self.z_size), mode='bilinear', align_corners=False)
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
                    F.interpolate(z, size=(self.z_size, self.z_size), mode='bilinear', align_corners=False)
                )
                return z, y

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        zs, ys = self.forward_random(imgs)

        # Cross-entropy losses
        ce_losses = [self.ce_loss(y, labels) for y in ys]
        total_ce = sum(ce_losses)

        # SIR losses vs largest scale
        ref = zs[-1]
        sir_losses = [
            self.mse_loss(
                F.interpolate(z, self.z_size, mode='bilinear', align_corners=False),
                F.interpolate(ref, self.z_size, mode='bilinear', align_corners=False)
            ) for z in zs[:-1]
        ]
        total_sir = sum(sir_losses)

        # Combined loss
        loss = total_ce + self.hparams.alpha * total_sir

        # Logging metrics
        logs = {}
        for i, (ce, sir, y) in enumerate(zip(ce_losses, sir_losses + [None], ys), start=1):
            logs[f'ce{i}'] = ce
            logs[f'sir{i}'] = sir if sir is not None else torch.tensor(0.0, device=self.device)
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
        opt = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate,
                                    weight_decay=self.hparams.weight_decay, momentum=0.9)
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
        MultiScaleVGG,
        ImageNetDataModule,
        save_config_callback=None,
        seed_everything_default=42
    )
