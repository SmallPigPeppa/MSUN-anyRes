import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import resnet50, densenet121, vgg16_bn, mobilenet_v2
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
import copy


class FixedResNet(lightning.LightningModule):

    def __init__(
            self,
            model_name: str = 'resnet50',
            num_classes: int = 1000,
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-4,
            max_epochs: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters()

        name = self.hparams.model_name.lower()
        if name == 'resnet50':
            self.base = resnet50(pretrained=False, num_classes=self.hparams.num_classes)
        elif name == 'densenet121':
            self.base = densenet121(pretrained=False, num_classes=self.hparams.num_classes)
        elif name == 'vgg16':
            self.base = vgg16_bn(pretrained=self.hparams.pretrained, num_classes=self.hparams.num_classes)
        elif name == 'mobilenetv2':
            self.base = mobilenet_v2(pretrained=self.hparams.pretrained, num_classes=self.hparams.num_classes)
        else:
            raise ValueError(f"Unsupported model: {name}")

        # losses
        self.ce_loss = nn.CrossEntropyLoss()
        self.acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.base(x)

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        logits = self.forward(imgs)
        loss = self.ce_loss(logits, labels)

        # log training loss
        self.log('train/ce_tot', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        # compute and log loss
        logits = self(imgs)
        loss = self.ce_loss(logits, labels)
        self.log('val/loss', loss, on_step=False, on_epoch=True)
        eval_scales = [32, 48, 96, 128, 176, 224]

        # evaluate different downscale -> upscale paths
        for scale in eval_scales:
            # downscale to (scale, scale)
            down = F.interpolate(imgs, size=(scale, scale), mode='bilinear', align_corners=False)
            # upscale back to (224, 224)
            up = F.interpolate(down, size=(224, 224), mode='bilinear', align_corners=False)

            # forward pass
            logits = self.forward(up)
            preds = torch.argmax(logits, dim=1)

            # update accuracy metric
            acc = self.acc(preds, labels)
            # log per-epoch accuracy
            self.log(f'val/acc{scale}', acc, on_step=False, on_epoch=True)

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
            "trainer.max_epochs", "model.max_epochs",
        )
        parser.add_lightning_class_args(ModelCheckpoint, 'model_checkpoint')
        parser.add_lightning_class_args(LearningRateMonitor, 'lr_monitor')


if __name__ == '__main__':
    CLI(
        FixedResNet,
        ImageNetDataModule,
        save_config_callback=None,
        seed_everything_default=42
    )
