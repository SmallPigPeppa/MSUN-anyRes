import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, densenet121, vgg16_bn, mobilenet_v2
import lightning
from lightning.pytorch import cli
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from lightning_datamodulev3 import ImageNetDataModule
import torchmetrics
from typing import Tuple
from lightning.pytorch.callbacks import StochasticWeightAveraging


class FixedResNet(lightning.LightningModule):

    def __init__(
            self,
            model_name: str = 'resnet50',
            num_classes: int = 1000,
            learning_rate: float = 1e-3,
            weight_decay: float = 1e-4,
            max_epochs: int = 100,
            pretrained: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        name = self.hparams.model_name.lower()
        if name == 'resnet50':
            self.base = resnet50(pretrained=self.hparams.pretrained, num_classes=self.hparams.num_classes)
        elif name == 'densenet121':
            self.base = densenet121(pretrained=self.hparams.pretrained, num_classes=self.hparams.num_classes)
        elif name == 'vgg16':
            self.base = vgg16_bn(pretrained=self.hparams.pretrained, num_classes=self.hparams.num_classes)
        elif name == 'mobilenetv2':
            self.base = mobilenet_v2(pretrained=self.hparams.pretrained, num_classes=self.hparams.num_classes)
        else:
            raise ValueError(f"Unsupported model: {name}")

        # losses
        self.ce_loss = nn.CrossEntropyLoss()
        self.acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_classes)

        # test_resolutions list
        self.test_resolutions = list(range(32, 225, 16))
        self.test_accs = nn.ModuleDict({
            f"acc_{r}": torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
            for r in self.test_resolutions
        })

    def forward(self, x: torch.Tensor) ->  torch.Tensor:
        return self.base(x)

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        logits = self.forward(imgs)
        loss = self.ce_loss(logits, labels)

        # log training loss
        self.log('train/ce_tot', loss, on_step=False, on_epoch=True)
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

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        # for each subnet and each resolution

        for r in self.test_resolutions:
            # resize → subnet → unified head → predict
            down = F.interpolate(imgs, size=(r, r), mode='bilinear', align_corners=False)
            # upscale back to (224, 224)
            up = F.interpolate(down, size=(224, 224), mode='bilinear', align_corners=False)
            y = self.forward(up)
            preds = y.argmax(dim=1)

            # update the right Accuracy instance
            key = f"acc_{r}"
            self.test_accs[key](preds, labels)

    def on_test_epoch_end(self):
        # prepare columns and rows
        cols = ["subnet"] + [str(r) for r in self.test_resolutions]
        rows = []

        # compute acc for each resolution
        accs = [
            self.test_accs[f"acc_{r}"].compute().item()
            for r in self.test_resolutions
        ]
        rows.append([f"subnet0", *accs])

        # reset all metrics in one go
        for m in self.test_accs.values():
            m.reset()

        self.logger.log_table(
            key="test/accuracy_table",
            columns=cols,
            data=rows
        )



class CLI(cli.LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments(
            "trainer.max_epochs", "model.max_epochs",
        )
        parser.add_lightning_class_args(ModelCheckpoint, 'model_checkpoint')
        parser.add_lightning_class_args(LearningRateMonitor, 'lr_monitor')
        parser.add_lightning_class_args(StochasticWeightAveraging, 'swa')


if __name__ == '__main__':
    CLI(
        FixedResNet,
        ImageNetDataModule,
        save_config_callback=None,
        seed_everything_default=42
    )
