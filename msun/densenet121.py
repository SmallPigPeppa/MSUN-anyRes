import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from torchvision.models import densenet121
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer
from imagenet_dali import ClassificationDALIDataModule
from args import parse_args

PRETRAINED = False

class MultiScaleDenseNet(pl.LightningModule):
    """Multi-scale DenseNet121 with SIR and explicit CE/SIR thresholds."""

    def __init__(
        self,
        num_classes: int = 1000,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 100,
        alpha: float = 1.0,
        pretrained: bool = PRETRAINED,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Define resolution groups
        res_lists = [list(range(32, 81, 16)),
                     list(range(96, 145, 16)),
                     list(range(160, 209, 16)),
                     [224]]
        base = densenet121(pretrained=pretrained)
        self._build_msun(res_lists, base)

        # Classifier and losses
        feat_dim = base.classifier.in_features
        self.classifier = nn.Linear(feat_dim, num_classes)
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
        self.acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def _build_msun(self, res_lists, base):
        # Per-resolution subnets
        configs = [
            {'k': 3, 's': 1, 'p': 1, 'pool': False, 'r': res_lists[0]},
            {'k': 5, 's': 1, 'p': 2, 'pool': True,  'r': res_lists[1]},
            {'k': 7, 's': 2, 'p': 3, 'pool': True,  'r': res_lists[2]},
            {'k': 7, 's': 2, 'p': 3, 'pool': True,  'r': res_lists[3]},
        ]
        self.res_lists = [c['r'] for c in configs]
        self.subnets = nn.ModuleList()
        for cfg in configs:
            layers = [
                nn.Conv2d(3, 64, cfg['k'], cfg['s'], cfg['p'], bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ]
            if cfg['pool']:
                layers.append(nn.MaxPool2d(3, 2, 1))
            # Attach first DenseNet block
            layers.append(base.features.denseblock1)
            self.subnets.append(nn.Sequential(*layers))

        # Unified head: reuse DenseNet feature extractor minus its initial layers
        u_net = densenet121(pretrained=self.hparams.pretrained)
        # Remove front layers to rely only on higher-level features
        u_net.features.conv0 = nn.Identity()
        u_net.features.norm0 = nn.Identity()
        u_net.features.relu0 = nn.Identity()
        u_net.features.pool0 = nn.Identity()
        u_net.features.denseblock1 = nn.Identity()
        # Build unified net
        self.unified_net = nn.Sequential(
            u_net.features,
            nn.AdaptiveAvgPool2d(1)
        )

        # Compute spatial size for interpolation
        with torch.no_grad():
            max_res = max(self.res_lists[-1])
            dummy = torch.zeros(1, 3, max_res, max_res, device=self.device)
            z = self.subnets[-1](dummy)
            self.unified_size = z.shape[-1]

    def encode_random(self, x):
        zs, ys = [], []
        for net, r_list in zip(self.subnets, self.res_lists):
            r = random.choice(r_list)
            z = net(F.interpolate(x, size=(r, r), mode='bilinear', align_corners=False))
            ys.append(
                self.unified_net(
                    F.interpolate(z, size=self.unified_size, mode='bilinear', align_corners=False)
                ).flatten(1)
            )
            zs.append(z)
        return zs, ys

    def encode_by_res(self, x):
        h = x.shape[2]
        for net, r_list in zip(self.subnets, self.res_lists):
            if h in r_list:
                z = net(x)
                y = self.unified_net(
                    F.interpolate(z, size=self.unified_size, mode='bilinear', align_corners=False)
                ).flatten(1)
                return z, y
        # Fallback: highest-res subnet
        z = self.subnets[-1](x)
        y = self.unified_net(
            F.interpolate(z, size=self.unified_size, mode='bilinear', align_corners=False)
        ).flatten(1)
        return z, y

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        zs, feats = self.encode_random(imgs)

        # Cross-entropy losses
        ce_losses = [self.ce_loss(self.classifier(y), labels) for y in feats]
        total_ce = sum(ce_losses)

        # SIR losses w.r.t. last (highest-res) subnet
        ref = zs[-1]
        sir_losses = [
            self.mse_loss(
                F.interpolate(z, self.unified_size, mode='bilinear', align_corners=False),
                F.interpolate(ref, self.unified_size, mode='bilinear', align_corners=False)
            )
            for z in zs[:-1]
        ]
        total_sir = sum(sir_losses)

        loss = total_ce + self.hparams.alpha * total_sir

        # Logging accuracies per scale
        for i, y in enumerate(feats, start=1):
            acc = self.acc(self.classifier(y), labels)
            self.log(f"train/acc{i}", acc, prog_bar=(i==1))
            self.log(f"train/ce{i}", ce_losses[i-1])
            self.log(f"train/sir{i}", sir_losses[i-1] if i < len(ce_losses) else torch.tensor(0.0, device=self.device))
        self.log("train/ce_tot", total_ce)
        self.log("train/sir_tot", total_sir)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        fixed = [32, 48, 96, 128, 176, 224]
        # Accuracies
        for r in fixed:
            _, y = self.encode_by_res(F.interpolate(imgs, (r, r), mode='bilinear', align_corners=False))
            acc = self.acc(self.classifier(y), labels)
            self.log(f"val/acc{r}", acc, prog_bar=(r==224))
        # SIR
        ref_z, _ = self.encode_by_res(F.interpolate(imgs, (224, 224), mode='bilinear', align_corners=False))
        for r in fixed[:-1]:
            z, _ = self.encode_by_res(F.interpolate(imgs, (r, r), mode='bilinear', align_corners=False))
            sir = self.mse_loss(
                F.interpolate(z, self.unified_size, mode='bilinear', align_corners=False),
                F.interpolate(ref_z, self.unified_size, mode='bilinear', align_corners=False)
            )
            self.log(f"val/sir{r}", sir)

    def configure_optimizers(self):
        opt = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            momentum=0.9
        )
        sched = LinearWarmupCosineAnnealingLR(
            opt,
            warmup_epochs=int(self.hparams.max_epochs * 0.05),
            max_epochs=self.hparams.max_epochs,
            warmup_start_lr=0.01 * self.hparams.learning_rate,
            eta_min=0.01 * self.hparams.learning_rate
        )
        return [opt], [sched]

if __name__ == "__main__":
    args = parse_args()
    pl.seed_everything(42)
    model = MultiScaleDenseNet(
        num_classes=args.num_classes,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        alpha=args.alpha if hasattr(args, 'alpha') else 1.0,
        pretrained=PRETRAINED
    )
    checkpoint = ModelCheckpoint(monitor="val/acc224", mode="max", dirpath=args.checkpoint_dir,
                                 save_top_k=1, save_last=True)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    logger = WandbLogger(name=f"{args.run_name}_MSUN", project=args.project,
                         entity=args.entity, offline=args.offline)
    datamodule = ClassificationDALIDataModule(
        train_data_path=os.path.join(args.dataset_path, 'train'),
        val_data_path=os.path.join(args.dataset_path, 'val'),
        num_workers=args.num_workers,
        batch_size=args.batch_size
    )
    trainer = Trainer.from_argparse_args(
        args,
        callbacks=[checkpoint, lr_monitor],
        logger=logger,
        precision=16,
        accelerator="ddp",
        gradient_clip_val=1.0,
        check_val_every_n_epoch=args.eval_every
    )
    trainer.fit(model, datamodule=datamodule)
