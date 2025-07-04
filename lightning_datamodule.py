from lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from torchvision.datasets import ImageFolder
import os


class ImageNetDataModule(LightningDataModule):
    """LightningDataModule for ImageNet with enhanced augmentations."""

    def __init__(self, data_dir, batch_size=64, num_workers=8, img_size=224):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # Training transforms with AutoAugment and random erasing
        self.train_tf = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0), ratio=(3/4, 4/3)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            AutoAugment(AutoAugmentPolicy.IMAGENET),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3))
        ])

        # Validation transforms
        self.val_tf = transforms.Compose([
            transforms.Resize(int(img_size * 256 / 224)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    def setup(self, stage=None):
        # Create train and val datasets
        self.train_ds = ImageFolder(os.path.join(self.data_dir, "train"), transform=self.train_tf)
        self.val_ds = ImageFolder(os.path.join(self.data_dir, "val"), transform=self.val_tf)

    def train_dataloader(self):
        """Return training DataLoader."""
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers,
                          prefetch_factor=2,
                          pin_memory=True)

    def val_dataloader(self):
        """Return validation DataLoader."""
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          pin_memory=True)
