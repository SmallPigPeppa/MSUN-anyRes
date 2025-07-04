from torch.utils.data import Dataset, DataLoader
from lightning import LightningDataModule
from torchvision import transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy


import os
import glob
import io
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import pandas as pd
from tqdm import tqdm

class ParquetImageDataset(Dataset):
    """A Dataset that reads image metadata (and optionally image bytes) from Parquet files,
       loading them in parallel with a progress bar."""
    def __init__(self,
                 parquet_dir: str,
                 image_dir: str,
                 transform=None,
                 read_workers: int = 32):
        """
        Args:
            parquet_dir: path to folder containing *.parquet files
            image_dir:   if images are on disk, root directory for img_path
            transform:   torchvision transforms to apply
            read_workers: number of threads for reading parquet files;
                          defaults to os.cpu_count() if None.
        """
        self.image_dir = image_dir
        self.transform = transform
        self.read_workers = read_workers or os.cpu_count()

        # find all .parquet files in this split
        files = sorted(glob.glob(os.path.join(parquet_dir, '*.parquet')))
        if not files:
            raise FileNotFoundError(f"No parquet files found in {parquet_dir}")

        # read & concatenate in parallel with a tqdm progress bar
        dfs = []
        with ThreadPoolExecutor(max_workers=self.read_workers) as executor:
            for df in tqdm(executor.map(pd.read_parquet, files),
                           total=len(files),
                           desc="Loading parquet files"):
                dfs.append(df)

        self.df = pd.concat(dfs, ignore_index=True)

        # expected columns: either 'image_bytes' OR 'img_path', and a 'label' column
        if 'image_bytes' not in self.df.columns and 'img_path' not in self.df.columns:
            raise KeyError("Parquet must have either 'image_bytes' or 'img_path' column")
        if 'label' not in self.df.columns:
            raise KeyError("Parquet must have a 'label' column")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1) If image_bytes column exists, decode directly
        if 'image_bytes' in row and row.image_bytes is not None:
            img = Image.open(io.BytesIO(row.image_bytes)).convert('RGB')
        # 2) Otherwise, load from disk via img_path
        else:
            img_path = row.img_path
            full_path = os.path.join(self.image_dir, img_path)
            with open(full_path, 'rb') as f:
                img = Image.open(io.BytesIO(f.read())).convert('RGB')

        if self.transform:
            img = self.transform(img)

        label = int(row.label)
        return img, label


class ImageNetParquetDataModule(LightningDataModule):
    """LightningDataModule for ImageNet stored in Parquet+disk format."""
    def __init__(self,
                 data_dir: str,
                 batch_size: int = 64,
                 num_workers: int = 8,
                 img_size: int = 224):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size

        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]

        # same train transforms as before
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

        # same val transforms as before
        self.val_tf = transforms.Compose([
            transforms.Resize(int(img_size * 256 / 224)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    def setup(self, stage=None):
        # train
        train_parquet_dir = os.path.join(self.data_dir, "train")
        self.train_ds = ParquetImageDataset(
            parquet_dir=train_parquet_dir,
            image_dir=self.data_dir,
            transform=self.train_tf
        )

        # val
        val_parquet_dir = os.path.join(self.data_dir, "val")
        self.val_ds = ParquetImageDataset(
            parquet_dir=val_parquet_dir,
            image_dir=self.data_dir,
            transform=self.val_tf
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
