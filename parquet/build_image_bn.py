#!/usr/bin/env python3
import os
import glob
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Configuration
# PARQUET_DIR = '/mnt/hdfs/byte_content_security/user/liuwenzhuo/datasets/parquet/imagenet'
PARQUET_DIR = './imagenet_parquet'
OUTPUT_DIR = '/mnt/bn/liuwenzhuo-lf/datasets/imagenet2'
SPLITS = ['train', 'val']
MAX_WORKERS =  128

def write_image(record):
    """Write image bytes to disk, recreating original directory structure."""
    rel_path = record['id']
    img_bytes = record['image_bytes']
    out_path = os.path.join(OUTPUT_DIR, rel_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if img_bytes:
        with open(out_path, 'wb') as f:
            f.write(img_bytes)


def reconstruct_split(split):
    """Reconstruct images for a given split from Parquet files."""
    pattern = os.path.join(PARQUET_DIR, split, '*.parquet')
    pq_files = glob.glob(pattern)
    for pq_file in tqdm(pq_files, desc=f"{split} Parquets", unit="file"):
        df = pd.read_parquet(pq_file, engine='pyarrow')
        desc = f"Writing images from {os.path.basename(pq_file)}"
        with ThreadPoolExecutor(MAX_WORKERS) as executor:
            list(tqdm(
                executor.map(write_image, df.to_dict('records')),
                total=len(df),
                desc=desc,
                unit='img',
                leave=False
            ))


def main():
    for split in tqdm(SPLITS, desc="Splits", unit="split"):
        reconstruct_split(split)
    print("Reconstruction complete.")


if __name__ == '__main__':
    main()
