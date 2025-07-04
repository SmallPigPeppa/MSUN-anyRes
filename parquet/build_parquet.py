import os
import glob
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# -------- Configuration --------
DATA_DIR    = '/mnt/bn/liuwenzhuo-hl-data/datasets'         # Root directory containing train/ and val/ subfolders
OUTPUT_DIR  = '~/imagenet_parquet'   # Where to write the .parquet files
SPLITS      = ['train', 'val']
COMPRESSION = 'snappy'
MAX_WORKERS = 64   # Adjust based on your machine’s CPU cores
# -------------------------------

def process_image(path_label):
    """
    Load one image file and return a record dict.
    path_label: tuple (full_path, split, label)
    """
    full_path, split, label = path_label
    try:
        with open(full_path, 'rb') as f:
            img_bytes = f.read()
    except Exception:
        img_bytes = None

    # Use the path relative to DATA_DIR as the unique id
    rel_path = os.path.relpath(full_path, DATA_DIR).replace(os.sep, '/')
    return {
        'id': rel_path,
        'split': split,
        'label': label,
        'image_bytes': img_bytes,
    }

def build_parquet_for_split(split: str):
    split_dir = os.path.join(DATA_DIR, split)
    # Gather all JPEG/PNG files under each class folder
    patterns = ['*.jpg', '*.jpeg', '*.png']
    all_paths = []
    for cls in os.listdir(split_dir):
        cls_dir = os.path.join(split_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        for pat in patterns:
            all_paths.extend(glob.glob(os.path.join(cls_dir, pat)))

    print(f"[{split}] Found {len(all_paths)} images. Starting processing...")

    # Prepare work items: (file_path, split_name, class_label)
    work_items = [
        (path, split, os.path.basename(os.path.dirname(path)))
        for path in all_paths
    ]

    # Process in parallel
    records = []
    with ThreadPoolExecutor(MAX_WORKERS) as pool:
        for rec in tqdm(pool.map(process_image, work_items),
                        total=len(work_items),
                        desc=f"Processing {split}"):
            records.append(rec)

    # Convert to DataFrame and write to Parquet
    df = pd.DataFrame(records)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{split}.parquet")
    df.to_parquet(output_path, engine='pyarrow', compression=COMPRESSION, index=False)
    print(f"[{split}] Wrote {len(df)} records to {output_path}")

def main():
    for split in SPLITS:
        build_parquet_for_split(split)

if __name__ == '__main__':
    main()
