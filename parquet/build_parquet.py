import os
import glob
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import math

# -------- Configuration --------
DATA_DIR    = '/mnt/bn/liuwenzhuo-hl-data/datasets/imagenet'         # Root directory containing train/ and val/ subfolders
OUTPUT_DIR  = '~/imagenet_parquet'   # Where to write the .parquet files
SPLITS      = ['train', 'val']
COMPRESSION = 'snappy'
MAX_WORKERS = 32  # Adjust based on your machine’s CPU cores
DEBUG_LIMIT = 10000  # Max rows per Parquet file
# --------------------------------

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
    patterns = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
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

    # Convert to DataFrame
    df = pd.DataFrame(records)

    # Create a subdirectory per split
    split_out_dir = os.path.join(OUTPUT_DIR, split)
    os.makedirs(split_out_dir, exist_ok=True)

    # Calculate number of parts
    total_rows = len(df)
    num_parts  = math.ceil(total_rows / DEBUG_LIMIT)
    print(f"[{split}] Splitting {total_rows} records into {num_parts} files (up to {DEBUG_LIMIT} rows each)")

    # Write each chunk
    for part in range(num_parts):
        start = part * DEBUG_LIMIT
        end   = min(start + DEBUG_LIMIT, total_rows)
        chunk = df.iloc[start:end]

        # Output path: OUTPUT_DIR/split/part{part}.parquet
        output_path = os.path.join(
            split_out_dir,
            f"part{part}.parquet"
        )

        chunk.to_parquet(
            output_path,
            engine='pyarrow',
            compression=COMPRESSION,
            index=False
        )
        print(f"[{split}] Wrote rows {start}-{end-1} to {output_path}")


def main():
    for split in SPLITS:
        build_parquet_for_split(split)


if __name__ == '__main__':
    main()
