import os
import shutil
import random
import argparse
from pathlib import Path

def prepare_dataset(source_dir: Path, dest_dir: Path, split_ratio: float):
    print("="*60)
    print("YOLO11 DATASET PREPARATION & SPLIT PROCESS")
    print("="*60)
    print(f"[*] Source Directory : {source_dir.resolve()}")
    print(f"[*] Destination Dir : {dest_dir.resolve()}")
    print(f"[*] Split Ratio      : {split_ratio * 100}% Train / {round((1.0 - split_ratio) * 100)}% Validation")
    print("-"*60)

    # 1. Gather all annotated image-label pairs
    img_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    annotated_pairs = []
    
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    for filename in os.listdir(source_dir):
        ext = os.path.splitext(filename)[1].lower()
        if ext in img_extensions:
            img_path = source_dir / filename
            txt_name = os.path.splitext(filename)[0] + ".txt"
            txt_path = source_dir / txt_name
            
            # Make sure a corresponding non-empty label file exists
            if txt_path.exists() and os.path.getsize(txt_path) > 0:
                annotated_pairs.append((img_path, txt_path))
                
    if not annotated_pairs:
        raise ValueError(f"No valid image-label pairs found in {source_dir}!")

    print(f"[+] Found {len(annotated_pairs)} valid annotated image-label pairs.")

    # 2. Shuffle and Split
    random.seed(42)
    random.shuffle(annotated_pairs)
    
    split_idx = int(len(annotated_pairs) * split_ratio)
    train_pairs = annotated_pairs[:split_idx]
    val_pairs = annotated_pairs[split_idx:]
    
    # Ensure val set is not empty
    if not val_pairs and len(annotated_pairs) > 1:
        train_pairs = annotated_pairs[:-1]
        val_pairs = [annotated_pairs[-1]]

    # 3. Create destination directory structure
    for folder in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        (dest_dir / folder).mkdir(parents=True, exist_ok=True)

    # Clean existing destination files if any to prevent stale data
    for folder in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        for file in (dest_dir / folder).iterdir():
            if file.is_file():
                file.unlink()

    # 4. Copy files to train/val directories
    def copy_pairs(pairs, split_name):
        for img_path, txt_path in pairs:
            # Copy image
            shutil.copy2(img_path, dest_dir / "images" / split_name / img_path.name)
            # Copy label
            shutil.copy2(txt_path, dest_dir / "labels" / split_name / txt_path.name)

    print("[*] Copying training files...")
    copy_pairs(train_pairs, "train")
    print("[*] Copying validation files...")
    copy_pairs(val_pairs, "val")
    
    print(f"[SUCCESS] Copied {len(train_pairs)} pairs to train and {len(val_pairs)} pairs to val.")

    # 5. Generate data.yaml pointing to the Colab Drive path
    yaml_content = """# Automatically generated YOLO11 dataset configuration for Google Colab
path: /content/drive/MyDrive/google_colab_ready/dataset
train: images/train
val: images/val

# Class names matching the 2-class layout annotation configuration
nc: 2
names:
  0: diagram
  1: text
"""
    yaml_path = dest_dir / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
        
    print(f"[SUCCESS] data.yaml successfully generated at: {yaml_path}")
    print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="YOLO11 Dataset Pre-Split Tool")
    parser.add_argument("--source", type=str, default="../../dataset/model-ready", help="Source folder containing raw png + txt pairs")
    parser.add_argument("--dest", type=str, default="../dataset", help="Destination dataset folder to split files into")
    parser.add_argument("--split", type=float, default=0.80, help="Train split ratio (e.g. 0.80 for 80% train, 20% val)")
    args = parser.parse_args()

    # Make paths relative to this script directory
    script_dir = Path(__file__).parent.resolve()
    source_dir = (script_dir / args.source).resolve()
    dest_dir = (script_dir / args.dest).resolve()

    prepare_dataset(source_dir, dest_dir, args.split)

if __name__ == "__main__":
    main()
