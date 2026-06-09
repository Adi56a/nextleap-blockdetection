import os
import argparse
from pathlib import Path
from PIL import Image

def verify_dataset(dataset_dir: Path):
    print("="*60)
    print("YOLO11 DATASET INTEGRITY VERIFICATION REPORT")
    print("="*60)
    print(f"[*] Scanning Dataset Folder: {dataset_dir.resolve()}")
    print("-"*60)

    splits = ['train', 'val']
    img_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    
    total_errors = 0
    total_warnings = 0
    
    report = {
        "train": {"images": 0, "labels": 0, "missing_labels": [], "missing_images": [], "empty_labels": [], "invalid_coords": [], "invalid_classes": [], "corrupt_images": []},
        "val": {"images": 0, "labels": 0, "missing_labels": [], "missing_images": [], "empty_labels": [], "invalid_coords": [], "invalid_classes": [], "corrupt_images": []}
    }

    for split in splits:
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split
        
        if not img_dir.exists():
            print(f"[ERROR] Directory does not exist: {img_dir}")
            total_errors += 1
            continue
        if not lbl_dir.exists():
            print(f"[ERROR] Directory does not exist: {lbl_dir}")
            total_errors += 1
            continue

        # Get all files
        img_files = {os.path.splitext(f)[0]: f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in img_extensions}
        lbl_files = {os.path.splitext(f)[0]: f for f in os.listdir(lbl_dir) if os.path.splitext(f)[1].lower() == '.txt'}
        
        report[split]["images"] = len(img_files)
        report[split]["labels"] = len(lbl_files)

        # 1. Check for missing labels
        for name in img_files:
            if name not in lbl_files:
                report[split]["missing_labels"].append(img_files[name])
                total_errors += 1
                
        # 2. Check for missing images
        for name in lbl_files:
            if name not in img_files:
                report[split]["missing_images"].append(lbl_files[name])
                total_errors += 1

        # 3. Verify images and labels contents
        for name, filename in img_files.items():
            img_path = img_dir / filename
            
            # Check image corruption
            try:
                with Image.open(img_path) as img:
                    img.verify()  # Check file structure
            except Exception as e:
                report[split]["corrupt_images"].append(filename)
                total_errors += 1

            # Check label content (if label exists)
            if name in lbl_files:
                lbl_path = lbl_dir / lbl_files[name]
                
                # Check empty label file
                if os.path.getsize(lbl_path) == 0:
                    report[split]["empty_labels"].append(lbl_files[name])
                    total_warnings += 1
                    continue
                
                try:
                    with open(lbl_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    
                    for line_idx, line in enumerate(lines, 1):
                        parts = line.strip().split()
                        if not parts:
                            continue
                        
                        if len(parts) != 5:
                            report[split]["invalid_coords"].append(f"{lbl_files[name]} (Line {line_idx}: Expected 5 fields, found {len(parts)})")
                            total_errors += 1
                            continue
                        
                        # Validate class id
                        try:
                            class_id = int(parts[0])
                            if class_id not in [0, 1]:  # Expecting 0 (diagram) or 1 (text)
                                report[split]["invalid_classes"].append(f"{lbl_files[name]} (Line {line_idx}: Class ID {class_id} is not 0 or 1)")
                                total_errors += 1
                        except ValueError:
                            report[split]["invalid_classes"].append(f"{lbl_files[name]} (Line {line_idx}: Non-integer class ID '{parts[0]}')")
                            total_errors += 1
                        
                        # Validate bounding box coordinates
                        coords_ok = True
                        for i, val_str in enumerate(parts[1:], 1):
                            try:
                                val = float(val_str)
                                if not (0.0 <= val <= 1.0):
                                    coords_ok = False
                            except ValueError:
                                coords_ok = False
                        
                        if not coords_ok:
                            report[split]["invalid_coords"].append(f"{lbl_files[name]} (Line {line_idx}: Box values '{' '.join(parts[1:])}' must be floats in [0.0, 1.0])")
                            total_errors += 1

                except Exception as e:
                    report[split]["invalid_coords"].append(f"{lbl_files[name]} (Failed to read: {e})")
                    total_errors += 1

    # Print Report Output
    print("[+] SCAN SUMMARY:")
    for split in splits:
        print(f"\n--- Split: {split.upper()} ---")
        print(f"    Images found         : {report[split]['images']}")
        print(f"    Labels found         : {report[split]['labels']}")
        print(f"    Missing labels (err) : {len(report[split]['missing_labels'])}")
        print(f"    Missing images (err) : {len(report[split]['missing_images'])}")
        print(f"    Empty labels (warn)  : {len(report[split]['empty_labels'])}")
        print(f"    Corrupt images (err) : {len(report[split]['corrupt_images'])}")
        print(f"    Invalid coords (err) : {len(report[split]['invalid_coords'])}")
        print(f"    Invalid classes (err): {len(report[split]['invalid_classes'])}")

    print("\n" + "-"*60)
    print(f"Scan finished. Total errors: {total_errors}, Total warnings: {total_warnings}")
    print("-"*60)

    # Detailed Error Reporting
    if total_errors > 0 or total_warnings > 0:
        print("\n[!] DETAILED DETECTED ISSUES:")
        for split in splits:
            if report[split]["missing_labels"]:
                print(f"  [{split}] Missing labels for images:")
                for item in report[split]["missing_labels"]:
                    print(f"    - {item}")
            if report[split]["missing_images"]:
                print(f"  [{split}] Missing images for labels:")
                for item in report[split]["missing_images"]:
                    print(f"    - {item}")
            if report[split]["empty_labels"]:
                print(f"  [{split}] Warning: Empty label files (background images):")
                for item in report[split]["empty_labels"]:
                    print(f"    - {item}")
            if report[split]["corrupt_images"]:
                print(f"  [{split}] Corrupt/unreadable images:")
                for item in report[split]["corrupt_images"]:
                    print(f"    - {item}")
            if report[split]["invalid_coords"]:
                print(f"  [{split}] Coordinate syntax/value errors:")
                for item in report[split]["invalid_coords"]:
                    print(f"    - {item}")
            if report[split]["invalid_classes"]:
                print(f"  [{split}] Class ID out-of-range/type errors:")
                for item in report[split]["invalid_classes"]:
                    print(f"    - {item}")

    print("="*60)
    if total_errors > 0:
        print("[FAIL] Dataset verification failed. Please resolve errors before training.")
        return False
    else:
        print("[SUCCESS] Dataset verification passed. Dataset is ready for YOLO11 training!")
        return True

def main():
    parser = argparse.ArgumentParser(description="YOLO11 Dataset Integrity Scanner")
    parser.add_argument("--dir", type=str, default="../dataset", help="Dataset root directory containing images/ and labels/")
    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()
    dataset_dir = (script_dir / args.dir).resolve()
    
    success = verify_dataset(dataset_dir)
    import sys
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
