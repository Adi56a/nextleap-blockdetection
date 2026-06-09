import os
import sys
import argparse
import shutil
from pathlib import Path

def check_gpu():
    """Verify GPU availability and device placement."""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"[+] CUDA GPU detected: {device_name}")
            return True
        else:
            print("[!] WARNING: No GPU detected. YOLO training on CPU will be extremely slow.")
            return False
    except ImportError:
        print("[!] PyTorch is not installed. Unable to run GPU diagnostics.")
        return False

def main():
    parser = argparse.ArgumentParser(description="YOLO11 Training CLI for Google Colab")
    parser.add_argument("--model", type=str, default="yolo11s.pt", help="Path to model weights or pre-trained name (e.g. yolo11s.pt, models/yolo11s_doc_layout.pt)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=-1, help="Batch size (-1 for auto-batching based on GPU VRAM)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training (default: 640)")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience epochs")
    parser.add_argument("--device", type=str, default="0", help="GPU device ID or 'cpu' (default: 0)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    # Resolve paths relative to script
    script_dir = Path(__file__).parent.resolve()
    package_dir = script_dir.parent
    
    # Path to data.yaml
    yaml_path = package_dir / "dataset" / "data.yaml"
    if not yaml_path.exists():
        print(f"[ERROR] data.yaml configuration file not found at: {yaml_path}")
        print("Please run scripts/prepare_dataset.py first or fix paths.")
        sys.exit(1)

    print("="*60)
    print("🤖 YOLO11 CLOUD TRAINING RUNNER INITIATING")
    print("="*60)
    print(f"[*] Training configuration: {yaml_path.resolve()}")
    print(f"[*] Base Model Weights     : {args.model}")
    print(f"[*] Epochs                 : {args.epochs}")
    print(f"[*] Batch Size             : {args.batch} (Autobatching if -1)")
    print(f"[*] Image Size             : {args.imgsz}")
    print(f"[*] Workers                : {args.workers}")
    print(f"[*] Patience               : {args.patience}")
    print(f"[*] Target Device          : {args.device}")
    print("-"*60)

    # Verify CUDA GPU if device == "0"
    if args.device == "0":
        check_gpu()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Ultralytics package is not installed. Please run: pip install ultralytics")
        sys.exit(1)

    # Paths for checkpoints
    checkpoint_dir = package_dir / "runs" / "detect" / "train" / "weights"
    last_checkpoint = checkpoint_dir / "last.pt"

    try:
        if args.resume:
            if not last_checkpoint.exists():
                print(f"[!] Warning: No previous checkpoint weights found at {last_checkpoint}.")
                print("[!] Starting a new training run instead...")
                model = YOLO(args.model)
                model.train(
                    data=str(yaml_path),
                    epochs=args.epochs,
                    batch=args.batch,
                    imgsz=args.imgsz,
                    device=args.device,
                    cache=True,
                    workers=args.workers,
                    patience=args.patience,
                    exist_ok=True
                )
            else:
                print(f"[+] Resuming training from checkpoint: {last_checkpoint}")
                model = YOLO(str(last_checkpoint))
                model.train(resume=True)
        else:
            # Check if using local model weight file or pre-trained string
            model_weight_path = args.model
            if not os.path.exists(model_weight_path) and (package_dir / model_weight_path).exists():
                model_weight_path = str(package_dir / model_weight_path)

            print(f"[+] Loading weights: {model_weight_path}")
            model = YOLO(model_weight_path)
            
            # Start training
            model.train(
                data=str(yaml_path),
                epochs=args.epochs,
                batch=args.batch,
                imgsz=args.imgsz,
                device=args.device,
                cache=True,
                workers=args.workers,
                patience=args.patience,
                exist_ok=True
            )

        print("\n" + "="*60)
        print("[SUCCESS] TRAINING COMPLETED SUCCESSFULLY!")
        
        # Copy best weights to models directory as custom_best.pt
        best_path = checkpoint_dir / "best.pt"
        if best_path.exists():
            models_dir = package_dir / "models"
            models_dir.mkdir(exist_ok=True)
            custom_best_path = models_dir / "custom_best.pt"
            shutil.copy2(best_path, custom_best_path)
            print(f"[SUCCESS] Copied best weights to: {custom_best_path}")
            
            # Export best model
            print("[+] Exporting fine-tuned model to ONNX format...")
            best_model = YOLO(str(custom_best_path))
            onnx_path = best_model.export(format="onnx")
            print(f"[SUCCESS] Exported ONNX model to: {onnx_path}")
            
        print("="*60 + "\n")

    except Exception as e:
        print(f"\n[ERROR] Training execution failed: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
