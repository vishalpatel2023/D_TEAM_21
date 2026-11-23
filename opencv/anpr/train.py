"""
YOLOv8 Training Script for License Plate Detection
===================================================

Dataset expected at:
    datasets/ANPR.v5i.yolov8/data.yaml
    or datasets/License Plate Detection.v1i.yolov8/data.yaml

Trained weights saved to:
    results/anpr/weights/best.pt

Usage:
    python -m opencv.anpr.train
"""

from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Check for new high-volume dataset first
DATASET = PROJECT_ROOT / "datasets" / "ANPR.v5i.yolov8" / "data.yaml"
if not DATASET.exists():
    DATASET = PROJECT_ROOT / "datasets" / "License Plate Detection.v1i.yolov8" / "data.yaml"

RESULTS_DIR = PROJECT_ROOT / "results"


def main():
    print("=" * 45)
    print("      ANPR — LICENSE PLATE DETECTION TRAINING")
    print("=" * 45)
    print(f"Dataset : {DATASET}")
    print(f"Output  : {RESULTS_DIR / 'anpr'}")
    print()

    if not DATASET.exists():
        print("ERROR: Dataset not found. Please ensure data.yaml exists at:")
        print(f"       {DATASET}")
        return

    # Load pretrained YOLOv8 Nano
    model = YOLO("yolov8n.pt")

    # Train with augmentations for Indian road angles
    model.train(
        data=str(DATASET),
        epochs=50,
        imgsz=640,
        batch=16,
        mosaic=1.0,
        degrees=10.0,
        scale=0.5,
        project=str(RESULTS_DIR),
        name="anpr",
        exist_ok=True,
    )

    print("\nTraining completed.")
    print(f"Weights saved to: {RESULTS_DIR / 'anpr' / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
