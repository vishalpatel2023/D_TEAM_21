"""
Training script for Triple Riding Violation Detection using YOLOv8.
(Pre-trained weights are already available at results/triple-riding/best.pt)
"""

from pathlib import Path
from ultralytics import YOLO

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Path to triple riding dataset configuration
DATASET_YAML = (
    PROJECT_ROOT
    / "datasets"
    / "triple riding dataset.v1i.darknet"
    / "data.yaml"
)

# Output directory for training results
RESULTS_DIR = PROJECT_ROOT / "results"


def train_triple_riding(epochs=50, imgsz=640, batch=16):
    """
    Train YOLOv8 model for Triple Riding detection.
    Note: Do not execute unless you wish to retrain the existing best.pt model.
    """
    print("=======================================")
    print("   TRIPLE RIDING DETECTION TRAINING   ")
    print("=======================================")
    print(f"Dataset path: {DATASET_YAML}")

    # Initialize base model (YOLOv8s recommended for multi-person detection)
    model = YOLO("yolov8s.pt")

    # Start training
    model.train(
        data=str(DATASET_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(RESULTS_DIR),
        name="triple-riding",
        exist_ok=True,
    )

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    # Dummy placeholder - pre-trained weights already exist in results/triple-riding/best.pt
    print("Pre-trained model already exists at: results/triple-riding/best.pt")
    print("To retrain, uncomment train_triple_riding() call.")
    # train_triple_riding()

