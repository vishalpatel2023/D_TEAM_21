from ultralytics import YOLO
from pathlib import Path


# Get project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset configuration
DATASET = (
    PROJECT_ROOT
    / "datasets"
    / "Bike Helmet Detection.v1i.yolov8"
    / "data.yaml"
)

# Training output directory
RESULTS_DIR = PROJECT_ROOT / "results"


def main():
    print("===================================")
    print("      HELMET DETECTION TRAINING")
    print("===================================")
    print(f"Dataset: {DATASET}")

    # Load pretrained YOLOv8 Nano
    model = YOLO("yolov8n.pt")

    # Train model
    model.train(
        data=str(DATASET),
        epochs=50,
        imgsz=416,
        batch=16,
        project=str(RESULTS_DIR),
        name="helmet_detection"
    )

    print("\nTraining completed.")


if __name__ == "__main__":
    main()