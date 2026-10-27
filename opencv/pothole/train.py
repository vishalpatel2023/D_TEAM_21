from ultralytics import YOLO
from pathlib import Path

# Get project root (road-safety-monitoring-system)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET = (
    PROJECT_ROOT
    / "datasets"
    / "Pothole-Detection.v1i.yolov8"
    / "data.yaml"
)

RESULTS_DIR = PROJECT_ROOT / "results"

def main():
    print("===================================")
    print("      POTHOLE DETECTION TRAINING")
    print("===================================")
    print(f"Dataset: {DATASET}")

    model = YOLO("yolov8n.pt")

    model.train(
        data=str(DATASET),
        epochs=50,
        imgsz=416,
        batch=16,
        project=str(RESULTS_DIR),
        name="pothole_detection"
    )
    print("\nTraining completed.")

if __name__ == "__main__":
    main()