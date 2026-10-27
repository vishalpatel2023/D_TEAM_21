from pathlib import Path
from ultralytics import YOLO
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "pothole_detection"
    / "weights"
    / "best.pt"
)

# Load model once
model = YOLO(str(MODEL_PATH))

def detect_pothole(image_path, conf=0.25):
    """
    Detect potholes in a single image.

    Returns:
        list[dict]: Detection information.
    """
    # Set the base project directory to runs/detect_pothole
    save_dir = PROJECT_ROOT / "runs" / "detect_pothole"

    results = model.predict(
        source=str(image_path),
        conf=conf,
        save=True,
        project=str(save_dir),
        name="predict-2",
        exist_ok=True
    )

    detections = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_name = result.names[class_id]

            detection = {
                "class": class_name,
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2],
                "hazard": True
            }

            detections.append(detection)

    return detections

if __name__ == "__main__":
    test_dir = (
        PROJECT_ROOT 
        / "datasets" 
        / "Pothole-Detection.v18i.yolov8" 
        / "Valid" 
        / "images"
    )
    
    if test_dir.exists():
        test_images = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        if test_images:
            test_image = test_dir / test_images[2]
            print(f"Running detection on: {test_image}...\n")
            
            detections = detect_pothole(test_image)
            
            print("\nDetections:")
            for detection in detections:
                print(detection)
        else:
            print(f"No images found in {test_dir}")
    else:
        print(f"Error: Could not find image directory at {test_dir}")