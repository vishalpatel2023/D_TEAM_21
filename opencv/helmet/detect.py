from pathlib import Path
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "helmet_detection"
    / "weights"
    / "best.pt"
)


# Load model once
model = YOLO(str(MODEL_PATH))


def detect_helmet(image_path, conf=0.5):
    """
    Detect helmets in a single image.

    Returns:
        list[dict]: Detection information.
    """

    results = model.predict(
        source=str(image_path),
        conf=conf,
        save=True
    )

    detections = []

    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            class_name = result.names[class_id]

            violation = class_name == "Without Helmet"

            detection = {
                "class": class_name,
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2],
                "violation": violation
            }

            detections.append(detection)

    return detections


if __name__ == "__main__":

    test_image = (
        PROJECT_ROOT
        / "datasets"
        / "Bike Helmet Detection.v1i.yolov8"
        / "test"
        / "images"
        / "BikesHelmets102_png_jpg.rf.594796e5a372daa68099f1a53880ec96.jpg"
    )

    detections = detect_helmet(test_image)

    print("\nDetections:")

    for detection in detections:
        print(detection)