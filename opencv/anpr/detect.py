"""
ANPR Detection Pipeline
=======================

Workflow for each image:
    1. Detect license plate bounding boxes  → YOLO (primary) or Haar cascade (fallback)
    2. Crop each plate region from the frame
    3. Preprocess the crop with OpenCV      → preprocess.py
    4. Extract text via EasyOCR             → ocr.py
    5. Annotate the original image          → preprocess.annotate_image
    6. Return detections + base64 annotated image

Usage (standalone):
    python -m opencv.anpr.detect <image_path>
"""

import base64
import sys
from pathlib import Path
from typing import Union

import cv2
import numpy as np

# ── Project paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "results" / "anpr" / "weights" / "best.pt"

# OpenCV's built-in Russian/EU plate Haar cascade used as fallback
HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"

# ── Lazy model singleton ──────────────────────────────────────────────────────
_model = None


def _get_model():
    global _model
    if _model is None:
        if MODEL_PATH.exists():
            from ultralytics import YOLO
            print(f"[ANPR] Loading YOLO model from {MODEL_PATH}")
            _model = YOLO(str(MODEL_PATH))
        else:
            print(
                f"[ANPR] WARNING: YOLO model not found at {MODEL_PATH}.\n"
                "       Falling back to OpenCV Haar Cascade.\n"
                "       Run 'python -m opencv.anpr.train' to train a model."
            )
            _model = "haar"
    return _model


# ── Detection helpers ─────────────────────────────────────────────────────────

def _detect_yolo(image: np.ndarray, model, conf: float) -> list[dict]:
    """Run YOLO inference and return raw bounding boxes."""
    results = model.predict(source=image, conf=conf, verbose=False)
    boxes = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append({
                "bbox":       [x1, y1, x2, y2],
                "confidence": float(box.conf[0]),
            })
    return boxes


def _detect_haar(image: np.ndarray) -> list[dict]:
    """Run Haar cascade and return raw bounding boxes (confidence always 1.0)."""
    cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plates  = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 20)
    )
    boxes = []
    if len(plates):
        for (x, y, w, h) in plates:
            boxes.append({
                "bbox":       [int(x), int(y), int(x + w), int(y + h)],
                "confidence": 1.0,
            })
    return boxes


# ── Main pipeline ─────────────────────────────────────────────────────────────

def detect_plates(
    image_input: Union[str, Path, np.ndarray],
    conf: float = 0.4,
) -> dict:
    """
    Full ANPR pipeline: detect → preprocess → OCR → annotate.

    Args:
        image_input: File path (str/Path) or BGR numpy array.
        conf:        YOLO detection confidence threshold (ignored for Haar).

    Returns:
        dict:
            detections           — list of detection dicts
            annotated_image_b64  — base64-encoded JPEG of annotated image
            total_plates         — number of plates found
    """
    # Lazy imports to avoid circular import at module level
    from opencv.anpr.preprocess import preprocess_plate, annotate_image
    from opencv.anpr.ocr import extract_plate_text

    # ── Load image ────────────────────────────────────────────────────────────
    if isinstance(image_input, (str, Path)):
        image = cv2.imread(str(image_input))
        if image is None:
            raise ValueError(f"Could not read image: {image_input}")
    else:
        image = image_input

    img_h, img_w = image.shape[:2]

    # ── Step 1: Plate detection ───────────────────────────────────────────────
    model    = _get_model()
    raw_boxes = _detect_yolo(image, model, conf) if model != "haar" else _detect_haar(image)

    # ── Steps 2-4: Crop → Preprocess → OCR ───────────────────────────────────
    detections = []

    for raw in raw_boxes:
        x1, y1, x2, y2 = raw["bbox"]

        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        plate_crop = image[y1:y2, x1:x2]

        preprocessed = preprocess_plate(plate_crop)
        ocr_result   = extract_plate_text(preprocessed)

        detections.append({
            "bbox":           [x1, y1, x2, y2],
            "confidence":     round(raw["confidence"], 4),
            "plate_text":     ocr_result["text"],
            "raw_ocr_text":   ocr_result["raw_text"],
            "ocr_confidence": round(ocr_result["ocr_confidence"], 4),
        })

    # ── Step 5: Annotate image ────────────────────────────────────────────────
    annotated = annotate_image(image, detections)

    _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    annotated_b64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "detections":          detections,
        "annotated_image_b64": annotated_b64,
        "total_plates":        len(detections),
    }


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m opencv.anpr.detect <image_path> [conf]")
        sys.exit(1)

    img_path = sys.argv[1]
    conf_arg = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4

    result = detect_plates(img_path, conf=conf_arg)
    print(f"\nFound {result['total_plates']} plate(s):")
    for i, d in enumerate(result["detections"], 1):
        print(
            f"  #{i}  Text: {d['plate_text'] or '(none)':20s}  "
            f"Det: {d['confidence']:.2f}  OCR: {d['ocr_confidence']:.2f}  "
            f"BBox: {d['bbox']}"
        )
