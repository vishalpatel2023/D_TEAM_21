"""
ANPR Detection Pipeline
=======================

Workflow for each image:
    1. Detect license plate bounding boxes  → YOLO (primary) or contour fallback
    2. Crop each plate region from the frame
    3. Preprocess the crop with OpenCV      → preprocess.py
    4. Extract text via EasyOCR             → ocr.py
    5. Annotate the original image          → preprocess.annotate_image
    6. Return detections + base64 annotated image

Usage (standalone):
    python opencv/anpr/detect.py <image_path>
"""

import base64
import io
import sys

# Fix Windows cp1252 terminal — EasyOCR's progress bar uses Unicode block chars
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
from pathlib import Path
from typing import Union

import cv2
import numpy as np

# ── Ensure project root is on sys.path so sibling packages resolve ────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from opencv.anpr.preprocess import preprocess_plate, annotate_image
from opencv.anpr.ocr import extract_plate_text

# ── Project paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = _PROJECT_ROOT
MODEL_PATH   = PROJECT_ROOT / "results" / "anpr" / "weights" / "best.pt"

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
                f"[ANPR] WARNING: No YOLO model found at:\n"
                f"         {MODEL_PATH}\n"
                "       Using contour-based fallback detector.\n"
                "       Run 'python opencv/anpr/train.py' to train a model."
            )
            _model = "contour"
    return _model


# ── Detection helpers ─────────────────────────────────────────────────────────

def _detect_yolo(image: np.ndarray, model, conf: float) -> list:
    """Run YOLO inference and return raw bounding boxes."""
    results = model.predict(source=image, conf=conf, verbose=False)
    boxes = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append({"bbox": [x1, y1, x2, y2], "confidence": float(box.conf[0])})
    return boxes


def _detect_contour(image: np.ndarray) -> list:
    """
    Contour-based license plate region detector (OpenCV 5 compatible).
    Finds rectangular regions with aspect ratio matching typical plates (2:1 - 6:1).
    """
    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.bilateralFilter(gray, 11, 17, 17)
    edges   = cv2.Canny(blurred, 30, 200)

    # Dilate to connect nearby edges
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Sort by area descending — plates are usually among largest rectangles
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

    img_h, img_w = image.shape[:2]
    img_area     = img_h * img_w
    boxes        = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h == 0:
            continue
        aspect = w / h
        area   = w * h

        # Typical plate aspect ratio: 2.0 - 6.5  |  area: 0.5% - 20% of image
        if 2.0 <= aspect <= 6.5 and 0.005 * img_area <= area <= 0.20 * img_area:
            boxes.append({
                "bbox":       [int(x), int(y), int(x + w), int(y + h)],
                "confidence": 1.0,
            })

    return _nms(boxes, iou_threshold=0.4)


def _nms(boxes: list, iou_threshold: float = 0.4) -> list:
    """
    Non-Maximum Suppression — removes overlapping duplicate bounding boxes.
    Keeps the box with the largest area when IoU with another box exceeds threshold.
    """
    if len(boxes) <= 1:
        return boxes

    # Sort by area descending (largest = most likely the true plate)
    boxes = sorted(boxes, key=lambda b: (b["bbox"][2]-b["bbox"][0]) * (b["bbox"][3]-b["bbox"][1]), reverse=True)

    kept = []
    for box in boxes:
        x1, y1, x2, y2 = box["bbox"]
        dominated = False
        for kept_box in kept:
            kx1, ky1, kx2, ky2 = kept_box["bbox"]
            # Intersection
            ix1, iy1 = max(x1, kx1), max(y1, ky1)
            ix2, iy2 = min(x2, kx2), min(y2, ky2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter == 0:
                continue
            # Union
            area_a = (x2 - x1) * (y2 - y1)
            area_b = (kx2 - kx1) * (ky2 - ky1)
            iou    = inter / (area_a + area_b - inter)
            if iou > iou_threshold:
                dominated = True
                break
        if not dominated:
            kept.append(box)

    return kept


# ── Main pipeline ─────────────────────────────────────────────────────────────

def detect_plates(
    image_input: Union[str, Path, np.ndarray],
    conf: float = 0.4,
) -> dict:
    """
    Full ANPR pipeline: detect → preprocess → OCR → annotate.

    Args:
        image_input: File path (str/Path) or BGR numpy array.
        conf:        YOLO confidence threshold (ignored for contour fallback).

    Returns:
        dict with keys: detections, annotated_image_b64, total_plates
    """
    # ── Load image ────────────────────────────────────────────────────────────
    if isinstance(image_input, (str, Path)):
        image = cv2.imread(str(image_input))
        if image is None:
            raise ValueError(f"Could not read image: {image_input}")
    else:
        image = image_input

    img_h, img_w = image.shape[:2]

    # ── Step 1: Plate detection ───────────────────────────────────────────────
    model     = _get_model()
    raw_boxes = _detect_yolo(image, model, conf) if model != "contour" else _detect_contour(image)

    # ── Steps 2-4: Crop → Preprocess → OCR ───────────────────────────────────
    detections = []
    for raw in raw_boxes:
        x1, y1, x2, y2 = raw["bbox"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        plate_crop   = image[y1:y2, x1:x2]
        preprocessed = preprocess_plate(plate_crop)
        ocr_result   = extract_plate_text(preprocessed)

        detections.append({
            "bbox":           [x1, y1, x2, y2],
            "confidence":     round(raw["confidence"], 4),
            "plate_text":     ocr_result["text"],
            "raw_ocr_text":   ocr_result["raw_text"],
            "ocr_confidence": round(ocr_result["ocr_confidence"], 4),
        })

    # ── Step 5: Annotate + encode ─────────────────────────────────────────────
    annotated     = annotate_image(image, detections)
    _, buffer     = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    annotated_b64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "detections":          detections,
        "annotated_image_b64": annotated_b64,
        "total_plates":        len(detections),
    }


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python opencv/anpr/detect.py <image_path> [conf]")
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

    # Save annotated image for easy inspection in VS Code
    if result.get("annotated_image_b64"):
        out_dir = PROJECT_ROOT / "runs" / "anpr"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "output.jpg"
        img_bytes = base64.b64decode(result["annotated_image_b64"])
        with open(out_path, "wb") as f:
            f.write(img_bytes)
        print(f"\nAnnotated image saved to: {out_path}")

