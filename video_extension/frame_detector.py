"""
Frame-level Object Detectors for Video Stream
=============================================
Runs in-memory inference directly on numpy array frames (cv2 BGR frames)
without saving intermediate image files to disk for real-time speed.
"""

from typing import Any
import numpy as np
import cv2

from video_extension.model_loader import get_model
from video_extension.config import DEFAULT_CONFIDENCES


# ── Helmet Detection on Frame ──────────────────────────────────────────────────
def detect_frame_helmet(frame: np.ndarray, conf: float = 0.30) -> list[dict[str, Any]]:
    """Detect helmets & helmet violations on a single video frame."""
    model = get_model("helmet")
    if model is None:
        return []

    results = model.predict(source=frame, conf=conf, verbose=False, save=False)
    detections = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_name = result.names.get(class_id, "unknown")
            # Class 1 is 'Without Helmet'
            violation = class_id == 1 or "without" in class_name.lower() or "no" in class_name.lower()

            detections.append({
                "module": "helmet",
                "class": class_name,
                "confidence": round(confidence, 3),
                "bbox": [x1, y1, x2, y2],
                "violation": violation,
            })

    return detections


# ── Pothole Detection on Frame ─────────────────────────────────────────────────
def detect_frame_pothole(frame: np.ndarray, conf: float = 0.25) -> list[dict[str, Any]]:
    """Detect potholes on a single video frame."""
    model = get_model("pothole")
    if model is None:
        return []

    results = model.predict(source=frame, conf=conf, verbose=False, save=False)
    detections = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_name = result.names.get(class_id, "pothole")

            detections.append({
                "module": "pothole",
                "class": class_name,
                "confidence": round(confidence, 3),
                "bbox": [x1, y1, x2, y2],
                "hazard": True,
            })

    return detections


# ── ANPR License Plate Detection on Frame ─────────────────────────────────────
def detect_frame_anpr(
    frame: np.ndarray,
    conf: float = 0.40,
    run_ocr: bool = False
) -> list[dict[str, Any]]:
    """
    Detect vehicle license plates on a single video frame.
    OCR can optionally be executed on plate crops.
    """
    model = get_model("anpr")
    detections = []

    if model is not None:
        results = model.predict(source=frame, conf=conf, verbose=False, save=False)
        for result in results:
            for box in result.boxes:
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                det = {
                    "module": "anpr",
                    "class": "license_plate",
                    "confidence": round(confidence, 3),
                    "bbox": [x1, y1, x2, y2],
                    "plate_text": "",
                }
                detections.append(det)

    return detections


# ── Triple Riding Detection on Frame ──────────────────────────────────────────
def _is_person_on_bike(person_box: list[int], bike_box: list[int]) -> bool:
    """Check spatial containment/alignment of person on two-wheeler."""
    px1, py1, px2, py2 = person_box
    bx1, by1, bx2, by2 = bike_box

    bw = bx2 - bx1
    bh = by2 - by1
    pcx = (px1 + px2) / 2.0

    # Horizontal alignment with 25% span margin
    h_aligned = (bx1 - 0.25 * bw) <= pcx <= (bx2 + 0.25 * bw)
    # Vertical alignment with upper half of bike
    v_aligned = (py2 >= by1 - 0.25 * bh) and (py1 <= by2)
    return h_aligned and v_aligned


def detect_frame_triple_riding(frame: np.ndarray, conf: float = 0.30) -> list[dict[str, Any]]:
    """
    Detect triple-riding violations on two-wheelers in a video frame.
    Recognizes the direct fine-tuned class (class 0 / 'Triple_riding')
    as well as person-on-motorcycle spatial heuristics.
    """
    model = get_model("triple_riding")
    if model is None:
        return []

    # Sensitive threshold for video frames
    infer_conf = max(0.10, min(conf * 0.7, 0.20))
    results = model.predict(source=frame, conf=infer_conf, verbose=False, save=False)

    detections = []
    persons = []
    motorcycles = []

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            c_name = str(result.names.get(cls_id, "")).lower()
            c_conf = float(box.conf[0])
            coords = list(map(int, box.xyxy[0]))

            # Direct class match: results/triple-riding/best.pt has class {0: '0'}
            is_direct_triple = (
                cls_id == 0
                or c_name in ["0", "triple_riding", "3riders", "tripleriding", "triple_rider", "overloaded"]
                or "triple" in c_name
                or "3rider" in c_name
            )

            if is_direct_triple:
                if c_conf >= conf:
                    detections.append({
                        "module": "triple_riding",
                        "class": "Triple Riding",
                        "confidence": round(c_conf, 3),
                        "bbox": coords,
                        "violation": True,
                        "rider_count": 3,
                    })
            elif "motorcycle" in c_name or "bike" in c_name or "two-wheeler" in c_name or "scooter" in c_name:
                if c_conf >= 0.12:
                    motorcycles.append({"bbox": coords, "confidence": c_conf})
            elif "person" in c_name:
                if c_conf >= 0.18:
                    persons.append({"bbox": coords, "confidence": c_conf})

    # Strategy A: Check riders on motorcycle if separate persons/bikes are predicted
    for mc in motorcycles:
        riders_on_mc = [p for p in persons if _is_person_on_bike(p["bbox"], mc["bbox"])]
        rider_count = len(riders_on_mc)

        if rider_count >= 3:
            all_boxes = [mc["bbox"]] + [r["bbox"] for r in riders_on_mc]
            comb_x1 = min(b[0] for b in all_boxes)
            comb_y1 = min(b[1] for b in all_boxes)
            comb_x2 = max(b[2] for b in all_boxes)
            comb_y2 = max(b[3] for b in all_boxes)
            c_conf = round(max(mc["confidence"], sum(r["confidence"] for r in riders_on_mc) / rider_count), 3)

            detections.append({
                "module": "triple_riding",
                "class": "Triple Riding",
                "confidence": c_conf,
                "bbox": [comb_x1, comb_y1, comb_x2, comb_y2],
                "violation": True,
                "rider_count": rider_count,
            })

    return detections


# ── Unified Scan on Frame ─────────────────────────────────────────────────────
def detect_frame_all(
    frame: np.ndarray,
    conf_dict: dict[str, float] | None = None
) -> list[dict[str, Any]]:
    """Run all 4 modules on a single video frame and aggregate detections."""
    confs = {**DEFAULT_CONFIDENCES, **(conf_dict or {})}

    all_dets = []
    all_dets.extend(detect_frame_helmet(frame, conf=confs["helmet"]))
    all_dets.extend(detect_frame_pothole(frame, conf=confs["pothole"]))
    all_dets.extend(detect_frame_anpr(frame, conf=confs["anpr"]))
    all_dets.extend(detect_frame_triple_riding(frame, conf=confs["triple_riding"]))

    return all_dets
