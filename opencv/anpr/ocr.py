"""
EasyOCR wrapper for license plate text extraction.

The Reader is initialised once (lazy singleton) to avoid the costly
model-load on every request.
"""

from pathlib import Path
import re

try:
    import easyocr
except ImportError:
    easyocr = None
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OCR_MODEL_DIR = PROJECT_ROOT / "results" / "anpr" / "ocr_models"

# ── Lazy singleton ────────────────────────────────────────────────────────────
_reader = None


def get_reader(languages: list[str] | None = None):
    """Return the shared EasyOCR Reader, creating it on first call."""
    global _reader
    if easyocr is None:
        return None
    if _reader is None:
        OCR_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_dir = str(OCR_MODEL_DIR)
        download_needed = not (OCR_MODEL_DIR / "craft_mlt_25k.pth").exists()
        _reader = easyocr.Reader(
            languages or ["en"],
            gpu=False,
            model_storage_directory=model_dir,
            user_network_directory=model_dir,
            download_enabled=True,
            verbose=False
        )
    return _reader


# ── Main function ─────────────────────────────────────────────────────────────
def extract_plate_text(plate_img: np.ndarray) -> dict:
    """
    Extract text from a preprocessed license plate image.
    Supports both single-line and two-line (motorcycle/scooter) Indian plates.

    Args:
        plate_img: Preprocessed image (grayscale or BGR numpy array).

    Returns:
        dict with:
            text           — cleaned plate string  (uppercase alphanumeric)
            raw_text       — raw concatenated OCR output
            ocr_confidence — average confidence across detected text segments
    """
    reader = get_reader()
    if reader is None:
        return {"text": "", "raw_text": "", "ocr_confidence": 0.0}

    results = reader.readtext(plate_img, detail=1, paragraph=False)
    if not results:
        return {"text": "", "raw_text": "", "ocr_confidence": 0.0}

    # Extract bounding box polygons, filter out tiny noise (e.g. screws, IND logo)
    valid_boxes = []
    for r in results:
        poly = np.array(r[0])
        min_x = float(np.min(poly[:, 0]))
        min_y = float(np.min(poly[:, 1]))
        max_y = float(np.max(poly[:, 1]))
        box_h = max_y - min_y
        raw_chunk = str(r[1]).strip()
        conf_val = float(r[2])

        # Strip symbols and keep alphanumeric
        clean_chunk = re.sub(r"[^A-Z0-9]", "", raw_chunk.upper())
        if len(clean_chunk) >= 1 and conf_val >= 0.15:
            valid_boxes.append({
                "min_x": min_x,
                "min_y": min_y,
                "box_h": box_h,
                "text": clean_chunk,
                "raw": raw_chunk,
                "conf": conf_val,
            })

    if not valid_boxes:
        return {"text": "", "raw_text": "", "ocr_confidence": 0.0}

    # Sort boxes vertically into lines (Row 1 on top, Row 2 on bottom)
    valid_boxes.sort(key=lambda b: b["min_y"])
    lines = []
    current_line = []
    current_y = None

    for item in valid_boxes:
        # If box is within 25px vertically, it belongs to the same horizontal line
        if current_y is None or abs(item["min_y"] - current_y) <= 25:
            current_line.append(item)
            current_y = item["min_y"] if current_y is None else (current_y + item["min_y"]) / 2.0
        else:
            current_line.sort(key=lambda b: b["min_x"])
            lines.append(current_line)
            current_line = [item]
            current_y = item["min_y"]

    if current_line:
        current_line.sort(key=lambda b: b["min_x"])
        lines.append(current_line)

    # Join lines: Line 1 (State/District) then Line 2 (Series/Number)
    line_strings = []
    all_confs = []
    raw_parts = []

    for line in lines:
        line_str = " ".join(item["text"] for item in line)
        line_strings.append(line_str)
        for item in line:
            all_confs.append(item["conf"])
            raw_parts.append(item["raw"])

    full_plate_text = " ".join(line_strings).strip()
    avg_conf = float(np.mean(all_confs)) if all_confs else 0.0

    return {
        "text": full_plate_text,
        "raw_text": " ".join(raw_parts),
        "ocr_confidence": avg_conf,
    }
    results = reader.readtext(plate_img, detail=1, paragraph=False)

    if not results:
        return {"text": "", "raw_text": "", "ocr_confidence": 0.0}

    # Sort text boxes left-to-right by their top-left x coordinate
    results_sorted = sorted(results, key=lambda r: r[0][0][0])

    raw_parts   = [r[1] for r in results_sorted]
    confidences = [float(r[2]) for r in results_sorted]

    raw_text       = " ".join(raw_parts)
    avg_confidence = float(np.mean(confidences)) if confidences else 0.0

    # Clean: keep A-Z, 0-9 and spaces only
    cleaned = re.sub(r"[^A-Z0-9\s]", "", raw_text.upper())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return {
        "text":           cleaned,
        "raw_text":       raw_text,
        "ocr_confidence": avg_confidence,
    }
