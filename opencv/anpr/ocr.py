"""
EasyOCR wrapper for license plate text extraction.

The Reader is initialised once (lazy singleton) to avoid the costly
model-load on every request.
"""

import re

import easyocr
import numpy as np

# ── Lazy singleton ────────────────────────────────────────────────────────────
_reader: easyocr.Reader | None = None


def get_reader(languages: list[str] | None = None) -> easyocr.Reader:
    """Return the shared EasyOCR Reader, creating it on first call."""
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(languages or ["en"], gpu=False)
    return _reader


# ── Main function ─────────────────────────────────────────────────────────────
def extract_plate_text(plate_img: np.ndarray) -> dict:
    """
    Extract text from a preprocessed license plate image.

    Args:
        plate_img: Preprocessed image (grayscale or BGR numpy array).

    Returns:
        dict with:
            text           — cleaned plate string  (uppercase alphanumeric)
            raw_text       — raw concatenated OCR output
            ocr_confidence — average confidence across detected text segments
    """
    reader  = get_reader()
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
