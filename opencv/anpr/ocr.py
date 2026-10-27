"""
EasyOCR wrapper for license plate text extraction.

The Reader is initialised once (lazy singleton) to avoid the costly
model-load on every request.
"""

from pathlib import Path
import re

import easyocr
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OCR_MODEL_DIR = PROJECT_ROOT / "results" / "anpr" / "ocr_models"

# ── Lazy singleton ────────────────────────────────────────────────────────────
_reader: easyocr.Reader | None = None


def get_reader(languages: list[str] | None = None) -> easyocr.Reader:
    """Return the shared EasyOCR Reader, creating it on first call."""
    global _reader
    if _reader is None:
        OCR_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_dir = str(OCR_MODEL_DIR)
        download_needed = not (OCR_MODEL_DIR / "craft_mlt_25k.pth").exists()
        _reader = easyocr.Reader(
            languages or ["en"],
            gpu=False,
            model_storage_directory=model_dir,
            user_network_directory=model_dir,
            download_enabled=download_needed,
            verbose=False
        )
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
