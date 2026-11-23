"""
OpenCV preprocessing utilities for license plate images.

Pipeline applied to each cropped plate region:
    1. Resize to standard height (preserving aspect ratio)
    2. Convert to grayscale
    3. Fast non-local means denoising
    4. Adaptive Gaussian thresholding
    5. Morphological closing (cleans small gaps in characters)
"""

import cv2
import numpy as np


def preprocess_plate(plate_img: np.ndarray) -> np.ndarray:
    """
    Apply fast OpenCV preprocessing pipeline to a license plate crop
    to maximize EasyOCR accuracy on both single-line and two-line plates.

    Optimized for speed and character clarity (uses CLAHE instead of slow CPU denoising).

    Args:
        plate_img: BGR image (numpy array) — raw crop from the original frame.

    Returns:
        Enhanced grayscale image ready for OCR.
    """
    h, w = plate_img.shape[:2]
    if h == 0 or w == 0:
        return plate_img

    # --- Step 1: Resize to high-clarity height (140px) -------------------
    # 64px is too small for two-row plates; 140px preserves clean letter strokes
    TARGET_HEIGHT = 140
    aspect = w / h
    new_w = max(int(TARGET_HEIGHT * aspect), 1)
    resized = cv2.resize(plate_img, (new_w, TARGET_HEIGHT), interpolation=cv2.INTER_CUBIC)

    # --- Step 2: Convert to Grayscale ------------------------------------
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # --- Step 3: Fast Edge-preserving Bilateral Filter -------------------
    # Fast smoothing (0.01s) without blurring sharp character edges
    filtered = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    # --- Step 4: CLAHE (Contrast Limited Adaptive Histogram Equalization) -
    # Increases contrast between embossed characters and white/yellow plate background
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(filtered)

    return enhanced


def annotate_image(image: np.ndarray, detections: list) -> np.ndarray:
    """
    Draw bounding boxes and plate text labels on the original image.

    Args:
        image:      BGR image (numpy array).
        detections: List of detection dicts (each has bbox, plate_text, confidence).

    Returns:
        Annotated BGR image (copy — original not modified).
    """
    annotated = image.copy()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        plate_text  = det.get("plate_text", "")
        confidence  = det.get("confidence", 0.0)

        # Bounding box colour: green if text extracted, amber otherwise
        color = (0, 200, 80) if plate_text else (0, 180, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label text
        label = f"{plate_text}  {confidence:.2f}" if plate_text else f"Plate  {confidence:.2f}"

        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness  = 2

        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Background rectangle for label
        label_y1 = max(y1 - text_h - baseline - 6, 0)
        cv2.rectangle(annotated, (x1, label_y1), (x1 + text_w + 6, y1), color, cv2.FILLED)

        # Label text (dark foreground for readability)
        cv2.putText(
            annotated,
            label,
            (x1 + 3, y1 - baseline - 2),
            font,
            font_scale,
            (10, 10, 10),
            thickness,
            cv2.LINE_AA,
        )

    return annotated
