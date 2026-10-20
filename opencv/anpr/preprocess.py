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
    Apply OpenCV preprocessing pipeline to a license plate crop
    to maximise EasyOCR accuracy.

    Args:
        plate_img: BGR image (numpy array) — raw crop from the original frame.

    Returns:
        Preprocessed grayscale image ready for OCR.
    """

    # --- Step 1: Resize to standard height --------------------------------
    TARGET_HEIGHT = 64
    h, w = plate_img.shape[:2]
    if h == 0 or w == 0:
        return plate_img

    aspect = w / h
    new_w = max(int(TARGET_HEIGHT * aspect), 1)
    resized = cv2.resize(plate_img, (new_w, TARGET_HEIGHT), interpolation=cv2.INTER_CUBIC)

    # --- Step 2: Grayscale -------------------------------------------------
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # --- Step 3: Denoise ---------------------------------------------------
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    # --- Step 4: Adaptive threshold ----------------------------------------
    thresh = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2,
    )

    # --- Step 5: Morphological closing ------------------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return cleaned


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
