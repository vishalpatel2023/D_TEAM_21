"""
OpenCV-based result visualization.

Draws bounding boxes, rider counts, violation labels, and optional
association lines on an image/frame for presentation-quality output.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from app.config import VisualizationConfig
from app.schemas import FrameResult

# ── Colour palette (BGR) ──────────────────────────────────────────────
_COLOUR_NORMAL_MOTO = (0, 200, 80)       # green
_COLOUR_VIOLATION_MOTO = (0, 0, 230)     # red
_COLOUR_PERSON = (255, 180, 0)           # cyan-blue
_COLOUR_ASSOCIATION_LINE = (0, 220, 255)  # yellow
_COLOUR_TEXT_BG = (30, 30, 30)            # dark grey
_COLOUR_TEXT_FG = (255, 255, 255)         # white
_COLOUR_VIOLATION_TEXT = (60, 60, 255)    # bright red

# ── Drawing helpers ───────────────────────────────────────────────────

def _scale_thickness(image_height: int) -> int:
    """Return line thickness appropriate for the image resolution."""
    if image_height >= 1080:
        return 3
    if image_height >= 720:
        return 2
    return 1


def _scale_font(image_height: int) -> float:
    """Return font scale appropriate for the image resolution."""
    if image_height >= 1080:
        return 0.8
    if image_height >= 720:
        return 0.6
    return 0.45


def _draw_label(
    image: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    fg_colour: Tuple[int, int, int] = _COLOUR_TEXT_FG,
    bg_colour: Tuple[int, int, int] = _COLOUR_TEXT_BG,
    font_scale: float = 0.6,
    thickness: int = 1,
) -> None:
    """Draw *text* at *origin* with a filled background rectangle."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = origin
    # Ensure the label stays inside the image
    y = max(th + baseline + 4, y)
    x = max(0, x)

    cv2.rectangle(
        image,
        (x, y - th - baseline - 4),
        (x + tw + 6, y + 2),
        bg_colour,
        cv2.FILLED,
    )
    cv2.putText(
        image, text, (x + 3, y - baseline - 2),
        font, font_scale, fg_colour, thickness, cv2.LINE_AA,
    )


# ── Public API ────────────────────────────────────────────────────────

def draw_results(
    image: np.ndarray,
    result: FrameResult,
    vis_config: VisualizationConfig,
) -> np.ndarray:
    """Annotate *image* with detection and violation information.

    Parameters:
        image: BGR image (will be modified **in-place** and returned).
        result: ``FrameResult`` from the processing pipeline.
        vis_config: Visualization options.

    Returns:
        The annotated image (same array as *image*).
    """
    h, w = image.shape[:2]
    thick = _scale_thickness(h)
    font_scale = _scale_font(h)

    for moto in result.motorcycles:
        mx1, my1, mx2, my2 = moto.bbox
        is_violation = moto.violation
        moto_colour = _COLOUR_VIOLATION_MOTO if is_violation else _COLOUR_NORMAL_MOTO

        # --- Motorcycle box ---
        cv2.rectangle(image, (mx1, my1), (mx2, my2), moto_colour, thick)

        # --- Primary label ---
        if is_violation:
            label = f"TRIPLE RIDING | Riders: {moto.rider_count}"
            fg = _COLOUR_VIOLATION_TEXT
        else:
            label = f"Motorcycle | Riders: {moto.rider_count}"
            fg = _COLOUR_TEXT_FG

        if vis_config.show_confidence:
            label += f" | {moto.confidence * 100:.0f}%"

        _draw_label(image, label, (mx1, my1 - 4), fg_colour=fg,
                    font_scale=font_scale, thickness=thick)

        # --- Riders ---
        for rider in moto.riders:
            rx1, ry1, rx2, ry2 = rider.bbox
            cv2.rectangle(image, (rx1, ry1), (rx2, ry2), _COLOUR_PERSON, thick)

            # Rider annotation
            parts = []
            if vis_config.show_person_ids:
                parts.append(f"P{rider.person_id}")
            if vis_config.show_association_score:
                parts.append(f"Sc:{rider.association_score:.2f}")
            if vis_config.show_confidence:
                parts.append(f"{rider.confidence * 100:.0f}%")
            if parts:
                _draw_label(image, " ".join(parts), (rx1, ry1 - 4),
                            font_scale=font_scale * 0.85, thickness=max(1, thick - 1))

            # --- Association line ---
            if vis_config.show_association_lines:
                person_bottom = (int((rx1 + rx2) / 2), ry2)
                moto_centre = (int((mx1 + mx2) / 2), int((my1 + my2) / 2))
                cv2.line(image, person_bottom, moto_centre,
                         _COLOUR_ASSOCIATION_LINE, max(1, thick - 1), cv2.LINE_AA)

    # --- Summary overlay (top-left) ---
    summary = (
        f"Motorcycles: {result.motorcycles_detected}  "
        f"Persons: {result.persons_detected}  "
        f"Violations: {result.violations}"
    )
    _draw_label(image, summary, (8, 28), font_scale=font_scale, thickness=thick)

    return image
