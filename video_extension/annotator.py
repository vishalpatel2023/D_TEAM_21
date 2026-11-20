"""
Renders high-visibility bounding boxes, label pills, violation badges,
and an informational HUD header directly onto video frames.
"""

from typing import Any
import cv2
import numpy as np

from video_extension.config import COLORS


def get_detection_color(detection: dict[str, Any]) -> tuple[int, int, int]:
    """Determine bounding box color based on module and violation status."""
    module = detection.get("module", "")
    cls_name = str(detection.get("class", "")).lower()

    if detection.get("violation", False) or "without helmet" in cls_name or "triple" in cls_name:
        return COLORS["danger"]
    if module == "pothole" or detection.get("hazard", False):
        return COLORS["hazard"]
    if module == "anpr" or "plate" in cls_name:
        return COLORS["plate"]
    if "with helmet" in cls_name:
        return COLORS["safe"]

    return COLORS["safe"]


def annotate_frame(
    frame: np.ndarray,
    detections: list[dict[str, Any]],
    frame_idx: int = 0,
    fps: float = 30.0,
    total_frames: int = 0,
    draw_hud: bool = True
) -> np.ndarray:
    """
    Draw bounding boxes and status HUD onto the given video frame.

    Args:
        frame: BGR image numpy array.
        detections: List of detection dictionaries.
        frame_idx: Current frame index.
        fps: Video framerate.
        total_frames: Total number of frames in the video.
        draw_hud: Whether to render the top informational overlay bar.

    Returns:
        Annotated BGR frame numpy array.
    """
    output = frame.copy()
    h, w = output.shape[:2]

    # Count violations on this specific frame
    violations_count = sum(1 for d in detections if d.get("violation", False))
    hazards_count = sum(1 for d in detections if d.get("hazard", False))

    # 1. Draw each detection bounding box
    for d in detections:
        bbox = d.get("bbox", [])
        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        color = get_detection_color(d)
        thickness = 2

        # Draw main rectangle
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)

        # Build label text
        cls_name = d.get("class", "object")
        conf = d.get("confidence", 0.0)
        label = f"{cls_name} {int(conf * 100)}%"
        if d.get("violation", False):
            label = f"[VIOLATION] {label}"
        elif d.get("hazard", False):
            label = f"[HAZARD] {label}"

        # Draw label background pill
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        font_thick = 1
        (txt_w, txt_h), baseline = cv2.getTextSize(label, font, font_scale, font_thick)

        label_y1 = max(0, y1 - txt_h - 8)
        label_y2 = y1
        label_x2 = min(w, x1 + txt_w + 10)

        # Background pill
        cv2.rectangle(output, (x1, label_y1), (label_x2, label_y2), color, -1)
        # White text inside pill
        cv2.putText(
            output,
            label,
            (x1 + 5, y1 - 4),
            font,
            font_scale,
            COLORS["white"],
            font_thick,
            cv2.LINE_AA,
        )

    # 2. Draw Top HUD Dashboard Banner
    if draw_hud:
        hud_h = 36
        # Semi-transparent top bar
        overlay = output.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_h), COLORS["banner_bg"], -1)
        cv2.addWeighted(overlay, 0.75, output, 0.25, 0, output)

        # HUD Text
        current_sec = round(frame_idx / max(1.0, fps), 2)
        total_sec = round(total_frames / max(1.0, fps), 2)
        time_str = f"Time: {current_sec:.1f}s / {total_sec:.1f}s | Frame: {frame_idx}/{total_frames}"

        cv2.putText(
            output,
            "ROAD SAFETY AI MONITOR",
            (12, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            COLORS["white"],
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            output,
            time_str,
            (w // 2 - 120, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        # Violation / Hazard alert indicator
        if violations_count > 0:
            alert_text = f"ALERT: {violations_count} VIOLATION"
            cv2.putText(
                output,
                alert_text,
                (w - 220, 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                COLORS["danger"],
                2,
                cv2.LINE_AA,
            )
        elif hazards_count > 0:
            hazard_text = f"CAUTION: {hazards_count} HAZARD"
            cv2.putText(
                output,
                hazard_text,
                (w - 220, 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                COLORS["hazard"],
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                output,
                "STATUS: NORMAL",
                (w - 180, 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                COLORS["safe"],
                1,
                cv2.LINE_AA,
            )

    return output
