"""
High-level image and video processing pipeline.

Orchestrates detection → association → counting → visualization.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from app.association import associate_persons_to_motorcycles
from app.config import AppConfig
from app.detector import ObjectDetector
from app.rider_counter import build_motorcycle_results
from app.schemas import Detection, FrameResult
from app.visualization import draw_results

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}


def _split_detections(
    detections: List[Detection],
) -> tuple[List[Detection], List[Detection]]:
    """Separate detections into persons and motorcycles."""
    persons = [d for d in detections if d.class_name == "person"]
    motorcycles = [d for d in detections if d.class_name == "motorcycle"]
    return persons, motorcycles


def _ensure_output_dir(output_path: str) -> None:
    """Create the parent directory of *output_path* if it doesn't exist."""
    parent = Path(output_path).parent
    parent.mkdir(parents=True, exist_ok=True)


class TripleRidingProcessor:
    """End-to-end triple-riding detection processor.

    Holds a single ``ObjectDetector`` instance and the configuration
    needed for association, counting, and visualization.

    Parameters:
        config: Application configuration.
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._detector = ObjectDetector(config.model)

    # ------------------------------------------------------------------
    # Single-frame pipeline
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> FrameResult:
        """Run the complete pipeline on a single BGR frame.

        Returns:
            A ``FrameResult`` with all motorcycle/rider/violation data.
        """
        h, w = frame.shape[:2]

        detections = self._detector.detect(frame)
        persons, motorcycles = _split_detections(detections)

        logger.debug(
            "Detected %d person(s) and %d motorcycle(s).",
            len(persons), len(motorcycles),
        )

        associations = associate_persons_to_motorcycles(
            persons, motorcycles, w, h, self._config.association,
        )

        result = build_motorcycle_results(
            persons, motorcycles, associations, w, h,
            self._config.rider_count,
        )

        return result

    # ------------------------------------------------------------------
    # Image
    # ------------------------------------------------------------------

    def process_image(
        self,
        input_path: str,
        output_path: Optional[str] = None,
    ) -> FrameResult:
        """Process a single image file.

        Parameters:
            input_path: Path to the source image.
            output_path: Where to save the annotated image (optional).

        Returns:
            Structured ``FrameResult``.

        Raises:
            FileNotFoundError: If *input_path* does not exist.
            ValueError: If the file cannot be read as an image.
        """
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Image not found: {input_path}")

        frame = cv2.imread(input_path)
        if frame is None:
            raise ValueError(f"Failed to read image (corrupt or unsupported): {input_path}")

        logger.info("Processing image: %s", input_path)
        result = self.process_frame(frame)

        violations = result.violations
        logger.info(
            "Detected %d person(s), %d motorcycle(s), %d violation(s).",
            result.persons_detected, result.motorcycles_detected, violations,
        )

        if output_path:
            _ensure_output_dir(output_path)
            annotated = draw_results(frame, result, self._config.visualization)
            cv2.imwrite(output_path, annotated)
            logger.info("Saved annotated image to %s", output_path)

        return result

    # ------------------------------------------------------------------
    # Video
    # ------------------------------------------------------------------

    def process_video(
        self,
        input_path: str,
        output_path: Optional[str] = None,
    ) -> List[FrameResult]:
        """Process a video file frame-by-frame.

        Parameters:
            input_path: Path to the source video.
            output_path: Where to save the annotated video (optional).

        Returns:
            List of ``FrameResult`` (one per processed frame).

        Raises:
            FileNotFoundError: If *input_path* does not exist.
            ValueError: If the video cannot be opened.
        """
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Video not found: {input_path}")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_fps = self._config.video.output_fps or fps
        frame_skip = self._config.video.frame_skip

        writer: Optional[cv2.VideoWriter] = None
        if output_path:
            _ensure_output_dir(output_path)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, out_fps, (width, height))
            if not writer.isOpened():
                cap.release()
                raise ValueError(f"Failed to create video writer for: {output_path}")

        logger.info(
            "Processing video: %s (%d frames, %.1f FPS, %dx%d)",
            input_path, total_frames, fps, width, height,
        )

        results: List[FrameResult] = []
        frame_idx = 0
        processed_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Frame skipping
                if frame_skip > 0 and (frame_idx % (frame_skip + 1)) != 0:
                    # Write un-annotated frame to maintain video continuity
                    if writer:
                        writer.write(frame)
                    frame_idx += 1
                    continue

                result = self.process_frame(frame)
                results.append(result)
                processed_count += 1

                if writer:
                    annotated = draw_results(frame, result, self._config.visualization)
                    writer.write(annotated)

                # Log progress periodically (every 100 processed frames)
                if processed_count % 100 == 0:
                    logger.info(
                        "Processed %d / %d frames …", frame_idx + 1, total_frames,
                    )

                frame_idx += 1

        finally:
            cap.release()
            if writer:
                writer.release()

        total_violations = sum(r.violations for r in results)
        logger.info(
            "Video processing complete: %d frames processed, %d total violation(s).",
            processed_count, total_violations,
        )

        if output_path:
            logger.info("Saved annotated video to %s", output_path)

        return results
