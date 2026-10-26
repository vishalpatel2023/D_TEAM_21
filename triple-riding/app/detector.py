"""
YOLO-based object detector for persons and motorcycles.

Loads an Ultralytics YOLO model once and provides a ``detect()``
method that returns structured ``Detection`` objects.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np

from app.config import ModelConfig
from app.schemas import Detection

logger = logging.getLogger(__name__)

# COCO class IDs that we care about
_COCO_PERSON_ID = 0
_COCO_MOTORCYCLE_ID = 3

_CLASS_MAP = {
    _COCO_PERSON_ID: "person",
    _COCO_MOTORCYCLE_ID: "motorcycle",
}


class ObjectDetector:
    """Wraps an Ultralytics YOLO model for person/motorcycle detection.

    The model is loaded once during ``__init__`` and reused for every
    subsequent call to ``detect()``.

    Parameters:
        model_config: A ``ModelConfig`` instance with model path and
            inference parameters.
    """

    def __init__(self, model_config: ModelConfig) -> None:
        self._config = model_config
        self._model = self._load_model(model_config.path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_model(model_path: str):
        """Load YOLO model, downloading it if necessary."""
        try:
            from ultralytics import YOLO  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "Ultralytics is required. Install with: pip install ultralytics"
            ) from exc

        path = Path(model_path)
        # Ultralytics handles downloading standard model names (e.g. yolov8n.pt)
        logger.info("Loading YOLO model from '%s' …", model_path)
        try:
            model = YOLO(str(path))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load YOLO model from '{model_path}': {exc}"
            ) from exc

        logger.info("YOLO model loaded successfully.")
        return model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on a single BGR frame.

        Returns a list of ``Detection`` objects filtered to only
        ``person`` and ``motorcycle`` classes.
        """
        results = self._model(
            frame,
            conf=self._config.confidence_threshold,
            iou=self._config.iou_threshold,
            imgsz=self._config.image_size,
            verbose=False,
        )

        detections: List[Detection] = []

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                cls_id = int(box.cls[0].item())
                if cls_id not in _CLASS_MAP:
                    continue

                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    Detection(
                        class_name=_CLASS_MAP[cls_id],
                        confidence=conf,
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                    )
                )

        return detections
