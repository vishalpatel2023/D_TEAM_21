"""
Manages thread-safe lazy singletons for all 4 YOLO models
so weights are loaded into memory once and reused across frames.
"""

import os
# Prevent Ultralytics from making blocking outbound network requests during import
os.environ["ULTRALYTICS_AUTOINSTALL"] = "0"
os.environ["YOLO_VERBOSE"] = "False"

from pathlib import Path
from ultralytics import YOLO

from video_extension.config import MODEL_PATHS

_models = {
    "helmet": None,
    "pothole": None,
    "anpr": None,
    "triple_riding": None,
}


def get_model(module_name: str):
    """
    Retrieve or lazily initialize the YOLO model for a given module.

    Args:
        module_name: One of 'helmet', 'pothole', 'anpr', 'triple_riding'.

    Returns:
        YOLO model instance or None if weights not found.
    """
    global _models

    if module_name not in _models:
        raise ValueError(f"Unknown module '{module_name}'. Valid: {list(_models.keys())}")

    if _models[module_name] is None:
        path = MODEL_PATHS.get(module_name)
        if path and path.exists():
            print(f"[VideoExtension] Loading {module_name} model from {path.name}...")
            _models[module_name] = YOLO(str(path))
        else:
            print(f"[VideoExtension] Warning: Weight file not found for '{module_name}' at {path}")
            _models[module_name] = None

    return _models[module_name]


def preload_models(modules: list[str] | None = None):
    """Preload specified models into memory before video streaming."""
    targets = modules or list(_models.keys())
    for mod in targets:
        get_model(mod)
