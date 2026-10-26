"""
Configuration management for Triple-Riding Detection.

Loads settings from config.yaml and provides defaults for all parameters.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

# Project root is the triple-riding/ directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ModelConfig:
    """YOLO model configuration."""
    path: str = "yolov8n.pt"
    confidence_threshold: float = 0.35
    iou_threshold: float = 0.45
    image_size: int = 640


@dataclass
class AssociationWeights:
    """Weights for person-motorcycle association scoring."""
    point_inside: float = 0.50
    horizontal_overlap: float = 0.25
    center_distance: float = 0.15
    vertical_relationship: float = 0.10


@dataclass
class AssociationConfig:
    """Person-motorcycle association configuration."""
    bbox_expansion_ratio: float = 0.20
    minimum_score: float = 0.35
    weights: AssociationWeights = field(default_factory=AssociationWeights)


@dataclass
class RiderCountConfig:
    """Rider counting / violation thresholds."""
    minimum_violation_riders: int = 3


@dataclass
class VideoConfig:
    """Video processing configuration."""
    frame_skip: int = 0
    output_fps: Optional[float] = None


@dataclass
class VisualizationConfig:
    """Visualization options."""
    show_association_lines: bool = True
    show_confidence: bool = True
    show_person_ids: bool = True
    show_association_score: bool = False


@dataclass
class AppConfig:
    """Top-level application configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    association: AssociationConfig = field(default_factory=AssociationConfig)
    rider_count: RiderCountConfig = field(default_factory=RiderCountConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)


def _merge_dict(target: dict, source: dict) -> dict:
    """Recursively merge *source* into *target*, returning *target*."""
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge_dict(target[key], value)
        else:
            target[key] = value
    return target


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Load configuration from a YAML file.

    Falls back to ``config.yaml`` in the project root when no path is given.
    Missing keys receive sensible defaults.
    """
    if config_path is None:
        config_path = str(PROJECT_ROOT / "config.yaml")

    cfg = AppConfig()

    if not os.path.isfile(config_path):
        logger.warning("Config file not found at %s — using defaults.", config_path)
        return cfg

    with open(config_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    # --- Model ---
    model_raw = raw.get("model", {})
    cfg.model.path = model_raw.get("path", cfg.model.path)
    cfg.model.confidence_threshold = float(
        model_raw.get("confidence_threshold", cfg.model.confidence_threshold)
    )
    cfg.model.iou_threshold = float(
        model_raw.get("iou_threshold", cfg.model.iou_threshold)
    )
    cfg.model.image_size = int(model_raw.get("image_size", cfg.model.image_size))

    # --- Association ---
    assoc_raw = raw.get("association", {})
    cfg.association.bbox_expansion_ratio = float(
        assoc_raw.get("bbox_expansion_ratio", cfg.association.bbox_expansion_ratio)
    )
    cfg.association.minimum_score = float(
        assoc_raw.get("minimum_score", cfg.association.minimum_score)
    )
    weights_raw = assoc_raw.get("weights", {})
    cfg.association.weights.point_inside = float(
        weights_raw.get("point_inside", cfg.association.weights.point_inside)
    )
    cfg.association.weights.horizontal_overlap = float(
        weights_raw.get("horizontal_overlap", cfg.association.weights.horizontal_overlap)
    )
    cfg.association.weights.center_distance = float(
        weights_raw.get("center_distance", cfg.association.weights.center_distance)
    )
    cfg.association.weights.vertical_relationship = float(
        weights_raw.get("vertical_relationship", cfg.association.weights.vertical_relationship)
    )

    # --- Rider count ---
    rc_raw = raw.get("rider_count", {})
    cfg.rider_count.minimum_violation_riders = int(
        rc_raw.get("minimum_violation_riders", cfg.rider_count.minimum_violation_riders)
    )

    # --- Video ---
    vid_raw = raw.get("video", {})
    cfg.video.frame_skip = int(vid_raw.get("frame_skip", cfg.video.frame_skip))
    fps_val = vid_raw.get("output_fps", cfg.video.output_fps)
    cfg.video.output_fps = float(fps_val) if fps_val is not None else None

    # --- Visualization ---
    vis_raw = raw.get("visualization", {})
    cfg.visualization.show_association_lines = bool(
        vis_raw.get("show_association_lines", cfg.visualization.show_association_lines)
    )
    cfg.visualization.show_confidence = bool(
        vis_raw.get("show_confidence", cfg.visualization.show_confidence)
    )
    cfg.visualization.show_person_ids = bool(
        vis_raw.get("show_person_ids", cfg.visualization.show_person_ids)
    )
    cfg.visualization.show_association_score = bool(
        vis_raw.get("show_association_score", cfg.visualization.show_association_score)
    )

    logger.info("Configuration loaded from %s", config_path)
    return cfg
