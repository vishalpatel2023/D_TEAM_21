"""
Data schemas for Triple-Riding Detection results.

All result types are plain dataclasses that are JSON-serializable
via the ``to_dict()`` helper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """A single object detection from YOLO.

    Attributes:
        class_name: ``"person"`` or ``"motorcycle"``.
        confidence: Detection confidence in ``[0, 1]``.
        bbox: ``(x1, y1, x2, y2)`` pixel coordinates.
    """
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]

    def to_dict(self) -> dict:
        return {
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": list(self.bbox),
        }


# ---------------------------------------------------------------------------
# Association
# ---------------------------------------------------------------------------

@dataclass
class Association:
    """Describes a person→motorcycle link.

    Attributes:
        person_id: Index of the person in the persons list.
        motorcycle_id: Index of the motorcycle in the motorcycles list.
        score: Association confidence in ``[0, 1]``.
    """
    person_id: int
    motorcycle_id: int
    score: float

    def to_dict(self) -> dict:
        return {
            "person_id": self.person_id,
            "motorcycle_id": self.motorcycle_id,
            "score": round(self.score, 4),
        }


# ---------------------------------------------------------------------------
# Rider info
# ---------------------------------------------------------------------------

@dataclass
class RiderInfo:
    """Information about a single rider associated with a motorcycle."""
    person_id: int
    bbox: Tuple[int, int, int, int]
    confidence: float
    association_score: float

    def to_dict(self) -> dict:
        return {
            "person_id": self.person_id,
            "bbox": list(self.bbox),
            "confidence": round(self.confidence, 4),
            "association_score": round(self.association_score, 4),
        }


# ---------------------------------------------------------------------------
# Motorcycle result
# ---------------------------------------------------------------------------

@dataclass
class MotorcycleResult:
    """Result for one motorcycle including rider info and violation status."""
    motorcycle_id: int
    bbox: Tuple[int, int, int, int]
    confidence: float
    rider_count: int
    violation: bool
    riders: List[RiderInfo] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "motorcycle_id": self.motorcycle_id,
            "bbox": list(self.bbox),
            "confidence": round(self.confidence, 4),
            "rider_count": self.rider_count,
            "violation": self.violation,
            "riders": [r.to_dict() for r in self.riders],
        }


# ---------------------------------------------------------------------------
# Frame / image result
# ---------------------------------------------------------------------------

@dataclass
class FrameResult:
    """Complete result for a single image or video frame."""
    module: str = "triple_riding"
    image_width: int = 0
    image_height: int = 0
    motorcycles_detected: int = 0
    persons_detected: int = 0
    violations: int = 0
    motorcycles: List[MotorcycleResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "module": self.module,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "motorcycles_detected": self.motorcycles_detected,
            "persons_detected": self.persons_detected,
            "violations": self.violations,
            "motorcycles": [m.to_dict() for m in self.motorcycles],
        }
