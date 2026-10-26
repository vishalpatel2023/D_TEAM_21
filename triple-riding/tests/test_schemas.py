"""
Unit tests for data schemas and JSON serialization.
"""

from __future__ import annotations

import json

import pytest

from app.schemas import (
    Association,
    Detection,
    FrameResult,
    MotorcycleResult,
    RiderInfo,
)


class TestDetection:
    def test_to_dict(self):
        d = Detection("person", 0.91234, (10, 20, 30, 40))
        result = d.to_dict()
        assert result["class_name"] == "person"
        assert result["confidence"] == 0.9123
        assert result["bbox"] == [10, 20, 30, 40]

    def test_json_serializable(self):
        d = Detection("motorcycle", 0.87, (100, 200, 300, 400))
        text = json.dumps(d.to_dict())
        assert isinstance(text, str)


class TestAssociation:
    def test_to_dict(self):
        a = Association(person_id=1, motorcycle_id=0, score=0.82345)
        result = a.to_dict()
        assert result["person_id"] == 1
        assert result["motorcycle_id"] == 0
        assert result["score"] == 0.8235


class TestRiderInfo:
    def test_to_dict(self):
        r = RiderInfo(person_id=2, bbox=(120, 100, 180, 300), confidence=0.94, association_score=0.88)
        result = r.to_dict()
        assert result["person_id"] == 2
        assert result["bbox"] == [120, 100, 180, 300]


class TestMotorcycleResult:
    def test_to_dict(self):
        mr = MotorcycleResult(
            motorcycle_id=0,
            bbox=(100, 200, 400, 500),
            confidence=0.89,
            rider_count=3,
            violation=True,
            riders=[
                RiderInfo(0, (120, 100, 180, 300), 0.94, 0.88),
                RiderInfo(1, (200, 110, 260, 310), 0.91, 0.85),
                RiderInfo(2, (280, 120, 340, 320), 0.88, 0.80),
            ],
        )
        result = mr.to_dict()
        assert result["motorcycle_id"] == 0
        assert result["violation"] is True
        assert result["rider_count"] == 3
        assert len(result["riders"]) == 3

    def test_json_serializable(self):
        mr = MotorcycleResult(0, (100, 200, 400, 500), 0.89, 3, True)
        text = json.dumps(mr.to_dict())
        assert isinstance(text, str)


class TestFrameResult:
    def test_default(self):
        fr = FrameResult()
        assert fr.module == "triple_riding"
        assert fr.violations == 0

    def test_to_dict_full(self):
        fr = FrameResult(
            image_width=1280,
            image_height=720,
            motorcycles_detected=2,
            persons_detected=5,
            violations=1,
            motorcycles=[
                MotorcycleResult(0, (100, 200, 400, 500), 0.89, 3, True),
                MotorcycleResult(1, (600, 200, 800, 500), 0.85, 2, False),
            ],
        )
        d = fr.to_dict()
        assert d["module"] == "triple_riding"
        assert d["image_width"] == 1280
        assert d["motorcycles_detected"] == 2
        assert d["violations"] == 1
        assert len(d["motorcycles"]) == 2

    def test_round_trip_json(self):
        fr = FrameResult(
            image_width=1920,
            image_height=1080,
            motorcycles_detected=1,
            persons_detected=3,
            violations=1,
            motorcycles=[
                MotorcycleResult(
                    0, (100, 200, 400, 500), 0.89, 3, True,
                    riders=[
                        RiderInfo(0, (120, 100, 180, 300), 0.94, 0.88),
                    ],
                ),
            ],
        )
        text = json.dumps(fr.to_dict(), indent=2)
        parsed = json.loads(text)
        assert parsed["module"] == "triple_riding"
        assert parsed["violations"] == 1
        assert parsed["motorcycles"][0]["riders"][0]["person_id"] == 0
