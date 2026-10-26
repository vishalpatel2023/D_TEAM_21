"""
Unit tests for rider counting and violation classification.
"""

from __future__ import annotations

import pytest

from app.config import RiderCountConfig
from app.rider_counter import build_motorcycle_results
from app.schemas import Association, Detection

IMG_W, IMG_H = 1280, 720


def _person(x1, y1, x2, y2, conf=0.90):
    return Detection("person", conf, (x1, y1, x2, y2))


def _moto(x1, y1, x2, y2, conf=0.85):
    return Detection("motorcycle", conf, (x1, y1, x2, y2))


class TestBuildMotorcycleResults:

    # Test 1 — 1 motorcycle, 1 rider → no violation
    def test_one_rider_no_violation(self):
        persons = [_person(150, 100, 250, 350)]
        motos = [_moto(100, 200, 400, 500)]
        assocs = [Association(person_id=0, motorcycle_id=0, score=0.8)]

        result = build_motorcycle_results(
            persons, motos, assocs, IMG_W, IMG_H, RiderCountConfig()
        )
        assert result.motorcycles[0].rider_count == 1
        assert result.motorcycles[0].violation is False
        assert result.violations == 0

    # Test 2 — 1 motorcycle, 2 riders → no violation
    def test_two_riders_no_violation(self):
        persons = [_person(150, 100, 250, 350), _person(300, 100, 400, 350)]
        motos = [_moto(100, 200, 500, 500)]
        assocs = [
            Association(person_id=0, motorcycle_id=0, score=0.8),
            Association(person_id=1, motorcycle_id=0, score=0.7),
        ]
        result = build_motorcycle_results(
            persons, motos, assocs, IMG_W, IMG_H, RiderCountConfig()
        )
        assert result.motorcycles[0].rider_count == 2
        assert result.motorcycles[0].violation is False
        assert result.violations == 0

    # Test 3 — 1 motorcycle, 3 riders → violation
    def test_three_riders_violation(self):
        persons = [
            _person(150, 100, 250, 350),
            _person(300, 100, 400, 350),
            _person(450, 100, 550, 350),
        ]
        motos = [_moto(100, 200, 600, 500)]
        assocs = [
            Association(person_id=0, motorcycle_id=0, score=0.8),
            Association(person_id=1, motorcycle_id=0, score=0.7),
            Association(person_id=2, motorcycle_id=0, score=0.6),
        ]
        result = build_motorcycle_results(
            persons, motos, assocs, IMG_W, IMG_H, RiderCountConfig()
        )
        assert result.motorcycles[0].rider_count == 3
        assert result.motorcycles[0].violation is True
        assert result.violations == 1

    # Test 4 — 1 motorcycle, 4 riders → violation
    def test_four_riders_violation(self):
        persons = [
            _person(150, 100, 250, 350),
            _person(300, 100, 400, 350),
            _person(450, 100, 550, 350),
            _person(600, 100, 700, 350),
        ]
        motos = [_moto(100, 200, 750, 500)]
        assocs = [
            Association(person_id=i, motorcycle_id=0, score=0.8 - i * 0.1)
            for i in range(4)
        ]
        result = build_motorcycle_results(
            persons, motos, assocs, IMG_W, IMG_H, RiderCountConfig()
        )
        assert result.motorcycles[0].rider_count == 4
        assert result.motorcycles[0].violation is True

    # Test — configurable minimum violation riders
    def test_custom_minimum_riders(self):
        persons = [_person(150, 100, 250, 350), _person(300, 100, 400, 350)]
        motos = [_moto(100, 200, 500, 500)]
        assocs = [
            Association(person_id=0, motorcycle_id=0, score=0.8),
            Association(person_id=1, motorcycle_id=0, score=0.7),
        ]
        # With minimum_violation_riders=2, two riders IS a violation
        config = RiderCountConfig(minimum_violation_riders=2)
        result = build_motorcycle_results(
            persons, motos, assocs, IMG_W, IMG_H, config
        )
        assert result.motorcycles[0].violation is True

    # Test 10 — empty detections → no crash
    def test_empty_detections(self):
        result = build_motorcycle_results([], [], [], IMG_W, IMG_H, RiderCountConfig())
        assert result.motorcycles_detected == 0
        assert result.persons_detected == 0
        assert result.violations == 0
        assert result.motorcycles == []

    # Test — motorcycle with no riders
    def test_motorcycle_no_riders(self):
        motos = [_moto(100, 200, 400, 500)]
        result = build_motorcycle_results([], motos, [], IMG_W, IMG_H, RiderCountConfig())
        assert result.motorcycles[0].rider_count == 0
        assert result.motorcycles[0].violation is False

    # Test — 2 motorcycles with different rider counts
    def test_two_motorcycles_mixed(self):
        persons = [
            _person(150, 100, 250, 350),
            _person(300, 100, 400, 350),
            _person(700, 100, 800, 350),
            _person(850, 100, 950, 350),
            _person(1000, 100, 1100, 350),
        ]
        motos = [
            _moto(100, 200, 500, 500),
            _moto(650, 200, 1150, 500),
        ]
        assocs = [
            Association(person_id=0, motorcycle_id=0, score=0.8),
            Association(person_id=1, motorcycle_id=0, score=0.7),
            Association(person_id=2, motorcycle_id=1, score=0.8),
            Association(person_id=3, motorcycle_id=1, score=0.7),
            Association(person_id=4, motorcycle_id=1, score=0.6),
        ]
        result = build_motorcycle_results(
            persons, motos, assocs, IMG_W, IMG_H, RiderCountConfig()
        )
        assert result.motorcycles[0].rider_count == 2
        assert result.motorcycles[0].violation is False
        assert result.motorcycles[1].rider_count == 3
        assert result.motorcycles[1].violation is True
        assert result.violations == 1

    # Test — result metadata
    def test_result_metadata(self):
        persons = [_person(150, 100, 250, 350)]
        motos = [_moto(100, 200, 400, 500)]
        assocs = [Association(person_id=0, motorcycle_id=0, score=0.8)]
        result = build_motorcycle_results(
            persons, motos, assocs, IMG_W, IMG_H, RiderCountConfig()
        )
        assert result.image_width == IMG_W
        assert result.image_height == IMG_H
        assert result.motorcycles_detected == 1
        assert result.persons_detected == 1
        assert result.module == "triple_riding"
