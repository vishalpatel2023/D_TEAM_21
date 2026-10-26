"""
Unit tests for the association module.

All tests use synthetic bounding boxes — no YOLO model is required.
This keeps the tests fast, deterministic, and reproducible.
"""

from __future__ import annotations

import pytest

from app.association import (
    associate_persons_to_motorcycles,
    compute_association_score,
    expand_bbox,
    get_person_reference_point,
)
from app.config import AssociationConfig, AssociationWeights
from app.schemas import Detection

# ── Helpers ───────────────────────────────────────────────────────────

IMG_W, IMG_H = 1280, 720

DEFAULT_WEIGHTS = AssociationWeights()
DEFAULT_CONFIG = AssociationConfig(
    bbox_expansion_ratio=0.20,
    minimum_score=0.35,
    weights=DEFAULT_WEIGHTS,
)


def _person(x1: int, y1: int, x2: int, y2: int, conf: float = 0.90) -> Detection:
    return Detection(class_name="person", confidence=conf, bbox=(x1, y1, x2, y2))


def _moto(x1: int, y1: int, x2: int, y2: int, conf: float = 0.85) -> Detection:
    return Detection(class_name="motorcycle", confidence=conf, bbox=(x1, y1, x2, y2))


# =====================================================================
# get_person_reference_point
# =====================================================================

class TestGetPersonReferencePoint:
    def test_basic(self):
        pt = get_person_reference_point((100, 50, 200, 300))
        assert pt == (150.0, 300.0)

    def test_single_pixel(self):
        pt = get_person_reference_point((0, 0, 0, 0))
        assert pt == (0.0, 0.0)

    def test_wide_box(self):
        pt = get_person_reference_point((10, 20, 110, 220))
        assert pt == (60.0, 220.0)


# =====================================================================
# expand_bbox
# =====================================================================

class TestExpandBbox:
    def test_normal_expansion(self):
        result = expand_bbox((100, 100, 300, 300), 1280, 720, 0.20)
        # Width 200 → expand by 40 each side
        # Height 200 → expand by 40 each side
        assert result == (60, 60, 340, 340)

    def test_clamped_to_image_top_left(self):
        result = expand_bbox((10, 10, 100, 100), 1280, 720, 0.50)
        # Width 90 → dx=45, height 90 → dy=45
        assert result[0] == 0  # clamped
        assert result[1] == 0  # clamped

    def test_clamped_to_image_bottom_right(self):
        result = expand_bbox((1200, 650, 1280, 720), 1280, 720, 0.50)
        assert result[2] == 1280  # clamped
        assert result[3] == 720   # clamped

    def test_zero_expansion(self):
        bbox = (100, 100, 300, 300)
        assert expand_bbox(bbox, 1280, 720, 0.0) == bbox

    def test_stays_inside_image(self):
        """Test 9 — bbox near image boundary remains valid after expansion."""
        result = expand_bbox((0, 0, 50, 50), 100, 100, 0.50)
        x1, y1, x2, y2 = result
        assert x1 >= 0 and y1 >= 0
        assert x2 <= 100 and y2 <= 100


# =====================================================================
# Association scoring
# =====================================================================

class TestComputeAssociationScore:
    def test_person_fully_on_motorcycle(self):
        """Person directly on top of motorcycle → high score."""
        person_bbox = (150, 100, 250, 350)
        moto_bbox = (100, 200, 400, 500)
        expanded = expand_bbox(moto_bbox, IMG_W, IMG_H, 0.20)
        score = compute_association_score(person_bbox, moto_bbox, expanded, DEFAULT_CONFIG)
        assert score > 0.6

    def test_person_far_away(self):
        """Person far from motorcycle → low / zero score."""
        person_bbox = (900, 100, 950, 200)
        moto_bbox = (100, 400, 300, 600)
        expanded = expand_bbox(moto_bbox, IMG_W, IMG_H, 0.20)
        score = compute_association_score(person_bbox, moto_bbox, expanded, DEFAULT_CONFIG)
        assert score < DEFAULT_CONFIG.minimum_score


# =====================================================================
# Full association pipeline
# =====================================================================

class TestAssociatePersonsToMotorcycles:

    # Test 1 — 1 motorcycle + 1 rider
    def test_one_moto_one_rider(self):
        persons = [_person(150, 100, 250, 350)]
        motos = [_moto(100, 200, 400, 500)]
        assocs = associate_persons_to_motorcycles(persons, motos, IMG_W, IMG_H, DEFAULT_CONFIG)
        assert len(assocs) == 1
        assert assocs[0].motorcycle_id == 0
        assert assocs[0].person_id == 0

    # Test 2 — 1 motorcycle + 2 riders
    def test_one_moto_two_riders(self):
        motos = [_moto(200, 300, 500, 550)]
        persons = [
            _person(220, 100, 300, 400),
            _person(350, 120, 430, 410),
        ]
        assocs = associate_persons_to_motorcycles(persons, motos, IMG_W, IMG_H, DEFAULT_CONFIG)
        assert len(assocs) == 2
        assert all(a.motorcycle_id == 0 for a in assocs)

    # Test 3 — 1 motorcycle + 3 riders (triple-riding)
    def test_one_moto_three_riders(self):
        motos = [_moto(200, 300, 600, 550)]
        persons = [
            _person(220, 100, 310, 420),
            _person(340, 110, 430, 430),
            _person(450, 120, 540, 440),
        ]
        assocs = associate_persons_to_motorcycles(persons, motos, IMG_W, IMG_H, DEFAULT_CONFIG)
        assert len(assocs) == 3
        assert all(a.motorcycle_id == 0 for a in assocs)

    # Test 4 — 1 motorcycle + 4 riders
    def test_one_moto_four_riders(self):
        motos = [_moto(150, 300, 650, 550)]
        persons = [
            _person(170, 100, 260, 420),
            _person(280, 110, 370, 430),
            _person(390, 120, 480, 440),
            _person(500, 130, 590, 450),
        ]
        assocs = associate_persons_to_motorcycles(persons, motos, IMG_W, IMG_H, DEFAULT_CONFIG)
        assert len(assocs) == 4

    # Test 5 — unrelated pedestrian far from motorcycle
    def test_unrelated_pedestrian(self):
        motos = [_moto(100, 400, 300, 550)]
        persons = [
            _person(120, 200, 200, 420),  # rider on motorcycle
            _person(900, 300, 950, 500),  # pedestrian far away
        ]
        assocs = associate_persons_to_motorcycles(persons, motos, IMG_W, IMG_H, DEFAULT_CONFIG)
        # The pedestrian should NOT be associated
        moto_0_riders = [a for a in assocs if a.motorcycle_id == 0]
        person_ids = {a.person_id for a in moto_0_riders}
        assert 1 not in person_ids  # pedestrian index 1 not associated

    # Test 6 — 2 motorcycles + 4 persons
    def test_two_motos_four_persons(self):
        motos = [
            _moto(50, 400, 250, 550),    # motorcycle A (left)
            _moto(700, 400, 900, 550),   # motorcycle B (right)
        ]
        persons = [
            _person(80, 200, 160, 420),   # rider on A
            _person(160, 210, 240, 430),  # rider on A
            _person(730, 200, 810, 420),  # rider on B
            _person(810, 210, 890, 430),  # rider on B
        ]
        assocs = associate_persons_to_motorcycles(persons, motos, IMG_W, IMG_H, DEFAULT_CONFIG)
        a_riders = {a.person_id for a in assocs if a.motorcycle_id == 0}
        b_riders = {a.person_id for a in assocs if a.motorcycle_id == 1}
        assert a_riders == {0, 1}
        assert b_riders == {2, 3}

    # Test 7 — 1 person close to 2 motorcycles → assigned to best only
    def test_person_near_two_motos(self):
        motos = [
            _moto(200, 400, 400, 550),   # motorcycle A
            _moto(350, 400, 550, 550),   # motorcycle B (overlapping)
        ]
        persons = [_person(310, 200, 380, 440)]  # between both
        assocs = associate_persons_to_motorcycles(persons, motos, IMG_W, IMG_H, DEFAULT_CONFIG)
        assert len(assocs) <= 1  # person assigned to at most one motorcycle

    # Test 8 — person completely outside all motorcycle regions
    def test_person_outside(self):
        motos = [_moto(100, 400, 300, 550)]
        persons = [_person(1000, 50, 1050, 150)]
        assocs = associate_persons_to_motorcycles(persons, motos, IMG_W, IMG_H, DEFAULT_CONFIG)
        assert len(assocs) == 0

    # Test 10 — empty detections
    def test_empty_persons(self):
        assocs = associate_persons_to_motorcycles([], [_moto(100, 200, 300, 400)], IMG_W, IMG_H, DEFAULT_CONFIG)
        assert assocs == []

    def test_empty_motorcycles(self):
        assocs = associate_persons_to_motorcycles([_person(100, 100, 200, 300)], [], IMG_W, IMG_H, DEFAULT_CONFIG)
        assert assocs == []

    def test_both_empty(self):
        assocs = associate_persons_to_motorcycles([], [], IMG_W, IMG_H, DEFAULT_CONFIG)
        assert assocs == []
