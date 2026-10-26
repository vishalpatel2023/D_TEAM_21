"""
Person–Motorcycle association logic.

This is the core computer-vision contribution of the module.  It uses
multiple geometric signals to decide which detected persons are riding
which detected motorcycles.

Signals
-------
1. **Point-inside** — Is the person's bottom-centre inside the expanded
   motorcycle bounding box?
2. **Horizontal overlap** — How much do the person and motorcycle
   bounding boxes overlap along the X-axis?
3. **Centre distance** — How far is the person's bottom-centre from the
   motorcycle's centre (normalised by motorcycle diagonal)?
4. **Vertical relationship** — Is the person vertically close to the
   motorcycle region?

Each signal produces a score in ``[0, 1]``.  The weighted sum yields an
overall *association score*.  A person is greedily assigned to the
motorcycle that gives the highest score above the configured threshold.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

from app.config import AssociationConfig
from app.schemas import Association, Detection


# ======================================================================
# Geometry helpers
# ======================================================================

def get_person_reference_point(person_bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """Return the bottom-centre of a person bounding box.

    This point approximates where the person meets the motorcycle seat
    and is more meaningful for association than the box centre.

    Parameters:
        person_bbox: ``(x1, y1, x2, y2)``

    Returns:
        ``(bottom_center_x, bottom_center_y)``
    """
    x1, y1, x2, y2 = person_bbox
    return ((x1 + x2) / 2.0, float(y2))


def expand_bbox(
    bbox: Tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    ratio: float,
) -> Tuple[int, int, int, int]:
    """Expand a bounding box by *ratio* on each side, clamped to image bounds.

    Parameters:
        bbox: ``(x1, y1, x2, y2)``
        image_width: Width of the image in pixels.
        image_height: Height of the image in pixels.
        ratio: Fractional expansion (e.g. ``0.20`` → 20 %).

    Returns:
        Expanded ``(x1, y1, x2, y2)`` guaranteed to stay inside the image.
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    dx = int(w * ratio)
    dy = int(h * ratio)

    nx1 = max(0, x1 - dx)
    ny1 = max(0, y1 - dy)
    nx2 = min(image_width, x2 + dx)
    ny2 = min(image_height, y2 + dy)
    return (nx1, ny1, nx2, ny2)


def _point_inside_bbox(
    point: Tuple[float, float],
    bbox: Tuple[int, int, int, int],
) -> bool:
    """Return whether *point* lies inside *bbox* (inclusive)."""
    px, py = point
    x1, y1, x2, y2 = bbox
    return x1 <= px <= x2 and y1 <= py <= y2


# ======================================================================
# Individual signal scorers
# ======================================================================

def _score_point_inside(
    person_ref: Tuple[float, float],
    expanded_moto_bbox: Tuple[int, int, int, int],
) -> float:
    """Signal 1 — 1.0 if bottom-centre is inside expanded motorcycle box."""
    return 1.0 if _point_inside_bbox(person_ref, expanded_moto_bbox) else 0.0


def _score_horizontal_overlap(
    person_bbox: Tuple[int, int, int, int],
    moto_bbox: Tuple[int, int, int, int],
) -> float:
    """Signal 2 — normalised horizontal overlap between person and motorcycle boxes."""
    p_x1, _, p_x2, _ = person_bbox
    m_x1, _, m_x2, _ = moto_bbox

    overlap_start = max(p_x1, m_x1)
    overlap_end = min(p_x2, m_x2)
    overlap = max(0, overlap_end - overlap_start)

    person_width = max(1, p_x2 - p_x1)
    return min(1.0, overlap / person_width)


def _score_center_distance(
    person_ref: Tuple[float, float],
    moto_bbox: Tuple[int, int, int, int],
) -> float:
    """Signal 3 — inverse normalised distance from person ref to motorcycle centre."""
    m_x1, m_y1, m_x2, m_y2 = moto_bbox
    moto_cx = (m_x1 + m_x2) / 2.0
    moto_cy = (m_y1 + m_y2) / 2.0

    diag = math.hypot(m_x2 - m_x1, m_y2 - m_y1)
    if diag < 1.0:
        return 0.0

    dist = math.hypot(person_ref[0] - moto_cx, person_ref[1] - moto_cy)
    normalised = dist / diag
    # Score falls linearly from 1.0 (dist=0) to 0.0 (dist>=diag)
    return max(0.0, 1.0 - normalised)


def _score_vertical_relationship(
    person_ref: Tuple[float, float],
    moto_bbox: Tuple[int, int, int, int],
) -> float:
    """Signal 4 — vertical proximity of person bottom to motorcycle region.

    Riders sit *on* the motorcycle, so the person's bottom-centre should
    be near (or inside) the motorcycle's vertical extent.  A person whose
    feet are far above or far below the motorcycle is likely a pedestrian.
    """
    _, m_y1, _, m_y2 = moto_bbox
    moto_height = max(1, m_y2 - m_y1)
    py = person_ref[1]

    if m_y1 <= py <= m_y2:
        return 1.0

    # Measure vertical gap as fraction of motorcycle height
    if py < m_y1:
        gap = m_y1 - py
    else:
        gap = py - m_y2

    normalised_gap = gap / moto_height
    return max(0.0, 1.0 - normalised_gap)


# ======================================================================
# Public association API
# ======================================================================

def compute_association_score(
    person_bbox: Tuple[int, int, int, int],
    moto_bbox: Tuple[int, int, int, int],
    expanded_moto_bbox: Tuple[int, int, int, int],
    config: AssociationConfig,
) -> float:
    """Compute a weighted association score in ``[0, 1]``.

    Parameters:
        person_bbox: Person detection ``(x1, y1, x2, y2)``.
        moto_bbox: Original motorcycle detection.
        expanded_moto_bbox: Motorcycle box after expansion.
        config: Association configuration (weights + thresholds).

    Returns:
        Association score in ``[0, 1]``.
    """
    person_ref = get_person_reference_point(person_bbox)
    w = config.weights

    s1 = _score_point_inside(person_ref, expanded_moto_bbox)
    s2 = _score_horizontal_overlap(person_bbox, moto_bbox)
    s3 = _score_center_distance(person_ref, moto_bbox)
    s4 = _score_vertical_relationship(person_ref, moto_bbox)

    score = (
        w.point_inside * s1
        + w.horizontal_overlap * s2
        + w.center_distance * s3
        + w.vertical_relationship * s4
    )
    return min(1.0, max(0.0, score))


def associate_persons_to_motorcycles(
    persons: List[Detection],
    motorcycles: List[Detection],
    image_width: int,
    image_height: int,
    config: AssociationConfig,
) -> List[Association]:
    """Assign each person to at most one motorcycle.

    The algorithm is greedy: for every person, the motorcycle with the
    highest score above ``config.minimum_score`` wins.  Each person is
    assigned to at most one motorcycle (no double-counting).

    Parameters:
        persons: Detected persons.
        motorcycles: Detected motorcycles.
        image_width: Frame width for bbox expansion clamping.
        image_height: Frame height for bbox expansion clamping.
        config: Association parameters.

    Returns:
        List of ``Association`` objects.
    """
    if not persons or not motorcycles:
        return []

    # Pre-compute expanded boxes
    expanded_boxes = [
        expand_bbox(m.bbox, image_width, image_height, config.bbox_expansion_ratio)
        for m in motorcycles
    ]

    associations: List[Association] = []

    for p_idx, person in enumerate(persons):
        best_score = -1.0
        best_m_idx = -1

        for m_idx, moto in enumerate(motorcycles):
            score = compute_association_score(
                person.bbox, moto.bbox, expanded_boxes[m_idx], config
            )
            if score >= config.minimum_score and score > best_score:
                best_score = score
                best_m_idx = m_idx

        if best_m_idx >= 0:
            associations.append(
                Association(
                    person_id=p_idx,
                    motorcycle_id=best_m_idx,
                    score=best_score,
                )
            )

    return associations
