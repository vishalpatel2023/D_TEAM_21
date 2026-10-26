"""
Rider counting and violation classification.

Takes association results, counts riders per motorcycle, and determines
whether the triple-riding threshold is met.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from app.config import RiderCountConfig
from app.schemas import (
    Association,
    Detection,
    FrameResult,
    MotorcycleResult,
    RiderInfo,
)


def build_motorcycle_results(
    persons: List[Detection],
    motorcycles: List[Detection],
    associations: List[Association],
    image_width: int,
    image_height: int,
    rider_config: RiderCountConfig,
) -> FrameResult:
    """Build a complete ``FrameResult`` from detections and associations.

    Parameters:
        persons: Detected person objects.
        motorcycles: Detected motorcycle objects.
        associations: Person→motorcycle associations.
        image_width: Source image width.
        image_height: Source image height.
        rider_config: Contains ``minimum_violation_riders``.

    Returns:
        A fully populated ``FrameResult``.
    """
    # Group associations by motorcycle ID
    moto_riders: Dict[int, List[Association]] = defaultdict(list)
    for assoc in associations:
        moto_riders[assoc.motorcycle_id].append(assoc)

    motorcycle_results: List[MotorcycleResult] = []
    total_violations = 0

    for m_idx, moto in enumerate(motorcycles):
        rider_assocs = moto_riders.get(m_idx, [])
        rider_count = len(rider_assocs)
        violation = rider_count >= rider_config.minimum_violation_riders

        riders: List[RiderInfo] = []
        for assoc in rider_assocs:
            person = persons[assoc.person_id]
            riders.append(
                RiderInfo(
                    person_id=assoc.person_id,
                    bbox=person.bbox,
                    confidence=person.confidence,
                    association_score=assoc.score,
                )
            )

        if violation:
            total_violations += 1

        motorcycle_results.append(
            MotorcycleResult(
                motorcycle_id=m_idx,
                bbox=moto.bbox,
                confidence=moto.confidence,
                rider_count=rider_count,
                violation=violation,
                riders=riders,
            )
        )

    return FrameResult(
        image_width=image_width,
        image_height=image_height,
        motorcycles_detected=len(motorcycles),
        persons_detected=len(persons),
        violations=total_violations,
        motorcycles=motorcycle_results,
    )
