from pathlib import Path
import os
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Search for trained model weights
POSSIBLE_MODEL_PATHS = [
    PROJECT_ROOT / "results" / "triple-riding" / "best.pt",
    PROJECT_ROOT / "results" / "triple-riding" / "weights" / "best.pt",
    PROJECT_ROOT / "results" / "triple_riding" / "best.pt",
    PROJECT_ROOT / "results" / "triple_riding" / "weights" / "best.pt",
]

MODEL_PATH = None
for p in POSSIBLE_MODEL_PATHS:
    if p.exists():
        MODEL_PATH = p
        break

if MODEL_PATH is None:
    MODEL_PATH = PROJECT_ROOT / "results" / "triple-riding" / "best.pt"

# Load model once
model = YOLO(str(MODEL_PATH))


def _is_person_on_motorcycle(person_box, motorcycle_box):
    """
    Check if a detected person is situated on / riding a detected motorcycle.
    """
    px1, py1, px2, py2 = person_box
    mx1, my1, mx2, my2 = motorcycle_box

    mw = mx2 - mx1
    mh = my2 - my1
    pcx = (px1 + px2) / 2.0

    # Horizontal alignment check: person center within motorcycle span with a 25% margin
    h_aligned = (mx1 - 0.25 * mw) <= pcx <= (mx2 + 0.25 * mw)

    # Vertical alignment check: person bottom overlaps or sits near motorcycle top half
    v_aligned = (py2 >= my1 - 0.25 * mh) and (py1 <= my2)

    return h_aligned and v_aligned


def _find_person_clusters(persons):
    """
    Fallback heuristic: when a two-wheeler is heavily occluded by 3+ riders,
    YOLO often detects the riders but misses the vehicle.
    This groups persons who are closely packed together horizontally and at
    similar vertical ground levels (sitting in tandem on a bike).
    """
    if len(persons) < 3:
        return []

    # Sort persons by x-center
    sorted_p = sorted(persons, key=lambda p: (p["bbox"][0] + p["bbox"][2]) / 2.0)
    clusters = []
    current_cluster = [sorted_p[0]]

    for p in sorted_p[1:]:
        prev = current_cluster[-1]
        prev_box = prev["bbox"]
        curr_box = p["bbox"]

        # Check horizontal distance between adjacent riders
        prev_w = prev_box[2] - prev_box[0]
        curr_w = curr_box[2] - curr_box[0]
        avg_w = (prev_w + curr_w) / 2.0

        prev_cx = (prev_box[0] + prev_box[2]) / 2.0
        curr_cx = (curr_box[0] + curr_box[2]) / 2.0
        h_dist = abs(curr_cx - prev_cx)

        # Check vertical baseline alignment (sitting at similar ground height)
        v_diff = abs(curr_box[3] - prev_box[3])
        avg_h = ((prev_box[3] - prev_box[1]) + (curr_box[3] - curr_box[1])) / 2.0

        if h_dist <= 1.2 * avg_w and v_diff <= 0.35 * avg_h:
            current_cluster.append(p)
        else:
            if len(current_cluster) >= 3:
                clusters.append(current_cluster)
            current_cluster = [p]

    if len(current_cluster) >= 3:
        clusters.append(current_cluster)

    return clusters


def detect_triple_riding(image_path, conf=0.35):
    """
    Detect triple-riding violations in a single image.

    Features:
    1. Sensitive two-wheeler inference: two-wheelers heavily occluded by 3-4 riders
       often have lower confidence scores in generic YOLO detectors. We run inference
       with a sensitive threshold and evaluate rider consensus.
    2. Overloaded motorcycle envelope: combines bike and rider boxes into one clear
       'Triple Riding' violation box.
    3. Occlusion fallback: clusters tandem seated riders even if the underlying
       bike is completely hidden by their bodies.
    4. Strict filtering: strips away unrelated classes (cars, phones, helmets, etc.).

    Returns:
        list[dict]: List of detection dictionaries.
    """
    save_dir = PROJECT_ROOT / "runs" / "detect_triple_riding"

    # Use a sensitive threshold for initial detection because heavily occluded
    # motorcycles often score between 12% - 25% confidence.
    infer_conf = max(0.10, min(conf * 0.4, 0.15))

    results = model.predict(
        source=str(image_path),
        conf=infer_conf,
        save=True,
        project=str(save_dir),
        name="predict",
        exist_ok=True,
        verbose=False,
    )

    detections = []
    persons = []
    motorcycles = []
    direct_violations = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            c = round(float(box.conf[0]), 4)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            raw_class = result.names[class_id]
            norm_class = raw_class.lower().replace("-", "_").replace(" ", "_")

            # 1. Direct custom triple riding class (including class '0' or 'Triple_riding')
            if norm_class in {"0", "triple_riding", "3riders", "tripleriding", "triple_rider", "overloaded"}:
                if c >= conf:
                    direct_violations.append({
                        "class": "Triple Riding",
                        "confidence": c,
                        "bbox": [x1, y1, x2, y2],
                        "violation": True,
                        "rider_count": 3,
                    })
            # 2. Two-wheeler / Motorcycle instances (allow slightly lower conf due to occlusion)
            elif norm_class in {"motorcycle", "motorbike", "bike"}:
                if c >= 0.12:
                    motorcycles.append({
                        "confidence": c,
                        "bbox": [x1, y1, x2, y2],
                    })
            # 3. Persons / Riders for rider-counting
            elif norm_class in {"person", "passenger", "rider"}:
                if c >= 0.18:
                    persons.append({
                        "confidence": c,
                        "bbox": [x1, y1, x2, y2],
                    })

    # If any direct custom class was detected
    detections.extend(direct_violations)

    # Track which persons have been associated with a vehicle
    assigned_persons = set()

    # Strategy A: Check riders on detected motorcycles
    for mc in motorcycles:
        riders_on_mc = []
        for idx, p in enumerate(persons):
            if _is_person_on_motorcycle(p["bbox"], mc["bbox"]):
                riders_on_mc.append(p)
                assigned_persons.add(idx)

        rider_count = len(riders_on_mc)

        if rider_count >= 3:
            # Overloaded motorcycle detected: Triple Riding Violation!
            all_boxes = [mc["bbox"]] + [r["bbox"] for r in riders_on_mc]
            comb_x1 = min(b[0] for b in all_boxes)
            comb_y1 = min(b[1] for b in all_boxes)
            comb_x2 = max(b[2] for b in all_boxes)
            comb_y2 = max(b[3] for b in all_boxes)

            # Consensus confidence: weighted average of motorcycle and rider confidences
            rider_conf_avg = sum(r["confidence"] for r in riders_on_mc) / rider_count
            consensus_conf = round(max(mc["confidence"], rider_conf_avg), 4)

            detections.append({
                "class": "Triple Riding",
                "confidence": consensus_conf,
                "bbox": [comb_x1, comb_y1, comb_x2, comb_y2],
                "violation": True,
                "rider_count": rider_count,
            })
        elif mc["confidence"] >= conf:
            # Normal motorcycle (1 or 2 riders)
            detections.append({
                "class": "Motorcycle",
                "confidence": mc["confidence"],
                "bbox": mc["bbox"],
                "violation": False,
                "rider_count": max(1, rider_count),
            })

    # Strategy B: Fallback when motorcycle is 100% occluded by 3+ tandem riders
    if not any(d.get("violation") for d in detections):
        unassigned = [p for idx, p in enumerate(persons) if idx not in assigned_persons]
        clusters = _find_person_clusters(unassigned)
        for cluster in clusters:
            comb_x1 = min(p["bbox"][0] for p in cluster)
            comb_y1 = min(p["bbox"][1] for p in cluster)
            comb_x2 = max(p["bbox"][2] for p in cluster)
            comb_y2 = max(p["bbox"][3] for p in cluster)

            avg_conf = round(sum(p["confidence"] for p in cluster) / len(cluster), 4)

            detections.append({
                "class": "Triple Riding",
                "confidence": avg_conf,
                "bbox": [comb_x1, comb_y1, comb_x2, comb_y2],
                "violation": True,
                "rider_count": len(cluster),
            })

    return detections


if __name__ == "__main__":
    # Test on the user's sample image
    test_img = PROJECT_ROOT / "datasets" / "triple riding dataset.v1i.darknet" / "train" / "10_jpg.rf.c822e3dd2221ac3798349dff7a72bf9b.jpg"
    if not test_img.exists():
        test_img = PROJECT_ROOT / "test_images" / "car_real.jpg"

    print(f"Testing Triple Riding Detection on: {test_img}\n")
    dets = detect_triple_riding(test_img, conf=0.45)

    print(f"Total detections at conf=0.45: {len(dets)}")
    for d in dets:
        print(d)
