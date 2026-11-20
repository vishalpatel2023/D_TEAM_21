"""
Reads video streams via OpenCV, executes frame-by-frame inference using
YOLO detectors, generates annotated video output, and compiles violation analytics.
"""

from pathlib import Path
from typing import Any
import time
import cv2

from video_extension.config import (
    VIDEO_OUTPUT_DIR,
    MAX_VIDEO_DURATION_SECONDS,
    DEFAULT_FRAME_SKIP,
    DEFAULT_CONFIDENCES,
)
from video_extension.frame_detector import (
    detect_frame_helmet,
    detect_frame_pothole,
    detect_frame_anpr,
    detect_frame_triple_riding,
    detect_frame_all,
)
from video_extension.annotator import annotate_frame


def _smooth_detections(current_dets: list[dict[str, Any]], active_tracks: list[dict[str, Any]], max_staleness: int = 2) -> list[dict[str, Any]]:
    """
    Keep bounding boxes alive for up to `max_staleness` frames if briefly dropped
    due to motion blur, preventing bounding box flicker in video.
    """
    new_active = []
    for d in current_dets:
        d_copy = dict(d)
        d_copy["staleness"] = 0
        new_active.append(d_copy)

    # Carry forward recently lost tracks if no overlapping box was detected
    for prev in active_tracks:
        pb = prev.get("bbox", [])
        if len(pb) != 4:
            continue

        has_match = False
        for cur in current_dets:
            cb = cur.get("bbox", [])
            if len(cb) != 4:
                continue
            # Check bounding box overlap
            ix1 = max(pb[0], cb[0])
            iy1 = max(pb[1], cb[1])
            ix2 = min(pb[2], cb[2])
            iy2 = min(pb[3], cb[3])
            if ix2 > ix1 and iy2 > iy1:
                has_match = True
                break

        if not has_match and prev.get("staleness", 0) < max_staleness:
            retained = dict(prev)
            retained["staleness"] = prev.get("staleness", 0) + 1
            retained["confidence"] = round(prev["confidence"] * 0.92, 3)
            new_active.append(retained)

    return new_active


def _get_detector_fn(module_name: str):
    """Map module name to corresponding frame detector function."""
    mod = module_name.lower().replace("-", "_")
    if mod in ["helmet", "helmet_detection"]:
        return lambda frame, conf: detect_frame_helmet(frame, conf=conf or DEFAULT_CONFIDENCES["helmet"])
    elif mod in ["pothole", "pothole_detection"]:
        return lambda frame, conf: detect_frame_pothole(frame, conf=conf or DEFAULT_CONFIDENCES["pothole"])
    elif mod in ["anpr", "plate", "license_plate"]:
        return lambda frame, conf: detect_frame_anpr(frame, conf=conf or DEFAULT_CONFIDENCES["anpr"])
    elif mod in ["triple_riding", "triple-riding"]:
        return lambda frame, conf: detect_frame_triple_riding(frame, conf=conf or DEFAULT_CONFIDENCES["triple_riding"])
    elif mod in ["all", "combined"]:
        return lambda frame, conf: detect_frame_all(frame)
    else:
        raise ValueError(f"Unsupported module '{module_name}'. Valid: helmet, pothole, anpr, triple_riding, all")


def process_video(
    input_path: str | Path,
    module: str = "all",
    conf: float | None = None,
    output_filename: str | None = None,
    frame_skip: int = DEFAULT_FRAME_SKIP,
    max_duration_sec: float = MAX_VIDEO_DURATION_SECONDS,
) -> dict[str, Any]:
    """
    Process an input video file: detect hazards/violations per frame and save annotated video.

    Args:
        input_path: Path to the input video file.
        module: Detection module ('helmet', 'pothole', 'anpr', 'triple_riding', 'all').
        conf: Optional confidence threshold override.
        output_filename: Optional custom output video filename.
        frame_skip: Interval between inference frames (1 = every frame, 2 = every 2nd frame).
        max_duration_sec: Maximum duration in seconds to process (default: 15s).

    Returns:
        Dictionary containing video metadata, violation summary, timeline, and output file path.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    # Ensure output directory exists
    VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open video: {input_path}")

    # Read video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or fps > 120:
        fps = 25.0

    total_frames_in_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration_sec = total_frames_in_file / fps if total_frames_in_file > 0 else 0.0

    # Cap maximum frames to process for safety (e.g., student project duration limit)
    max_frames_to_process = int(max_duration_sec * fps)
    frames_to_read = min(total_frames_in_file, max_frames_to_process) if total_frames_in_file > 0 else max_frames_to_process

    # Prepare output path and VideoWriter
    if not output_filename:
        stem = input_path.stem
        output_filename = f"{stem}_annotated_{module}.mp4"

    output_path = VIDEO_OUTPUT_DIR / output_filename

    # Try codecs: 'mp4v' is standard cross-platform, 'avc1' for modern browsers
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        # Fallback to XVID if mp4v fails
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        output_path = output_path.with_suffix(".avi")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    detector_fn = _get_detector_fn(module)

    # Analytics accumulators
    start_time = time.time()
    frame_idx = 0
    cached_detections: list[dict[str, Any]] = []

    timeline_events: list[dict[str, Any]] = []
    total_helmet_violations = 0
    total_potholes = 0
    total_plates = 0
    total_triple_riding = 0
    frames_with_violations = 0
    frames_with_hazards = 0

    try:
        while cap.isOpened() and frame_idx < frames_to_read:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # Run inference on target frames, reuse cached detections on skipped frames
            if frame_idx % frame_skip == 0:
                raw_dets = detector_fn(frame, conf)
                cached_detections = _smooth_detections(raw_dets, cached_detections, max_staleness=2)

            # Analyze frame detections
            frame_violations = [d for d in cached_detections if d.get("violation", False)]
            frame_hazards = [d for d in cached_detections if d.get("hazard", False)]

            if frame_violations:
                frames_with_violations += 1
            if frame_hazards:
                frames_with_hazards += 1

            # Log incident to timeline if any violation/hazard exists on this frame
            if frame_violations or frame_hazards:
                timestamp = round(frame_idx / fps, 2)
                timeline_events.append({
                    "frame": frame_idx,
                    "timestamp_sec": timestamp,
                    "violations": [d["class"] for d in frame_violations],
                    "hazards": [d["class"] for d in frame_hazards],
                    "count": len(frame_violations) + len(frame_hazards),
                })

            # Accumulate module-specific totals
            for d in cached_detections:
                mod = d.get("module", "")
                if mod == "helmet" and d.get("violation", False):
                    total_helmet_violations += 1
                elif mod == "pothole":
                    total_potholes += 1
                elif mod == "anpr":
                    total_plates += 1
                elif mod == "triple_riding" and d.get("violation", False):
                    total_triple_riding += 1

            # Annotate and write frame
            annotated_frame_img = annotate_frame(
                frame=frame,
                detections=cached_detections,
                frame_idx=frame_idx + 1,
                fps=fps,
                total_frames=frames_to_read,
                draw_hud=True,
            )
            writer.write(annotated_frame_img)
            frame_idx += 1

            # Live terminal progress logging every 25 frames
            if frame_idx % 25 == 0 or frame_idx == frames_to_read:
                pct = round((frame_idx / max(1, frames_to_read)) * 100, 1)
                elapsed = max(0.01, time.time() - start_time)
                cur_fps = round(frame_idx / elapsed, 1)
                print(f"[VideoExtension] Frame {frame_idx}/{frames_to_read} ({pct}%) | Speed: {cur_fps} FPS | Violations: {frames_with_violations}")

    finally:
        cap.release()
        writer.release()

    # Transcode with H.264 YUV420P so web browsers can natively play HTML5 video
    try:
        import imageio_ffmpeg
        import subprocess
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        web_temp_path = output_path.with_name(f"web_{output_path.stem}.mp4")

        cmd = [
            ffmpeg_exe, "-y",
            "-i", str(output_path),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "ultrafast",
            "-crf", "23",
            str(web_temp_path)
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if res.returncode == 0 and web_temp_path.exists() and web_temp_path.stat().st_size > 1000:
            output_path.unlink(missing_ok=True)
            web_temp_path.rename(output_path)
            print("[VideoExtension] Transcoded to web-compatible H.264 MP4 successfully.")
    except Exception as e:
        print(f"[VideoExtension] Warning: Web transcode skipped ({e}). Using raw output.")

    total_time_taken = round(time.time() - start_time, 2)
    processed_fps = round(frame_idx / max(0.01, total_time_taken), 1)
    processed_duration_sec = round(frame_idx / fps, 2)

    return {
        "status": "success",
        "module": module,
        "input_file": input_path.name,
        "output_file": output_path.name,
        "output_path": str(output_path),
        "video_metadata": {
            "resolution": f"{width}x{height}",
            "original_fps": round(fps, 1),
            "original_duration_sec": round(video_duration_sec, 2),
            "processed_frames": frame_idx,
            "processed_duration_sec": processed_duration_sec,
            "frame_skip": frame_skip,
        },
        "performance": {
            "total_processing_time_sec": total_time_taken,
            "processing_fps": processed_fps,
        },
        "analytics": {
            "has_violations": frames_with_violations > 0,
            "has_hazards": frames_with_hazards > 0,
            "frames_with_violations": frames_with_violations,
            "frames_with_hazards": frames_with_hazards,
            "helmet_violations_detected": total_helmet_violations,
            "potholes_detected": total_potholes,
            "license_plates_detected": total_plates,
            "triple_riding_violations_detected": total_triple_riding,
        },
        "timeline_events": timeline_events[:50],  # Return up to 50 key timeline events
    }
