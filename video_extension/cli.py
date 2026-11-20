"""
Allows testing and processing videos directly from the command line.

Usage:
    python -m video_extension.cli --input path/to/video.mp4 --module all
    python -m video_extension.cli --input path/to/video.mp4 --module helmet --conf 0.5 --skip 2
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from video_extension.video_processor import process_video


def main():
    parser = argparse.ArgumentParser(
        description="Run Road Safety hazard and violation detection on a video clip."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        type=str,
        help="Path to input video file (e.g., test.mp4)",
    )
    parser.add_argument(
        "--module", "-m",
        default="all",
        choices=["helmet", "pothole", "anpr", "triple_riding", "all"],
        help="Detection module to execute (default: all)",
    )
    parser.add_argument(
        "--conf", "-c",
        type=float,
        default=None,
        help="Optional confidence threshold override (e.g., 0.4)",
    )
    parser.add_argument(
        "--skip", "-s",
        type=int,
        default=1,
        help="Frame skip interval (1 = all frames, 2 = every 2nd frame)",
    )
    parser.add_argument(
        "--max-sec",
        type=float,
        default=15.0,
        help="Maximum video duration in seconds to process (default: 15.0)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Optional output filename",
    )

    args = parser.parse_args()

    print(f"\n[VideoExtension] Starting processing for: {args.input}")
    print(f"[VideoExtension] Module: {args.module} | Frame Skip: {args.skip} | Max Duration: {args.max_sec}s")

    try:
        result = process_video(
            input_path=args.input,
            module=args.module,
            conf=args.conf,
            output_filename=args.output,
            frame_skip=args.skip,
            max_duration_sec=args.max_sec,
        )

        print("\n" + "=" * 60)
        print("VIDEO PROCESSING COMPLETE")
        print("=" * 60)
        print(f"Output File:     {result['output_path']}")
        print(f"Processed:       {result['video_metadata']['processed_frames']} frames ({result['video_metadata']['processed_duration_sec']}s)")
        print(f"Speed:           {result['performance']['processing_fps']} FPS ({result['performance']['total_processing_time_sec']}s total)")
        print(f"Violations:      {result['analytics']['frames_with_violations']} frames flagged")
        print(f"Hazards:         {result['analytics']['frames_with_hazards']} frames flagged")
        print("=" * 60)
        print(json.dumps(result["analytics"], indent=2))

    except Exception as e:
        print(f"\n[ERROR] Video processing failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
