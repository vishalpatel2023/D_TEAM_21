#!/usr/bin/env python3
"""
Triple-Riding Detection — Command-Line Interface.

Usage:
    python run.py --input inputs/test.jpg --output outputs/result.jpg
    python run.py --input inputs/traffic.mp4 --output outputs/result.mp4
    python run.py --input inputs/test.jpg --output outputs/result.jpg --confidence 0.4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path so ``app`` is importable
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.config import load_config
from app.main import setup_logging
from app.processor import TripleRidingProcessor

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Triple-Riding Detection — detect motorcycles with 3+ riders.",
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to an image or video file.",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Path to save the annotated output (image or video).",
    )
    parser.add_argument(
        "--config", "-c", default=None,
        help="Path to config.yaml (default: triple-riding/config.yaml).",
    )
    parser.add_argument(
        "--confidence", type=float, default=None,
        help="Override model confidence threshold.",
    )
    parser.add_argument(
        "--json", dest="json_output", action="store_true",
        help="Print structured JSON result to stdout.",
    )
    return parser


def main() -> None:
    setup_logging()
    args = _build_parser().parse_args()

    input_path = os.path.abspath(args.input)
    ext = Path(input_path).suffix.lower()

    if ext not in _IMAGE_EXTENSIONS and ext not in _VIDEO_EXTENSIONS:
        logger.error("Unsupported file extension: %s", ext)
        sys.exit(1)

    if not os.path.isfile(input_path):
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    # Load config and apply CLI overrides
    config = load_config(args.config)
    if args.confidence is not None:
        config.model.confidence_threshold = args.confidence

    processor = TripleRidingProcessor(config)

    output_path = os.path.abspath(args.output) if args.output else None

    try:
        if ext in _IMAGE_EXTENSIONS:
            result = processor.process_image(input_path, output_path)
            result_dict = result.to_dict()
        else:
            frame_results = processor.process_video(input_path, output_path)
            # Aggregate summary for CLI output
            total_violations = sum(r.violations for r in frame_results)
            result_dict = {
                "module": "triple_riding",
                "video": input_path,
                "total_frames_processed": len(frame_results),
                "total_violations": total_violations,
            }
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        sys.exit(1)

    if args.json_output:
        print(json.dumps(result_dict, indent=2))
    else:
        logger.info("Result: %s", json.dumps(result_dict, indent=2))


if __name__ == "__main__":
    main()
