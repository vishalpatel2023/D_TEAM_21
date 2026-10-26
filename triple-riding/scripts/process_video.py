#!/usr/bin/env python3
"""Process a video for triple-riding detection."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.config import load_config
from app.main import setup_logging
from app.processor import TripleRidingProcessor


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="Process a video.")
    parser.add_argument("--input", "-i", required=True, help="Input video path.")
    parser.add_argument("--output", "-o", default=None, help="Output video path.")
    parser.add_argument("--config", "-c", default=None, help="Config YAML path.")
    args = parser.parse_args()

    config = load_config(args.config)
    processor = TripleRidingProcessor(config)

    results = processor.process_video(
        os.path.abspath(args.input),
        os.path.abspath(args.output) if args.output else None,
    )

    total_violations = sum(r.violations for r in results)
    summary = {
        "module": "triple_riding",
        "total_frames_processed": len(results),
        "total_violations": total_violations,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
