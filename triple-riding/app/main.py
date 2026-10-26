"""
Application entry-point helpers.

Provides ``setup_logging()`` and ``get_processor()`` so that both
``run.py`` and the helper scripts share the same initialisation logic.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

from app.config import AppConfig, load_config
from app.processor import TripleRidingProcessor


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the root logger with a human-readable format."""
    logging.basicConfig(
        level=level,
        format="%(levelname)-5s — %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def get_processor(config_path: Optional[str] = None) -> TripleRidingProcessor:
    """Load config and instantiate the processor."""
    config = load_config(config_path)
    return TripleRidingProcessor(config)
