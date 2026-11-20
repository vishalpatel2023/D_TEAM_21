"""
Configuration settings for the Video Processing Extension.
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Model weights
MODEL_PATHS = {
    "helmet": PROJECT_ROOT / "results" / "helmet_detection" / "weights" / "best.pt",
    "pothole": PROJECT_ROOT / "results" / "pothole_detection" / "weights" / "best.pt",
    "anpr": PROJECT_ROOT / "results" / "anpr" / "weights" / "best.pt",
    "triple_riding": PROJECT_ROOT / "results" / "triple-riding" / "best.pt",
}

# Video output and temp storage
VIDEO_OUTPUT_DIR = PROJECT_ROOT / "runs" / "video_output"
VIDEO_TEMP_DIR = PROJECT_ROOT / "runs" / "video_temp"

# Limits for student project execution (safe for laptop CPU/GPU)
MAX_VIDEO_DURATION_SECONDS = 15.0  # Max length: 15s (recommended 8-10s)
MAX_FILE_SIZE_MB = 100.0           # Maximum upload file size in MB
DEFAULT_FRAME_SKIP = 1             # 1 = process all frames, 2 = process every 2nd frame (2x faster)

# Supported input formats
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# Default confidence thresholds (optimized for video with motion blur)
DEFAULT_CONFIDENCES = {
    "helmet": 0.30,
    "pothole": 0.25,
    "anpr": 0.35,
    "triple_riding": 0.30,
}

# Color palette in BGR (OpenCV format)
COLORS = {
    "danger": (38, 38, 220),       # Red: Violations (Without Helmet, Triple Riding)
    "safe": (74, 163, 22),         # Green: Compliant (With Helmet, Normal Bike)
    "hazard": (6, 119, 217),       # Amber: Potholes
    "plate": (199, 132, 2),        # Blue: License plates
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "banner_bg": (25, 25, 25),     # Dark background for status overlay HUD
}
