# Video Processing Extension
### Real-Time Video Safety Hazard & Violation Detection for Road Safety Monitoring System

> **Note:** This extension is kept completely separate in `video_extension/` without modifying any existing project folders or files.

---

## Overview

This extension enables the Road Safety Monitoring System to process **video clips (8–10 seconds)** in addition to static photos. It performs frame-by-frame inference across all 4 detection modules:
1. **Helmet Violation Detection** (With Helmet / Without Helmet)
2. **Pothole Detection** (Road surface hazards)
3. **ANPR License Plate Localization**
4. **Triple Riding Detection** (Multi-person spatial heuristic)

It renders an annotated output video with bounding boxes, alert tags, and a live HUD status banner, and compiles complete violation analytics across time.

---

## Extension Architecture

```
video_extension/
├── __init__.py           # Extension metadata
├── config.py             # Constraints (15s max, 100MB limit, codecs, colors)
├── model_loader.py       # Thread-safe lazy singleton model manager
├── frame_detector.py     # In-memory numpy frame inference (no slow disk I/O)
├── annotator.py          # Bounding boxes, status pills, and top HUD overlay
├── video_processor.py    # Core OpenCV VideoCapture & VideoWriter pipeline
├── api.py                # Dedicated FastAPI application & router
├── cli.py                # Command-line testing tool
└── README.md             # This documentation
```

---

## How to Run

### Method 1: Run the Standalone Video API Server
You can run the video extension server on port **8001** alongside your existing photo API on port 8000:

```powershell
# From the project root with your virtual environment activated:
uvicorn video_extension.api:app --reload --port 8001
```

Once running, open your browser:
* **Interactive API Documentation:** [http://localhost:8001/docs](http://localhost:8001/docs)
* **Health Check:** [http://localhost:8001/video/health](http://localhost:8001/video/health)

---

### Method 2: Process a Video via Command Line (CLI)
You can directly process any test video from the terminal without launching a web server:

```powershell
# Process video with all 4 models:
python -m video_extension.cli --input path/to/video.mp4 --module all

# Process for helmet detection with 2x speedup (skipping every 2nd frame):
python -m video_extension.cli --input path/to/video.mp4 --module helmet --skip 2

# Process for potholes with custom confidence:
python -m video_extension.cli --input path/to/video.mp4 --module pothole --conf 0.30
```

Annotated video outputs are automatically saved to `runs/video_output/`.

---

## API Endpoints Reference

| Method | Endpoint | Query / Body | Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/video/health` | — | Checks extension status, duration limits, and model readiness. |
| `POST` | `/video/process` | `file` (multipart), `module` (`helmet`/`pothole`/`anpr`/`triple_riding`/`all`), `conf` (optional), `frame_skip` (1–5) | Processes video and returns analytics JSON + download link. |
| `GET` | `/video/download/{filename}` | `filename` (path parameter) | Streams or downloads the processed annotated `.mp4` video. |
| `GET` | `/video/list` | — | Lists all processed videos currently saved on the server. |

---

## Sample API Response (`POST /video/process`)

```json
{
  "status": "success",
  "module": "all",
  "input_file": "traffic_clip.mp4",
  "output_file": "annotated_4a1b8c2d_all.mp4",
  "download_url": "/video/download/annotated_4a1b8c2d_all.mp4",
  "video_metadata": {
    "resolution": "1280x720",
    "original_fps": 30.0,
    "original_duration_sec": 8.5,
    "processed_frames": 255,
    "processed_duration_sec": 8.5,
    "frame_skip": 1
  },
  "performance": {
    "total_processing_time_sec": 6.8,
    "processing_fps": 37.5
  },
  "analytics": {
    "has_violations": true,
    "has_hazards": true,
    "frames_with_violations": 42,
    "frames_with_hazards": 18,
    "helmet_violations_detected": 42,
    "potholes_detected": 18,
    "license_plates_detected": 64,
    "triple_riding_violations_detected": 0
  },
  "timeline_events": [
    {
      "frame": 35,
      "timestamp_sec": 1.17,
      "violations": ["Without Helmet"],
      "hazards": [],
      "count": 1
    }
  ]
}
```

---

## ⚙️ Key Design Features & Optimizations

1. **In-Memory Frame Processing:**  
   Unlike single-image detection scripts that save temporary images to disk on every prediction, `video_extension` passes frames directly as in-memory NumPy arrays (`source=frame, save=False`). This avoids hundreds of disk writes per video.

2. **Frame Skipping Support (`--skip 2`):**  
   Processing every 2nd frame (`skip=2`) doubles processing speed ($\sim 2\times$) while retaining smooth playback by reusing the previous frame's bounding box coordinates.

3. **Built-in Safety Limits for Student Hardware:**  
   To prevent memory exhaustion on laptop CPUs/GPUs, video length is capped at **15 seconds** (recommended 8–10s clips). Longer videos are automatically and safely trimmed to the first 15 seconds.

4. **HUD Dashboard Overlay:**  
   Output videos include a professional header bar indicating current timestamp, frame index, and a color-coded alert badge (`STATUS: NORMAL`, `ALERT: VIOLATION`, `CAUTION: HAZARD`).
