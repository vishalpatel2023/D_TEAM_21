"""
FastAPI HTTP API — ANPR Service
================================
Exposes Automatic Number Plate Recognition (ANPR) as a RESTful FastAPI service.
Matches the structure and patterns of helmet and pothole detection.

Endpoints:
    GET  /                 → Service health check and info
    GET  /health           → Service health check and model status
    POST /detect           → Detects plates, extracts text, returns bboxes & detections
    GET  /docs             → Interactive Swagger API documentation

Run Server:
    python opencv/anpr/server.py
    or: uvicorn opencv.anpr.server:app --port 5001 --reload
"""

import os
import sys
import shutil
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from opencv.anpr.detect import MODEL_PATH, detect_plate, detect_plates

# ── App configuration ─────────────────────────────────────────────────────────
app = FastAPI(
    title="Road Safety Monitoring System — ANPR API",
    description="Backend API for Automatic Number Plate Recognition (ANPR) using YOLOv8 & OCR",
    version="1.0.0"
)

# Enable CORS for React/Vite (ports 5173, 3000, or any frontend origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ── Health Endpoints ──────────────────────────────────────────────────────────
@app.get("/")
@app.get("/health")
def health():
    """Health check endpoint returning module status and model availability."""
    model_exists = MODEL_PATH.exists()
    return {
        "status": "online",
        "service": "Automatic Number Plate Recognition (ANPR) API",
        "version": "1.0.0",
        "model_loaded": model_exists,
        "model_path": str(MODEL_PATH),
        "engine": "YOLOv8 + EasyOCR" if model_exists else "Contour Fallback",
        "endpoints": {
            "detect": "POST /detect (accepts image upload)",
            "docs": "GET /docs (interactive Swagger API docs)"
        }
    }


# ── Detection Endpoint ────────────────────────────────────────────────────────
@app.post("/detect")
async def predict_plate(
    file: UploadFile = File(...),
    conf: float = Query(0.4, ge=0.05, le=0.95, description="Confidence threshold")
):
    """
    Accept an uploaded image file, run license plate detection and OCR,
    and return structured results matching helmet and pothole detection.
    """
    # 1. Validate file extension
    file_ext = Path(file.filename or "").suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{file_ext}'. Allowed formats: {list(ALLOWED_EXTENSIONS)}"
        )

    start_time = time.time()
    temp_file_path = None

    try:
        # 2. Save uploaded file to a temporary file for model inference
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name

        # 3. Run full ANPR pipeline
        result = detect_plates(temp_file_path, conf=conf)
        detections = result.get("detections", [])
        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "status": "success",
            "filename": file.filename,
            "total_detections": len(detections),
            "detections": detections,
            "annotated_image_b64": result.get("annotated_image_b64", ""),
            "time_taken_ms": elapsed_ms,
            "conf_threshold": conf
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    finally:
        # 4. Clean up temporary file from disk
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError:
                pass


# ── Server Launch ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("ANPR_PORT", 5001))
    print("=" * 55)
    print("     ROAD SAFETY MONITORING — ANPR FASTAPI SERVER")
    print("=" * 55)
    print(f"API running at: http://127.0.0.1:{port}")
    print(f"Swagger docs  : http://127.0.0.1:{port}/docs")
    print(f"Health check  : http://127.0.0.1:{port}/health")
    print(f"Detect route  : http://127.0.0.1:{port}/detect (POST)")
    print("=" * 55)
    uvicorn.run("opencv.anpr.server:app", host="0.0.0.0", port=port, reload=True)
