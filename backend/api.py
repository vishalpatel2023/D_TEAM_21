import os
import sys
import shutil
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Ensuring the project root is in Python's import search path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from opencv.helmet.detect import detect_helmet
from opencv.pothole.detect import detect_pothole
from opencv.anpr.detect import detect_plate

# Import triple riding detection module safely
try:
    from opencv.triple_riding.detect import detect_triple_riding
except ImportError:
    try:
        import importlib.util
        _tr_path = PROJECT_ROOT / "opencv" / "triple-riding" / "detect.py"
        if _tr_path.exists():
            _spec = importlib.util.spec_from_file_location("triple_riding_detect", str(_tr_path))
            _mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            detect_triple_riding = _mod.detect_triple_riding
        else:
            detect_triple_riding = None
    except Exception:
        detect_triple_riding = None

# Initialize FastAPI application
app = FastAPI(
    title="Road Safety Monitoring System - Unified API",
    description="Unified Backend API for Road Safety: Helmet Violation, Pothole Detection, ANPR & Triple Riding",
    version="1.2.0"
)

# Enable CORS for React/Vite (ports 5173, 3000, or any local origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _save_upload_temp_file(file: UploadFile) -> tuple[str, str]:
    """Validate file extension and save upload to temporary disk file."""
    file_ext = Path(file.filename or "").suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{file_ext}'. Allowed formats: {sorted(ALLOWED_EXTENSIONS)}"
        )
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        return temp_file.name, file_ext


@app.get("/")
@app.get("/health")
def health_check():
    """Health check endpoint to verify backend status and loaded AI modules."""
    return {
        "status": "online",
        "service": "Road Safety Monitoring System API",
        "version": "1.2.0",
        "active_modules": {
            "helmet_detection": True,
            "pothole_detection": True,
            "anpr": True,
            "triple_riding": detect_triple_riding is not None
        },
        "endpoints": {
            "detect_helmet": "POST /detect or POST /detect/helmet",
            "detect_pothole": "POST /detect/pothole",
            "detect_anpr": "POST /detect/anpr (or /detect/plate)",
            "detect_triple_riding": "POST /detect/triple-riding (or /detect/triple_riding)",
            "detect_all": "POST /detect/all (multi-model unified scan)",
            "docs": "GET /docs (interactive Swagger API docs)"
        }
    }


# ── HELMET DETECTION ──────────────────────────────────────────────────────────
@app.post("/detect")
@app.post("/detect/helmet")
async def predict_helmet(
    file: UploadFile = File(...),
    conf: float = Query(0.5, ge=0.05, le=0.95, description="Confidence threshold")
):
    """
    Accept an uploaded image file, run helmet detection,
    and return structured violation results for the React frontend.
    """
    start_time = time.time()
    temp_file_path, _ = _save_upload_temp_file(file)

    try:
        detections = detect_helmet(temp_file_path, conf=conf)
        violations = [d for d in detections if d.get("violation", False)]
        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "status": "success",
            "module": "helmet_detection",
            "filename": file.filename,
            "total_detections": len(detections),
            "violations_count": len(violations),
            "has_violation": len(violations) > 0,
            "time_taken_ms": elapsed_ms,
            "conf_threshold": conf,
            "detections": detections
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error in helmet detection: {str(e)}")

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError:
                pass


# ── POTHOLE DETECTION ─────────────────────────────────────────────────────────
@app.post("/detect/pothole")
async def predict_pothole(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.05, le=0.95, description="Confidence threshold")
):
    """
    Accept an uploaded image file, detect road potholes/hazards,
    and return structured results for the React frontend.
    """
    start_time = time.time()
    temp_file_path, _ = _save_upload_temp_file(file)

    try:
        detections = detect_pothole(temp_file_path, conf=conf)
        hazards = [d for d in detections if d.get("hazard", True)]
        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "status": "success",
            "module": "pothole_detection",
            "filename": file.filename,
            "total_detections": len(detections),
            "hazards_count": len(hazards),
            "has_hazard": len(hazards) > 0,
            "time_taken_ms": elapsed_ms,
            "conf_threshold": conf,
            "detections": detections
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error in pothole detection: {str(e)}")

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError:
                pass


# ── ANPR (LICENSE PLATE RECOGNITION) ──────────────────────────────────────────
@app.post("/detect/anpr")
@app.post("/detect/plate")
async def predict_plate(
    file: UploadFile = File(...),
    conf: float = Query(0.4, ge=0.05, le=0.95, description="Confidence threshold")
):
    """
    Accept an uploaded image file, detect license plates, extract alphanumeric text via OCR,
    and return structured results for the React frontend.
    """
    start_time = time.time()
    temp_file_path, _ = _save_upload_temp_file(file)

    try:
        detections = detect_plate(temp_file_path, conf=conf)
        plates_with_text = [d for d in detections if d.get("plate_text")]
        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "status": "success",
            "module": "anpr",
            "filename": file.filename,
            "total_detections": len(detections),
            "plates_count": len(detections),
            "recognized_plates": [d.get("plate_text") for d in plates_with_text],
            "time_taken_ms": elapsed_ms,
            "conf_threshold": conf,
            "detections": detections
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error in ANPR: {str(e)}")

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError:
                pass


# ── TRIPLE RIDING DETECTION ───────────────────────────────────────────────────
@app.post("/detect/triple-riding")
@app.post("/detect/triple_riding")
async def predict_triple_riding(
    file: UploadFile = File(...),
    conf: float = Query(0.35, ge=0.05, le=0.95, description="Confidence threshold")
):
    """
    Accept an uploaded image file, detect triple-riding (motorcycle overloading),
    and return structured violation results for the React frontend.
    """
    if detect_triple_riding is None:
        raise HTTPException(status_code=503, detail="Triple riding detection module is unavailable.")

    start_time = time.time()
    temp_file_path, _ = _save_upload_temp_file(file)

    try:
        detections = detect_triple_riding(temp_file_path, conf=conf)
        violations = [d for d in detections if d.get("violation", False)]
        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "status": "success",
            "module": "triple_riding",
            "filename": file.filename,
            "total_detections": len(detections),
            "violations_count": len(violations),
            "has_violation": len(violations) > 0,
            "time_taken_ms": elapsed_ms,
            "conf_threshold": conf,
            "detections": detections
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error in triple riding detection: {str(e)}")

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError:
                pass


# ── UNIFIED SCAN (ALL MODULES) ────────────────────────────────────────────────
@app.post("/detect/all")
async def predict_all(
    file: UploadFile = File(...),
    helmet_conf: float = Query(0.5, ge=0.05, le=0.95),
    pothole_conf: float = Query(0.25, ge=0.05, le=0.95),
    anpr_conf: float = Query(0.4, ge=0.05, le=0.95),
    triple_conf: float = Query(0.35, ge=0.05, le=0.95)
):
    """
    Run all detection models (Helmet, Pothole, ANPR, Triple Riding) on a single uploaded traffic image.
    Ideal for a full road safety inspection view.
    """
    start_time = time.time()
    temp_file_path, _ = _save_upload_temp_file(file)

    try:
        helmet_dets = detect_helmet(temp_file_path, conf=helmet_conf)
        pothole_dets = detect_pothole(temp_file_path, conf=pothole_conf)
        anpr_dets = detect_plate(temp_file_path, conf=anpr_conf)

        triple_dets = []
        if detect_triple_riding is not None:
            triple_dets = detect_triple_riding(temp_file_path, conf=triple_conf)

        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        helmet_violations = [d for d in helmet_dets if d.get("violation", False)]
        pothole_hazards = [d for d in pothole_dets if d.get("hazard", True)]
        triple_violations = [d for d in triple_dets if d.get("violation", False)]

        has_alert = (len(helmet_violations) > 0 or len(pothole_hazards) > 0 or len(triple_violations) > 0)

        return {
            "status": "success",
            "filename": file.filename,
            "time_taken_ms": elapsed_ms,
            "summary": {
                "helmet_violations": len(helmet_violations),
                "potholes_found": len(pothole_hazards),
                "plates_found": len(anpr_dets),
                "triple_riding_violations": len(triple_violations),
                "has_safety_alert": has_alert
            },
            "helmet": {
                "total_detections": len(helmet_dets),
                "violations_count": len(helmet_violations),
                "has_violation": len(helmet_violations) > 0,
                "detections": helmet_dets
            },
            "pothole": {
                "total_detections": len(pothole_dets),
                "hazards_count": len(pothole_hazards),
                "has_hazard": len(pothole_hazards) > 0,
                "detections": pothole_dets
            },
            "anpr": {
                "total_detections": len(anpr_dets),
                "plates_count": len(anpr_dets),
                "detections": anpr_dets
            },
            "triple_riding": {
                "total_detections": len(triple_dets),
                "violations_count": len(triple_violations),
                "has_violation": len(triple_violations) > 0,
                "detections": triple_dets
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error in unified scan: {str(e)}")

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError:
                pass


if __name__ == "__main__":
    import uvicorn
    # Direct execution starts server on port 8000
    uvicorn.run("backend.api:app", host="127.0.0.1", port=8000, reload=True)
