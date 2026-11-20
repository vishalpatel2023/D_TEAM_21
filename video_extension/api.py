"""
Provides REST API endpoints for video upload, batch frame inference,
downloading processed videos, and checking video processing health.
"""

from pathlib import Path
import shutil
import tempfile
import uuid

from fastapi import FastAPI, APIRouter, UploadFile, File, Query, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from video_extension.config import (
    ALLOWED_VIDEO_EXTENSIONS,
    MAX_VIDEO_DURATION_SECONDS,
    DEFAULT_FRAME_SKIP,
    VIDEO_OUTPUT_DIR,
    MODEL_PATHS,
)
from video_extension.video_processor import process_video

# Create an APIRouter so it can be mounted into any FastAPI app
video_router = APIRouter(prefix="/video", tags=["Video Processing Extension"])


@video_router.get("/health")
def video_health_check():
    """Verify video processing extension readiness and model availability."""
    available_models = {
        name: path.exists() for name, path in MODEL_PATHS.items()
    }
    return {
        "status": "online",
        "extension": "Video Processing Extension",
        "version": "1.0.0",
        "max_duration_seconds": MAX_VIDEO_DURATION_SECONDS,
        "default_frame_skip": DEFAULT_FRAME_SKIP,
        "supported_extensions": list(ALLOWED_VIDEO_EXTENSIONS),
        "models_available": available_models,
        "endpoints": [
            "GET  /video/health",
            "POST /video/process",
            "GET  /video/download/{filename}",
            "GET  /video/list",
        ],
    }


@video_router.post("/process")
async def process_video_endpoint(
    file: UploadFile = File(...),
    module: str = Query("all", description="Detection module: helmet, pothole, anpr, triple_riding, all"),
    conf: float | None = Query(None, ge=0.05, le=0.95, description="Confidence threshold override"),
    frame_skip: int = Query(DEFAULT_FRAME_SKIP, ge=1, le=5, description="Process every Nth frame (1=all, 2=2x faster)"),
):
    """
    Upload an 8-10 second video clip, run multi-hazard detection,
    and receive an annotated output video with violation metrics.
    """
    file_ext = Path(file.filename or "").suffix.lower()
    if file_ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format '{file_ext}'. Allowed: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}",
        )

    # Save uploaded video to temporary file
    temp_dir = tempfile.gettempdir()
    unique_id = uuid.uuid4().hex[:8]
    temp_input_path = Path(temp_dir) / f"upload_{unique_id}{file_ext}"

    try:
        with open(temp_input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        output_filename = f"annotated_{unique_id}_{module}.mp4"

        # Execute video processing engine
        result = process_video(
            input_path=temp_input_path,
            module=module,
            conf=conf,
            output_filename=output_filename,
            frame_skip=frame_skip,
            max_duration_sec=MAX_VIDEO_DURATION_SECONDS,
        )

        # Inject download URL into response
        result["download_url"] = f"/video/download/{output_filename}"
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")

    finally:
        # Clean up uploaded raw video from system temp
        if temp_input_path.exists():
            try:
                temp_input_path.unlink()
            except Exception:
                pass


@video_router.get("/download/{filename}")
def download_processed_video(filename: str):
    """Download or stream an annotated video output."""
    file_path = VIDEO_OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Processed video '{filename}' not found.")

    media_type = "video/mp4" if file_path.suffix.lower() == ".mp4" else "video/x-msvideo"
    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=filename,
    )


@video_router.get("/list")
def list_processed_videos():
    """List all recently processed annotated videos."""
    if not VIDEO_OUTPUT_DIR.exists():
        return {"total": 0, "videos": []}

    videos = []
    for f in VIDEO_OUTPUT_DIR.iterdir():
        if f.is_file() and f.suffix.lower() in [".mp4", ".avi"]:
            videos.append({
                "filename": f.name,
                "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
                "modified": f.stat().st_mtime,
                "download_url": f"/video/download/{f.name}",
            })

    # Sort newest first
    videos.sort(key=lambda x: x["modified"], reverse=True)
    return {"total": len(videos), "videos": videos}


# Standalone FastAPI App (allows running the video service on its own port, e.g., 8001)
app = FastAPI(
    title="Road Safety Monitoring System - Video Extension API",
    description="Video Processing Extension for Helmet, Pothole, ANPR, and Triple-Riding Detection",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(video_router)


@app.get("/")
def root():
    return {
        "message": "Road Safety Video Processing Extension is active.",
        "documentation": "/docs",
        "health": "/video/health",
    }
