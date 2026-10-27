"""
Flask HTTP API — ANPR Service
==============================
Exposes Automatic Number Plate Recognition (ANPR) as a RESTful HTTP service.
Easily consumable by React frontend, Node.js/Express backend, or any HTTP client.

Endpoints:
    GET  /health            → Service health check and model status
    GET  /api/anpr/health   → Alias for health check
    POST /detect            → Detects plates, extracts text, returns bboxes & annotated image
    POST /api/anpr/detect   → Alias for detect endpoint

Accepted Inputs for POST /detect:
    1. multipart/form-data:
       - File field: 'image'
       - Optional field: 'conf' (float 0.1 - 0.95, default 0.4)

    2. application/json:
       - { "image_base64": "<base64 encoded image string>", "conf": 0.4 }
       - or { "image_path": "path/to/image.jpg", "conf": 0.4 }

Run Server:
    python opencv/anpr/server.py
"""

import base64
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from opencv.anpr.detect import MODEL_PATH, detect_plates

# ── App configuration ─────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for frontend access
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB max request size


# ── Health Endpoints ──────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
@app.route("/api/anpr/health", methods=["GET"])
def health():
    """Health check returning module status and model availability."""
    model_exists = MODEL_PATH.exists()
    return jsonify({
        "status": "healthy",
        "service": "Automatic Number Plate Recognition (ANPR)",
        "model_loaded": model_exists,
        "model_path": str(MODEL_PATH),
        "engine": "YOLOv8 + EasyOCR" if model_exists else "Contour Fallback + EasyOCR"
    }), 200


# ── Detection Endpoints ───────────────────────────────────────────────────────
@app.route("/detect", methods=["POST"])
@app.route("/api/anpr/detect", methods=["POST"])
def detect():
    """
    Primary plate detection and recognition endpoint.
    Accepts image via multipart file upload or JSON payload.
    """
    start_time = time.time()
    image = None
    conf = 0.4

    # 1. Parse confidence parameter (from query args, form-data, or json)
    conf_raw = request.args.get("conf")
    if not conf_raw and request.form:
        conf_raw = request.form.get("conf")

    # 2. Check for multipart file upload ('image')
    if "image" in request.files:
        file = request.files["image"]
        if file.filename != "":
            file_bytes = file.read()
            np_arr = np.frombuffer(file_bytes, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # 3. Check for JSON payload if no file was uploaded
    elif request.is_json:
        data = request.get_json(silent=True) or {}
        if not conf_raw:
            conf_raw = data.get("conf")

        # Case A: Base64 string
        if "image_base64" in data or "image" in data:
            b64_str = data.get("image_base64") or data.get("image")
            # Strip data URL prefix if present (e.g., 'data:image/jpeg;base64,...')
            if "," in b64_str:
                b64_str = b64_str.split(",", 1)[1]
            try:
                img_bytes = base64.b64decode(b64_str)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            except Exception as e:
                return jsonify({"success": False, "error": f"Invalid base64 string: {e}"}), 400

        # Case B: Local image file path
        elif "image_path" in data:
            path_str = data["image_path"]
            if os.path.isfile(path_str):
                image = cv2.imread(path_str)
            else:
                return jsonify({"success": False, "error": f"Image file not found: {path_str}"}), 404

    # Validate that an image was successfully loaded
    if image is None:
        return jsonify({
            "success": False,
            "error": "No valid image provided. Send a multipart file with key 'image' or JSON with 'image_base64'/'image_path'."
        }), 400

    # Parse and clamp confidence
    if conf_raw is not None:
        try:
            conf = float(conf_raw)
            conf = max(0.05, min(0.95, conf))
        except (ValueError, TypeError):
            conf = 0.4

    # 4. Run the full ANPR pipeline
    try:
        result = detect_plates(image, conf=conf)
        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        return jsonify({
            "success": True,
            "total_plates": result.get("total_plates", 0),
            "detections": result.get("detections", []),
            "annotated_image_b64": result.get("annotated_image_b64", ""),
            "time_taken_ms": elapsed_ms,
            "conf_threshold": conf
        }), 200

    except Exception as exc:
        return jsonify({
            "success": False,
            "error": f"ANPR pipeline failed: {str(exc)}"
        }), 500


# ── Server Launch ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("ANPR_PORT", 5001))
    print("=" * 55)
    print("     ROAD SAFETY MONITORING — ANPR API SERVER")
    print("=" * 55)
    print(f"API running at: http://127.0.0.1:{port}")
    print(f"Health check  : http://127.0.0.1:{port}/health")
    print(f"Detect route  : http://127.0.0.1:{port}/detect (POST)")
    print("=" * 55)
    app.run(host="0.0.0.0", port=port, debug=False)
