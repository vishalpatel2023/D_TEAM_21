"""
Flask HTTP API — ANPR Service
==============================

Endpoints:
    GET  /health   → service liveness check
    POST /detect   → accepts multipart image, returns JSON detections

Run:
    python -m opencv.anpr.server
    # or
    python opencv/anpr/server.py
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request

# Allow imports from project root when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from opencv.anpr.detect import detect_plates

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "ANPR"}), 200


@app.route("/detect", methods=["POST"])
def detect():
    # Validate file presence
    if "image" not in request.files:
        return jsonify({"error": "No image provided. Send a multipart file with key 'image'."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    # Decode image bytes → numpy array
    file_bytes = file.read()
    np_arr     = np.frombuffer(file_bytes, np.uint8)
    image      = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Could not decode image. Ensure it is a valid JPEG/PNG/WebP/BMP."}), 400

    # Optional confidence threshold
    try:
        conf = float(request.form.get("conf", 0.4))
        conf = max(0.1, min(0.95, conf))  # clamp to [0.1, 0.95]
    except ValueError:
        conf = 0.4

    try:
        result = detect_plates(image, conf=conf)
        return jsonify({"success": True, **result}), 200
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("ANPR_PORT", 5001))
    print(f"[ANPR] Service starting on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
