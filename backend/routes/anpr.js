const express  = require("express");
const multer   = require("multer");
const axios    = require("axios");
const FormData = require("form-data");

const router = express.Router();

// ── Multer: keep files in memory ────────────────────────────────────────────
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 16 * 1024 * 1024 }, // 16 MB
  fileFilter: (_req, file, cb) => {
    const allowed = ["image/jpeg", "image/png", "image/webp", "image/bmp"];
    if (allowed.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error("Invalid file type. Only JPEG, PNG, WebP and BMP are supported."));
    }
  },
});

const PYTHON_SERVICE_URL = process.env.PYTHON_SERVICE_URL || "http://localhost:5001";

// ── POST /api/anpr/detect ────────────────────────────────────────────────────
/**
 * Accepts an image upload via multipart/form-data (field name: "image"),
 * forwards it to the Python ANPR service, and returns detection results.
 *
 * Optional body field:
 *   conf  {number}  YOLO detection confidence threshold (default 0.4)
 */
router.post("/detect", upload.single("image"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "No image file uploaded. Use the field name 'image'." });
  }

  try {
    const form = new FormData();
    form.append("image", req.file.buffer, {
      filename:    req.file.originalname,
      contentType: req.file.mimetype,
    });

    if (req.body.conf) {
      form.append("conf", String(req.body.conf));
    }

    const response = await axios.post(`${PYTHON_SERVICE_URL}/detect`, form, {
      headers: form.getHeaders(),
      timeout: 60_000, // 60 s — OCR can be slow on first run
    });

    return res.status(200).json(response.data);
  } catch (err) {
    if (err.response) {
      // Python service returned an error
      return res.status(err.response.status).json(err.response.data);
    }
    console.error("[ANPR] Python service error:", err.message);
    return res.status(503).json({
      error: "ANPR service unreachable. Make sure the Python server is running on port 5001.",
    });
  }
});

// ── GET /api/anpr/health ─────────────────────────────────────────────────────
router.get("/health", async (_req, res) => {
  try {
    const response = await axios.get(`${PYTHON_SERVICE_URL}/health`, { timeout: 5_000 });
    return res.status(200).json({ backend: "ok", python_service: response.data });
  } catch {
    return res.status(503).json({ backend: "ok", python_service: "unreachable" });
  }
});

module.exports = router;
