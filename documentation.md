# Road Safety Monitoring System — Project Documentation

A technical reference guide detailing the architecture, computer vision models, backend API, and frontend dashboard of the Road Safety Monitoring System.

---

## 1. Project Overview

The **Road Safety Monitoring System** is an image-processing and deep-learning web application designed to automate traffic safety monitoring. The application detects traffic violations and road hazards from roadway photos and provides visual feedback through an interactive dashboard.

### Core Capabilities
1. **Helmet Violation Detection**: Identifies whether two-wheeler riders are wearing helmets (`With Helmet`) or not (`Without Helmet`).
2. **Pothole Detection**: Detects road surface damage and potholes to identify road hazards.
3. **Automatic Number Plate Recognition (ANPR)**: Localizes vehicle license plates using a trained YOLOv8 model and extracts alphanumeric plate numbers using EasyOCR.
4. **Unified Scan**: Executes all models concurrently on a single traffic frame and aggregates violations and hazard counts.

---

## 2. Team Members & Module Responsibilities

| Name | Registration No. | Module Responsibility | Implementation Status |
|---|---|---|---|
| **Vishal Patel** | 20233314 | Helmet Violation Detection | Completed & Integrated |
| **Virajsingh Rajput** | 20233360 | Road Pothole Detection | Completed & Integrated |
| **Shibchandan Mistry** | 20233263 | License Plate Recognition (ANPR) | Completed & Integrated |
| **Sojwal Vilhekar** | 20233359 | Triple-Riding Detection | In Progress / Planned |

---

## 3. System Architecture

The project follows a standard three-tier architecture:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                   1. PRESENTATION LAYER (React + Vite)                 │
│                                                                        │
│  - Image selection via drag-and-drop or file picker                    │
│  - Module selection tabs (Helmet, Pothole, ANPR, Unified)              │
│  - Confidence threshold slider (15% to 95%)                            │
│  - HTML5 Canvas drawing color-coded bounding boxes & tags              │
│  - Live statistics and detection summary tables                        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    │ HTTP POST (Multipart Form-Data)
                                    │ HTTP Response (JSON Detections)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    2. API LAYER (FastAPI + Uvicorn)                    │
│                                                                        │
│  - Serves REST endpoints on port 8000                                  │
│  - Manages CORS middleware for frontend communication                  │
│  - Handles image upload streaming and temporary disk persistence      │
│  - Invokes computer vision inference routines                          │
│  - Returns standardized JSON payloads with timing and violation counts │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    │ Python Function Invocations
                                    │ Model Weights Inference
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│            3. COMPUTER VISION & DEEP LEARNING (YOLOv8 + OCR)           │
│                                                                        │
│  - Helmet Model:  results/helmet_detection/weights/best.pt             │
│  - Pothole Model: results/pothole_detection/weights/best.pt            │
│  - ANPR Model:    results/anpr/weights/best.pt + EasyOCR               │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Deep Dive into Computer Vision Modules

All computer vision modules follow a standardized inference contract:
```python
def detect_*(image_path: str, conf: float = 0.4) -> list[dict]:
    ...
```
Each function returns a list of dictionaries where each detection contains:
- `bbox`: Coordinates in `[x1, y1, x2, y2]` integer pixel format.
- `class`: String name of the detected class.
- `confidence`: Confidence score as a float rounded to 4 decimals.
- Module-specific metadata (e.g., `violation: bool`, `hazard: bool`, `plate_text: str`).

---

### A. Helmet Detection (`opencv/helmet/`)

- **Script:** `opencv/helmet/detect.py`
- **Trained Weights:** `results/helmet_detection/weights/best.pt`
- **Classes:**
  - `0`: `With Helmet`
  - `1`: `Without Helmet`
- **Violation Flag:** A detection is flagged with `"violation": True` whenever the detected class is `"Without Helmet"`.
- **Training Script:** `opencv/helmet/train.py` runs transfer learning on `yolov8n.pt` using 50 epochs on the annotated helmet dataset.

**Detection Output Format:**
```json
{
  "bbox": [142, 60, 280, 245],
  "class": "Without Helmet",
  "confidence": 0.8841,
  "violation": true
}
```

---

### B. Pothole Detection (`opencv/pothole/`)

- **Script:** `opencv/pothole/detect.py`
- **Trained Weights:** `results/pothole_detection/weights/best.pt`
- **Classes:**
  - `0`: `pothole`
- **Hazard Flag:** A detection is flagged with `"hazard": True` whenever a pothole is identified on the road surface.
- **Training Script:** `opencv/pothole/train.py` trains YOLOv8 on road surface hazard data.

**Detection Output Format:**
```json
{
  "bbox": [320, 410, 580, 560],
  "class": "pothole",
  "confidence": 0.7612,
  "hazard": true
}
```

---

### C. License Plate Recognition (ANPR) (`opencv/anpr/`)

- **Scripts:**
  - `opencv/anpr/detect.py`: Main detection pipeline.
  - `opencv/anpr/ocr.py`: EasyOCR wrapper with bounding-box cropping and text cleaning.
  - `opencv/anpr/preprocess.py`: Image pre-processing utilities (grayscale, noise reduction).
- **Trained Weights:** `results/anpr/weights/best.pt`
- **Pipeline Workflow:**
  1. YOLOv8 locates the vehicle license plate bounding box (`[x1, y1, x2, y2]`).
  2. The bounding box region is cropped from the input image.
  3. Preprocessing cleans contrast and resizes the cropped plate.
  4. EasyOCR extracts alphanumeric characters. Non-alphanumeric noise is stripped to return a normalized plate string (e.g., `DL8CAF5030`).

**Detection Output Format:**
```json
{
  "bbox": [215, 340, 430, 405],
  "class": "license_plate",
  "confidence": 0.8924,
  "plate_text": "DL8CAF5030"
}
```

---

### D. Triple-Riding Detection (Planned Module)

- **Planned Approach:**
  - Utilize person and motorcycle detection bounding boxes.
  - Compute Intersection over Union (IoU) or spatial overlap between rider boxes and motorcycle boxes.
  - Flag an infraction if greater than 2 person bounding boxes overlap a single motorcycle bounding box.

---

## 5. Backend API Reference (`backend/api.py`)

The backend is built with FastAPI and runs with Uvicorn on port `8000`. It provides CORS support for `localhost:5173`.

### 1. Health Check
- **Endpoint:** `GET /health` (also aliased at `GET /`)
- **Purpose:** Verifies that the API is running and checks whether model weights files are found.
- **Response Example:**
  ```json
  {
    "status": "online",
    "service": "Road Safety Monitoring API",
    "version": "1.0.0",
    "modules": {
      "helmet_detection": true,
      "pothole_detection": true,
      "anpr": true
    }
  }
  ```

### 2. Helmet Violation Detection
- **Endpoint:** `POST /detect/helmet` (also aliased at `POST /detect`)
- **Query Parameter:** `conf` (float, default: `0.4`)
- **Body:** `file` (Multipart form-data image)
- **Response Example:**
  ```json
  {
    "status": "success",
    "module": "helmet_detection",
    "total_detections": 2,
    "violations_count": 1,
    "has_violation": true,
    "detections": [
      {
        "bbox": [140, 50, 275, 230],
        "class": "Without Helmet",
        "confidence": 0.89,
        "violation": true
      },
      {
        "bbox": [450, 60, 560, 210],
        "class": "With Helmet",
        "confidence": 0.92,
        "violation": false
      }
    ],
    "time_taken_ms": 78.4
  }
  ```

### 3. Pothole Detection
- **Endpoint:** `POST /detect/pothole`
- **Query Parameter:** `conf` (float, default: `0.4`)
- **Body:** `file` (Multipart form-data image)
- **Response Example:**
  ```json
  {
    "status": "success",
    "module": "pothole_detection",
    "total_detections": 1,
    "hazards_count": 1,
    "has_hazard": true,
    "detections": [
      {
        "bbox": [200, 310, 480, 450],
        "class": "pothole",
        "confidence": 0.81,
        "hazard": true
      }
    ],
    "time_taken_ms": 64.2
  }
  ```

### 4. License Plate Detection (ANPR)
- **Endpoint:** `POST /detect/anpr` (also aliased at `POST /detect/plate`)
- **Query Parameter:** `conf` (float, default: `0.4`)
- **Body:** `file` (Multipart form-data image)
- **Response Example:**
  ```json
  {
    "status": "success",
    "module": "anpr",
    "total_detections": 1,
    "plates_count": 1,
    "detections": [
      {
        "bbox": [110, 280, 320, 360],
        "class": "license_plate",
        "confidence": 0.91,
        "plate_text": "HR26DK8337"
      }
    ],
    "time_taken_ms": 142.1
  }
  ```

### 5. Unified All-in-One Detection
- **Endpoint:** `POST /detect/all`
- **Body:** `file` (Multipart form-data image)
- **Response Example:**
  ```json
  {
    "status": "success",
    "module": "unified_all",
    "summary": {
      "helmet_violations": 1,
      "potholes_found": 1,
      "plates_found": 1,
      "has_safety_alert": true
    },
    "helmet": { ... },
    "pothole": { ... },
    "anpr": { ... },
    "time_taken_ms": 284.5
  }
  ```

---

## 6. Frontend Dashboard Architecture (`frontend/`)

The frontend is built using **React 19** and **Vite**, styled with clean, accessible CSS designed specifically for an academic project presentation.

### Key Components & Capabilities
1. **Module Tabs (`tabs-nav`)**: Allows the user to toggle between `Helmet Detection`, `Pothole Detection`, `License Plate (ANPR)`, and `All Models (Combined)`.
2. **HTML5 Canvas Renderer (`canvasRef`)**:
   - Loads the user-selected image.
   - Dynamically renders bounding boxes and tag badges over the image based on the backend response.
   - Color coding:
     - 🔴 **Red (`#dc2626`)**: Helmet violations (`Without Helmet`).
     - 🟢 **Green (`#16a34a`)**: Compliant riders (`With Helmet`).
     - 🟠 **Amber (`#d97706`)**: Road potholes.
     - 🔵 **Blue (`#0284c7`)**: License plates.
3. **Interactive Controls**:
   - **Confidence Slider**: Real-time slider adjusting model confidence from 15% to 95%.
   - **Toggle Boxes**: Button to show or hide bounding boxes to compare raw vs. detected outputs.
   - **Save Image**: Exports the annotated canvas as a `.png` file.
   - **Reset**: Clears the current preview, canvas, and detection state.
4. **Backend Health Polling**: Periodically polls `http://127.0.0.1:8000/health` every 5 seconds to provide visual connection status (`Online` vs. `Offline`).

---

## 7. Repository Directory Structure

```text
road-safety-monitoring-system/
├── backend/
│   └── api.py                  # FastAPI server with all detection endpoints
├── frontend/                   # React + Vite frontend dashboard
│   ├── public/
│   ├── src/
│   │   ├── App.jsx             # React component containing controls & canvas logic
│   │   ├── App.css             # Local styling overrides
│   │   ├── index.css           # Clean academic layout stylesheet
│   │   └── main.jsx            # React application entry point
│   ├── index.html              # HTML shell
│   ├── package.json            # Node dependencies
│   └── vite.config.js          # Vite bundler configuration
├── opencv/                     # Computer vision & deep learning code
│   ├── helmet/
│   │   ├── detect.py           # Helmet violation detection logic
│   │   └── train.py            # Helmet model training script
│   ├── pothole/
│   │   ├── detect.py           # Road pothole detection logic
│   │   └── train.py            # Pothole model training script
│   └── anpr/
│       ├── detect.py           # Plate detection logic
│       ├── ocr.py              # EasyOCR text extraction module
│       ├── preprocess.py       # Plate image preprocessing
│       ├── server.py           # Standalone FastAPI server for ANPR
│       ├── train.py            # ANPR local training script
│       └── ANPR_YOLOv8_Training_Colab.ipynb # Google Colab notebook
├── results/                    # YOLOv8 model weights
│   ├── helmet_detection/weights/best.pt
│   ├── pothole_detection/weights/best.pt
│   └── anpr/weights/best.pt
├── test_images/                # Sample road images for testing
│   ├── bus.jpg
│   ├── car_real.jpg
│   ├── synthetic_plate.jpg
│   └── test_car.jpg
├── .gitignore                  # Git ignore rules (datasets, cache, runs)
├── documentation.md            # Technical project documentation (this file)
├── README.md                   # Repository overview and setup guide
└── requirements.txt            # Python dependencies
```

---

## 8. Step-by-Step Setup and Execution Guide

### Prerequisites
- Python 3.10+
- Node.js 18+ and npm

### 1. Python Environment Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate on Windows PowerShell
.venv\Scripts\Activate.ps1

# Activate on Windows Command Prompt
.venv\Scripts\activate.bat

# Activate on Linux / macOS
source .venv/bin/activate

# Install required Python packages
pip install -r requirements.txt
```

### 2. Start Backend API
```bash
uvicorn backend.api:app --reload --port 8000
```
- API Base URL: `http://127.0.0.1:8000`
- Interactive Swagger Documentation: `http://127.0.0.1:8000/docs`

### 3. Start Frontend Dashboard
Open a new terminal window:
```bash
cd frontend
npm install
npm run dev
```
- Frontend URL: `http://localhost:5173`

### 4. Running Standalone Module Tests
To test individual models from the command line without the web interface:

```bash
# Test Helmet Detection
python opencv/helmet/detect.py

# Test Pothole Detection
python opencv/pothole/detect.py

# Test License Plate Recognition (ANPR)
python opencv/anpr/detect.py
```
