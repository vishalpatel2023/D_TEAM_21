# 🛡️ Road Safety Monitoring System — Complete Project Documentation

> **A Comprehensive, Beginner-Friendly Guide to the Architecture, Codebase, and How Everything Connects.**

---

## 📋 Table of Contents
1. [Project Overview](#1-project-overview)
2. [Team Members & Assigned Modules](#2-team-members--assigned-modules)
3. [System Architecture (The Big Picture)](#3-system-architecture-the-big-picture)
4. [Deep Dive into Each Component](#4-deep-dive-into-each-component)
   - [A. Datasets & YOLO Labels](#a-datasets--yolo-labels)
   - [B. AI Model Training (`train.py`)](#b-ai-model-training-trainpy)
   - [C. Computer Vision Inference (`detect.py`)](#c-computer-vision-inference-detectpy)
   - [D. Backend API (`backend/api.py`)](#d-backend-api-backendapipy)
   - [E. Frontend Dashboard (`frontend/`)](#e-frontend-dashboard-frontend)
5. [Complete Directory Structure & File Map](#5-complete-directory-structure--file-map)
6. [Step-by-Step Guide: How to Run the Project](#6-step-by-step-guide-how-to-run-the-project)
7. [Frequently Asked Questions & Troubleshooting](#7-frequently-asked-questions--troubleshooting)
8. [Roadmap for Remaining Team Modules](#8-roadmap-for-remaining-team-modules)

---

## 1. Project Overview

The **Road Safety Monitoring System** is an intelligent computer vision and web application designed to automatically detect road-safety violations and road hazards from traffic camera imagery and video feeds.

### The Problem It Solves
Manual monitoring of traffic cameras by traffic authorities is slow, labor-intensive, and prone to human oversight. By leveraging deep learning (YOLOv8) and a real-time web dashboard, this system automatically flags safety violations (such as two-wheeler riders without helmets) and organizes incidents for authorities to review.

---

## 2. Team Members & Assigned Modules

| Name | Registration No. | Module Responsibility | Status |
| :--- | :--- | :--- | :--- |
| **Vishal Patel** | **20233314** | **Helmet Violation Detection** | **Completed & Integrated** ✅ |
| **Sojwal Vilhekar** | 20233359 | Triple-Riding Detection | In Progress ⏳ |
| **Virajsingh Rajput** | 20233360 | Pothole Detection | In Progress ⏳ |
| **Shibchandan Mistry** | 20233263 | Automatic Number Plate Recognition (ANPR) | In Progress ⏳ |

---

## 3. System Architecture (The Big Picture)

The system is organized into **Three Main Layers**:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   1. PRESENTATION LAYER (React + Vite)                 │
│                                                                        │
│  - User uploads traffic image via browser drag-and-drop               │
│  - Displays live violation counters & statistics                       │
│  - Canvas renders color-coded bounding boxes on the photo:             │
│       🟢 Green = Compliant Rider (With Helmet)                         │
│       🔴 Red   = Safety Violation (Without Helmet)                     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                       HTTP POST /detect (Image file)
                       HTTP Response (JSON Detections)
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   2. INTEGRATION LAYER (FastAPI Backend)               │
│                                                                        │
│  - Receives uploaded image from the browser                            │
│  - Enables Cross-Origin Resource Sharing (CORS)                        │
│  - Passes the image to the Computer Vision Engine                      │
│  - Calculates violation counts and sends structured JSON back          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                       Calls detect_helmet(image_path)
                       Returns bounding boxes & confidence
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│             3. COMPUTER VISION & AI LAYER (YOLOv8 + PyTorch)           │
│                                                                        │
│  - Dataset: 1,376 annotated photos ('With Helmet' / 'Without Helmet')  │
│  - Model: Custom-trained YOLOv8 Nano weights (results/.../best.pt)     │
│  - Script: opencv/helmet/detect.py detects riders and tags violations  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Deep Dive into Each Component

### A. Datasets & YOLO Labels
The project contains two datasets under `datasets/`:

1. **`Bike Helmet Detection.v1i.yolov8/` (Active Dataset for Helmet Detection)**
   - **Total Images:** 1,376 images (416x416 pixels).
   - **2 Classes:**
     - `0`: `With Helmet`
     - `1`: `Without Helmet`
   - **Split:**
     - `train/` (1,185 images ~86%): The "textbook" the AI studies to learn patterns.
     - `valid/` (127 images ~9%): The "practice test" checked after each round (epoch) to prevent memorization (overfitting).
     - `test/` (64 images ~5%): The "final exam" used for unbiased scoring.

2. **What does `data.yaml` do?**
   `data.yaml` is the **master configuration file** for YOLO. Instead of passing 1,000+ files to Python, you give YOLO `data.yaml`. YOLO reads it and automatically discovers where the `train`, `valid`, and `test` images and labels are located.

3. **Why are coordinates decimals between 0 and 1?**
   In label files like `BikesHelmets0.txt`, you see:
   `0 0.509 0.776 0.981 0.446`
   These are **normalized coordinates**:
   $$\text{Normalized Value} = \frac{\text{Pixel Coordinate}}{\text{Total Image Dimension}}$$
   This ensures that if an image is resized from 416x416 to 1280x720, the labels remain 100% accurate without having to recalculate pixel values.

---

### B. AI Model Training (`train.py`)
Located in [opencv/helmet/train.py](file:///c:/Users/HP/OneDrive/Desktop/road-safety-monitoring-system/opencv/helmet/train.py):

* **Base Model:** Uses `yolov8n.pt` (YOLOv8 Nano, the fastest and lightest version).
* **Transfer Learning:** Starts with a model already pre-trained on millions of everyday objects (COCO dataset) and fine-tunes it on the helmet dataset for 50 epochs.
* **Output:** Produces the optimized weights file: [results/helmet_detection/weights/best.pt](file:///c:/Users/HP/OneDrive/Desktop/road-safety-monitoring-system/results/helmet_detection/weights/best.pt).

---

### C. Computer Vision Inference (`detect.py`)
Located in [opencv/helmet/detect.py](file:///c:/Users/HP/OneDrive/Desktop/road-safety-monitoring-system/opencv/helmet/detect.py):

* Loads `best.pt` once.
* Takes an image input and runs `model.predict()`.
* Iterates through all detected bounding boxes:
  - Extracts the box corner coordinates `[x1, y1, x2, y2]`.
  - Determines the class name (`'With Helmet'` or `'Without Helmet'`).
  - Sets `violation = (class_name == "Without Helmet")`.
* Returns a list of structured Python dictionaries:
  ```python
  {
      "class": "Without Helmet",
      "confidence": 0.892,
      "bbox": [140, 85, 320, 410],
      "violation": True
  }
  ```

---

### D. Backend API (`backend/api.py`)
Located in [backend/api.py](file:///c:/Users/HP/OneDrive/Desktop/road-safety-monitoring-system/backend/api.py):

#### Why do we need an API?
A web browser running React speaks JavaScript and runs inside the user's browser. It **cannot directly run Python files or execute PyTorch `.pt` models** on your computer. The API acts as the bridge.

#### Key Features of the API:
1. **FastAPI Framework:** Extremely high performance, automatic data validation, and built-in interactive Swagger documentation at `/docs`.
2. **CORS Middleware:** Permits React (running on `http://localhost:5173`) to call the API without browser cross-origin security errors.
3. **Automatic Cleanup:** Saves the uploaded image to a temporary file, executes detection, and guarantees removal from disk via a `finally:` block.
4. **Endpoints:**
   - `GET /`: Health-check endpoint returning server status.
   - `POST /detect`: Accepts an uploaded image file (`file: UploadFile`) and returns JSON detection metrics.

---

### E. Frontend Dashboard (`frontend/`)
Located in [frontend/](file:///c:/Users/HP/OneDrive/Desktop/road-safety-monitoring-system/frontend):

Built with **React (Vite)** and a custom **Vanilla CSS** design system (no bulky external CSS frameworks required):

1. **Visual Inspection Canvas ([src/App.jsx](file:///c:/Users/HP/OneDrive/Desktop/road-safety-monitoring-system/frontend/src/App.jsx)):**
   * Uses an HTML5 `<canvas>` to draw the photo and dynamically superimpose color-coded bounding boxes.
   * **Safe rider:** Emerald Green box with `✓ With Helmet` label.
   * **Violating rider:** Bright Rose Red box with `⚠️ Without Helmet` label and danger highlight.
2. **Live Analytics Cards:**
   * Total Scans Performed.
   * Violations Caught.
   * Helmets Verified.
   * Violation Ratio (%).
3. **Detection Breakdown Table:**
   * Lists every detected rider, their class, compliance badge, confidence score, and raw `[x1, y1, x2, y2]` coordinates.
4. **Controls & Utilities:**
   * Confidence threshold slider (filter out low-confidence detections).
   * One-click "Download Annotated Image" button.
   * "Toggle Bounding Boxes" button to inspect the raw vs. annotated image.
   * Real-time backend connection status badge (`Online` / `Offline`).

---

## 5. Complete Directory Structure & File Map

```
road-safety-monitoring-system/
│
├── .gitignore                   # Tells Git which files to ignore (.venv, datasets, cache)
├── documentation.md             # This comprehensive architecture & project documentation
├── help.txt                     # Dataset & project reference guide (3-class dataset)
├── help_data.txt                # Dataset guide (2-class Bike Helmet dataset)
├── how_to_compile_run.txt       # Quick terminal command cheatsheet
├── requirements.txt             # Python dependencies (PyTorch, Ultralytics, OpenCV, etc.)
├── README.md                    # Project overview & team member register
│
├── backend/                     # API & Server Layer
│   └── api.py                   # FastAPI application connecting React to YOLOv8
│
├── datasets/                    # Datasets (Ignored in Git due to large file size)
│   ├── Bike Helmet Detection.../ # 2-class dataset (With Helmet / Without Helmet)
│   └── helmet/                  # 3-class dataset (Bike / Helmet / Passenger)
│
├── frontend/                    # User Interface (React + Vite)
│   ├── index.html               # Main HTML entry with Google Fonts (Outfit & JetBrains Mono)
│   ├── package.json             # NPM dependencies (React, Vite)
│   ├── vite.config.js           # Vite build configuration
│   └── src/
│       ├── App.jsx              # Main dashboard application with Canvas renderer
│       ├── index.css            # Dark glassmorphic design system
│       └── main.jsx             # React DOM root mounting script
│
├── opencv/                      # Computer Vision & Deep Learning Engine
│   ├── helmet/
│   │   ├── __init__.py          # Package indicator
│   │   ├── detect.py            # Inference script returning bounding boxes & violation flags
│   │   └── train.py             # Script to train YOLOv8 on data.yaml
│   └── other_team_work/         # Workspace reserved for Triple-Riding, Potholes & ANPR
│
└── results/                     # Model weights and training outputs
    └── helmet_detection/
        └── weights/
            └── best.pt          # The trained YOLOv8 model weights file (6.2 MB)
```

---

## 6. Step-by-Step Guide: How to Run the Project

### Prerequisites
Make sure you have installed on your computer:
* **Python 3.10+**
* **Node.js 18+ & npm**

---

### Step 1: Open Terminal & Activate the Virtual Environment

On Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```
*(Or in Windows Command Prompt: `.venv\Scripts\activate`)*

Ensure your Python API packages are installed:
```bash
pip install fastapi uvicorn python-multipart
```

---

### Step 2: Start the Backend API Server

From the project root:
```bash
uvicorn backend.api:app --reload --port 8000
```
* **API Address:** `http://127.0.0.1:8000`
* **Interactive Test UI:** `http://127.0.0.1:8000/docs` (You can upload test photos and see JSON directly).

---

### Step 3: Start the Frontend Dashboard

Open a **second terminal window** and run:
```bash
cd frontend
npm run dev
```

* **Frontend Dashboard:** `http://localhost:5173`

---

### Step 4: Test Detection

1. Open your browser to `http://localhost:5173`.
2. Notice the green **"YOLOv8 Backend: Online"** indicator in the top-right corner.
3. Drag and drop any traffic image (or click to browse a `.jpg` or `.png`).
4. Click **"🔍 Scan for Helmet Violations"**.
5. Watch the scanner animation run and inspect the red/green bounding boxes, statistics, and breakdown table!

---

## 7. Frequently Asked Questions & Troubleshooting

#### Q: The frontend shows "Backend Offline (:8000)". Why?
* Make sure you ran `uvicorn backend.api:app --reload --port 8000` in your first terminal and that it has not encountered an error.

#### Q: Where is `best.pt` stored?
* The trained weights are stored at `results/helmet_detection/weights/best.pt`. The script [detect.py](file:///c:/Users/HP/OneDrive/Desktop/road-safety-monitoring-system/opencv/helmet/detect.py) automatically resolves this path using `Path(__file__).resolve().parents[2]`.

#### Q: How can I change the detection sensitivity?
* Adjust the **Confidence Threshold slider** on the dashboard. A higher value (e.g., 70%) only shows detections the model is extremely certain about. A lower value (e.g., 35%) captures faint or distant riders.

---

## 8. Roadmap for Remaining Team Modules

As other team members build their modules, here is how they integrate into the system:

1. **Triple-Riding Detection (Sojwal Vilhekar):**
   * Use `datasets/helmet` (which detects `Bike` and `Passenger`).
   * Logic: If number of `Passenger` bounding boxes overlapping a single `Bike` box > 2, flag a **Triple-Riding Violation**.
2. **Pothole Detection (Virajsingh Rajput):**
   * Train YOLO on a road-hazard / pothole dataset.
   * Add a `detect_potholes()` function and a new endpoint `POST /detect/pothole`.
3. **Automatic Number Plate Recognition - ANPR (Shibchandan Mistry):**
   * Detect vehicle license plates with YOLO, crop the plate box, and pass it to **EasyOCR** or Tesseract to extract the alphanumeric registration string.
   * Return `"plate_number": "MH12AB1234"` in the API response.


[ Upload 8-sec Video ]
         │
         ▼
[ OpenCV VideoCapture ]  ──>  Splits video into individual frames (Frame 1, Frame 2 ... Frame 240)
         │
         ▼
   [ While Loop ]
   ├── Frame 1  ──> YOLO detects helmet/no-helmet ──> Draws red/green boxes ──> Saves to Output Video
   ├── Frame 2  ──> YOLO detects helmet/no-helmet ──> Draws red/green boxes ──> Saves to Output Video
   └── ... up to Frame 240
         │
         ▼
[ OpenCV VideoWriter ]  ──>  Stitches all 240 processed frames back into "annotated_result.mp4"
         │
         ▼
[ Send to Frontend ]    ──>  React plays the processed video with violations highlighted!
