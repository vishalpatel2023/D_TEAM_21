# Road Safety Monitoring System

A computer vision-based web application developed for monitoring road safety and traffic violations using deep learning (YOLOv8) and optical character recognition (OCR).

This project detects two-wheeler helmet violations, road potholes, and vehicle license plates from road images through a unified web interface.

---

## Features

- **Helmet Violation Detection**: Identifies whether motorcycle riders are wearing a helmet (`With Helmet`) or not (`Without Helmet`).
- **Pothole Detection**: Detects road surface damage and potholes to help flag hazardous road conditions.
- **License Plate Recognition (ANPR)**: Detects vehicle license plates using YOLOv8 and extracts the plate number using EasyOCR.
- **Unified Multi-Model Scan**: Runs helmet, pothole, and license plate models together on a single uploaded traffic photo.
- **Interactive Web Dashboard**: React frontend allowing users to upload images, adjust confidence thresholds, view bounding boxes on an HTML5 canvas, and download annotated results.
- **RESTful API**: FastAPI backend providing dedicated endpoints for each model and a health check endpoint.

---

## Tech Stack

- **Computer Vision & Deep Learning**: Python, Ultralytics YOLOv8, PyTorch, OpenCV, EasyOCR
- **Backend**: FastAPI, Uvicorn, Python-Multipart
- **Frontend**: React 19, Vite, Plain CSS (HTML5 Canvas)

---

## Datasets

All datasets used for training and evaluating the computer vision models are publicly accessible via Google Drive:

📁 **[Download Road Safety Datasets (Google Drive)](https://drive.google.com/drive/folders/1Fwo6gKeLIZs2_RAX-cXUG2pI6j1U0Pea?usp=drive_link)**

### Included Datasets:
1. **Bike Helmet Detection Dataset (`Bike Helmet Detection.v1i.yolov8/`)**:
   - Annotated photos of two-wheeler riders categorized into `With Helmet` and `Without Helmet` in YOLO format (`train/`, `valid/`, and `test/` splits).
2. **Pothole Detection Dataset (`Pothole-Detection.v18i.yolov8/`)**:
   - Annotated road imagery with bounding boxes labeled for `pothole` hazard detection.
3. **Car License Plate (ANPR) Dataset**:
   - Annotated vehicle photos used for license plate localization and EasyOCR text extraction.

> **Note**: Pre-trained weights (`best.pt`) for all three models are already included in the `results/` folder, so you can run and test the complete system immediately without downloading the datasets. If you wish to retrain any model using `train.py`, download the datasets from the link above and place them inside a `datasets/` directory in the project root.

---

## Project Folder Structure

```text
road-safety-monitoring-system/
├── backend/
│   └── api.py                  # FastAPI REST API serving all CV models
├── frontend/                   # React + Vite web dashboard
│   ├── public/
│   ├── src/
│   │   ├── App.jsx             # Main dashboard UI & canvas rendering
│   │   ├── index.css           # Styling for clean dashboard layout
│   │   └── main.jsx            # React root mount
│   ├── package.json
│   └── vite.config.js
├── opencv/                     # Computer vision modules
│   ├── helmet/
│   │   ├── detect.py           # Helmet detection inference function
│   │   └── train.py            # YOLOv8 training script for helmet dataset
│   ├── pothole/
│   │   ├── detect.py           # Pothole detection inference function
│   │   └── train.py            # YOLOv8 training script for pothole dataset
│   └── anpr/
│       ├── detect.py           # License plate detection inference function
│       ├── ocr.py              # EasyOCR text reader for cropped plates
│       ├── preprocess.py       # Image resizing and grayscale helpers
│       ├── server.py           # Standalone FastAPI runner for ANPR
│       ├── train.py            # Local training script for license plates
│       └── ANPR_YOLOv8_Training_Colab.ipynb # Google Colab training notebook
├── results/                    # Trained YOLOv8 model weights
│   ├── helmet_detection/
│   │   └── weights/best.pt     # Trained weights for helmet detection
│   ├── pothole_detection/
│   │   └── weights/best.pt     # Trained weights for pothole detection
│   └── anpr/
│       └── weights/best.pt     # Trained weights for license plate detection
├── test_images/                # Sample test images for verification
│   ├── bus.jpg
│   ├── car_real.jpg
│   ├── synthetic_plate.jpg
│   └── test_car.jpg
├── requirements.txt            # Python dependencies
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Node.js (v18+) and npm
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/vishalpatel2023/road-safety-monitoring-system.git
cd road-safety-monitoring-system
```

### 2. Set Up Python Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate on Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Activate on Windows (Command Prompt)
.venv\Scripts\activate.bat

# Activate on Linux / macOS
source .venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

---

## Running the Application

### 1. Start the FastAPI Backend

From the project root:

```bash
uvicorn backend.api:app --reload --port 8000
```

The backend server will start at `http://127.0.0.1:8000`.  
Interactive API documentation (Swagger UI) is available at `http://127.0.0.1:8000/docs`.

### 2. Start the React Frontend

In a second terminal window:

```bash
cd frontend
npm run dev
```

Open your browser and visit: `http://localhost:5173`

---

## Running Individual Models from Terminal

You can test each computer vision module directly via Python scripts:

- **Helmet Detection:**
  ```bash
  python opencv/helmet/detect.py
  ```

- **Pothole Detection:**
  ```bash
  python opencv/pothole/detect.py
  ```

- **License Plate (ANPR):**
  ```bash
  python opencv/anpr/detect.py
  ```

---

## API Endpoints

The FastAPI backend exposes the following endpoints:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check and status of loaded YOLO models |
| `POST` | `/detect/helmet` | Detects helmet usage (`With Helmet` / `Without Helmet`) |
| `POST` | `/detect/pothole` | Detects road surface potholes |
| `POST` | `/detect/anpr` | Detects license plates and extracts plate text |
| `POST` | `/detect/all` | Runs all models on a single image and aggregates results |

### Example API Request (cURL)

```bash
curl -X POST "http://127.0.0.1:8000/detect/helmet?conf=0.4" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_images/car_real.jpg"
```

---

## Team & Project Roles

| Name | Registration No. | Module Responsibility | Status |
|---|---|---|---|
| **Vishal Patel** | 20233314 | Helmet Violation Detection | Completed |
| **Virajsingh Rajput** | 20233360 | Pothole Detection | Completed |
| **Shibchandan Mistry** | 20233263 | License Plate Recognition (ANPR) | Completed |
| **Sojwal Vilhekar** | 20233359 | Triple-Riding Detection | In Progress |
