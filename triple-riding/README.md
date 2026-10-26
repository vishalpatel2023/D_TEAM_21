# Triple-Riding Detection Module

> **Road Safety Monitoring System** — Sojwal Vilhekar  
> Detects motorcycles carrying **3 or more riders** using computer vision.

---

## 📌 Project Objective

Build a computer-vision system that receives a road image or video and
identifies **triple-riding violations** — motorcycles carrying three or more
persons.

The module uses a pretrained **YOLOv8** model (COCO-trained) for object
detection, then applies a custom **geometric association algorithm** to
link detected persons to motorcycles and count riders.

---

## 🏗️ Architecture

```
Image / Video
      ↓
 YOLO Object Detection
      ↓
 Detect PERSON + MOTORCYCLE
      ↓
 Person–Motorcycle Association
      ↓
 Rider Counting
      ↓
 Violation Classification (riders ≥ 3)
      ↓
 Visualization + Structured JSON Result
```

### Module Structure

```
triple-riding/
├── app/
│   ├── __init__.py          # Package init
│   ├── config.py            # Configuration management
│   ├── schemas.py           # Data models (Detection, Association, Result)
│   ├── detector.py          # YOLO-based object detection
│   ├── association.py       # Person–motorcycle geometric association
│   ├── rider_counter.py     # Rider counting + violation classification
│   ├── visualization.py     # OpenCV annotation and rendering
│   ├── processor.py         # End-to-end image/video pipeline
│   └── main.py              # Shared initialisation helpers
├── tests/
│   ├── test_association.py  # Association unit tests (10 scenarios)
│   ├── test_rider_counter.py# Rider counting tests
│   └── test_schemas.py      # JSON serialization tests
├── scripts/
│   ├── process_image.py     # Standalone image processing
│   └── process_video.py     # Standalone video processing
├── inputs/                  # Place input images/videos here
├── outputs/                 # Annotated results are saved here
├── models/                  # Downloaded/cached YOLO model
├── config.yaml              # All configurable parameters
├── requirements.txt
├── run.py                   # Main CLI entry point
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### Prerequisites

- **Python 3.9+** (tested on 3.10 / 3.11)
- pip

### Setup

```bash
# Navigate to the triple-riding directory
cd triple-riding

# Create a virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/macOS)
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

The YOLO model (`yolov8n.pt`) will be **automatically downloaded** by
Ultralytics on first run if not already present.

---

## 🚀 Usage

### Run the Model on Windows

Open a terminal in the `triple-riding` project folder (the folder containing
`run.py`). Use the virtual-environment Python explicitly so the required
packages, including OpenCV, are available. These commands work in both
PowerShell and Command Prompt:

```powershell
.\.venv\Scripts\python.exe run.py --input inputs\single.png --output outputs\single_result.jpg
.\.venv\Scripts\python.exe run.py --input inputs\double.png --output outputs\double_result.jpg
.\.venv\Scripts\python.exe run.py --input inputs\triple.png --output outputs\triple_result.jpg
```

Replace the example input filename with the name of your image. The annotated
result is saved to the path supplied to `--output`.

Do not type the literal placeholder `your_image.png`; replace it with an
existing filename from `inputs`. For example, to process `triple2.png`:

```powershell
.\.venv\Scripts\python.exe run.py --input inputs\triple2.png --output outputs\triple2_result.jpg --json
```

If you want to activate the environment instead, use
`.\.venv\Scripts\Activate.ps1` in PowerShell or
`.\.venv\Scripts\activate.bat` in Command Prompt. Then `python run.py ...`
will use the environment. If activation does not work, use the explicit
`.venv\Scripts\python.exe` command above.

To print the structured detection result as JSON as well:

```powershell
.\.venv\Scripts\python.exe run.py --input inputs\double.png --output outputs\double_result.jpg --json
```

For a video, use a video filename and output extension:

```powershell
.\.venv\Scripts\python.exe run.py --input inputs\traffic.mp4 --output outputs\result.mp4
```

The first run downloads `yolov8n.pt` if it is not already available, so an
internet connection is required then. This is a command-line program: it
processes the supplied file and exits; it does not launch a web server.

If you have not set up the virtual environment yet, follow the installation
steps above first. If you see `Input file not found`, check that the file is in
`inputs` or pass its full path after `--input`.

### Sample Image Results

The sample images were run with the default configuration:

| Input | Riders detected | Violations | Annotated output |
|---|---:|---:|---|
| `inputs/single.png` | 1 | 0 | `outputs/single_result.jpg` |
| `inputs/double.png` | 2 | 0 | `outputs/double_result.jpg` |
| `inputs/triple.png` | 3 | 1 | `outputs/triple_result.jpg` |

### Process an Image

```bash
python run.py --input inputs/test.jpg --output outputs/result.jpg
```

### Process a Video

```bash
python run.py --input inputs/traffic.mp4 --output outputs/result.mp4
```

### Override Confidence Threshold

```bash
python run.py --input inputs/test.jpg --output outputs/result.jpg --confidence 0.4
```

### Get JSON Output

```bash
python run.py --input inputs/test.jpg --output outputs/result.jpg --json
```

### Use a Custom Config

```bash
python run.py --input inputs/test.jpg --output outputs/result.jpg --config my_config.yaml
```

### Helper Scripts

```bash
# Image only
python scripts/process_image.py --input inputs/test.jpg --output outputs/result.jpg

# Video only
python scripts/process_video.py --input inputs/traffic.mp4 --output outputs/result.mp4
```

---

## 🔧 Configuration

All parameters are defined in `config.yaml`:

| Section | Parameter | Default | Description |
|---------|-----------|---------|-------------|
| `model.path` | `yolov8n.pt` | — | YOLO model name or path |
| `model.confidence_threshold` | `0.35` | — | Minimum detection confidence |
| `model.iou_threshold` | `0.45` | — | NMS IoU threshold |
| `model.image_size` | `640` | — | Inference image size |
| `association.bbox_expansion_ratio` | `0.20` | — | Expand motorcycle box by this ratio |
| `association.minimum_score` | `0.35` | — | Minimum association score |
| `rider_count.minimum_violation_riders` | `3` | — | Riders ≥ this = violation |
| `video.frame_skip` | `0` | — | 0 = every frame; N = skip N frames |
| `visualization.show_association_lines` | `true` | — | Draw person→motorcycle lines |

---

## 🧮 Association Algorithm

The association engine evaluates every (person, motorcycle) pair using
four geometric signals:

### Signal 1 — Point Inside (weight: 0.50)
The person's **bottom-centre** point is checked against the **expanded**
motorcycle bounding box.  If inside → score = 1.0.

### Signal 2 — Horizontal Overlap (weight: 0.25)
Normalised horizontal overlap between person and motorcycle bounding boxes.

### Signal 3 — Centre Distance (weight: 0.15)
Inverse of the normalised distance from the person's bottom-centre to the
motorcycle's centre.  Closer → higher score.

### Signal 4 — Vertical Relationship (weight: 0.10)
Penalises persons whose feet are far above or below the motorcycle region
(likely pedestrians, not riders).

**Final score** = weighted sum, normalised to `[0, 1]`.

### Assignment Rule
Each person is assigned to **at most one** motorcycle — the one with the
highest valid score above the configured threshold.  This prevents
double-counting.

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest tests/test_association.py -v
```

All tests use **synthetic bounding boxes** — no GPU or YOLO model download
required.

### Test Scenarios

| # | Scenario | Expected |
|---|----------|----------|
| 1 | 1 motorcycle + 1 rider | `rider_count=1`, no violation |
| 2 | 1 motorcycle + 2 riders | `rider_count=2`, no violation |
| 3 | 1 motorcycle + 3 riders | `rider_count=3`, **violation** |
| 4 | 1 motorcycle + 4 riders | `rider_count=4`, **violation** |
| 5 | Motorcycle + far pedestrian | Pedestrian not counted |
| 6 | 2 motorcycles + 4 persons | Correct assignment to each |
| 7 | 1 person near 2 motorcycles | Assigned to best match only |
| 8 | Person completely outside | Not associated |
| 9 | Bbox near image boundary | Expanded box stays valid |
| 10 | Empty detections | No crash, zero results |

---

## 📊 Output Format

### Image Result (JSON)

```json
{
  "module": "triple_riding",
  "image_width": 1280,
  "image_height": 720,
  "motorcycles_detected": 2,
  "persons_detected": 5,
  "violations": 1,
  "motorcycles": [
    {
      "motorcycle_id": 0,
      "bbox": [100, 200, 400, 500],
      "confidence": 0.89,
      "rider_count": 3,
      "violation": true,
      "riders": [
        {
          "person_id": 0,
          "bbox": [120, 100, 180, 300],
          "confidence": 0.94,
          "association_score": 0.88
        }
      ]
    }
  ]
}
```

---

## ⚠️ Limitations

- **Heavy occlusion** may cause missed riders (occluded persons not detected
  by YOLO).
- **Very small persons** at long range may fall below the confidence threshold.
- **Dense traffic** can create ambiguous associations when motorcycles and
  persons overlap significantly.
- **Camera angle** affects association quality — the algorithm is tuned for
  typical surveillance/road-level views.
- **Frame-by-frame processing** may produce slightly inconsistent rider counts
  across consecutive video frames (no temporal tracking in the MVP).
- Detection performance depends on **YOLO model size** and available
  **hardware** (CPU vs GPU).

---

## 🔮 Future Improvements

- **Object tracking** (e.g., SORT/DeepSORT) for temporal consistency across
  video frames.
- **Perspective-aware geometry** to improve association in angled views.
- **Custom fine-tuned model** trained on motorcycle-rider datasets.
- **Multi-camera processing** for larger-scale surveillance.
- **Real-time CCTV streaming** support via RTSP/RTMP.
- **GPU optimization** with TensorRT or ONNX Runtime for edge deployment.
- **Confidence calibration** for more reliable violation thresholds.

---

## 📝 Supported File Formats

| Type | Extensions |
|------|-----------|
| Images | `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp` |
| Videos | `.mp4`, `.avi`, `.mov`, `.mkv`, `.wmv`, `.flv` |

---

## 👤 Author

**Sojwal Vilhekar** — Triple-Riding Detection Module  
B.Tech CSE — Image Processing & Computer Vision Project
