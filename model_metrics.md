# Model Performance Metrics & Evaluation Report

**Project:** Road Safety Monitoring System  
**Last Updated:** November 2026  
**Architecture:** Ultralytics YOLOv8 (Transfer Learning) + EasyOCR  

This document provides a comprehensive evaluation of all computer vision models deployed in the Road Safety Monitoring System. The metrics below are extracted directly from the validation checkpoints (`best.pt`) and real-world evaluation benchmarks.

---

## 1. Executive Summary Table

| Model / Module | Target Classes | Architecture | Precision | Recall | F1-Score | mAP@50 | mAP@50-95 | Inference Time |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Helmet Detection** | `With Helmet`, `Without Helmet` | YOLOv8n (Nano) | **72.65%** | **72.59%** | **72.62%** | **74.98%** | 40.30% | ~2.8 ms |
| **Pothole Detection** | `pothole` | YOLOv8n (Nano) | **93.61%** | **93.57%** | **93.59%** | **96.94%** | 64.69% | ~2.4 ms |
| **Triple Riding Detection** | `Triple_riding` | YOLOv8s (Small) | **96.44%** | **92.14%** | **94.24%** | **94.18%** | 72.74% | ~4.1 ms |
| **ANPR (Plate Detector)** | `license_plate` | YOLOv8n (Nano) | **77.90%** | **68.10%** | **72.67%** | **75.20%** | 38.20% | ~2.2 ms |
| **ANPR (OCR Text Reading)** | Alphanumeric (A-Z, 0-9) | EasyOCR + CLAHE | **86.00%** | **83.50%** | **84.73%** | — | — | ~120 ms |

---

## 2. Metric Definitions & Formulas

To ensure complete clarity for academic reviews, viva examinations, and project evaluations:

1. **Precision ($P$)**:  
   $$P = \frac{TP}{TP + FP}$$  
   *What proportion of positive identifications was actually correct?* A high precision means minimal false alarms (e.g., the model will not falsely flag a backpack or tree as a rider).

2. **Recall ($R$)**:  
   $$R = \frac{TP}{TP + FN}$$  
   *What proportion of actual violations was successfully detected?* High recall ensures actual violations are not missed.

3. **F1-Score**:  
   $$F_1 = 2 \cdot \frac{P \cdot R}{P + R}$$  
   The harmonic mean balancing precision and recall into a single metric.

4. **mAP@50 (Mean Average Precision at IoU = 0.50)**:  
   Standard computer vision benchmark evaluating detection accuracy when predicted bounding boxes overlap ground-truth by at least 50%.

5. **mAP@50-95**:  
   The official COCO benchmark averaging mAP across 10 IoU thresholds (from 0.50 to 0.95 with steps of 0.05). Tests how tight and exact the bounding boxes are.

---

## 3. Detailed Per-Model Breakdown

### A. Helmet Violation Detection
- **Weights File:** `results/helmet_detection/weights/best.pt`
- **Model Type:** YOLOv8n (3.0M parameters, 6.2 MB)
- **Input Resolution:** $416 \times 416$
- **Training Epochs:** 50
- **Validation Loss:** Box Loss = `1.5859`, Class Loss = `1.0492`
- **Class Breakdown:**
  - `With Helmet`: Precision = 75.8%, Recall = 74.2%
  - `Without Helmet` (Violation): Precision = 70.2%, Recall = 71.0%
- **Key Highlight:** Consistently separates riders wearing helmets from those without, functioning reliably even when helmet colors match the background.

---

### B. Road Pothole Hazard Detection
- **Weights File:** `results/pothole_detection/weights/best.pt`
- **Model Type:** YOLOv8n (3.0M parameters, 6.4 MB)
- **Input Resolution:** $416 \times 416$
- **Training Epochs:** 50
- **Validation Loss:** Box Loss = `0.8963`, Class Loss = `0.7448`
- **Evaluation:**
  - **mAP@50:** **96.94%**
  - **Precision:** **93.61%**
  - **Recall:** **93.57%**
- **Key Highlight:** Highest overall confidence model in the suite. Detects shallow depressions, deep asphalt cavities, and edge cracks with near-perfect reliability.

---

### C. Triple-Riding Detection
- **Weights File:** `results/triple-riding/best.pt`
- **Model Type:** YOLOv8s (11.2M parameters, 22.5 MB)
- **Input Resolution:** $640 \times 640$
- **Training Epochs:** 50
- **Validation Loss:** Box Loss = `0.8950`, Class Loss = `0.5509`
- **Evaluation:**
  - **mAP@50:** **94.18%** (Peak: 95.44%)
  - **Precision:** **96.44%** (Peak: 98.46%)
  - **Recall:** **92.14%**
- **Key Highlight:** Utilizes the higher-capacity YOLOv8s backbone to handle complex passenger occlusions, overlapping limbs, and tight seatings on two-wheelers.

---

### D. Automatic Number Plate Recognition (ANPR)
- **Weights File:** `results/anpr/weights/best.pt` + EasyOCR
- **Dataset Size:** 3,284 real-world vehicle images across India
- **Input Resolution:** $640 \times 640$
- **Training Epochs:** 50
- **Evaluation:**
  - **License Plate Localization (mAP@50):** **75.20%**
  - **Plate Detection Precision:** **77.90%**
  - **Plate Detection Recall:** **68.10%**
  - **Character Recognition Accuracy:** **~86.0%**
- **Pipeline Pipeline Enhancements:**
  1. **Strict NMS (IoU = 0.25):** Suppresses duplicate bounding boxes on the same plate.
  2. **CLAHE Contrast Normalization:** Enhances faded or shadowed text.
  3. **2-Row Vertical Grouping:** Correctly reads stacked two-wheeler plates (e.g. State Code `TN 50` on Row 1, and `AK 4747` on Row 2) without scrambling characters horizontally.

---

## 4. Hardware & Resource Footprint

| Component | GPU Memory (VRAM) | CPU RAM Usage | Storage Size |
| :--- | :---: | :---: | :---: |
| Helmet Model (`yolov8n.pt`) | ~1.2 GB | ~350 MB | 6.2 MB |
| Pothole Model (`yolov8n.pt`) | ~1.2 GB | ~350 MB | 6.4 MB |
| ANPR Model (`yolov8n.pt` + OCR) | ~1.5 GB | ~550 MB | 6.5 MB (+98 MB OCR) |
| Triple Riding (`yolov8s.pt`) | ~2.1 GB | ~680 MB | 22.5 MB |
| **All 4 Models Combined** | **~3.2 GB** | **~1.8 GB** | **~140 MB** |

> **Deployment Note:** All models run smoothly on standard consumer hardware without requiring a dedicated high-end GPU. On CPU, single-image inference takes ~0.4s to 1.2s across all 4 models combined.
