# 3D Printer First Layer Defect Detection System
### Real-Time Computer Vision & Dual-Failure Classification Architecture

This project provides an end-to-end image-based detection system for identifying first-layer printing failures in a 3D printer in real time. It analyzes images and video captured around the nozzle tip, print path, and bed contact region to classify printing status into two distinct failure conditions:

1. **Model 1 – Filament Not Deposited Due to Supply/Extrusion Failure (`FILAMENT_NOT_DEPOSITED`)**
   - **Fault Description**: Spool tangle/jam, extruder stepper slipping, or clogged nozzle resulting in no or insufficient filament emerging from nozzle.
   - **Sub-Classes**: `NORMAL_EXTRUSION`, `NO_EXTRUSION`, `INSUFFICIENT_EXTRUSION`, `INTERMITTENT_EXTRUSION`.
   - **Visual Indicators**: Empty deposition path, absence of filament at nozzle orifice, path continuity break.

2. **Model 2 – Filament Extruding but Not Adhering to Bed (`FILAMENT_NOT_ADHERING`)**
   - **Fault Description**: Filament exits nozzle correctly but fails to stick/adhere to print bed surface.
   - **Sub-Classes**: `NORMAL_ADHESION`, `FLOATING_THREAD`, `DRAGGING`, `POOR_ADHESION`.
   - **Visual Indicators**: Thread-like structure hanging in air, floating wavy strand, dragging behind moving nozzle, lack of bed contact.

3. **Class 0 – Normal Extrusion & Adhesion (`NORMAL`)**
   - Correct filament output touching print bed, forming continuous squished first-layer line.

4. **Class 3 – Uncertain / Other (`UNCERTAIN`)**
   - Ambiguous, out-of-focus, or low-light frames.

---

## 🚀 Quick Start Guide

### 1. Requirements
Ensure Python 3.8+ and required libraries are installed:
```bash
pip install -r requirements.txt
```

### 2. Generate Synthetic Dataset
To bootstrap the multi-severity dataset across all failure conditions:
```bash
python generate_dataset.py
```
This generates 330+ realistic images categorized under `dataset/normal`, `dataset/filament_not_deposited`, `dataset/filament_not_adhering`, and `dataset/uncertain`.

### 3. Train ML Model
Train the physics-aware Random Forest classifier:
```bash
python train_model.py --model rf
```
*(Optional: Use `--model cnn` if TensorFlow/Keras is installed).*

### 4. Launch Production Server & Industrial UI Dashboard
Run the Flask server:
```bash
python app.py
```
Open your browser and navigate to:
- **Interactive Web UI Dashboard**: `http://localhost:5001/` or `first_layer_detector.html`
- **Video Stream Endpoint**: `http://localhost:5001/video_feed`
- **JSON Prediction API**: `http://localhost:5001/api/predict`

---

## 🛠 Project Architecture

```
├── app.py                      # Flask API server & video streaming
├── generate_dataset.py         # Synthetic multi-severity dataset generator
├── train_model.py              # Physics-aware model trainer (RF & CNN)
├── first_layer_detector.html   # Modern Industrial Web UI Dashboard
├── backend/
│   └── detector.py             # Feature extractor, ROI HUD overlay, & simulation engine
├── dataset/                    # Training and validation image datasets
│   ├── normal/
│   ├── filament_not_deposited/
│   ├── filament_not_adhering/
│   └── uncertain/
├── models/
│   └── model.pkl               # Trained Random Forest classifier
└── defect_images/              # Auto-saved timestamped failure snapshots
```

---

## 📊 Test Simulation Toolbar
The Web Dashboard includes built-in interactive simulation buttons so you can test all fault conditions directly from the browser:
- 🟢 **Live Camera / Normal Print**
- 🔴 **Model 1: No Extrusion** (Spool Tangle / Supply Failure)
- 🔴 **Model 1: Insufficient Extrusion** (75% Reduction)
- 🟠 **Model 2: Floating Thread** (No Bed Contact)
- 🟠 **Model 2: Dragging Strand** (Poor Adhesion)
