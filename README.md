# 3D Printing First-Layer Defect Detection

An industrial-grade, AI-powered computer vision system designed to monitor and classify first-layer 3D printing defects in real-time.

## Features

- **Live Camera Tracking:** Real-time MJPEG HTTP streaming with AI metric overlays (~20 FPS).
- **Static Image Evaluation:** Seamless drag-and-drop functionality built directly into the tracking dashboard.
- **Multithreaded Processing:** OpenCV `cv2.VideoCapture` loops asynchronously inside the background to prevent server bandwidth choking.
- **Model Agnosticism:** Train via Scikit-Learn `Random Forest` models (default) or Keras `TensorFlow CNN` architectures (optional).
- **Interactive UI:** Responsive Vanilla JavaScript & HTML dashboard mapping accuracy datasets dynamically using `chart.js`.

---

## 🚀 How to Run the Project

### 1. Prerequisites
Ensure you have Python 3.10+ installed natively on your Windows environment. 

Install the required core machine-learning and server dependencies via PIP:
```bash
pip install -r requirements.txt
```

### 2. Booting the Application Server
Since the backend utilizes a multithreaded Flask microservice architecture, you must launch the core application from the `backend/` directory routing script:

```bash
python backend/app.py
```

### 3. Accessing the Interface
Once the terminal outputs that the models are loaded and the camera is bound:
1. Open any modern Web Browser (Chrome, Edge, Firefox).
2. Type in the local address: `http://localhost:8080/`
3. Hit `Enter`. The live video matrix should seamlessly fade into view inside your dashboard!

---

## 🛠 Model Management

If you have added new images to the `dataset/0ver/`, `dataset/under/`, or `dataset/correct/` folders, you can automatically recalculate the Random Forest trees by running:

```bash
python train_model.py --model rf
```

*(This will parse the OpenCV edge contours and resave `models/model.pkl` structurally before you boot `app.py`!)*
