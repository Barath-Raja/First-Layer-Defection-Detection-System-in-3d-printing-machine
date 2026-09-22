import cv2
import numpy as np
import joblib
import json
import os
import threading
import warnings
from datetime import datetime
import time
import base64

# Disable OpenCV resource warnings on macOS
warnings.filterwarnings("ignore", message="resource_tracker")
warnings.filterwarnings("ignore", category=ResourceWarning)

# Email alert import
try:
    from .email_alert import send_defect_alert
except ImportError:
    send_defect_alert = None

os.makedirs('defect_images', exist_ok=True)
os.makedirs('logs', exist_ok=True)

class DefectDetector:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, model_path='model.pkl'):
        if hasattr(self, 'initialized'):
            return
        try:
            self.model = joblib.load(model_path)
        except:
            print("Model load failed, using dummy")
            self.model = None
        self.cap = None
        self.lock = threading.Lock()
        self.predictions = []
        self.last_alert = 0
        self.cooldown = 30
        self.initialized = True
        self._open_camera()  # Init camera
    
    def _open_camera(self):
        print("Opening camera...")
        with self.lock:
            if self.cap:
                self.cap.release()
            self.cap = None
            for i in range(3):
                cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    self.cap = cap
                    print("Camera {} opened".format(i))
                    return
            print("No camera available")
    
    def get_features(self, frame):
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return np.array([[np.sum(edges > 0) / (h*w), np.mean(gray), np.std(gray)]])
    
    def detect(self):
        with self.lock:
            if not self.cap or not self.cap.isOpened():
                self._open_camera()
            if not self.cap:
                return {'status': 'No Camera', 'confidence': 0}
            
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self._open_camera()
                return {'status': 'Read Error', 'confidence': 0}
        
        if self.model:
            feature = self.get_features(frame)
            pred = self.model.predict(feature)[0]
            proba = self.model.predict_proba(feature)[0]
            conf = np.max(proba)
        else:
            pred = 0
            conf = 0.9
        
        self.predictions.append(pred)
        if len(self.predictions) > 10:
            self.predictions.pop(0)
        
        labels = ['Correct', 'Under', 'Over']
        status = labels[pred]
        is_defect = pred != 0
        
        result = {'status': status, 'is_defect': is_defect, 'confidence': conf}
        
        return result
    
    def gen_frames(self):
        print("Starting video stream...")
        while True:
            with self.lock:
                if not self.cap or not self.cap.isOpened():
                    print("Camera reconnect in gen_frames")
                    self._open_camera()
                    time.sleep(0.1)
                    continue
                
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    print("Frame read failed")
                    self._open_camera()
                    time.sleep(0.1)
                    continue
            
            # Debug print first frames
            print("Frame captured", frame.shape)
            
            overlay = frame.copy()
            cv2.putText(overlay, "LIVE STREAM", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            ret, jpeg = cv2.imencode('.jpg', overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ret:
                print("JPEG encode failed")
                continue
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(0.03)
    
    def cleanup(self):
        print("Cleaning up...")
        with self.lock:
            if self.cap:
                self.cap.release()
        cv2.destroyAllWindows()

