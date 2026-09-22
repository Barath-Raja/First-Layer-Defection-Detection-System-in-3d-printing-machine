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

# Disable OpenCV resource tracker warnings on macOS
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
        def _open_camera(self):
            return
        self.model = joblib.load(model_path)
        self.cap = None
        self.lock = threading.Lock()
        self.predictions = []
        self.last_alert = 0
        self.cooldown = 30
        self.initialized = True
        
        with self.lock:
            if self.cap:
        def get_features(self, frame):
            self.cap = None
            for i in [0, 1, 2]:
            img = cv2.resize(frame, (200, 200))
                if cap.isOpened():
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    self.cap = cap
                    print(f"✅ Using camera {i} with AVFoundation")
                    return
            print("❌ No camera found. Check System Preferences > Security & Privacy > Camera.")
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / edges.size
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        return np.array([[edge_ratio, mean_val, std_val]])
    
    def detect(self):
        with self.lock:
            # Reconnect if needed
            if self.cap is None or not self.cap.isOpened():
            
                self._open_camera()
            
            if self.cap is None:
            feature = self.get_features(frame)
            
            ret, frame = self.cap.read()
            if not ret:
            pred = self.model.predict(feature)[0]
                self._open_camera()
                if self.cap:
            proba = self.model.predict_proba(feature)[0]
                if not ret:
            conf = np.max(proba)
        
        self.predictions.append(pred)
        if len(self.predictions) > 10:
            self.predictions.pop(0)
        final_pred = max(set(self.predictions), key=self.predictions.count)
        
        labels = ['Correct Printing', 'Under Extrusion', 'Over Extrusion']
        status = labels[final_pred]
        is_defect = final_pred != 0
        
        overlay_frame = frame.copy()
        color = (0, 255, 0) if not is_defect else (0, 0, 255)
        cv2.putText(overlay_frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
        cv2.putText(overlay_frame, f"{round(conf*100)}%", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        now = time.time()
        should_alert = is_defect and (now - self.last_alert > self.cooldown)
        result = {
            'status': status,
            'is_defect': is_defect,
            'confidence': conf
        }
        
        if should_alert:
            ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            img_path = f'defect_images/defect_{status.replace(" ", "_")}_{ts_str}.jpg'
            success = cv2.imwrite(img_path, frame)
            if success:
                result['defect_image'] = img_path
                log_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'type': status,
                    'image': img_path,
                    'confidence': conf
                }
                try:
                    with open('logs/defects.jsonl', 'a') as f:
                        f.write(json.dumps(log_entry) + '\n')
                    if send_defect_alert:
                        send_defect_alert(status, datetime.now().isoformat(), img_path)
                except Exception as e:
                    print(f"Log/email error: {e}")
            self.last_alert = now
        
        _, buffer = cv2.imencode('.jpg', overlay_frame)
        result['frame_base64'] = base64.b64encode(buffer).decode('utf-8')
        
        return result
    
    def gen_frames(self):
        while True:
            if self.cap is None or not self.cap.isOpened():
                print("Gen_frames reconnect...")
                self._open_camera()
                time.sleep(0.5)
                continue
            
            ret, frame = self.cap.read()
            if not ret:
                print("Gen_frames read failed, reconnect...")
                if self.cap:
                    self.cap.release()
                self._open_camera()
                time.sleep(0.5)
                continue
            
            feature = self.get_features(frame)
            pred = self.model.predict(feature)[0]
            proba = self.model.predict_proba(feature)[0]
            conf = np.max(proba)
            labels_short = ['Correct', 'Under', 'Over']
            status = labels_short[pred]
            
            overlay_frame = frame.copy()
            color = (0, 255, 0) if pred == 0 else (0, 0, 255)
            cv2.putText(overlay_frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(overlay_frame, f"{round(conf*100)}%", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            ret, buffer = cv2.imencode('.jpg', overlay_frame)
            frame_bytes = buffer.tobytes()
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
            time.sleep(0.033)  # ~30 FPS cap
    
    def release(self):
        print("Releasing camera...")
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
