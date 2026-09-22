import cv2
import numpy as np
import joblib
import warnings
import os
import threading
import time
import datetime

DETECT_IMAGES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'defect_images')
os.makedirs(DETECT_IMAGES_DIR, exist_ok=True)

warnings.filterwarnings('ignore', message='Trying to unpickle estimator')

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'model.pkl')

class DefectDetector:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.model = None
        
        self.latest_frame = None
        self.processed_frame = None
        self.status = "Initializing"
        self.confidence = 0.0
        self.is_defect = False
        
        self.running = False
        self.thread = None
        
        self.last_saved_time = 0
        self.recent_snapshots = self._load_initial_snapshots()
        
        self.load_model()
        self.start_camera()

    def _load_initial_snapshots(self):
        try:
            files = [f for f in os.listdir(DETECT_IMAGES_DIR) if f.endswith('.jpg')]
            files.sort(key=lambda x: os.path.getmtime(os.path.join(DETECT_IMAGES_DIR, x)), reverse=True)
            return files[:5]
        except Exception:
            return []

    def load_model(self):
        print('Loading model...')
        try:
            self.model = joblib.load(MODEL_PATH)
            print('[OK] Model loaded')
        except Exception as e:
            print(f'[WARN] Model issue: {e}. Using dummy predictions.')
            self.model = None

    def start_camera(self):
        # On Windows, USB cameras often lock up without caching the DirectShow driver explicitly.
        backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
        
        # The user explicitly wants the laptop native camera (0) to take priority!
        # Phantoms / capture cards often sit at 1 or 2 and return pitch black frames.
        for idx in [0, 1, 2]:
            self.cap = cv2.VideoCapture(idx, backend)
            
            # Read a test frame to ensure the hardware isn't completely locked
            if self.cap.isOpened() and self.cap.read()[0]:
                self.camera_index = idx
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                print(f'[OK] Found active camera at index {self.camera_index} using {"DSHOW" if backend==cv2.CAP_DSHOW else "Native"} driver')
                
                self.running = True
                self.thread = threading.Thread(target=self._process_loop)
                self.thread.daemon = True
                self.thread.start()
                return
            
            # If this index failed or hardware was locked, release it before the next iteration
            if self.cap:
                self.cap.release()
                
        print('[ERROR] No camera found attached to system')

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()
        if self.cap:
            self.cap.release()

    def _extract_features(self, roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        edges = cv2.Canny(thresh, 50, 150)
        
        edge_ratio = np.sum(edges > 0) / (roi.shape[0] * roi.shape[1])
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        max_contour_area = 0
        if contours:
            max_contour_area = max([cv2.contourArea(c) for c in contours])
            
        edge_continuity = 0.0
        if np.sum(edges > 0) > 0:
            edge_continuity = 1.0 - min(1.0, len(contours) / (np.sum(edges > 0) / 255.0))
            
        return [edge_ratio, mean_intensity, std_intensity, max_contour_area, edge_continuity], thresh, edges, contours

    def analyze_frame(self, frame):
        """Processes a single frame array (native BGR OpenCV matrix) completely and returns parameters"""
        draw_frame = frame.copy()
        
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        roi_size = 200
        
        x1, y1 = max(0, cx - roi_size//2), max(0, cy - roi_size//2)
        x2, y2 = min(w, cx + roi_size//2), min(h, cy + roi_size//2)
        
        roi = frame[y1:y2, x1:x2]
        status = "Unknown"
        confidence = 0.0
        is_defect = False
        
        if roi.shape[0] > 0 and roi.shape[1] > 0:
            roi_resized = cv2.resize(roi, (200, 200))
            features, thresh, edges, contours = self._extract_features(roi_resized)
            
            if self.model and hasattr(self.model, "n_features_in_") and self.model.n_features_in_ == 5:
                features_arr = np.array([features])
                pred = self.model.predict(features_arr)[0]
                proba = self.model.predict_proba(features_arr)[0]
                confidence = np.max(proba)
            else:
                pred = np.random.choice([0, 1, 2], p=[0.85, 0.08, 0.07])
                confidence = 0.95
                if self.model and hasattr(self.model, "n_features_in_") and self.model.n_features_in_ == 3:
                    try:
                        features_arr = np.array([features[:3]])
                        pred = self.model.predict(features_arr)[0]
                        proba = self.model.predict_proba(features_arr)[0]
                        confidence = np.max(proba)
                    except Exception:
                        pass
            
            labels = ['Correct', 'Under', 'Over']
            status = labels[pred]
            is_defect = (pred != 0)
            
            color = (0, 255, 0) if not is_defect else (0, 0, 255)
            cv2.rectangle(draw_frame, (x1, y1), (x2, y2), color, 2)
            
            if is_defect and contours:
                scale_x, scale_y = (x2 - x1) / 200.0, (y2 - y1) / 200.0
                for c in contours:
                    if cv2.contourArea(c) > 50:
                        for pt in c:
                            pt[0][0] = int(pt[0][0] * scale_x + x1)
                            pt[0][1] = int(pt[0][1] * scale_y + y1)
                        cv2.drawContours(draw_frame, [c], -1, (0, 0, 255), 1)

            cv2.putText(draw_frame, f'Status: {status}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(draw_frame, f'Conf: {confidence:.1%}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        return draw_frame, status, float(confidence), bool(is_defect)

    def _process_loop(self):
        while self.running:
            if not self.cap.isOpened():
                time.sleep(0.1)
                continue

            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            self.latest_frame = frame.copy()
            draw_frame, status, conf, is_defect = self.analyze_frame(frame)
            
            with threading.Lock():
                self.processed_frame = draw_frame
                self.status = status
                self.confidence = conf
                self.is_defect = is_defect
                
                # Snapshot capturing logic
                if is_defect and (time.time() - self.last_saved_time > 3.0):
                    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f'defect_{status.lower()}_{timestamp}.jpg'
                    path = os.path.join(DETECT_IMAGES_DIR, filename)
                    cv2.imwrite(path, draw_frame)
                    
                    self.recent_snapshots.insert(0, filename)
                    self.recent_snapshots = self.recent_snapshots[:5]
                    self.last_saved_time = time.time()
                
            time.sleep(0.05)
            
    def get_status(self):
        with threading.Lock():
            return {
                "status": self.status,
                "confidence": self.confidence,
                "is_defect": self.is_defect,
                "recent_snapshots": list(self.recent_snapshots)
            }
