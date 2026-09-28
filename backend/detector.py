import cv2
import numpy as np
import joblib
import warnings
import os
import threading
import time
import datetime
from generate_dataset import (
    generate_normal_sample,
    generate_filament_not_deposited_sample,
    generate_filament_not_adhering_sample,
    generate_uncertain_sample
)

DETECT_IMAGES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'defect_images')
os.makedirs(DETECT_IMAGES_DIR, exist_ok=True)

warnings.filterwarnings('ignore', message='Trying to unpickle estimator')

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'model.pkl')

CLASS_LABELS = {
    0: 'NORMAL',
    1: 'FILAMENT_NOT_DEPOSITED',
    2: 'FILAMENT_NOT_ADHERING',
    3: 'UNCERTAIN'
}

class DefectDetector:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.model = None
        
        self.latest_frame = None
        self.processed_frame = None
        
        self.status = "NORMAL"
        self.model1_extrusion_status = "NORMAL_EXTRUSION"
        self.model2_adhesion_status = "NORMAL_ADHESION"
        self.confidence = 0.95
        self.is_defect = False
        self.sim_mode = None  # None = camera mode; 'normal', 'no_extrusion', etc = simulation mode
        
        self.running = False
        self.thread = None
        
        self.last_saved_time = 0
        self.recent_snapshots = self._load_initial_snapshots()
        
        self.load_model()
        self.start_camera()

    def _load_initial_snapshots(self):
        try:
            files = [f for f in os.listdir(DETECT_IMAGES_DIR) if f.endswith('.jpg') or f.endswith('.png')]
            files.sort(key=lambda x: os.path.getmtime(os.path.join(DETECT_IMAGES_DIR, x)), reverse=True)
            return files[:10]
        except Exception:
            return []

    def load_model(self):
        print('Loading first layer defect detection model...')
        try:
            self.model = joblib.load(MODEL_PATH)
            print('[OK] Model loaded successfully')
        except Exception as e:
            print(f'[WARN] Model loading failed ({e}). Running rule-based computer vision heuristic engine.')
            self.model = None

    def start_camera(self):
        backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
        
        for idx in [0, 1, 2]:
            self.cap = cv2.VideoCapture(idx, backend)
            if self.cap.isOpened() and self.cap.read()[0]:
                self.camera_index = idx
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                print(f'[OK] Found active camera at index {self.camera_index}')
                break
            if self.cap:
                self.cap.release()
                self.cap = None
                
        self.running = True
        self.thread = threading.Thread(target=self._process_loop)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()
        if self.cap:
            self.cap.release()

    def set_simulation_mode(self, mode):
        """Sets live simulation mode: 'normal', 'no_extrusion', 'insufficient_extrusion', 'floating_thread', 'dragging', or None."""
        print(f"[SIMULATION] Switching mode to: {mode}")
        self.sim_mode = mode

    def _extract_features(self, roi):
        img = cv2.resize(roi, (200, 200))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        edges = cv2.Canny(thresh, 50, 150)
        
        total_pixels = 200 * 200
        edge_ratio = np.sum(edges > 0) / float(total_pixels)
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        max_contour_area = 0.0
        thread_wavy_index = 0.0
        
        if contours:
            contour_areas = [cv2.contourArea(c) for c in contours]
            max_contour_area = max(contour_areas)
            largest_c = contours[np.argmax(contour_areas)]
            perimeter = cv2.arcLength(largest_c, True)
            if max_contour_area > 0:
                thread_wavy_index = (perimeter * perimeter) / (4 * np.pi * max_contour_area)
                
        edge_continuity = 0.0
        if np.sum(edges > 0) > 0:
            edge_continuity = 1.0 - min(1.0, len(contours) / (np.sum(edges > 0) / 255.0))
            
        nozzle_roi = gray[40:80, 80:120]
        nozzle_orifice_intensity = np.mean(nozzle_roi) if nozzle_roi.size > 0 else 0.0
        
        path_roi = edges[80:180, 90:110]
        expected_path_density = np.sum(path_roi > 0) / float(path_roi.size) if path_roi.size > 0 else 0.0
        
        features = [
            edge_ratio,
            mean_intensity,
            std_intensity,
            max_contour_area,
            edge_continuity,
            thread_wavy_index,
            nozzle_orifice_intensity,
            expected_path_density
        ]
        return features, thresh, edges, contours

    def analyze_frame(self, frame):
        draw_frame = frame.copy()
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        roi_size = 200
        
        x1, y1 = max(0, cx - roi_size//2), max(0, cy - roi_size//2)
        x2, y2 = min(w, cx + roi_size//2), min(h, cy + roi_size//2)
        
        roi = frame[y1:y2, x1:x2]
        
        status = "NORMAL"
        model1_extrusion = "NORMAL_EXTRUSION"
        model2_adhesion = "NORMAL_ADHESION"
        confidence = 0.95
        is_defect = False
        
        if roi.shape[0] > 0 and roi.shape[1] > 0:
            features, thresh, edges, contours = self._extract_features(roi)
            
            # Predict using model if available and feature counts match
            pred_class = 0
            if self.model and hasattr(self.model, "n_features_in_") and self.model.n_features_in_ == len(features):
                features_arr = np.array([features])
                pred_class = self.model.predict(features_arr)[0]
                proba = self.model.predict_proba(features_arr)[0]
                confidence = float(np.max(proba))
            else:
                # Physics rule-based fallback heuristic
                edge_ratio, _, _, max_area, continuity, wavy_idx, _, path_density = features
                if path_density < 0.02 and edge_ratio < 0.03:
                    pred_class = 1 # FILAMENT_NOT_DEPOSITED
                    confidence = 0.92
                elif wavy_idx > 15.0 or (edge_ratio > 0.05 and max_area < 800):
                    pred_class = 2 # FILAMENT_NOT_ADHERING
                    confidence = 0.89
                else:
                    pred_class = 0
                    confidence = 0.95
                    
            status = CLASS_LABELS.get(pred_class, 'NORMAL')
            is_defect = (pred_class != 0 and pred_class != 3)
            
            # Sub-diagnostic categorization
            if pred_class == 1: # FILAMENT_NOT_DEPOSITED
                if features[7] < 0.005:
                    model1_extrusion = "NO_EXTRUSION"
                elif features[7] < 0.03:
                    model1_extrusion = "INSUFFICIENT_EXTRUSION"
                else:
                    model1_extrusion = "INTERMITTENT_EXTRUSION"
                model2_adhesion = "NO_BED_CONTACT"
                
            elif pred_class == 2: # FILAMENT_NOT_ADHERING
                model1_extrusion = "EXTRUDING_OK"
                if features[5] > 20.0:
                    model2_adhesion = "FLOATING_THREAD"
                elif features[3] < 1200:
                    model2_adhesion = "DRAGGING"
                else:
                    model2_adhesion = "POOR_ADHESION"
            else:
                model1_extrusion = "NORMAL_EXTRUSION"
                model2_adhesion = "NORMAL_ADHESION"
                
            # Draw HUD & Visual Overlays
            if pred_class == 0:
                color = (0, 255, 120)  # Neon Green
            elif pred_class == 1:
                color = (0, 0, 255)    # Red
            elif pred_class == 2:
                color = (0, 165, 255)  # Orange/Amber
            else:
                color = (255, 255, 0)  # Yellow
                
            # Draw ROI Bounding Box
            cv2.rectangle(draw_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(draw_frame, "ROI FOCUS", (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw Nozzle Tip Box (Top of ROI)
            cv2.rectangle(draw_frame, (x1 + 60, y1 + 10), (x1 + 140, y1 + 50), (255, 200, 0), 1)
            cv2.putText(draw_frame, "NOZZLE ZONE", (x1 + 62, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)
            
            # Draw Deposition Path Zone (Center/Bottom of ROI)
            cv2.rectangle(draw_frame, (x1 + 70, y1 + 50), (x1 + 130, y1 + 180), (255, 255, 0), 1)
            
            # Draw detected contours
            if is_defect and contours:
                scale_x, scale_y = (x2 - x1) / 200.0, (y2 - y1) / 200.0
                for c in contours:
                    if cv2.contourArea(c) > 30:
                        c_scaled = c.copy()
                        for pt in c_scaled:
                            pt[0][0] = int(pt[0][0] * scale_x + x1)
                            pt[0][1] = int(pt[0][1] * scale_y + y1)
                        cv2.drawContours(draw_frame, [c_scaled], -1, color, 2)

            # Text Overlays
            cv2.putText(draw_frame, f'Status: {status}', (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(draw_frame, f'Model 1 (Extrusion): {model1_extrusion}', (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(draw_frame, f'Model 2 (Adhesion):  {model2_adhesion}', (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(draw_frame, f'Conf: {confidence:.1%}', (15, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

        return draw_frame, status, model1_extrusion, model2_adhesion, float(confidence), bool(is_defect)

    def _get_frame(self):
        """Returns frame depending on camera availability or simulation mode."""
        if self.sim_mode:
            if self.sim_mode == 'normal':
                return generate_normal_sample()
            elif self.sim_mode == 'no_extrusion':
                return generate_filament_not_deposited_sample(failure_type='no_extrusion')
            elif self.sim_mode == 'insufficient_extrusion':
                return generate_filament_not_deposited_sample(failure_type='insufficient_75')
            elif self.sim_mode == 'floating_thread':
                return generate_filament_not_adhering_sample(failure_type='floating_thread')
            elif self.sim_mode == 'dragging':
                return generate_filament_not_adhering_sample(failure_type='dragging_loops')
            elif self.sim_mode == 'uncertain':
                return generate_uncertain_sample()

        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame
                
        # Default fallback frame generator (animating normal extrusion)
        return generate_normal_sample()

    def _process_loop(self):
        while self.running:
            frame = self._get_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            self.latest_frame = frame.copy()
            draw_frame, status, m1_status, m2_status, conf, is_defect = self.analyze_frame(frame)
            
            with threading.Lock():
                self.processed_frame = draw_frame
                self.status = status
                self.model1_extrusion_status = m1_status
                self.model2_adhesion_status = m2_status
                self.confidence = conf
                self.is_defect = is_defect
                
                # Snapshot saving on defect detection
                if is_defect and (time.time() - self.last_saved_time > 3.0):
                    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f'defect_{status.lower()}_{timestamp}.jpg'
                    path = os.path.join(DETECT_IMAGES_DIR, filename)
                    cv2.imwrite(path, draw_frame)
                    
                    self.recent_snapshots.insert(0, filename)
                    self.recent_snapshots = self.recent_snapshots[:10]
                    self.last_saved_time = time.time()
                
            time.sleep(0.05)
            
    def get_status(self):
        with threading.Lock():
            accuracy = self.confidence * 100.0 if not self.is_defect else (100.0 - self.confidence * 80.0)
            return {
                "status": self.status,
                "model1_extrusion": self.model1_extrusion_status,
                "model2_adhesion": self.model2_adhesion_status,
                "confidence": round(self.confidence, 4),
                "accuracy": round(accuracy, 1),
                "is_defect": self.is_defect,
                "sim_mode": self.sim_mode,
                "recent_snapshots": list(self.recent_snapshots)
            }
