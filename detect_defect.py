import cv2
import numpy as np
import joblib
import warnings
import os
import json
import time
from datetime import datetime

warnings.filterwarnings('ignore', message='Trying to unpickle estimator')
warnings.filterwarnings('ignore', category=ResourceWarning)

# Optional email
try:
    from email_alert import send_defect_alert
    HAS_EMAIL = True
except:
    send_defect_alert = None
    HAS_EMAIL = False

MODEL_PATH = 'model.pkl'
DETECT_IMAGES_DIR = 'defect_images'
LOGS_DIR = 'logs'
os.makedirs(DETECT_IMAGES_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

print('Loading model...')
try:
    model = joblib.load(MODEL_PATH)
    print('[OK] Model loaded')
except Exception as e:
    print(f'[WARNING] Model issue: {e}. Using dummy predictions.')
    model = None

cap = None
last_alert = 0
cooldown = 30

def open_camera():
    global cap
    if cap:
        cap.release()
    cap = None
    for i in range(3):
        test_cap = cv2.VideoCapture(i) # Removed cv2.CAP_AVFOUNDATION which is MacOS only
        if test_cap.isOpened():
            test_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            test_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            test_cap.release()
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                print(f'[OK] Camera {i} ready')
                return True
    print('[ERROR] No camera found')
    return False

def get_features(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return np.array([[np.sum(edges > 0) / (frame.shape[0]*frame.shape[1]), np.mean(gray), np.std(gray)]])

def predict_defect(frame):
    global model
    if model:
        features = get_features(frame)
        pred = model.predict(features)[0]
        proba = model.predict_proba(features)[0]
        conf = np.max(proba)
    else:
        pred = np.random.choice([0,1,2], p=[0.85,0.08,0.07])
        conf = 0.95
    labels = ['Correct', 'Under', 'Over']
    return labels[pred], conf, pred != 0

def save_defect(frame, defect_type, conf):
    global last_alert
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'defect_{defect_type}_{timestamp}.jpg'
    path = os.path.join(DETECT_IMAGES_DIR, filename)
    cv2.imwrite(path, frame)
    
    log_entry = {
        'timestamp': timestamp,
        'defect_type': defect_type,
        'confidence': conf,
        'image': filename
    }
    log_path = os.path.join(LOGS_DIR, 'defects.jsonl')
    with open(log_path, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
    
    print(f'[ALERT] Saved defect: {filename} ({conf:.1%})')
    if HAS_EMAIL and time.time() - last_alert > cooldown:
        send_defect_alert(defect_type, conf, path)
        last_alert = time.time()

if __name__ == '__main__':
    if not open_camera():
        print('Cannot open camera. Test with "python test_camera.py"')
        exit(1)
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print('Failed to grab frame, retrying...')
            open_camera()
            time.sleep(0.5)
            continue
        
        status, conf, is_defect = predict_defect(frame)
        
        # Draw
        color = (0, 255, 0) if not is_defect else (0, 0, 255)
        cv2.putText(frame, f'{status}: {conf:.1%}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        if is_defect:
            save_defect(frame, status.lower(), conf)
        
        cv2.imshow('3D Print Defect Detector', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print('Detector stopped.')

