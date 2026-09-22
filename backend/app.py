import os
import cv2
import numpy as np
import threading
import time
import base64
from flask import Flask, Response, jsonify, send_from_directory, request
from flask_cors import CORS

from detector import DefectDetector

# Setup Flask with correct static folder relative to backend/
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static')
app = Flask(__name__, static_folder=static_dir)
CORS(app)

detector = DefectDetector(camera_index=0)

# Disable browser caching for development updates
@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    return r

def generate_frames():
    """Generator to continuously yield JPEG frames from the background thread."""
    while True:
        frame = detector.processed_frame
        if frame is None:
            time.sleep(0.1)
            continue
            
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            time.sleep(0.1)
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Stream at roughly 20 FPS
        time.sleep(0.05)

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

@app.route('/api/predict', methods=['GET'])
def api_predict():
    """Return JSON with {status, confidence, is_defect}"""
    try:
        status_data = detector.get_status()
        return jsonify(status_data)
    except Exception as e:
        return jsonify({"error": str(e), "status": "Unknown", "confidence": 0.0, "is_defect": False}), 500

@app.route('/api/upload', methods=['POST'])
def api_upload():
    """Process an uploaded static image"""
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided."}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Empty filename."}), 400
        
    try:
        # Read file bytes to numpy array
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Invalid image architecture"}), 400
            
        # Analyze without modifying the camera stream values
        draw_frame, status, conf, is_defect = detector.analyze_frame(img)
        
        # Convert annotated frame back to base64
        ret, buffer = cv2.imencode('.jpg', draw_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "status": status,
            "confidence": conf,
            "is_defect": is_defect,
            "image": img_base64
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/video_feed')
def video_feed():
    try:
        return Response(
            generate_frames(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    print("=" * 50)
    print("  3D Print Defect Detection API Server  ")
    print("  Server is running on http://127.0.0.1:8080")
    print("  Press Ctrl+C to stop")
    print("=" * 50)
    
    try:
        app.run(host='0.0.0.0', port=8080, debug=False, threaded=True, use_reloader=False)
    finally:
        detector.stop()
