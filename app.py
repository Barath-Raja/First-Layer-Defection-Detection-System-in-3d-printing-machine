from flask import Flask, Response, send_file, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import time
import os
from backend.detector import DefectDetector

app = Flask(__name__)
CORS(app)  # Allow HTML dashboard to connect

# Initialize Real ML Defect Detector
detector = DefectDetector(camera_index=0)

def generate_frames():
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
        
        time.sleep(0.05)

@app.route('/')
@app.route('/first_layer_detector.html')
def index():
    import os
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'first_layer_detector.html')
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"<h1>Error loading dashboard:</h1> <p>{str(e)}</p>"

@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/status')
def status():
    return jsonify({"status": "online", "camera": "connected", "ai_model": "loaded"})

@app.route('/api/predict')
def api_predict():
    try:
        return jsonify(detector.get_status())
    except Exception as e:
        return jsonify({"error": str(e), "status": "Unknown", "confidence": 0.0, "is_defect": False, "recent_snapshots": []}), 500

@app.route('/defect_images/<path:filename>')
def serve_defect_image(filename):
    defect_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'defect_images')
    return send_from_directory(defect_dir, filename)
    
if __name__ == '__main__':
    print("=" * 50)
    print("  3D Print First Layer Production Camera Server")
    print("  Stream: http://localhost:5001/video_feed")
    print("  Dashboard: open first_layer_detector.html")
    print("  Press Ctrl+C to stop")
    print("=" * 50)
    try:
        app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
    finally:
        detector.stop()

