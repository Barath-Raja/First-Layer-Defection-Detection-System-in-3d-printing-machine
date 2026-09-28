from flask import Flask, Response, send_file, jsonify, send_from_directory, request
from flask_cors import CORS
import cv2
import time
import os
from backend.detector import DefectDetector

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

detector = DefectDetector(camera_index=0)

def generate_frames():
    while True:
        frame = detector.processed_frame
        if frame is None:
            time.sleep(0.05)
            continue
            
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            time.sleep(0.05)
            continue
            
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.04)

@app.route('/')
@app.route('/first_layer_detector.html')
def index():
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
@app.route('/api/predict')
def api_predict():
    try:
        return jsonify(detector.get_status())
    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "UNKNOWN",
            "model1_extrusion": "UNKNOWN",
            "model2_adhesion": "UNKNOWN",
            "confidence": 0.0,
            "accuracy": 0.0,
            "is_defect": False,
            "recent_snapshots": []
        }), 500

@app.route('/api/simulate/<mode>', methods=['GET', 'POST'])
def api_simulate(mode):
    valid_modes = ['normal', 'no_extrusion', 'insufficient_extrusion', 'floating_thread', 'dragging', 'uncertain', 'reset']
    if mode not in valid_modes:
        return jsonify({"error": f"Invalid mode. Choose from {valid_modes}"}), 400
        
    set_mode = None if mode == 'reset' else mode
    detector.set_simulation_mode(set_mode)
    return jsonify({
        "success": True,
        "mode": set_mode,
        "message": f"Simulation mode set to '{set_mode}'" if set_mode else "Switched to live camera feed"
    })

@app.route('/defect_images/<path:filename>')
def serve_defect_image(filename):
    defect_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'defect_images')
    return send_from_directory(defect_dir, filename)

if __name__ == '__main__':
    print("=" * 60)
    print("  3D Printer First Layer Defect Detection Server")
    print("  Stream:    http://localhost:5001/video_feed")
    print("  Dashboard: http://localhost:5001/first_layer_detector.html")
    print("  API Status:http://localhost:5001/api/predict")
    print("=" * 60)
    try:
        app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
    finally:
        detector.stop()
