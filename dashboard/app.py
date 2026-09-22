import os
import sys
import json
import time
from flask import Flask, Response, send_from_directory, jsonify
from flask_socketio import SocketIO

# Disable resource tracker to avoid semaphore leaks on macOS
os.environ['PYTHONWARNINGS'] = 'ignore::ResourceWarning'

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from backend.detector_fixed import DefectDetector

app = Flask(__name__, static_folder='static', static_url_path='')
app.config['SECRET_KEY'] = '3dprint_monitor_secret!'
app.config['DEBUG'] = False
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

detector = DefectDetector()

# ------------------ BACKGROUND DETECTION TASK ------------------
def run_detection():
    """SocketIO background task - better for threading stability"""
    while True:
        try:
            result = detector.detect()
            socketio.emit('detection_update', result, namespace='/')
            print(f"Detection: {result['status']} ({result['confidence']:.1%})")
        except Exception as e:
            print(f"Detection error: {e}")
        time.sleep(0.5)  # 2 FPS

# ------------------ ROUTES ------------------
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/video_feed')
def video_feed():
    """Stable video streaming"""
    return Response(detector.gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

@app.route('/api/predict')
def api_predict():
    try:
        result = detector.detect()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "status": "error"})

@app.route('/api/predict_status')
def predict_status():
    """Added per original TODO"""
    try:
        result = detector.detect()
        return jsonify({
            "status": result['status'],
            "is_defect": result['is_defect'],
            "confidence": result['confidence']
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/logs')
def get_logs():
    logs = []
    try:
        log_file = 'logs/defects.jsonl'
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                for line in f.readlines()[-100:]:  # Last 100
                    logs.append(json.loads(line))
    except Exception as e:
        print(f"Logs error: {e}")
    return jsonify({'logs': logs})

@app.route('/defect_images/<path:filename>')
def serve_defect_image(filename):
    return send_from_directory('defect_images', filename)

# ------------------ MAIN ------------------
if __name__ == '__main__':
    os.makedirs('logs', exist_ok=True)
    os.makedirs('defect_images', exist_ok=True)
    
    # Start detection as SocketIO background task (more stable than Thread)
    socketio.start_background_task(run_detection)
    
    print("🚀 3D Print Monitor running on http://127.0.0.1:5000")
    print("📹 Check camera permissions in System Preferences > Security & Privacy")
    print("💡 Open: http://127.0.0.1:5000")
    
    # Fixed 403: localhost only, debug=False
    socketio.run(app, host='127.0.0.1', port=8080, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)
