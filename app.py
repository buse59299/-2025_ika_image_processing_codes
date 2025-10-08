from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
import threading

app = Flask(__name__)

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Using nano model for faster inference

# Global variables for camera
camera = None
camera_lock = threading.Lock()

def get_camera():
    """Get or initialize the camera."""
    global camera
    with camera_lock:
        if camera is None:
            camera = cv2.VideoCapture(0)
            # Set camera properties for better performance
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return camera

def generate_frames():
    """Generate frames with object detection."""
    cam = get_camera()
    
    while True:
        success, frame = cam.read()
        if not success:
            break
        
        # Perform object detection
        results = model(frame, verbose=False)
        
        # Draw detection results on the frame
        annotated_frame = results[0].plot()
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        
        # Yield frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)
