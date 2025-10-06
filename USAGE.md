# YOLOv8 Flask Web Application - Usage Guide

## Overview
This Flask web application provides real-time object detection using YOLOv8 (You Only Look Once version 8) on a live camera feed. The application streams video from your webcam and performs real-time object detection, displaying bounding boxes around detected objects.

## Features

### Core Functionality
- **Real-time Video Streaming**: Live camera feed streamed to web browser
- **Object Detection**: YOLOv8 nano model for fast, accurate detection
- **Web Interface**: Clean, responsive web UI accessible via browser
- **Multi-threading**: Efficient handling of video streams and detection
- **80 Object Classes**: Detects common objects from COCO dataset

### Technical Features
- **Flask Framework**: Lightweight web server
- **OpenCV Integration**: Video capture and processing
- **YOLOv8 Nano Model**: Optimized for speed and accuracy balance
- **Thread-safe Camera Access**: Prevents resource conflicts
- **MJPEG Streaming**: Efficient video streaming protocol

## Installation

### Prerequisites
- Python 3.8 or higher
- Webcam or external camera
- Internet connection (for first-time model download)

### Step-by-step Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/buse59299/-2025_ika_image_processing_codes.git
   cd -2025_ika_image_processing_codes
   ```

2. **Create virtual environment** (recommended)
   ```bash
   # On Linux/Mac
   python3 -m venv venv
   source venv/bin/activate

   # On Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   This will install:
   - Flask 3.0.0 - Web framework
   - opencv-python 4.8.1.78 - Computer vision library
   - ultralytics 8.0.220 - YOLOv8 implementation
   - torch 2.1.0 - PyTorch deep learning framework
   - torchvision 0.16.0 - PyTorch vision utilities

4. **First run - Model download**
   On first run, YOLOv8 will automatically download the nano model (yolov8n.pt, ~6MB).

## Running the Application

### Basic Usage

1. **Start the server**
   ```bash
   python app.py
   ```

2. **Access the web interface**
   Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

3. **Allow camera access**
   Your browser may ask for camera permissions - allow it.

4. **View detections**
   The live feed with object detections will appear automatically.

### Advanced Usage

#### Custom Port
```bash
# Edit app.py and change:
app.run(debug=True, host='0.0.0.0', port=8080, threaded=True)
```

#### Production Deployment
For production, use a WSGI server like Gunicorn:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Application Architecture

### File Structure
```
-2025_ika_image_processing_codes/
├── app.py                 # Main Flask application
├── templates/
│   └── index.html        # Web interface template
├── requirements.txt      # Python dependencies
├── .gitignore           # Git ignore rules
└── README.md            # Project documentation
```

### Code Architecture

#### app.py Components

1. **Imports and Setup**
   - Flask for web framework
   - OpenCV for video capture
   - YOLO for object detection
   - Threading for concurrent access

2. **Model Loading**
   ```python
   model = YOLO('yolov8n.pt')
   ```
   Loads YOLOv8 nano model on startup

3. **Camera Management**
   ```python
   def get_camera():
   ```
   Thread-safe camera initialization and access

4. **Frame Generation**
   ```python
   def generate_frames():
   ```
   - Captures frames from camera
   - Runs YOLO detection
   - Annotates frames with bounding boxes
   - Encodes as JPEG
   - Yields for streaming

5. **Routes**
   - `/` - Main page (renders HTML template)
   - `/video_feed` - Video stream endpoint

## How It Works

### Detection Pipeline

1. **Camera Capture**: OpenCV captures frames from webcam (640x480)
2. **Object Detection**: YOLOv8 processes each frame
3. **Annotation**: Detected objects are marked with bounding boxes
4. **Encoding**: Frame is encoded as JPEG
5. **Streaming**: Frame is sent to browser via MJPEG stream
6. **Display**: Browser displays the annotated video feed

### Performance Optimization

- **YOLOv8 Nano**: Fastest YOLO variant for real-time performance
- **Resolution**: Set to 640x480 for speed
- **Threading**: Prevents blocking during video capture
- **Verbose=False**: Suppresses detection logs for cleaner output

## Detected Object Classes

YOLOv8 can detect 80 object classes from the COCO dataset, including:

**People & Animals**: person, cat, dog, horse, bird, etc.
**Vehicles**: car, motorcycle, airplane, bus, train, truck
**Indoor Objects**: chair, couch, TV, laptop, keyboard, cell phone
**Outdoor Objects**: traffic light, fire hydrant, stop sign, bench
**Food**: banana, apple, sandwich, pizza, donut, cake
**And many more...**

## Troubleshooting

### Camera Not Found
- Ensure your camera is connected and working
- Check camera index (change `0` to `1` or `2` in `cv2.VideoCapture(0)`)
- Verify camera permissions in your OS settings

### Slow Performance
- Use a GPU-enabled PyTorch installation for better performance
- Reduce video resolution in camera settings
- Consider using YOLOv8n (nano) instead of larger models

### Model Download Issues
- Ensure internet connection for first run
- Manually download yolov8n.pt from Ultralytics website
- Check firewall settings

### Port Already in Use
- Change the port number in app.py
- Kill the process using port 5000:
  ```bash
  # Linux/Mac
  lsof -ti:5000 | xargs kill -9
  
  # Windows
  netstat -ano | findstr :5000
  taskkill /PID <PID> /F
  ```

## Customization

### Using Different YOLO Models
```python
# Faster but less accurate
model = YOLO('yolov8n.pt')  # Nano (current)

# More accurate but slower
model = YOLO('yolov8s.pt')  # Small
model = YOLO('yolov8m.pt')  # Medium
model = YOLO('yolov8l.pt')  # Large
model = YOLO('yolov8x.pt')  # Extra Large
```

### Adjusting Camera Resolution
```python
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
```

### Modifying Detection Confidence
```python
results = model(frame, conf=0.5)  # Only show detections > 50% confidence
```

## Security Considerations

- **Local Network Only**: By default, runs on all interfaces (0.0.0.0)
- **Debug Mode**: Disable debug mode in production
- **Authentication**: Add authentication for public deployments
- **HTTPS**: Use HTTPS in production environments

## License
Educational purposes only.

## Support
For issues and questions, please refer to the GitHub repository issues page.
