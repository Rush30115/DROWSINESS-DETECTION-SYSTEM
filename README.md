# Driver Sleep Alert System

A real-time drowsiness detection system using computer vision.

## Features
- Face Detection
- Eye Blink & Drowsiness Detection (EAR)
- Glasses Detection
- Real-time Audio Alerts
- Event Logging

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. There are two ways to run the system:
   * **Standalone OpenCV mode** – run `python main.py` and a local window will
     show the camera feed along with EAR/status.
   * **Web dashboard mode** – start the Flask server which provides a
     live MJPEG stream and JSON data used by `dashboard.html`:
     ```bash
     python server.py
     ```
     Then open a browser to <http://localhost:5000> (or your host/port).

## Controls
- When running `main.py`, press 'q' to quit.
- On the web dashboard you can stop detection with the <kbd>Stop Detection</kbd>
  button or Ctrl+Q; logs and EAR are refreshed every second.
