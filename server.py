import threading
import time
import cv2
import json
from flask import Flask, Response, jsonify, request, send_from_directory

from detection.face_detector import FaceDetector
from detection.eye_detector import EyeDetector
from detection.glasses_detector import GlassesDetector
from utils.alarm import Alarm
from utils.notifier import TwilioNotifier

app = Flask(__name__, static_folder='.', template_folder='.')

# shared state between detection thread and HTTP handlers
state_lock = threading.Lock()
latest_frame = None  # BGR image
latest_ear = "--"
is_drowsy_state = False
glasses_status_state = "Unknown"
detection_active = True
ear_threshold = 0.25
logs = []
notifier = TwilioNotifier()


def _append_log(message):
    ts = time.strftime("%H:%M:%S")
    with state_lock:
        logs.append(f"[{ts}] {message}")
        logs[:] = logs[-200:]


def detection_loop():
    """Grab frames from webcam, run detectors and update shared state."""
    global latest_frame, latest_ear, is_drowsy_state, glasses_status_state, logs

    face_detector = FaceDetector()
    eye_detector = EyeDetector(ear_threshold=ear_threshold, consecutive_frames=20)
    glasses_detector = GlassesDetector()
    alarm = Alarm()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        with state_lock:
            logs.append("Unable to open webcam")
        return

    while True:
        with state_lock:
            active = detection_active
            eye_detector.ear_threshold = ear_threshold

        if not active:
            alarm.stop()
            time.sleep(0.1)
            continue

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        landmarks = face_detector.detect(frame)

        ear_val = 0.0
        is_drowsy = False
        glasses_status = "Unknown"

        if landmarks:
            left_eye, right_eye = [], []
            try:
                left_eye, right_eye = eye_detector.get_eye_landmarks(landmarks, (h, w, 3))
                is_drowsy, ear_val = eye_detector.check_drowsiness(left_eye, right_eye)
            except Exception:
                pass

            if is_drowsy:
                alarm.start()
            else:
                alarm.stop()

            notifier.handle_detection(
                is_drowsy,
                ear_val,
                on_result=lambda ok, msg: _append_log(msg),
            )

            glasses_status = glasses_detector.detect(frame, landmarks)
            for (x, y) in left_eye:
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            for (x, y) in right_eye:
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        else:
            alarm.stop()
            notifier.handle_detection(False, 0.0)

        with state_lock:
            latest_ear = ear_val
            is_drowsy_state = is_drowsy
            glasses_status_state = glasses_status
            if is_drowsy:
                ts = time.strftime("%H:%M:%S")
                logs.append(f"[{ts}] Drowsy alert — EAR={ear_val:.3f}")
                logs = logs[-200:]
            latest_frame = frame

            try:
                with open("data.json", "w") as f:
                    json.dump({"ear": latest_ear, "logs": logs}, f)
            except Exception:
                pass

        time.sleep(0.03)

    cap.release()
    alarm.stop()


# ------ Flask endpoints -------------------------------------------------

@app.route('/')
def index():
    return send_from_directory('.', 'dashboard.html')


@app.route('/data.json')
def data():
    with state_lock:
        return jsonify(
            ear=latest_ear,
            logs=list(logs),
            is_drowsy=is_drowsy_state,
            glasses=glasses_status_state,
            active=detection_active,
            threshold=ear_threshold,
            twilio_configured=notifier.is_configured(),
            twilio_missing=notifier.missing_fields(),
            twilio_detections=notifier.detection_count,
            twilio_alert_after=notifier.alert_after,
        )


@app.route('/api/test_sms', methods=['POST'])
def test_sms():
    ok, detail = notifier.send_test_message()
    _append_log(detail)
    return jsonify(ok=ok, detail=detail), (200 if ok else 400)


@app.route('/api/toggle', methods=['POST'])
def toggle_detection():
    global detection_active, is_drowsy_state
    with state_lock:
        detection_active = not detection_active
        if not detection_active:
            is_drowsy_state = False
        ts = time.strftime("%H:%M:%S")
        logs.append(f"[{ts}] Detection {'resumed' if detection_active else 'paused'}")
        logs[:] = logs[-200:]
        return jsonify(active=detection_active)


@app.route('/api/clear_logs', methods=['POST'])
def clear_logs():
    with state_lock:
        logs.clear()
    return jsonify(ok=True)


@app.route('/api/threshold', methods=['POST'])
def set_threshold():
    global ear_threshold
    payload = request.get_json(silent=True) or {}
    try:
        val = float(payload.get("threshold", ear_threshold))
        val = max(0.10, min(0.45, val))
    except (TypeError, ValueError):
        return jsonify(error="Invalid threshold"), 400
    with state_lock:
        ear_threshold = round(val, 2)
        ts = time.strftime("%H:%M:%S")
        logs.append(f"[{ts}] EAR threshold set to {ear_threshold:.2f}")
        logs[:] = logs[-200:]
        return jsonify(threshold=ear_threshold)


@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with state_lock:
                frame = latest_frame
            if frame is None:
                time.sleep(0.02)
                continue
            ret, buf = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame_bytes = buf.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.03)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    threading.Thread(target=detection_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)
