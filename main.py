import cv2
import time
import mediapipe as mp
from detection.face_detector import FaceDetector
from detection.eye_detector import EyeDetector
from detection.glasses_detector import GlassesDetector
from utils.alarm import Alarm
from utils.helpers import setup_logger
from utils.notifier import TwilioNotifier

def main():
    # Setup Logger
    logger = setup_logger("logs/events.log")
    logger.info("System Started (MediaPipe Mode)")

    # Initialize Detectors
    face_detector = FaceDetector()
    eye_detector = EyeDetector(ear_threshold=0.25, consecutive_frames=20)
    glasses_detector = GlassesDetector()
    alarm = Alarm()
    notifier = TwilioNotifier()

    # Open Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not open webcam.")
        return

    logger.info("Webcam opened. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame for mirror view
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Detect Face
        landmarks = face_detector.detect(frame)

        status = "Awake"
        glasses_status = "Unknown"
        ear_value = 0.0

        if landmarks:
            # Draw face mesh landmarks for visualization
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp.solutions.drawing_utils.draw_landmarks(
                    frame_rgb,
                    landmarks,
                    face_detector.mp_face_mesh.FACEMESH_TESSELATION,
                    mp.solutions.drawing_utils.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
                    mp.solutions.drawing_utils.DrawingSpec(color=(0,0,255), thickness=1)
                )
                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            except Exception:
                # If drawing fails, continue without crashing
                pass

            # Draw bounding box around the detected face (based on landmarks)
            try:
                xs = [int(lm.x * w) for lm in landmarks.landmark]
                ys = [int(lm.y * h) for lm in landmarks.landmark]
                x_min, x_max = max(min(xs) - 10, 0), min(max(xs) + 10, w)
                y_min, y_max = max(min(ys) - 10, 0), min(max(ys) + 10, h)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            except Exception:
                pass
            # Get Eye Landmarks
            left_eye, right_eye = eye_detector.get_eye_landmarks(landmarks, (h, w, 3))
            
            # Check Drowsiness
            is_drowsy, ear_value = eye_detector.check_drowsiness(left_eye, right_eye)
            
            # Detect Glasses
            glasses_status = glasses_detector.detect(frame, landmarks)

            # Draw Eyes (larger markers for visibility) and connect with small lines
            for (x, y) in left_eye:
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            for (x, y) in right_eye:
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            # Optionally draw a thicker center point for the first landmark of each eye
            if left_eye:
                cv2.circle(frame, left_eye[0], 4, (0, 200, 0), -1)
            if right_eye:
                cv2.circle(frame, right_eye[0], 4, (0, 200, 0), -1)

            

            # Handle Alert
            if is_drowsy:
                status = "DROWSY!"
                alarm.start()
                cv2.putText(frame, "DROWSINESS DETECTED!", (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                logger.warning(f"Drowsiness detected! EAR: {ear_value:.2f}")
            else:
                alarm.stop()

            notifier.handle_detection(
                is_drowsy,
                ear_value,
                on_result=lambda ok, msg: logger.info(msg) if ok else logger.error(msg),
            )

        # Display Info
        cv2.putText(frame, f"EAR: {ear_value:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Status: {status}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if status == "Awake" else (0, 0, 255), 2)
        cv2.putText(frame, f"Glasses: {glasses_status}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Driver Sleep Alert System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    alarm.stop()
    logger.info("System Stopped")

if __name__ == "__main__":
    main()
