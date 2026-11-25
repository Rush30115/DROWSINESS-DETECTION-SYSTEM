import sys
sys.path.insert(0, r'c:\Users\rusha\Downloads\driver_sleep_alert-master\driver_sleep_alert-master')
import cv2
from detection.face_detector import FaceDetector
from detection.eye_detector import EyeDetector

fd = FaceDetector()
ed = EyeDetector()
cap = cv2.VideoCapture(0)
count = 0
while count < 50:
    ret, frame = cap.read()
    if not ret:
        print('no frame')
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    landmarks = fd.detect(frame)
    if not landmarks:
        print(count, 'no landmarks')
    else:
        left_eye, right_eye = ed.get_eye_landmarks(landmarks, (h, w, 3))
        print(count, 'landmarks:', len(left_eye), len(right_eye), 'first left:', left_eye[0] if left_eye else None)
    count += 1
cap.release()
print('done')
