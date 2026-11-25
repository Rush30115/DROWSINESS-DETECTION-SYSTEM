import cv2
import mediapipe as mp
import logging

class FaceDetector:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
    def detect(self, image):
        """
        Detects face landmarks in the image.
        :param image: BGR image from OpenCV.
        :return: face landmarks if face detected, else None.
        """
        if image is None:
            return None

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Performance optimization: mark read-only while processing
        image_rgb.flags.writeable = False

        # Process the image
        results = self.face_mesh.process(image_rgb)

        # restore writeable flag (good practice if image is reused)
        image_rgb.flags.writeable = True

        # Log whether detection found landmarks using the project's logger
        try:
            found = bool(results and results.multi_face_landmarks)
        except Exception:
            found = False
        logger = logging.getLogger()
        logger.debug(f"[FaceDetector] landmarks found: {found}")

        if results and results.multi_face_landmarks:
            return results.multi_face_landmarks[0]

        return None