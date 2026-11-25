from utils.helpers import calculate_ear

class EyeDetector:
    def __init__(self, ear_threshold=0.20, consecutive_frames=40):
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames
        self.frame_counter = 0
        
        # MediaPipe Face Mesh Indices for Eyes (468 point mesh)
        # Left Eye
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        # Right Eye
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]

    def get_eye_landmarks(self, face_landmarks, image_shape):
        """
        Extracts eye landmarks from face landmarks.
        """
        h, w, _ = image_shape
        left_eye_coords = []
        right_eye_coords = []

        for idx in self.LEFT_EYE:
            lm = face_landmarks.landmark[idx]
            left_eye_coords.append((int(lm.x * w), int(lm.y * h)))
            
        for idx in self.RIGHT_EYE:
            lm = face_landmarks.landmark[idx]
            right_eye_coords.append((int(lm.x * w), int(lm.y * h)))

        return left_eye_coords, right_eye_coords


    def check_drowsiness(self, left_eye, right_eye):
        """
        Calculates EAR and checks for drowsiness.
        :return: (is_drowsy, avg_ear)
        """
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0

        if avg_ear < self.ear_threshold:
            self.frame_counter += 1
        else:
            self.frame_counter = 0

        is_drowsy = self.frame_counter >= self.consecutive_frames
        return is_drowsy, avg_ear
