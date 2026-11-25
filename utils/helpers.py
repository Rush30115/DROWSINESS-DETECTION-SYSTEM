import logging
import os
from scipy.spatial import distance as dist

def setup_logger(log_file):
    """Sets up the logger to write to a file and console."""
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

def calculate_ear(eye):
    """
    Calculates the Eye Aspect Ratio (EAR).
    :param eye: List of 6 (x, y) coordinates for the eye landmarks.
    :return: EAR value
    """
    # Vertical distances
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # Horizontal distance
    C = dist.euclidean(eye[0], eye[3])

    # EAR formula
    ear = (A + B) / (2.0 * C)
    return ear
