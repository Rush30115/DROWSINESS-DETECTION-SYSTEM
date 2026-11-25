import unittest
from utils.helpers import calculate_ear

class TestEAR(unittest.TestCase):
    def test_calculate_ear_open(self):
        # Coordinates for an open eye (approximate)
        eye = [
            (0, 0), (1, 1), (2, 1), 
            (3, 0), (2, -1), (1, -1)
        ]
        # Vertical distances: A=2, B=2. Horizontal: C=3. EAR = (2+2)/(2*3) = 4/6 = 0.66
        ear = calculate_ear(eye)
        self.assertAlmostEqual(ear, 0.666, places=2)

    def test_calculate_ear_closed(self):
        # Coordinates for a closed eye (flat)
        eye = [
            (0, 0), (1, 0), (2, 0), 
            (3, 0), (2, 0), (1, 0)
        ]
        # Vertical distances: A=0, B=0. Horizontal: C=3. EAR = 0
        ear = calculate_ear(eye)
        self.assertEqual(ear, 0.0)

if __name__ == '__main__':
    unittest.main()
