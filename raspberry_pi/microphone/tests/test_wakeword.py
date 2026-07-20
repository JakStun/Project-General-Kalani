import unittest

import numpy as np

from microphone.wakeword import WakeWordDetector


class DummyModel:
    def __init__(self, score=0.95):
        self.score = score

    def predict(self, frame):
        return {"hey_jarvis": self.score}


class WakeWordDetectorTests(unittest.TestCase):
    def test_debounce_suppresses_immediate_repeated_detections(self):
        detector = WakeWordDetector(threshold=0.5, debounce=1.0, model=DummyModel())

        first_score = detector.process(np.zeros(48000, dtype=np.int16))
        second_score = detector.process(np.zeros(48000, dtype=np.int16))

        self.assertGreater(first_score, detector.threshold)
        self.assertEqual(second_score, 0.0)

    def test_requires_multiple_hits_before_detecting(self):
        detector = WakeWordDetector(threshold=0.5, debounce=1.0, model=DummyModel())

        first_score = detector.process(np.zeros(48000, dtype=np.int16))
        second_score = detector.process(np.zeros(48000, dtype=np.int16))

        self.assertGreater(first_score, detector.threshold)
        self.assertEqual(second_score, 0.0)


if __name__ == "__main__":
    unittest.main()
