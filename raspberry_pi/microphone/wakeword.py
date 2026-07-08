import numpy as np

from openwakeword.model import Model

class WakeWordDetector:
    def __init__(self, wakeword: str = "hey_jarvis", threshold: float = 0.5) -> None:
        self.wakeword = wakeword
        self.threshold = threshold

        self.model = Model()

    def process(self, frame: np.ndarray) -> bool:
        """
        Params:
        frame -> Mono int16 PCM at 48 kHz

        Returns:
        bool -> true if wakkeword detected
        """

        frame16 = frame[::3]

        predictions = self.model.predict(frame16)

        score = predictions[self.wakeword]

        return score >= self.threshold
