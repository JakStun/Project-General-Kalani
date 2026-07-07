import numpy as np

from openwakeword.model import Model

class WakeWordDetector:
    def __init__(self, wakeword: str = "hey_jarvis", threshold: float = 0.5) -> None:
        self.threshold = threshold

        self.model = Model(
            wakeword_models=[wakeword],
        )

    def process(self, frame: np.ndarray) -> bool:
        """
        Params:
        frame -> Mono int16 PCM at 48 kHz

        Returns:
        bool -> true if wakkeword detected
        """

        frame16 = frame[::3]

        prediction = self.model.predict(frame16)

        score = prediction["hey_jarvis"]

        return score >= self.threshold

    @property
    def detected(self) -> bool:
        pass