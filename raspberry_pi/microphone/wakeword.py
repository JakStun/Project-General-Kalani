import time
import numpy as np

from logging import getLogger
from openwakeword.model import Model


class WakeWordDetector:
    def __init__(
        self,
        wakeword: str = "hey_jarvis",
        threshold: float = 0.9,
        debounce: float = 1.5,
        model=None,
        min_hits: int = 1,
    ):
        self.logger = getLogger("main")

        self.wakeword = wakeword
        self.threshold = threshold
        self.debounce = debounce
        self.min_hits = max(1, min_hits)

        self.model = model if model is not None else Model()

        self._last_detection_time = 0.0
        self._hit_count = 0
        self._is_blocked = False

    def process(self, frame: np.ndarray) -> float:
        """
        frame:
            mono int16 @ 48 kHz
        """

        if self._is_blocked:
            return 0.0

        frame16 = frame[::3]

        predictions = self.model.predict(frame16)
        score = predictions[self.wakeword]

        if score < self.threshold:
            self._hit_count = 0
            return 0.0

        now = time.monotonic()

        if (
            self._last_detection_time > 0
            and now - self._last_detection_time < self.debounce
        ):
            return 0.0

        self._hit_count += 1

        if self._hit_count < self.min_hits:
            return 0.0

        self._hit_count = 0
        self._last_detection_time = now
        self._is_blocked = True

        return float(score)

    def reset(self):
        """
        Reset only the neural network state.

        Do NOT clear debounce.
        """
        self.model.reset()
        self._hit_count = 0

    def release(self):
        """
        Called after the whole interaction has finished.
        """
        self._is_blocked = False