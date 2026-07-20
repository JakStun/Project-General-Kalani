import numpy as np
import time

from logging import getLogger
from openwakeword.model import Model


class WakeWordDetector:
    def __init__(
        self,
        wakeword: str = "hey_jarvis",
        threshold: float = 0.9,
        debounce: float = 3.0,
        model=None,
        min_hits: int = 1,
    ) -> None:
        self.logger = getLogger("main")

        self.wakeword = wakeword
        self.threshold = threshold
        self.debounce = debounce
        self.min_hits = max(1, min_hits)

        self.model = model if model is not None else Model()
        self._last_detection_time = 0.0
        self._is_blocked = False
        self._hit_count = 0

    def process(self, frame: np.ndarray) -> float:
        """
        Params:
        frame -> Mono int16 PCM at 48 kHz

        Returns:
        float -> confidence score for the wakeword
        """

        t0 = time.perf_counter()
        frame16 = frame[::3]

        predictions = self.model.predict(frame16)
        score = predictions[self.wakeword]

        dt = (time.perf_counter() - t0) * 1000

        # print(f"[WAKEWORD] Score: {score:.3f} (took {dt:.1f} ms)")
        # print(f"[WAKEWORD] {time.monotonic():.3f} detected -> {score:.3f}")

        if self._is_blocked:
            return 0.0

        now = time.monotonic()
        if self._last_detection_time > 0.0 and (now - self._last_detection_time) < self.debounce:
            return 0.0

        if score < self.threshold:
            self._hit_count = 0
            return 0.0

        self._hit_count += 1
        if self._hit_count < self.min_hits:
            return 0.0

        self._last_detection_time = now
        self._is_blocked = True
        self._hit_count = 0
        return float(score)

    def reset(self):
        self._last_detection_time = 0.0
        self._is_blocked = False
        self._hit_count = 0
        self.model.reset()

    def release(self):
        self._is_blocked = False
        self._hit_count = 0