import numpy as np
import time

from openwakeword.model import Model

class WakeWordDetector:
    def __init__(self, wakeword: str = "hey_jarvis", threshold: float = 0.5) -> None:
        self.wakeword = wakeword
        self.threshold = threshold

        self.model = Model()

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
        print(f"[WAKEWORD] {time.monotonic():.3f} detected -> {score:.3f}")

        return float(score)

    def reset(self):
        self.model.reset()