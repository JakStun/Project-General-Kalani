from collections import deque
import numpy as np

class AudioBuffer:
    def __init__(self, seconds: float, sample_rate: int, frame_size: int) -> None:
        self.max_frames = int(seconds * sample_rate / frame_size)

        self.frames = deque(maxlen=self.max_frames)

    def push(self, frame: np.ndarray):
        self.frames.append(frame.copy())

    def clear(self):
        self.frames.clear()

    def get_audio(self) -> np.ndarray:
        if not self.frames:
            return np.empty((0,), dtype=np.int16)

        return np.concatenate(self.frames, axis=0)