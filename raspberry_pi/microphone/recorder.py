import numpy as np

class Recorder:
    async def read_frame(self) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement this method.")