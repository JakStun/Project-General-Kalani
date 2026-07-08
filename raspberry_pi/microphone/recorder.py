import asyncio
import sounddevice as sd
import numpy as np

class Recorder:
    FRAME_MS = 30

    def __init__(self, device=2) -> None:
        device_info = sd.query_devices(device)

        self.device = device

        self.sample_rate = 48000
        self.channels = 1


        self.frame_size = int(
            self.sample_rate * self.FRAME_MS / 1000
        )

        self.stream = sd.InputStream(
            samplerate=48000,
            channels=2,
            dtype="int16",
            device=2,
            blocksize=self.frame_size,
        )

        self.stream.start()

    async def read_frame(self) -> np.ndarray:
        frame, overflow = self.stream.read(self.frame_size)
        frame = frame[:, 0].copy() # convert to mono         
        
        return frame, overflow

    def close(self) -> None:
        self.stream.stop()
        self.stream.close()

if __name__ == "__main__":
    import asyncio

    recorder = Recorder()

    async def main():
        while True:
            frame = await recorder.read_frame()
            print(f"Read {len(frame)} samples")

    asyncio.run(main())