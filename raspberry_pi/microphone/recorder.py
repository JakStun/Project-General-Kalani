import asyncio
import sounddevice as sd
import numpy as np
# import threading
import queue

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
            callback=self._audio_callback,
        )

        self.stream.start()

        self.queue = queue.Queue(maxsize=10)

        # self.running = True

        # self.thread = threading.Thread(
        #     target=self._capture_loop,
        #     daemon=True
        # )

        # self.thread.start()

    async def read_frame(self) -> np.ndarray:
        return await asyncio.to_thread(
            self.queue.get
        )

    def close(self) -> None:
        # self.running = False
        
        # self.thread.join(timeout=1)

        self.stream.stop()
        self.stream.close()

    def _audio_callback(self, indata, frames, time, status):
        overflow = status.input_overflow

        frame = indata[:, 0].copy()
        
        try:
            self.queue.put_nowait((frame, overflow))
        except queue.Full:
            self.queue.get_nowait()
            self.queue.put_nowait((frame, overflow))

    def _capture_loop(self) -> None:
        while self.running:
            frame, overflow = self.stream.read(self.frame_size)

            frame = frame[:, 0].copy()

            if self.queue.full():
                self.queue.get_nowait()

            self.queue.put_nowait((frame, overflow))

if __name__ == "__main__":
    import asyncio

    recorder = Recorder()

    async def main():
        while True:
            frame = await recorder.read_frame()
            print(f"Read {len(frame)} samples")

    asyncio.run(main())