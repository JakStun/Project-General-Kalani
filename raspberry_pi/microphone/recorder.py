import asyncio
import queue

import numpy as np
import sounddevice as sd


class Recorder:
    FRAME_MS = 30

    def __init__(self, device=2):
        self.device = device

        self.sample_rate = 48000
        self.channels = 1

        self.frame_size = int(
            self.sample_rate * self.FRAME_MS / 1000
        )

        self.queue = queue.Queue(maxsize=10)

        self.stream = sd.InputStream(
            device=self.device,
            samplerate=self.sample_rate,
            channels=2,
            dtype="int16",
            blocksize=self.frame_size,
            callback=self._audio_callback,
        )

        self.stream.start()

    async def read_frame(self):
        return await asyncio.to_thread(
            self.queue.get
        )

    def pause(self):
        if self.stream.active:
            self.stream.stop()

    def resume(self):
        if not self.stream.active:
            self.stream.start()

    def close(self):
        self.pause()
        self.stream.close()

    def clear(self):
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break

    def _audio_callback(self, indata, frames, time, status):
        frame = indata[:, 0].copy()

        overflow = status.input_overflow

        try:
            self.queue.put_nowait(
                (frame, overflow)
            )

        except queue.Full:
            self.queue.get_nowait()
            self.queue.put_nowait(
                (frame, overflow)
            )


if __name__ == "__main__":
    async def main():
        recorder = Recorder()

        while True:
            frame, overflow = await recorder.read_frame()

            print(frame.shape, overflow)

    asyncio.run(main())