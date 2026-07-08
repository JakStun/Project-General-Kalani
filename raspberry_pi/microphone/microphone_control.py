import queue
import threading

from logging import getLogger

from .audio_buffer import AudioBuffer
from .recorder import Recorder
from .vad import VoiceActivityDetector
from .wakeword import WakeWordDetector

class MicrophoneControl:
    def __init__(self):
        self.logger = getLogger("main")

        self.wakeword_queue = queue.Queue(maxsize=10)

        self.wakeword_thread = threading.Thread(
            target=self._wakeword_worker,
            daemon=True
        )

        self.wakeword_thread.start()

        self.recorder = Recorder()

        self.buffer = AudioBuffer(
            seconds=3,
            sample_rate=self.recorder.sample_rate,
            frame_size=self.recorder.frame_size
        )

        self.wakeword = WakeWordDetector()

        self.vad = VoiceActivityDetector(sample_rate=self.recorder.sample_rate)
        self.speaking = False

        self.speech_frames = 0
        self.silence_frames = 0

        self.start_threshold = 3
        self.stop_threshold = 10

        self.wakeword_detected = False

    async def run(self):
        self.logger.info("[MIC] Listening ...")

        while True:

            frame, overflow = await self.recorder.read_frame()
            
            self.buffer.push(frame)

            is_speech = self.vad.is_speech(frame)

            try:
                self.wakeword_queue.put_nowait(frame.copy())
            except queue.Full:
                pass

            if self.wakeword_detected:
                self.logger.info("[MIC] Wakeword detected!")
                self.wakeword_detected = False


            if is_speech and not self.speaking:
                self.speech_frames += 1
                self.silence_frames = 0
            elif not is_speech and self.speaking:
                self.silence_frames += 1
                self.speech_frames = 0

            if not self.speaking and self.speech_frames >= self.start_threshold:
                self.speaking = True
                self.logger.info("[MIC] Speech started")
            elif self.speaking and self.silence_frames >= self.stop_threshold:
                self.speaking = False
                self.logger.info("[MIC] Speech stopped")

    def _wakeword_worker(self):
        while True:
            frame = self.wakeword_queue.get()

            detected = self.wakeword.process(frame)

            if detected:
                self.wakeword_detected = True

if __name__ == "__main__":
    import asyncio

    mic = MicrophoneControl()

    asyncio.run(mic.run())