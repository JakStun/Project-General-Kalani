import numpy as np

from logging import getLogger

from .audio_buffer import AudioBuffer
from .recorder import Recorder
from .vad import VoiceActivityDetector
from .wakeword import WakeWordDetector

class MicrophoneControl:
    def __init__(self):
        self.logger = getLogger("main")
        
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

    async def run(self):
        self.logger.info("[MIC] Listening ...")

        while True:

            frame, overflow = await self.recorder.read_frame()
            
            self.buffer.push(frame)

            if overflow:
                self.logger.warning(f"[MIC] Overflow")

            detected = self.wakeword.process(frame)

            if detected:
                self.logger.info("[MIC] Wake word detected")

            rms = np.sqrt(np.mean(frame.astype(np.float32) ** 2))

            if rms < 200:
                self.speaking = False
            else:
                is_speech = self.vad.is_speech(frame)

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

if __name__ == "__main__":
    import asyncio

    mic = MicrophoneControl()

    asyncio.run(mic.run())