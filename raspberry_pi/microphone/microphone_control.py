from logging import getLogger

from .recorder import Recorder
from .audio_buffer import AudioBuffer
from .vad import VoiceActivityDetector

class MicrophoneControl:
    def __init__(self):
        self.logger = getLogger("main")
        
        self.recorder = Recorder()

        self.buffer = AudioBuffer(
            seconds=3,
            sample_rate=self.recorder.sample_rate,
            frame_size=self.recorder.frame_size
        )

        self.vad = VoiceActivityDetector(sample_rate=self.recorder.sample_rate)
        self.speaking = False

    async def run(self):
        self.logger.info("[MIC] Listening ...")

        while True:

            frame, overflow = await self.recorder.read_frame()
            
            self.buffer.push(frame)

            if overflow:
                self.logger.warning(f"[MIC] Overflow")

            is_speech = self.vad.is_speech(frame)

            if is_speech and not self.speaking:
                self.speaking = True
                self.logger.info(f"[MIC] Started speaking")
            elif not is_speech and self.speaking:
                self.speaking = False
                self.logger.info(f"[MIC] Stopped speaking")

            # self.logger.debug(frame.shape)

if __name__ == "__main__":
    import asyncio

    mic = MicrophoneControl()

    asyncio.run(mic.run())