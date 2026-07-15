import numpy as np
import time
import wave

from datetime import datetime
from logging import getLogger

from events import Event

from .audio_buffer import AudioBuffer
from .recorder import Recorder
from .vad import VoiceActivityDetector
from .wakeword import WakeWordDetector

from .microphone_state import MicrophoneState

class MicrophoneControl:
    def __init__(self, event_queue):
        self.logger = getLogger("main")

        self.event_queue = event_queue

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

        # self.wakeword_detected = False

        self.state = MicrophoneState.LISTENING

        self.recording_frames = []

        self.recording_silence = 0

        self.recording_stop_frames = 40

    async def run(self):
        self.logger.info("[MIC] Listening ...")

        while True:

            await self.wait_for_wakeword()

            audio = await self.record_interaction()

            await self.queue.put(audio)

    async def wait_for_wakeword(self):
        while True:
            frame = await self.recorder.read_frame()

            self.buffer.push(frame)

            if not self.vad.is_speech(frame):
                continue

            score = self.wakeword.process(frame)

            if score >= self.wakeword.threshold:
                return
            
    async def record_interaction(self):
        audio = self.buffer.get_audio()

        silence = 0

        while True:
            frame = await self.recorder.read_frame()

            audio.append(frame)

            if self.vad.is_speech(frame):
                silence = 0
            else:
                silence += 1

            if silence >= 40:
                break

        return concatenate(auido)

               
if __name__ == "__main__":
    import asyncio

    mic = MicrophoneControl()

    asyncio.run(mic.run())