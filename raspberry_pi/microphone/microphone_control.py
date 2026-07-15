import numpy as np

from logging import getLogger

from events import Event

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
            frame_size=self.recorder.frame_size,
        )

        self.vad = VoiceActivityDetector(
            sample_rate=self.recorder.sample_rate
        )

        self.wakeword = WakeWordDetector()

        self.recording_stop_frames = 40

    async def wait_for_interaction(self):
        self.logger.info("[MIC] Listening...")

        while True:

            await self.wait_for_wakeword()

            self.logger.info("[MIC] Wakeword detected")

            audio = await self.record_interaction()

            self.logger.info("[MIC] Recording finished")

    async def wait_for_wakeword(self):
        self.buffer.clear()
        self.wakeword.reset()

        while True:

            frame, overflow = await self.recorder.read_frame()

            self.buffer.push(frame)

            if not self.vad.is_speech(frame):
                continue

            score = self.wakeword.process(frame)

            if score >= self.wakeword.threshold:
                return

    async def record_interaction(self):

        recording_frames = [
            self.buffer.get_audio()
        ]

        silence = 0

        while True:

            frame, overflow = await self.recorder.read_frame()

            recording_frames.append(frame.copy())

            if self.vad.is_speech(frame):
                silence = 0
            else:
                silence += 1

            if silence >= self.recording_stop_frames:
                break

        return np.concatenate(recording_frames)