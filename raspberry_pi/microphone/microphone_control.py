import asyncio

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
            frame_size=self.recorder.frame_size,
        )

        self.vad = VoiceActivityDetector(
            sample_rate=self.recorder.sample_rate
        )

        self.wakeword = WakeWordDetector(
            debounce=1.5
        )

        self.recording_stop_frames = 40

        self.listening_enabled = True

    def pause_listening(self):
        self.listening_enabled = False

    def resume_listening(self):
        self.listening_enabled = True

    async def wait_for_interaction(self):
        self.logger.info("[MIC] Listening...")

        while True:

            if not self.listening_enabled:
                await asyncio.sleep(0.1)
                continue

            self.recorder.clear()
            self.buffer.clear()

            self.wakeword.reset()

            await self.wait_for_wakeword()

            self.logger.info("[MIC] Wakeword detected")

            audio = await self.record_interaction()

            self.logger.info("[MIC] Recording finished")

            self.wakeword.release()

            if audio.size == 0:
                continue

            return audio

    async def wait_for_wakeword(self):

        while True:

            frame, overflow = await self.recorder.read_frame()

            self.buffer.push(frame)

            if not self.vad.is_speech(frame):
                continue

            score = self.wakeword.process(frame)

            if score >= self.wakeword.threshold:
                self.logger.info(
                    "[MIC] Wakeword score %.3f",
                    score,
                )
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