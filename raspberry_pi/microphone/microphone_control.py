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

            frame, overflow = await self.recorder.read_frame()
            
            self.buffer.push(frame)

            if self.state == MicrophoneState.LISTENING:
                await self._handle_listening(frame)

            elif self.state == MicrophoneState.RECORDING:
                await self._handle_recording(frame)

        
    async def _handle_listening(self, frame):
        if self.state != MicrophoneState.LISTENING:
            return
           
        is_speech = self.vad.is_speech(frame)
        
        if not is_speech:
            return
        
        score = self.wakeword.process(frame)

        if score < self.wakeword.threshold:
            return
        

        self.state = MicrophoneState.RECORDING

        self.recording_frames = [
            self.buffer.get_audio()
        ]    

        self.recording_silence = 0

        self.logger.info("[MIC] Wakeword detected, entering RECORDING mode!")


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

    async def _handle_recording(self, frame):
        
        self.recording_frames.append(frame.copy())

        if self.vad.is_speech(frame):
            self.recording_silence = 0
        else:
            self.recording_silence += 1

        if self.recording_silence >= self.recording_stop_frames:
            self.logger.info("[MIC] Recording finished, entering PROCESSING mode!")

            self.state = MicrophoneState.PROCESSING

            self.recorder.pause() # stop recording immediately

            audio = np.concatenate(
                self.recording_frames
            )

            filename = datetime.now().strftime("recordings/recording_%Y%m%d_%H%M%S.wav")

            with wave.open(filename, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(48000)

                wav.writeframes(audio.tobytes())

            self.ignore_wakeword_frames = 20

            self.recording_frames.clear()

            self.buffer.clear()

            self.wakeword.reset()

            await self.event_queue.put(
                (Event.MIC_PROCESSING, None)
            )

            return

    def finish_interaction(self):
        self.logger.info("[MIC] Interacton finished")

        self.buffer.clear()

        self.recording_frames.clear()

        self.recording_silence = 0

        self.wakeword.reset()
        
        self.recorder.resume() # restart recording

        self.state = MicrophoneState.LISTENING

        self.wakeword.recording = False

        while not self.recorder.queue.empty():
            self.recorder.queue.get_nowait()

        self.logger.info("[MIC] Entering LISTENING mode!")


               
if __name__ == "__main__":
    import asyncio

    mic = MicrophoneControl()

    asyncio.run(mic.run())