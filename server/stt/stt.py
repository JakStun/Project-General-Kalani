import time
from typing import Iterable
from rich.segment import Segment
from faster_whisper import WhisperModel
from logging import getLogger

class SpeechToText:
    def __init__(self, model_size="small", device="cuda", compute_type="float16"):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

        self.logger = getLogger("main")

    def transcribe_audio(self, audio) -> Iterable[Segment]:
        self.logger.info(f"Transcribing audio: {audio}")

        start_time = time.time()
        segments, info = self.model.transcribe(audio)
        end_time = time.time()

        self.logger.info(f"Transcription completed in {end_time - start_time:.2f} seconds")

        return segments
    
if __name__ == "__main__":
    stt = SpeechToText()
    segments = stt.transcribe_audio(r"C:\Code\Github\Project-General-Kalani\models\voice_samples\voice_sample1.wav")

    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))