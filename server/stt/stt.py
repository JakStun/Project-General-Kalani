import time
from typing import Iterable
from rich.segment import Segment
from faster_whisper import WhisperModel
from logging import getLogger
import torch

class SpeechToText:
    def __init__(self, model_size="tiny", device="auto", compute_type="float16"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
        
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

        self.logger = getLogger("main")

    def transcribe_audio(self, audio) -> Iterable[Segment]:
        self.logger.info(f"Transcribing audio: {audio}")

        start_time = time.time()
        segments, info = self.model.transcribe(audio)

        text = " ".join([segment.text for segment in segments])

        end_time = time.time()

        self.logger.info(f"Transcription completed in {end_time - start_time:.2f} seconds")

        return text.strip()
    
if __name__ == "__main__":
    stt = SpeechToText()
    segments = stt.transcribe_audio(r"C:\Code\Github\Project-General-Kalani\models\voice_samples\voice_sample1.wav")

    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))