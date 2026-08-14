import time
from logging import getLogger
from faster_whisper import WhisperModel

import torch


class SpeechToText:
    def __init__(self, model_size="tiny", device="auto", compute_type=None):
        self.logger = getLogger("main")

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if compute_type is None:
            compute_type = "float16" if device == "cuda" else "int8"

        self.device = device
        self.compute_type = compute_type
        self.model = self._load_model(model_size, self.device, self.compute_type)

    def _load_model(self, model_size, device, compute_type):
        try:
            return WhisperModel(model_size, device=device, compute_type=compute_type)
        except RuntimeError as exc:
            error_text = str(exc).lower()
            if device == "cuda" and (
                "cublas" in error_text
                or "not found" in error_text
                or "cannot be loaded" in error_text
            ):
                self.logger.warning(
                    "CUDA DLLs are missing or incompatible for Whisper. Falling back to CPU transcription."
                )
                return WhisperModel(model_size, device="cpu", compute_type="int8")
            raise

    def transcribe_audio(self, audio) -> str:
        self.logger.info(f"Transcribing audio: {audio}")

        start_time = time.time()
        segments, info = self.model.transcribe(audio)

        text = " ".join([segment.text for segment in segments])

        end_time = time.time()

        self.logger.info(f"Transcription completed in {end_time - start_time:.2f} seconds")
        self.logger.info(f"Transcribed text: {text}")

        return text.strip()


if __name__ == "__main__":
    stt = SpeechToText()
    segment = stt.transcribe_audio(r"C:\Code\Github\Project-General-Kalani\V1\models\voice_samples\question1.wav")

    print(segment)