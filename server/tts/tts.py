from TTS.api import TTS
from pathlib import Path
import torch

class TextToSpeech:
    def __init__(self, voice_sample_path=None):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tts = TTS(
            "tts_models/multilingual/multi-dataset/xtts_v2",
            gpu=(device == "cuda"),
            progress_bar=False
        )

        self.voice_sample_path = (
            voice_sample_path
            or Path(__file__).parent.parent.parent
            / "models"
            / "voice_samples"
            / "voice_sample3_cleared.wav"
        )

    def create_speech(self, text, output_path):
        self.tts.tts_to_file(
            text=text,
            speaker_wav=str(self.voice_sample_path),
            file_path=str(output_path),
            language="en",
            temperature=0.75,
            length_penalty=1.0,
            repetition_penalty=2.0,
            emotion=0.5
        )