from TTS.api import TTS
from pathlib import Path

class TextToSpeech:
    def __init__(self, voice_sample_path=None):
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True, progress_bar=False)
        self.voice_sample_path = voice_sample_path or Path(__file__).parent.parent.parent / "models" / "voice_samples" / "voice_sample1.wav"

    def create_speech(self, text, output_path):
        self.tts.tts_to_file(
            text=text,
            speaker_wav=str(self.voice_sample_path),
            file_path=str(output_path),
            language="en"
        )