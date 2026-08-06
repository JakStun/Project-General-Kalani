from TTS.api import TTS

# Load the model
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

# Clone voice and generate speech
tts.tts_to_file(
    text="Your text here",
    speaker_wav="C:\\Code\\Github\\Project-General-Kalani\\models\\voice_samples\\voice_sample1.wav",
    file_path="output.wav"
)

