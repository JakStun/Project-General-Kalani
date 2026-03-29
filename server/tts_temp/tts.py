import pyttsx3

class TextToSpeech:
    def __init__(self):
        self.engine = pyttsx3.init()

    def create_speech(self, text, output_path):
        self.engine.save_to_file(text, str(output_path))
        self.engine.runAndWait()