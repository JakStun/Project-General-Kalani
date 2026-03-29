from .config import TEMP_DIR

class AudioService:
    def __init__(self):
        self.temp_dir = TEMP_DIR
        self.temp_dir.mkdir(exist_ok=True)

    def post_audio(self, audio_file):
        '''
        Upload the .wav audio so that Lucrehulk can process it.
        '''
        
        try:
            with self.temp_dir.joinpath(audio_file.filename).open("wb") as f:
                f.write(audio_file.file.read())
                
        except Exception as e:
            return {"message": f"Error occurred while processing audio: {str(e)}"}

        return {"message": "Audio received successfully."}