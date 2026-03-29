from fastapi.responses import FileResponse

from stt import SpeechToText
from tts_temp import TextToSpeech
from llm import ResponseGenerator

from .config import TEMP_DIR

class AudioService:
    def __init__(self):
        self.temp_dir = TEMP_DIR
        self.temp_dir.mkdir(exist_ok=True)

        self.stt = SpeechToText()
        self.tts = TextToSpeech()
        self.llm = ResponseGenerator()

    async def process_audio(self, audio_file):
        '''
        Upload the .wav audio so that Lucrehulk can process it.
        '''
        
        try:
            file_path = self.temp_dir / audio_file.filename

            with open(file_path, "wb") as f:
                content = await audio_file.read()
                f.write(content)

            # I. Process text from audio:
            user_text = await self._process_audio(file_path)

            # II. Generate response from LLM:
            response_text = await self._generate_response(user_text)

            # III. Create audio response:
            tts_path = await self._create_audio_response(response_text, audio_file)

            return {
                "transcription": user_text,
                "response_text": response_text,
                "response_audio": FileResponse(tts_path, media_type="audio/wav", filename=tts_path.name)
            }

        except Exception as e:
            return {"message": f"Error occurred while processing audio: {str(e)}"}
    

    # --------------------- Helper Funcs ---------------------

    async def _process_audio(self, file_path):
        user_text = self.stt.transcribe_audio(file_path)

        return user_text
    
    async def _generate_response(self, user_text):
        response_text = self.llm.generate_response(user_text)

        return response_text
    
    async def _create_audio_response(self, response_text, audio_file):
        tts_path = self.temp_dir / f"response_{audio_file.filename}"
        self.tts.create_speech(response_text, tts_path)

        return tts_path