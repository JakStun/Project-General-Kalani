from fastapi.responses import FileResponse
import time
import logging
import asyncio

from stt import SpeechToText
from tts import TextToSpeech
from llm import ResponseGenerator

from .config import TEMP_DIR

class AudioService:
    def __init__(self):
        self.temp_dir = TEMP_DIR
        self.temp_dir.mkdir(exist_ok=True)

        self.stt = SpeechToText()
        self.tts = TextToSpeech()
        self.llm = ResponseGenerator()
        self.logger = logging.getLogger("main")

    async def process_audio(self, audio_file):
        '''
        Upload the .wav audio so that Lucrehulk can process it.
        '''
        start_total = time.time()
        try:
            file_path = self.temp_dir / audio_file.filename

            with open(file_path, "wb") as f:
                content = await audio_file.read()
                f.write(content)

            # I. Process text from audio:
            start_stt = time.time()
            user_text = await self._process_audio(file_path)
            stt_time = time.time() - start_stt
            self.logger.info(f"STT took {stt_time:.2f}s")

            # II. Generate response from LLM:
            start_llm = time.time()
            response_text = await self._generate_response(user_text)
            llm_time = time.time() - start_llm
            self.logger.info(f"LLM took {llm_time:.2f}s")

            # III. Create audio response:
            start_tts = time.time()
            tts_path = await self._create_audio_response(response_text, audio_file)
            tts_time = time.time() - start_tts
            self.logger.info(f"TTS took {tts_time:.2f}s")

            total_time = time.time() - start_total
            self.logger.info(f"Total processing took {total_time:.2f}s")

            return {
                "transcription": user_text,
                "response_text": response_text,
                "response_audio": FileResponse(tts_path, media_type="audio/wav", filename=tts_path.name)
            }

        except Exception as e:
            return {"message": f"Error occurred while processing audio: {str(e)}"}
    

    # --------------------- Helper Funcs ---------------------

    async def _process_audio(self, file_path):
        user_text = await asyncio.to_thread(self.stt.transcribe_audio, file_path)

        return user_text
    
    async def _generate_response(self, user_text):
        response_text = await asyncio.to_thread(self.llm.generate_response, user_text)

        return response_text
    
    async def _create_audio_response(self, response_text, audio_file):
        tts_path = self.temp_dir / f"response_{audio_file.filename}"
        await asyncio.to_thread(self.tts.create_speech, response_text, tts_path)

        return tts_path