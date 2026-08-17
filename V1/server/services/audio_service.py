import time
import logging
import asyncio
import torch

from fastapi import UploadFile
from fastapi.responses import Response

from stt import SpeechToText
from tts import PiperService
from llm import ResponseGenerator

from .config import TEMP_DIR

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

class AudioService:
    def __init__(self):
        self.temp_dir = TEMP_DIR
        self.temp_dir.mkdir(exist_ok=True)

        self.stt = SpeechToText()
        self.tts = PiperService()
        self.llm = None
        self.logger = logging.getLogger("main")

    async def process_audio(self, robot_id: str, audio_file: UploadFile):
        '''
        Upload the .wav audio so that Lucrehulk can process it.
        '''
        start_total = time.time()
        try:
            file_path = f"{self.temp_dir}/{audio_file.filename}"

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
            response_text = await self._generate_response(str(user_text))
            llm_time = time.time() - start_llm
            self.logger.info(f"LLM took {llm_time:.2f}s")

            # III. Create audio response:
            start_tts = time.time()
            audio_data = await self._create_audio_response(
                response_text
            )
            tts_time = time.time() - start_tts
            self.logger.info(f"TTS took {tts_time:.2f}s")

            total_time = time.time() - start_total
            self.logger.info(f"Total processing took {total_time:.2f}s")

            return Response(
                content=audio_data,
                media_type="audio/wav",
                headers={
                    "Content-Disposition": f'attachment; filename="{robot_id}.wav"'
                },
            )

        except Exception as e:
            return {"message": f"Error occurred while processing audio: {str(e)}"}
    

    # --------------------- Helper Funcs ---------------------

    async def _process_audio(self, file_path):
        user_text = await asyncio.to_thread(self.stt.transcribe_audio, file_path)

        return user_text
    
    async def _generate_response(self, user_text: str):
        if self.llm is None:
            self.llm = ResponseGenerator()
        response_text = await asyncio.to_thread(self.llm.generate_response, user_text)

        return response_text
    
    async def _create_audio_response(
        self,
        response_text: str,
    ) -> bytes:

        return await self.tts.generate_async(response_text)