import torch
import soundfile as sf
import time

from qwen_tts import Qwen3TTSModel
from logging import getLogger

from ..config import REF_AUDIO_PATH, REF_TEXT, VERSION

class TextToSpeech:
    def __init__(self, ref_audio_path: str = REF_AUDIO_PATH, ref_text: str = REF_TEXT) -> None:
        self.ref_audio_path = ref_audio_path
        self.ref_text = ref_text
        self.logger = getLogger("main")

        # Try to load model with CUDA, fall back to CPU if it fails
        try:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            
            self.model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                device_map="cuda",
                dtype=torch.bfloat16,
                # use_cuda_graph=True
            )

            self.logger.info(f"Model loaded successfully on {device}")
        except Exception as e:
            self.logger.warning(f"Failed to load model on CUDA: {e}. Falling back to CPU...")
            try:
                self.model = Qwen3TTSModel.from_pretrained(
                    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                    device_map="cpu",
                    dtype=torch.float32,
                )
                self.logger.info("Model loaded successfully on CPU")
            except Exception as cpu_error:
                self.logger.error(f"Failed to load model on CPU: {cpu_error}")
                raise

        self.voice_clone_prompt = self._create_voice_clone(self.ref_audio_path, self.ref_text)

        # Warmup the model with a dummy input to ensure it's ready for real requests
        _ = self.model.generate_voice_clone("warmup", language="English", voice_clone_prompt=self.voice_clone_prompt)
        torch.cuda.synchronize()

    def create_speech(self, robot_id: str, text: str, version: dict[str, str] = VERSION) -> None:
        try:
            params = version["params"]
            wavs, sr = self.model.generate_voice_clone(
                text,
                language="English",
                voice_clone_prompt=self.voice_clone_prompt,
                **params
            )
            sf.write(f"temp/{robot_id}.wav", wavs[0], sr)

            self.logger.info(f"Generated response for version: {version['name']}")

        except Exception as e:
            self.logger.error(f"Error generating response for version {version['name']}: {e}")

    def _create_voice_clone(self, ref_audio_path: str, ref_text: str):
        start = time.time()
        
        voice_clone_prompt = self.model.create_voice_clone_prompt(
            ref_audio=ref_audio_path,
            ref_text=ref_text,
        )

        end = time.time()

        self.logger.info(f"Created voice clone prompt in {end - start:.2f} seconds")

        return voice_clone_prompt
