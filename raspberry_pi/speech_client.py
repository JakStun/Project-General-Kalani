import httpx
from pathlib import Path
from typing import Optional

import sounddevice as sd
import soundfile as sf

class SpeechClient:
    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8000/v1/audio/post",
        robot_id: str = "nanoid",
    ) -> None:
        self.server_url = server_url
        self.robot_id = robot_id

        self.client = httpx.AsyncClient(
            timeout=1200
        )

    async def process(self, filename: str = "current.wav") -> Optional[Path]:
        print(f"Uploading {filename}")

        path = Path(filename)

        with path.open("rb") as audio:
            response = await self.client.post(
                self.server_url,
                headers={
                    "X-Robot-ID": self.robot_id
                },
                files={
                    "audio_file": (
                        path.name,
                        audio,
                        "audio/wav",
                    )
                },
            )

        response.raise_for_status()

        try:
            payload = response.json()
            return payload
        except (UnicodeDecodeError, ValueError):
            output_path = path.with_suffix(".response.wav")
            output_path.write_bytes(response.content)
            return output_path
    

    async def say_response(self):
        filename = r"/home/pi/Code/Github/Project-General-Kalani/raspberry_pi/recordings/current.response.wav"

        data,sample_rate = sf.read(filename, dtype="float32")

        sd.play(data, sample_rate)

        sd.wait()


    async def close(self):
        await self.client.aclose()