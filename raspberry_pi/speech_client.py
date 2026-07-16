import asyncio
import json
import urllib.request
from pathlib import Path


class SpeechClient:
    def __init__(self, server_url: str = "http://127.0.0.1:8000/process") -> None:
        self.server_url = server_url

    async def process(self, audio_path: Path) -> str:
        print(f"Uploading {audio_path}")

        data = audio_path.read_bytes()

        def _request() -> str:
            req = urllib.request.Request(
                self.server_url,
                data=data,
                headers={"Content-Type": "application/octet-stream"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
                return payload.get("response_text", "")

        return await asyncio.to_thread(_request)