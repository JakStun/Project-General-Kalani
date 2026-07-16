import asyncio
import io
import json
import urllib.request
from pathlib import Path
from typing import Optional


class SpeechClient:
    def __init__(
        self,
        server_url: str = "http://127.0.0.1:8000/v1/audio/post",
        robot_id: str = "nanoid",
    ) -> None:
        self.server_url = server_url
        self.robot_id = robot_id

    async def process(self, audio_path: Path) -> Optional[Path]:
        print(f"Uploading {audio_path}")

        data = audio_path.read_bytes()

        def _request() -> Optional[Path]:
            boundary = "----RaspberryPiBoundary"
            body = []
            body.append(f"--{boundary}\r\n".encode())
            body.append(b'Content-Disposition: form-data; name="x_robot_id"\r\n\r\n')
            body.append(f"{self.robot_id}\r\n".encode())
            body.append(f"--{boundary}\r\n".encode())
            body.append(
                b'Content-Disposition: form-data; name="audio_file"; filename="input.wav"\r\n'
            )
            body.append(b"Content-Type: audio/wav\r\n\r\n")
            body.append(data)
            body.append(f"\r\n--{boundary}--\r\n".encode())

            payload = b"".join(body)
            req = urllib.request.Request(
                self.server_url,
                data=payload,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=60) as response:
                content_type = response.headers.get("Content-Type", "audio/wav")
                if "audio/wav" not in content_type:
                    raise RuntimeError(f"Unexpected content type: {content_type}")

                response_bytes = response.read()

                output_path = audio_path.with_suffix(".response.wav")
                output_path.write_bytes(response_bytes)
                return output_path

        return await asyncio.to_thread(_request)