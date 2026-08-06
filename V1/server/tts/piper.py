from pathlib import Path
import asyncio
import subprocess


class PiperService:
    def __init__(self):
        self.model = Path(
            r"C:\Code\Github\Project-General-Kalani\server\model\en_US-lessac-medium.onnx"
        )

    def generate_audio(
        self,
        text: str,
        output_path: str
    ) -> None:

        subprocess.run(
            [
                "piper",
                "--model",
                str(self.model),
                "--output_file",
                output_path,
            ],
            input=text.encode(),
            check=True,
        )

    async def generate_async(
        self,
        text: str,
        output_path: str
    ):
        await asyncio.to_thread(
            self.generate_audio,
            text,
            output_path,
        )