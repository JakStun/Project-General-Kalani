from pathlib import Path
import asyncio
import subprocess
import tempfile
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL = PROJECT_ROOT / "model" / "en_US-lessac-medium.onnx"
TEMP_DIR = PROJECT_ROOT / "temp"

class PiperService:
    def __init__(self):
        self.model = MODEL
        TEMP_DIR.mkdir(exist_ok=True)

    def generate_audio(self, text: str) -> bytes:
        # Create a temporary file in our project's temp directory (not system temp)
        temp_output = None
        try:
            # Use a file descriptor to create the temp file safely
            fd, temp_output = tempfile.mkstemp(suffix=".wav", dir=str(TEMP_DIR))
            os.close(fd)  # Close the file descriptor so piper can write to it
            
            result = subprocess.run(
                [
                    "piper",
                    "--model",
                    str(self.model),
                    "--output_file",
                    temp_output,
                ],
                input=text.encode(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            
            # Read the generated audio file
            with open(temp_output, "rb") as f:
                audio_bytes = f.read()
            
            return audio_bytes
            
        except subprocess.CalledProcessError as e:
            stderr_msg = e.stderr.decode() if e.stderr else "No stderr output"
            raise RuntimeError(f"Piper command failed: {stderr_msg}")
        except Exception as e:
            raise RuntimeError(f"Error occurred while generating audio: {str(e)}")
        finally:
            # Clean up temporary file
            if temp_output and os.path.exists(temp_output):
                try:
                    os.remove(temp_output)
                except Exception:
                    pass  # Ignore cleanup errors

    async def generate_async(self, text: str) -> bytes:
        return await asyncio.to_thread(
            self.generate_audio,
            text,
        )

if __name__ == "__main__":

    path = Path(r"C:\Code\Github\Project-General-Kalani\V1\server\model\en_US-lessac-medium.onnx")

    print(path.exists())
    piper = PiperService()
    # piper.generate_audio()