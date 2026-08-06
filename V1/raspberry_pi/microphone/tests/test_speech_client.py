import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from speech_client import SpeechClient


class DummyResponse:
    def __init__(self, payload: dict | None = None, content: bytes | None = None) -> None:
        self._payload = payload or {"status": "ok"}
        self._content = content if content is not None else b""
        self.headers = {"content-type": "application/json"}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        if self._content:
            raise UnicodeDecodeError("utf-8", self._content, 0, 1, "invalid start byte")
        return self._payload

    @property
    def content(self) -> bytes:
        return self._content


class SpeechClientTests(unittest.TestCase):
    def test_process_posts_audio_with_httpx_file_tuple(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            handle.write(b"dummy-audio")
            path = Path(handle.name)

        async def fake_post(url, headers=None, files=None):
            self.assertEqual(url, "http://127.0.0.1:8000/v1/audio/post")
            self.assertEqual(headers["X-Robot-ID"], "nanoid")
            self.assertIsInstance(files["audio_file"], tuple)
            self.assertEqual(files["audio_file"][0], path.name)
            self.assertEqual(files["audio_file"][2], "audio/wav")
            return DummyResponse({"status": "ok"})

        with patch("speech_client.httpx.AsyncClient.post", side_effect=fake_post):
            client = SpeechClient(server_url="http://127.0.0.1:8000/v1/audio/post")
            response = asyncio.run(client.process(path))

        self.assertEqual(response, {"status": "ok"})
        path.unlink(missing_ok=True)

    def test_process_returns_raw_bytes_for_binary_response(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            handle.write(b"dummy-audio")
            path = Path(handle.name)

        async def fake_post(url, headers=None, files=None):
            return DummyResponse(content=b"\x90\x91binary-payload")

        with patch("speech_client.httpx.AsyncClient.post", side_effect=fake_post):
            client = SpeechClient(server_url="http://127.0.0.1:8000/v1/audio/post")
            response = asyncio.run(client.process(path))

        self.assertIsInstance(response, Path)
        self.assertTrue(response.exists())
        self.assertEqual(response.read_bytes(), b"\x90\x91binary-payload")
        response.unlink(missing_ok=True)
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
