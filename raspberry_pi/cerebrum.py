'''Main Brain Control'''

import asyncio
import logging
import logging.config
import time
import wave
import yaml

from pathlib import Path

from events import Event
from leds import LEDsControl
from microphone import MicrophoneControl
from radar import RadarControl
from servos import ServoControlPCA9685

from speech_client import SpeechClient

LOG_CONFIG_PATH = Path(__file__).parent / "logger_config.yml"

with open(LOG_CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)
    logging.config.dictConfig(config)

RECORDING_DIR = Path("recordings")
RECORDING_DIR.mkdir(exist_ok=True)

CURRENT_AUDIO = RECORDING_DIR / "current.wav"

async def main():
    cerebrum = Cerebrum()
    await cerebrum.run()


class Cerebrum:

    def __init__(self):
        self.logger = logging.getLogger("main")

        self.event_queue = asyncio.Queue()

        self.radar = RadarControl(self.event_queue)
        self.leds = LEDsControl()
        self.servos = ServoControlPCA9685()

        self.microphone = MicrophoneControl()

        self.speech_client = SpeechClient()

        self.awake = False

        self.last_request_time = 0
        self.request_cooldown = 30.0

    async def startup(self):

        await self.servos.calibrate()

        asyncio.create_task(
            self.radar.run()
        )

        self.logger.info("[CEREBRUM] Startup complete")

    async def run(self):

        await self.startup()

        await asyncio.gather(
            self.event_loop(),
            self.microphone_loop(),
        )

    async def microphone_loop(self):

        while True:

            audio = await self.microphone.wait_for_interaction()

            duration = len(audio) / 48000
            self.logger.info(f"[CEREBRUMs] Recording length: {duration:.2f}s")

            if duration < 2.0:
                self.logger.info(
                    f"[CEREBRUM] Recording too short ({duration:.2f}s), ignoring."
                )
                continue

            now = time.monotonic()

            if now - self.last_request_time < self.request_cooldown:
                self.logger.info("[CEREBRUM] Request ignored (cooldown active)")
                continue

            self.last_request_time = now

            self.microphone.pause_listening()
            try:
                path = self.save_audio(audio)
                response = await self.speech_client.process(path)
                path.unlink(missing_ok=True)
                self.logger.info(f"[CEREBRUM] Response: {response}")
            finally:
                self.microphone.resume_listening()

            await asyncio.sleep(3)

    async def event_loop(self):

        while True:

            event, payload = await self.event_queue.get()

            match event:

                case Event.RADAR_DETECTED:
                    await self.wake_up()

                case Event.RADAR_LOST:
                    await self.sleep()

            self.event_queue.task_done()

    async def wake_up(self):

        if self.awake:
            return

        self.awake = True

        self.logger.info("[CEREBRUM] WAKING UP FROM SLUMBER")

        await asyncio.gather(
            self.leds.wake_animation(),
            self.servos.wake_animation(),
        )

        await self.leds.start_active()

    async def sleep(self):

        if not self.awake:
            return

        self.awake = False

        self.logger.info("[CEREBRUM] GOING BACK TO SLUMBER MODE")

        await self.leds.stop_active()

        await asyncio.gather(
            self.leds.sleep_animation(),
            self.servos.sleep_animation(),
        )

    def save_audio(self, audio) -> Path:
        with wave.open(str(CURRENT_AUDIO), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(48000)

            wav.writeframes(audio.tobytes())

        return CURRENT_AUDIO


if __name__ == "__main__":
    asyncio.run(main())