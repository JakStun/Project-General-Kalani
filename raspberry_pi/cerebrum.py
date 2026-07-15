'''Main Brain Control'''

import asyncio
import logging
import logging.config
import wave
import yaml

from datetime import datetime
from pathlib import Path

from events import Event
from leds import LEDsControl
from microphone import MicrophoneControl
from radar import RadarControl
from servos import ServoControlPCA9685


LOG_CONFIG_PATH = Path(__file__).parent / "logger_config.yml"

with open(LOG_CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)
    logging.config.dictConfig(config)


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

        self.awake = False

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

            filename = datetime.now().strftime(
                "recordings/recording_%Y%m%d_%H%M%S.wav"
            )

            with wave.open(filename, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(48000)
                wav.writeframes(audio.tobytes())

            self.logger.info("[CEREBRUM] Simulating processing")

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


if __name__ == "__main__":
    asyncio.run(main())