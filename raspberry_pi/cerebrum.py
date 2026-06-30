'''Main Brain Control'''

import asyncio
import logging
import logging.config
import yaml

from events import Event
from leds import LEDsControl
from microphone import MicrophoneControl
from radar import RadarControl
from servos import ServoControl

from pathlib import Path

# --> Loading config file / Config setup <-- #
LOG_CONFIG_PATH = Path(__file__).parent / "logger_config.yml"

with open(LOG_CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)
    logging.config.dictConfig(config)

async def main() -> None:
    cerebrum = Cerebrum()

    await cerebrum.run()

class Cerebrum:
    '''
    Center of all thoughts and source of strategic truth
        -> hears out every request
        -> obeys everything
    '''

    def __init__(self) -> None:
        self.logger = logging.getLogger("main")

        self.event_queue = asyncio.Queue()

        self.radar = RadarControl(self.event_queue)
        self.leds = LEDsControl()
        # self.servos = ServoControl()

        self.awake = False


    async def startup(self) -> None:
        # await self.radar.startup()

        # move head back to init position:
        # await self.servos.move(-60)

        asyncio.create_task(
            self.radar.run()
        )

        self.logger.info("[CEREBRUM] Startup complete")

    async def run(self) -> None:
        await self.startup()

        await self.event_loop()

    async def event_loop(self) -> None:
        while True:
            event, payload = await self.event_queue.get()

            match event:
                case Event.RADAR_DETECTED:
                    await self.wake_up()

                case Event.RADAR_LOST:
                    await self.sleep()

            self.event_queue.task_done()


    async def wake_up(self) -> None:

        if self.awake: # no idea how else...
            return
        
        self.awake = True

        self.logger.info("[CEREBRUM] WAKING UP FROM SLUMBER")

        await asyncio.gather(
            self.leds.wake_animation(),
            # self.servos.wake_animation()
        )

        await self.leds.start_active()

    async def sleep(self) -> None:

        if not self.awake: # no idea how else...
            return
        
        self.awake = False

        self.logger.info("[CEREBRUM] GOING BACK TO SLUMBER MODE")

        await self.leds.stop_active()

        await asyncio.gather(
            self.leds.sleep_animation(),
            # self.servos.sleep_animation(),
        )


if __name__ == "__main__":
    asyncio.run(main())