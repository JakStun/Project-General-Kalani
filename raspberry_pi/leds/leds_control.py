import asyncio
import random

from rpi_ws281x import PixelStrip, Color
from logging import getLogger

from .led_state import LEDState

LED_COUNT = 7
LED_PIN = 21

class LEDsControl:
    def __init__(self) -> None:
        self.logger = getLogger("main")

        self.strip = PixelStrip(
            LED_COUNT,
            LED_PIN,
        )

        self.strip.begin()

        self.base_color = (255, 60, 0)
        
        self.state = LEDState.OFF
        self._active_task = None


    async def wake_animation(self) -> None:
        self.logger.info("[OCULI] Opening has commenced")
        
        self.state = LEDState.STARTING

        self._all_off()
        await asyncio.sleep(0.2)

        self._fill(20, 0, 0) # startup power pulse
        await asyncio.sleep(0.1)

        self._all_off()

        # Random damaged flickers in the eyes
        for _ in range(8):
            for i in range(self.strip.numPixels()):
                if random.random() > 0.5: # 50% chance of flicker, maybe experiment more
                    self.strip.setPixelColor(i, Color(30, 10, 0))
                else:
                    self.strip.setPixelColor(i, Color(0, 0, 0))

            self.strip.show()
            await asyncio.sleep(random.uniform(0.05, 0.15))

        # Gradual wake up
        for brightness in range(20, 180, 20):
            for i in range(self.strip.numPixels()):

                # Simulate one damaged LED, for now just one set
                if i == 3 and brightness < 120:
                    self.strip.setPixelColor(i, Color(0, 0, 0))
                else:
                    self.strip.setPixelColor(
                        i,
                        Color(brightness, brightness // 3, 0)
                    )

            self.strip.show()
            await asyncio.sleep(0.1)

        # glitching, servitor has some rough days behind it
        self._fill(255, 255, 255)
        await asyncio.sleep(0.05)

        self._all_off()
        await asyncio.sleep(0.05)

        self._fill(255, 255, 255)
        await asyncio.sleep(0.05)

        # final state -> golden/orange look
        self._fill(255, 60, 0)

        self.state = LEDState.ACTIVE

        self.logger.info("[OCULI] Opening is hereby finished successfully")

    async def sleep_animation(self) -> None:
        self.logger.info("[OCULI] Shuting down has commenced")

        self.state = LEDState.SLEEPING

        # dimming LED lights out
        for brightness in range(180, 0, -15):
            for i in range(self.strip.numPixels()):
                
                if random.random() < 0.15:
                    self.strip.setPixelColor(i, Color(0, 0, 0))
                else:
                    self.strip.setPixelColor(i, Color(brightness, brightness // 3, 0))

            self.strip.show()

            await asyncio.sleep(0.08)

        # final brief flicker
        self._fill(255, 255, 255)
        await asyncio.sleep(0.04)

        self._all_off()

        # single flicker/light going out
        # rand_led = random.randint(0,5)
        # self.strip.setPixelColor(rand_led, Color(180, 30, 0))

        # for brightness in range(180, 0, -15):
        #     self.strip.setPixelColor(rand_led, Color(brightness, brightness // 4, 0))

        # self._all_off()

        self.state = LEDState.OFF

    async def start_active(self) -> None:
        if self._active_task:
            return

        self._active_task = asyncio.create_task(
            self.active_loop()
        )

    async def stop_active(self) -> None:
        if self._active_task:

            task = self._active_task
            self._active_task = None

            await task

    async def active_loop(self) -> None:
        while self.state == LEDState.ACTIVE:
            if random.random() < 0.03:
                await self.damaged_flicker()

            await asyncio.sleep(0.05)

    async def damaged_flicker(self) -> None:
        count = random.randint(1, 4)

        affected = random.sample(
            range(self.strip.numPixels()),
            count
        )

        duration = random.uniform(0.05, 0.25)
        loop = asyncio.get_running_loop()
        end_time = loop.time() + duration

        # destabilizing
        while loop.time() < end_time:
            for led in affected:
                if random.random() < 0.6:
                    self.strip.setPixelColor(led, Color(0, 0, 0))
                else:
                    factor = random.uniform(0.1, 1.5)

                    r = min(255, int(self.base_color[0] * factor))
                    g = min(255, int(self.base_color[1] * factor))
                    b = min(255, int(self.base_color[2] * factor))

                    self.strip.setPixelColor(led, Color(r, g, b))

            self.strip.show()

            await asyncio.sleep(random.uniform(0.005, 0.03))

        # stabilizing
        for led in affected:
            self.strip.setPixelColor(led, Color(*self.base_color))

        self.strip.show()


    def _fill(self, r, g, b) -> None:
        '''Light up LEDs (as a compromise, like in the can for 20 fucking years)'''

        for i in range(self.strip.numPixels()):
            self.strip.setPixelColor(i, Color(r, g, b))

        self.strip.show()

    def _all_off(self) -> None:
        '''Shutdown LEDs'''

        for i in range(self.strip.numPixels()):
            self.strip.setPixelColor(i, Color(0, 0, 0))

        self.strip.show()

if __name__ == "__main__":
    pass