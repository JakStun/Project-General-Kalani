import time

from events import Event
from aio_ld2410 import LD2410
from logging import getLogger

MOVING_CONFIG = [90, 90, 90]
STATIC_CONFIG = [90, 90, 90]

class RadarControl:
    def __init__(self, event_queue) -> None:
        self.logger = getLogger("main")

        self.event_queue = event_queue

        self.detected = False
        self.distance = 999

        self.last_seen = 0

        self._device = None

    async def run(self) -> None:
        self.logger.info("[LD2410] Starting radar module")

        self._device = LD2410("/dev/serial0")

        async with self._device:
            await self._configure()

            self.logger.info("[LD2410] Observation has commenced")
            async for report in self._device.get_reports():
                await self._handle_report(report)


    async def _configure(self) -> None:
        self.logger.info("[LD2410] Sensor activated ... configuring ...")

        self.logger.debug("[LD2410] Before config mode")
        async with self._device.configure():
            self.logger.debug("[LD2410] In config mode")

            await self._device.set_parameters(
                moving_max_distance_gate=2,
                static_max_distance_gate=2,
                presence_timeout=15,
            )

            for i in range(len(MOVING_CONFIG)):
                await self._device.set_gate_sensitivity(
                    distance_gate=i,
                    moving_threshold=MOVING_CONFIG[i],
                    static_threshold=STATIC_CONFIG[i],
                )

        self.logger.info("[LD2410] Configuration DONE")

    async def _handle_report(self, report) -> None:
        # if self.detected == True:
        #     return
        
        basic = report.basic

        currently_detected = bool(
            basic.target_status != 0 and 30 < basic.moving_distance <= 55
        )

        if currently_detected and not self.detected:
            self.last_seen = time.monotonic()

            self.distance = basic.detection_distance

            self.detected = True

            self.logger.info(f"[LD2410] TARGET DETECTED ({self.distance}cm)")

            await self.event_queue.put(
                (
                    Event.RADAR_DETECTED,
                    {
                        "distance": self.distance,
                    }
                )
            )

        elif (
            not currently_detected 
            and self.detected 
            and time.monotonic() - self.last_seen > 5
        ):
            self.detected = False

            self.logger.info("[LD2410] TARGET LOST")

            await self.event_queue.put(
                (
                    Event.RADAR_LOST,
                    None
                )
            )

if __name__ == "__main__":
    import asyncio 

    async def main():
        event_queue = asyncio.Queue()
        radar = RadarControl(event_queue)

        asyncio.create_task(
            radar.run()
        )

        while True:
            event, payload = await event_queue.get()

            # match event:
            #     case Event.RADAR_DETECTED:
            #         await self.wake_up()

            #     case Event.RADAR_LOST:
            #         await self.sleep()

            event_queue.task_done()


    asyncio.run(main())