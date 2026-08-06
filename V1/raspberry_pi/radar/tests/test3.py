import asyncio
from aio_ld2410 import LD2410

async def main():
    print("Opening")

    async with LD2410("/dev/serial0") as device:
        print("device opened")

        async for report in device.get_reports():
            print(report)

        # print("entering config")

        # async with device.configure():
        #     print("in config")

        #     await device.set_parameters(
        #         moving_max_distance_gate=2,
        #         static_max_distance_gate=2,
        #         presence_timeout=15,
        #     )

        #     for i in range(3):
        #         await device.set_gate_sensitivity(
        #             distance_gate=i,
        #             moving_threshold=90,
        #             static_threshold=90,
        #         )

        #     cfg = await device.get_parameters()
        #     print(cfg)

asyncio.run(main())