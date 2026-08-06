from livekit.wakeword import WakeWordModel, WakeWordListener
import inspect
import asyncio
import pyaudio
p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    print(i, info["name"], info["defaultSampleRate"])

print(inspect.getsource(WakeWordListener.__aenter__))
async def main():
    model = WakeWordModel()

    async with WakeWordListener(model) as listener:
        print("Listening")

        while True:
            detection = await listener.wait_for_detection()

            # print(type(detection))
            # print(detection)
            # print(dir(detection))
            # print(detection.__dict__)

asyncio.run(main())