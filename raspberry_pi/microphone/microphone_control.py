from .recorder import Recorder

class MicrophoneControl:
    def __init__(self):
        self.recorder = Recorder()

    async def run(self):

        while True:
            frame = await self.recorder.read_frame()

if __name__ == "__main__":
    pass