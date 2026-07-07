import webrtcvad

class VoiceActivityDetector:
    def __init__(self, aggressiveness: int = 2, sample_rate: int = 48000) -> None:
        """
        aggresiveness:
            0 -> least
            3 -> most (but can miss quiet speech)
        """

        self.sample_rate = sample_rate

        self.vad = webrtcvad.Vad(aggressiveness)

    def is_speech(self, frame) -> bool:
        """
        Params:
        frame -> np.ndarray(dtype=int16)
        """

        return self.vad.is_speech(frame.tobytes(), sample_rate=self.sample_rate)