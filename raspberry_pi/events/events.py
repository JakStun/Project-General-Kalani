from enum import Enum, auto


class Event(Enum):
    RADAR_DETECTED = auto()
    RADAR_LOST = auto()

    MIC_PROCESSING = auto()