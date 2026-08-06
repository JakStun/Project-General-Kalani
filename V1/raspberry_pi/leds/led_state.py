from enum import Enum, auto

class LEDState(Enum):
    OFF = auto()

    STARTING = auto()

    ACTIVE = auto()

    SLEEPING = auto()

    LISTENING = auto()

    THINKING = auto()

    SPEAKING = auto()