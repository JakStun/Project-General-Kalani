import sounddevice as sd

print(sd.query_devices())


from inspect import signature
from openwakeword.model import Model

print(signature(Model))

m = Model()

print(m.models.keys())