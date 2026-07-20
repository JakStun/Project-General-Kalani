import sounddevice as sd
import soundfile as sf

print(sd.query_devices())

filename = r"/home/pi/Code/Github/Project-General-Kalani/models/voice_samples/voice_sample3_cleared.wav"

data,sample_rate = sf.read(filename, dtype="float32")

sd.play(data, sample_rate)

sd.wait()
