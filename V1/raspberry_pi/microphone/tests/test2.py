import sounddevice as sd
import soundfile as sf

samplerate = 48000
duration = 5 #secs

audio = sd.rec(
    int(duration * samplerate),
    samplerate=samplerate,
    channels=1,
    dtype='int32'
)

sd.wait()

sf.write("test.wav", audio, samplerate)
print("DONE")