import time
from faster_whisper import WhisperModel

model = WhisperModel("small", device="cuda") # large-v3

start = time.time()

segments, info = model.transcribe(r"C:\Code\Github\Project-General-Kalani\models\voice_samples\voice_sample1.wav")

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

end = time.time()

print("Time taken: %.2fs" % (end - start))