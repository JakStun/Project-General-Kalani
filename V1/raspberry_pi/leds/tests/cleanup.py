from rpi_ws281x import PixelStrip, Color
import time

print("creating strip")
strip = PixelStrip(7, 18)

print("begin")
strip.begin()

print("Sending off")
for i in range(7):
    strip.setPixelColor(i, Color(255,0,0))

strip.show()
print("done")
time.sleep(10)