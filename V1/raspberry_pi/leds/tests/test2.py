from rpi_ws281x import PixelStrip, Color
import time

LED_COUNT = 7
LED_PIN = 18

strip = PixelStrip(LED_COUNT, LED_PIN)
strip.begin()

# for i in range(LED_COUNT):
#     strip.setPixelColor(i, Color(255, 0, 0))
#     strip.show()
#     time.sleep(0.5)

# time.sleep(2)

# for i in range(LED_COUNT):
#     strip.setPixelColor(i, Color(0,0,0))U

# strip.show()

for i in range(LED_COUNT):
    strip.setPixelColor(i, Color(0,0,0))

strip.setPixelColor(0, Color(255, 100, 0))
strip.setPixelColor(1, Color(255, 100, 0))
strip.show()
time.sleep(0.5)

strip.setPixelColor(5, Color(255, 100, 0))
strip.setPixelColor(2, Color(255, 100, 0))
strip.show()
time.sleep(0.5)

strip.setPixelColor(4, Color(255, 100, 0))
strip.setPixelColor(3, Color(255, 100, 0))
strip.show()
time.sleep(0.5)

strip.setPixelColor(6, Color(255, 100, 0))
strip.show()
time.sleep(0.5)


time.sleep(2)

for i in range(LED_COUNT):
    strip.setPixelColor(i, Color(0,0,0))

strip.show()