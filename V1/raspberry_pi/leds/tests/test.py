import board
import neopixel
import time


PIXEL_PIN = board.D18
NUM_PIXELS = 7

pixels = neopixel.NeoPixel(
    PIXEL_PIN,
    NUM_PIXELS,
    brightness=0.2,
    auto_write=True
)

while True:
    pixels.fill((255,0,0))
    time.sleep(1)

    pixels.fill((0,255,0))
    time.sleep(1)

    pixels.fill((0,0,255))
    time.sleep(1)

    pixels.fill((0,0,0))
    time.sleep(1)