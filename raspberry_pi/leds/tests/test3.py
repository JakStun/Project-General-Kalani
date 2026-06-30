from rpi_ws281x import PixelStrip, Color
import time
import random

LED_COUNT = 7
LED_PIN = 18

base_color = (255, 60, 0)

strip = PixelStrip(LED_COUNT, LED_PIN)
strip.begin()


def all_off():
    for i in range(LED_COUNT):
        strip.setPixelColor(i, Color(0, 0, 0))
    strip.show()


def fill(r, g, b):
    for i in range(LED_COUNT):
        strip.setPixelColor(i, Color(r, g, b))
    strip.show()

def damaged_flicker(strip, base_color):
    count = random.randint(1, 4)

    affected = random.sample(
        range(strip.numPixels()),
        count
    )

    duration = random.uniform(0.05, 0.25)
    end_time = time.time() + duration

    while time.time() < end_time:

        for led in affected:
            if random.random() < 0.6:
                strip.setPixelColor(led, Color(0,0,0))
            else:
                factor = random.uniform(0.1, 1.5)

                r = min(255, int(base_color[0] * factor))
                g = min(255, int(base_color[1] * factor))
                b = min(255, int(base_color[2] * factor))

                strip.setPixelColor(
                    led,
                    Color(r,g,b)
                )

        strip.show()

        time.sleep(random.uniform(0.005, 0.03))


    # Now stabilizing:
    for led in affected:
        strip.setPixelColor(
            led,
            Color(*base_color)
        )

    strip.show()


all_off()
time.sleep(0.2)

# Initial power pulse
fill(20, 0, 0)
time.sleep(0.1)
all_off()

# Random damaged flickers
for _ in range(8):
    for i in range(LED_COUNT):
        if random.random() > 0.5:
            strip.setPixelColor(i, Color(30, 10, 0))
        else:
            strip.setPixelColor(i, Color(0, 0, 0))

    strip.show()
    time.sleep(random.uniform(0.05, 0.15))

# Gradual wake-up
for brightness in range(20, 180, 20):
    for i in range(LED_COUNT):

        # Simulate one damaged LED
        if i == 3 and brightness < 120:
            strip.setPixelColor(i, Color(0, 0, 0))
        else:
            strip.setPixelColor(
                i,
                Color(brightness, brightness // 3, 0)
            )

    strip.show()
    time.sleep(0.1)

# Electrical glitch
fill(255, 255, 255)
time.sleep(0.05)

all_off()
time.sleep(0.05)

fill(255, 255, 255)
time.sleep(0.05)

# Final state
fill(255, 60, 0)

try:
    while True:
        if random.random() < 0.03:
            damaged_flicker(strip, base_color)

        time.sleep(0.05)

except KeyboardInterrupt:
    print("Shutting down")

finally:
    for i in range(strip.numPixels()):
        strip.setPixelColor(i, Color(0,0,0))

    strip.show()

    del strip

    time.sleep(0.1)