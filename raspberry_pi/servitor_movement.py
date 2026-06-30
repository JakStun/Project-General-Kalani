from time import sleep
from human_radar import get_distance

from gpiozero import AngularServo
from gpiozero.pins.pigpio import PiGPIOFactory

import math

factory = PiGPIOFactory()

right_servo = AngularServo(
    13, 
    min_pulse_width=0.0006, 
    max_pulse_width=0.0023,
    min_angle=-90,
    max_angle=90,
    initial_angle=None,
    pin_factory=factory
)

left_servo = AngularServo(
    12,
    min_pulse_width=0.0006, 
    max_pulse_width=0.0023,
    min_angle=-90,
    max_angle=90,
    initial_angle=None,
    pin_factory=factory
)

# must turn off signal right at start or else the servo goes back very fast to 0 degrees
right_servo.detach()
left_servo.detach()

def slow_move(servo1, servo2, target, duration=5, steps=50):
    if servo1.angle is None:
        try:
            with open("servo_pos.txt") as f:
                last = float(f.read())
        except:
            last = 0

        servo1.angle = last
        servo2.angle = -last
    
        sleep(1)

    start = servo1.angle or 0

    for i in range(steps + 1):
        t = i / steps
        eased = start + (target - start) * (0.5 - 0.5 * math.cos(math.pi * t))
        servo1.angle = eased
        servo2.angle = -eased

        # TODO: add logging
        print(servo1.angle, servo2.angle)

        with open("servo_pos.txt", "w") as f:
            f.write(str(servo1.angle))
        
        sleep(duration / steps)

def calibrate(servo1, servo2):
    slow_move(servo1=servo1, servo2=servo2, target=0, duration=2.5)

def activate():
    slow_move(right_servo, left_servo, -20, 2.5)

def sleep_position():
    slow_move(right_servo, left_servo, 60, 2.5)

# INIT Poistion
sleep_position()

active = False
cooldown = 0

while True:
    dist = get_distance()

    # if dist is not None:
    #     print(dist)

    # Activate zone (40-50 cm):
    if dist is not None and 40 <= dist <=55:
        cooldown = 0

        if not active:
            print("Booting up")
            activate()
            active = True

    # Deactivation: no presence for 10 secs
    else:
        cooldown +=1

        if cooldown > 30 and active:
            print("Deactivating Servitor")
            sleep_position()
            active = False

    sleep(0.05)