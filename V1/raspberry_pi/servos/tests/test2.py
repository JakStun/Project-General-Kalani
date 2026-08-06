from gpiozero import AngularServo
from gpiozero.pins.pigpio import PiGPIOFactory
from time import sleep
import math

factory = PiGPIOFactory()

right_servo = AngularServo(
    13, 
    min_pulse_width=0.0006, 
    max_pulse_width=0.0023,
    min_angle=-90,
    max_angle=90,
    initial_angle=None,Problem:
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

# Must turn off signal
right_servo.detach()
left_servo.detach()

# sleep(2)

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

        print(servo1.angle, servo2.angle)

        with open("servo_pos.txt", "w") as f:
            f.write(str(servo1.angle))
        
        sleep(duration / steps)

def calibrate(servo1, servo2):
    slow_move(servo1=servo1, servo2=servo2, target=0, duration=2.5)

calibrate(servo1=right_servo, servo2=left_servo)

sleep(0.5)

while True:
    slow_move(servo1=right_servo, servo2=left_servo, target=60, duration=2.5)
    sleep(2)


    slow_move(servo1=right_servo, servo2=left_servo, target=-60, duration=2.5)
    sleep(2)
