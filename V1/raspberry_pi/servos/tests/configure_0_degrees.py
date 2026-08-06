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

# Must turn off signal
right_servo.detach()
left_servo.detach()

right_servo.angle = 0
left_servo.angle = 0