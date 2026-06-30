import pigpio
import time

pi = pigpio.pi()

print(pi.connected)

while True:
    pi.set_servo_pulsewidth(13, 1000)
    print("1000")
    time.sleep(1)

    pi.set_servo_pulsewidth(13, 2000)
    print("2000")
    time.sleep(1)