import asyncio
from aio_ld2410 import LD2410, TargetStatus
from gpiozero import AngularServo
from gpiozero.pins.pigpio import PiGPIOFactory
import math
from time import sleep

factory = PiGPIOFactory()

right_servo = AngularServo(
    13, min_pulse_width=0.0006, max_pulse_width=0.0023,
    min_angle=-90, max_angle=90, initial_angle=None,
    pin_factory=factory
)

left_servo = AngularServo(
    12, min_pulse_width=0.0006, max_pulse_width=0.0023,
    min_angle=-90, max_angle=90, initial_angle=None,
    pin_factory=factory
)

print("Detaching servos initially...")
right_servo.detach()
left_servo.detach()

def slow_move(servo1, servo2, target, duration=2.5, steps=50):
    print(f"[SERVO] slow_move to {target}° over {duration}s")
    if servo1.angle is None:
        try:
            with open("servo_pos.txt") as f:
                last = float(f.read())
                print(f"[SERVO] Loaded last angle: {last}")
        except:
            last = 0
            print("[SERVO] No last angle, default 0")

        servo1.angle = last
        servo2.angle = -last
        sleep(1)

    start = servo1.angle or 0
    # print(f"[SERVO] Start angle: {start}")

    for i in range(steps + 1):
        t = i / steps
        eased = start + (target - start) * (0.5 - 0.5 * math.cos(math.pi * t))
        servo1.angle = eased
        servo2.angle = -eased

        with open("servo_pos.txt", "w") as f:
            f.write(str(servo1.angle))

        sleep(duration / steps)

def activate():
    print("[STATE] ACTIVATING (target 0°)")
    slow_move(right_servo, left_servo, -20)

def sleep_position():
    print("[STATE] SLEEP POSITION (target 60°)")
    slow_move(right_servo, left_servo, 60)

async def radar_task(device, state):
    print("[RADAR] Starting radar_task...")
    async for report in device.get_reports():
        b = report.basic
        status = b.target_status

        if status:
            dist = b.detection_distance
            state["distance"] = dist
            state["detected"] = True
            state["moving_energy"] = b.moving_energy
            state["static_energy"] = b.static_energy
            print(f"[RADAR] DETECTED | dist={dist} | status={status}")
        else:
            if state["detected"]:
                print("[RADAR] LOST target")
            state["detected"] = False

        await asyncio.sleep(0.05)

async def servo_task(state):
    print("[SERVO] Starting servo_task...")
    active = False
    cooldown = 0
    ignore_until = 0

    sleep_position()

    while True:
        now = asyncio.get_event_loop().time()

        if now < ignore_until:
            await asyncio.sleep(0.05)
            continue

        dist = state.get("distance", 999)
        detected = state.get("detected", False)
        moving_energy = state.get("moving_energy", 0)
        static_energy = state.get("static_energy", 0)

        print(f"[LOOP] detected={detected} | dist={dist} | active={active} | cooldown={cooldown} | moving energy={moving_energy} | static energy={static_energy}")

        # energy = max(moving_energy, static_energy)

        if detected and 40 <= dist <= 60 and moving_energy > 20: # detected and 30 <= dist <= 55
            cooldown = 0
            if not active:
                print("[LOGIC] In range 40–55cm → ACTIVATE")
                activate()
                active = True

                ignore_until = now + 1.0
        else:
            cooldown += 1
            if cooldown > 100 and active:
                print("[LOGIC] Out of range / no target for a while → SLEEP")
                sleep_position()
                active = False

                ignore_until = now + 1.5

        await asyncio.sleep(0.05)

async def main():
    state = {"distance": 999, "detected": False}

    print("[MAIN] Opening LD2410 on /dev/serial0 ...")
    async with LD2410('/dev/serial0') as device:
        print("[MAIN] Device opened.")

        async with device.configure():
            print("[MAIN] Configuring device...")

            MOVING_CONFIG = [80, 80, 80]
            STATIC_CONFIG = [80, 80, 80]
            
            await device.set_parameters(
                moving_max_distance_gate=2,
                static_max_distance_gate=2,
                presence_timeout=5,
            )

            for i in range(len(MOVING_CONFIG)):
                await device.set_gate_sensitivity(
                    distance_gate=i,
                    moving_threshold=MOVING_CONFIG[i],
                    static_threshold=STATIC_CONFIG[i],
                )
            print("[MAIN] Parameters set.")

        print("[MAIN] Starting radar + servo tasks...")
        await asyncio.gather(
            radar_task(device, state),
            servo_task(state)
        )

if __name__ == "__main__":
    asyncio.run(main())