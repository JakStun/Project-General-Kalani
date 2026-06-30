import asyncio

from math import pi, cos
from gpiozero import AngularServo
from gpiozero.pins.pigpio import PiGPIOFactory
from logging import getLogger


class ServoControl:
    def __init__(self) -> None:
        self.logger = getLogger("main")

        self.factory = PiGPIOFactory()

        self.right_servo = AngularServo(
            13,
            min_pulse_width=0.0006,
            max_pulse_width=0.0023,
            min_angle=-90,
            max_angle=90,
            initial_angle=None,
            pin_factory=self.factory,
        )

        self.left_servo = AngularServo(
            12,
            min_pulse_width=0.0006,
            max_pulse_width=0.0023,
            min_angle=-90,
            max_angle=90,
            initial_angle=None,
            pin_factory=self.factory,
        )

        # Need to detach servos so they won't jump into init position (0 degrees)
        self.right_servo.detach()
        self.left_servo.detach()

        self._set_init_angle()


    async def wake_animation(self) -> None:
        self.logger.info("[SERVO] Moving gears to awoken stature")

        await self.move(40) #TODO: Experiment
        await self.move(20, duration=1)

    async def sleep_animation(self) -> None:
        self.logger.info("[SERVO] Moving gears to asleep stature")

        await self.move(-40)
        await self.move(-60, duration=0.5)

    async def move(self, target: float, duration: float = 2.5, steps: int = 50) -> None:
        '''Func that moves servos to set postion'''
        
        self.logger.info(f"[SERVO] Moving to {target}° over {duration} secs")

        start = self.right_servo.angle or 0

        for step in range(steps + 1):
            t = step / steps
            eased_step = start + (target - start) * (0.5 - 0.5 * cos(pi * t))

            self.right_servo.angle = eased_step
            self.left_servo.angle = - eased_step

            await asyncio.sleep(duration / steps)

        self._save_position()

        self.logger.info("[SERVO] Moving procedure finished")

    def _save_position(self) -> None:
        with open("servo_pos.txt", "w") as file:
            file.write(str(self.right_servo.angle))

    def _set_init_angle(self) -> None:
        '''Helper func setting the true init pos for both servos'''

        self.logger.info("[SERVO] Setting init position")

        if self.right_servo.angle is None:
            try:
                with open("servo_pos.txt") as f:
                    last = float(f.read())
                    self.logger.info(f"[SERVO] Loaded last angle: {last}")
            except:
                last = 0
                self.logger.info("[SERVO] No last angle, default 0")

            self.right_servo.angle = last
            self.left_servo.angle = -last

        self.logger.info("[SERVO] Finished setting init position successfully")

if __name__ == "__main__":
    pass