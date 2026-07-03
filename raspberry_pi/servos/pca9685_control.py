import asyncio
import board 
import busio 

from math import pi, cos 
from logging import getLogger
from adafruit_servokit import ServoKit 

class ServoControlPCA9685: 
    RIGHT_CHANNEL = 0 
    LEFT_CHANNEL = 1 

    def __init__(self) -> None: 
        self.logger = getLogger("main") 
        
        i2c = busio.I2C(board.SCL, board.SDA) 
        
        self.kit = ServoKit(
            channels=16, 
            i2c=i2c,
        ) 
        
        #TODO: Calibrating pulse for SG90 servos, needs further testing 
        self.kit.servo[self.RIGHT_CHANNEL].set_pulse_width_range(
            500, 
            2500
        ) 
        
        
        self.kit.servo[self.LEFT_CHANNEL].set_pulse_width_range(
            1000,
            2000
        ) 
        
        self.current_angle = 0 
        
        self._set_init_angle() 

    async def calibrate(self) -> None: 
        with open("servo_pos.txt") as f: 
            last = float(f.read()) 
        
        await asyncio.sleep(1) 
        
        if last > -29: 
            try: 
                await self.move(-30) 
                self.logger.info("[SERVO] Calibrating SLUMBER POSITION successful") 
            except Exception: 
                self.logger.exception("[SERVO] Failed to calibrate into SLUMBER POSITION") 
        else: 
            self.logger.info("[SERVO] Already in SLUMBER POSITION") 

    async def wake_animation(self) -> None: 
        self.logger.info("[SERVO] Servitors gears moving to awoken stature...") 
        
        await self.move(40) # TODO: experiment 
        await self.move(20, duration=0.5) 

    async def sleep_animation(self) -> None: 
        self.logger.info("[SERVO] Servitors gears moving to slumber stature...") 
        
        await self.move(-40) 
        await self.move(-60, duration=0.5) 

    async def move(self, target: float, duration: float = 2.5, steps: int = 50) -> None: 
        """ Moves servitor joints to set position <-90, 90> """ 
        self.logger.info(f"[SERVO] Moving to {target}° over {duration} secs") 
        start = self.current_angle 

        for step in range(steps + 1): 
            t = step / steps 
            eased = start + (target- start) * (0.5 - 0.5 * cos(pi * t)) 

            self._set_pair_angle(eased) 

            await asyncio.sleep(duration / steps) 

        self.current_angle = target 

        self._save_position() # disable PM output 

        self._detach() 

        self.logger.info("[SERVO] Moving procedure finished") 

    def _set_pair_angle(self, angle: float) -> None: 
        """ inside class -> -90 ... 90 servokit wants -> 0 ... 180 """ 
        right = 90 + angle 
        left = 90 - angle 

        self.kit.servo[self.RIGHT_CHANNEL].angle = right 
        self.kit.servo[self.LEFT_CHANNEL].angle = left 

    def _detach(self) -> None: 
        self.kit.servo[self.RIGHT_CHANNEL].angle = None 
        self.kit.servo[self.LEFT_CHANNEL].angle = None 

    def _save_position(self) -> None: 
        with open("servo_pos.txt", "w") as file: 
            file.write(str(self.current_angle)) 

    def _set_init_angle(self) -> None: 
        self.logger.info("[SERVO] Setting last position") 

        try:
            with open("servo_pos.txt") as f:
                last = float(f.read())

            self.logger.info(f"[SERVO] Loaded last position: {last}°")

        except Exception:
            last = 0

            self.logger.warning("[SERVO] Failed to load last position, setting 0°")

        self.current_angle = last

        self._set_pair_angle(last)

        self.logger.info("[SERVO] Last position set successfully")

if __name__ == "__main__":
    import time

    async def main():
        servo_control = ServoControlPCA9685()

        await servo_control.calibrate()

        # await servo_control.wake_animation()
        # await asyncio.sleep(1)
        # await servo_control.sleep_animation()
        # await servo_control.move(target=0)

        # time.sleep(10)

        while True:
            await servo_control.move(target=5, steps=30)
            time.sleep(2)

            await servo_control.move(target=-30, steps=100, duration=3)
            time.sleep(2)

    asyncio.run(main())