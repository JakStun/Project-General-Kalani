import serial
import time

ser = serial.Serial('/dev/serial0', 256000, timeout=0.5)

def send(cmd):
    ser.write(bytes.fromhex(cmd))
    time.sleep(0.1)

send("FD FC 01 00 00 01 02")

print('success')