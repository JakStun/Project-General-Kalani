# import serial
# import time

# ser = serial.Serial('/dev/serial0', 256000, timeout=0.5)

# def send(cmd):
#     ser.write(bytes.fromhex(cmd))
#     time.sleep(0.1)

# # I. enter engineering mode
# send("FD FC 01 00 01 01")

# # II. adjust motion sensitivity for all gates 100
# # send("FD FC 03 09 64 64 64 64 64 64 64 64 64")

# # III. adjust static sensitivity for all gates 80
# # send("FD FC 04 09 50 50 50 50 50 50 50 50 50")

# # IV. enable presence report
# # send("FD FC 06 01 01")

# send("FD FC 02 01 01 04")

# send("FD FC 03 01 01 05")

# send("FD FC 04 09 64 64 00 00 00 00 00 00 00")


# # V. save settings
# send("FD FC 05 00 05")

# # VI. exit engineering modesend("FD FC 01 00 01 01")

# send("FD FC 01 00 00 02")

# print("config successful")

# import serial
# import time

# ser = serial.Serial('/dev/serial0', 256000, timeout=0.5)

# def send(cmd):
#     ser.write(bytes.fromhex(cmd))
#     time.sleep(0.1)

# def read_response():
#     data = ser.read(200)
#     return data.hex(" ")

# print("=== Reading LD2410C configuration ===")

# # 1. Enter engineering mode (required to read config)
# send("FD FC 01 00 01 01")

# # 2. Request configuration dump
# #    Command: FD FC 0A 00 0A
# send("FD FC 0A 00 0A")

# # 3. Read response
# resp = read_response()
# print("\nRAW RESPONSE:\n", resp)

# # Try to parse it
# bytes_list = bytes.fromhex(resp)

# # Look for header FD FC
# idx = resp.find("fd fc")
# if idx == -1:
#     print("No valid config frame found.")
#     exit()

# # Convert to byte array
# frame = bytes_list

# print("\n=== PARSED CONFIG ===")

# # Byte layout (LD2410C config frame):
# # FD FC 0A LL HH <payload...>

# # Moving farthest gate = payload[0]
# moving_gate = frame[5]
# print("Moving farthest gate:", moving_gate)

# # Static farthest gate = payload[1]
# static_gate = frame[6]
# print("Static farthest gate:", static_gate)

# # Sensitivity gates 0–8 = payload[2..10]
# sens = frame[7:16]
# for i, v in enumerate(sens):
#     print(f"Gate {i} sensitivity:", v)

# # 4. Exit engineering mode
# send("FD FC 01 00 00 01 02")

# print("\n=== DONE ===")

import serial
import time

ser = serial.Serial('/dev/serial0', 256000, timeout=0.5)

HEADER = b'\xF4\xF3'

def read_frame():
    data = ser.read(300)
    idx = data.find(HEADER)
    if idx == -1:
        return None
    frame = data[idx:idx+40]
    if len(frame) < 16:
        return None
    return frame

while True:
    frame = read_frame()
    if frame is None:
        continue

    # State byte (presence)
    state = frame[8]
    presence = state != 0
    moving = bool(state & 0x01)
    static = bool(state & 0x02)

    # Distance (little endian)
    dist = frame[9] | (frame[10] << 8)

    # Motion energy (little endian)
    motion_energy = frame[11] | (frame[12] << 8)

    # Static energy (little endian)
    static_energy = frame[13] | (frame[14] << 8)

    print(
        f"Distance: {dist} cm | "
        f"State: {state} | "
        f"Presence: {presence} | "
        f"Moving: {moving} | "
        f"Static: {static} | "
        f"MotionEnergy: {motion_energy} | "
        f"StaticEnergy: {static_energy}"
    )

    time.sleep(0.05)
