import serial
import time

ser = serial.Serial('/dev/serial0', 256000, timeout=0.5)

HEADER = b'\xF4\xF3'

while True:
    data = ser.read(300)
    # print(frame[:10])
    idx = data.find(HEADER)
    
    if idx == -1:
        print("No header found")
        continue


    frame = data[idx:idx+40]

    if len(frame) < 10:
        continue

    # print("Frame:", frame[:10])
    # print("Energy:", frame[7], frame[8], frame[9], frame[10])
    # print("Motion:", frame[7], "Static:", frame[8])


    state = frame[8]
    has_moving = bool(state & 0x03)
    has_static = bool(state & 0x02)
    presence = state != 0

    print(
        frame[:16].hex(' '),
        "raw state:", state,
        "presence:", presence,
        "moving:", has_moving,
        "static:", has_static
    )

    # if presence == 1:
    #     print("Human")
    # else:
    #     print("no human")
    # if b'\xFD\xFC' in data:
    #     print('human detected!')
    # else:print("Energy:", frame[7, frame[8], frame[9], frame[10]])
    #     print('no human')
    time.sleep(0.1)