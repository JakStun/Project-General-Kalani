import serial
from collections import deque

ser = serial.Serial('/dev/serial0', 256000, timeout=0.5)

HEADER = b'\xF4\xF3'

window = deque(maxlen=5)

def read_frame():
    data = ser.read(300)
    idx = data.find(HEADER)

    if idx == -1:
        print("No header found")
        return None
    
    frame = data[idx:idx+40]

    if len(frame) < 10:
        return None
    
    return frame

def get_distance():
    frame = read_frame()

    if frame is None:
        return None
    
    state = frame[8]
    presence = state != 0
    moving = bool(state & 0x01)
    static = bool(state & 0x02)

    if state == 0:
        return None # no one's home
    
    dist = frame[9] | (frame[10] << 8)

    # smoother distance
    window.append(dist)
    if len(window) < window.maxlen:
        return dist
    
    median = sorted(window)[len(window)//2]

    # TODO: Change into logging
    print(
        f"Distance: {dist} cm | "
        f"Median: {median} cm | "
        f"State: {state} | "
        f"Presence: {presence} | "
        f"Moving: {moving} | "
        f"Static: {static} | "
    )

    # ignoring distance spikes (irrelevant)
    if dist > median * 1.8:
        return median
    
    return median 
    
