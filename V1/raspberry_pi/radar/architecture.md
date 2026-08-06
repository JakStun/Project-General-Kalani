# Radar Architecture

- Used sensors: ```Hi-Link LD2410, LD2410C, LD2450```

- Pros:
    - Very Good/easy to use packages in python for operating these sensors (aio-ld2410...)
    - Easy wiring and RPi operating
    - Cheap

- Cons:
    - I needed to tinker with them alot (took me ~2 weeks to setup it corectly for this project)
    - at least in my case, one LD2410C unexpectedly burned down during testing, even though it was wired 100% correctly (no external power supply, everything was wired into RPi 4 (GND, 5V, RX, TX))

- Idea:
    - radar_control.py -> first thing that is called in cerebrum.py (after init callibration), will handle human detection
        - person walks to servitor (<55 cm, therefore I need to disable some gates in sensor and change moving threshold and static threshold for activa gates)
        - sensor sends signal to cerebrum -> which activates servitor
        - after person leaves (or there is no movement), sensor sends signal to cerebrum to shut down itself

    - PROBLEMS:
        - after shutting down, sensor sees this procedure -> sends signal to activate again -> infinite loop
            - FIX - filter out any movement (moving and static alike) that is very close to servitor (<30 cm), seems working 90% of times
        - I put my chair in front of the servitor (<50 cm) -> after I leave servitor activates itself, even though there is no one in front of it
            - FIX - instead of using detection_distence in if statement that decides whether there is someone in front of servitor, use moving_distance (makes sense, took me too long to come up with this fix...)