# V1 - Chilion Mahlon

## Full Cycle Preview:

```
I.   Human comes in front of Servitor (< 55 cm)

II.   Servitor activates itself, because it listens to LD2410S radar, that uncovers every movement in front of it:
        - servo moves head into 'awake' position
        - LEDs activate through startup  animation
        - MIC is set to active (NOT recording yet)
        - Optional: Servitor says something along these lines: At your service, Your word is my command...

III.  Human speaks Servitors name, thus activating MIC -> LEDs change color to Listening Mode

IV.   After humans request is recorded (human stops speaking and there is a slight/noticable pause), servitor sends their request for wisdom to Lucrehulk (Main Cogitator) and initiates Processing Mode, LEDs change to signal change

V.    Lucrehulk processes request for knowledge and returns appropriate answer to servitor (for more knowledge about Lucrehulk seek server folder)

VI.   Servitor spreads the unknown misterious knowledge to the keen listener and initiates Preaching Mode, again LEDs indicate this change

VII.  Human, satisfied with this new wisdom leaves servitor

VIII. After sensing no activity for a long time (~5 mins) Servitor starts 'sleep' funcs to preserve power and prepares itself for slumber: 
        - servo moves to 'sleep' position
        - LEDS start to jitter and at the end shut down
        - MIC is set to off
```

---

## Before launching code (Cerebrum.py):

- Interface Options (```sudo raspi-config```):
    - enable I2S (needed for MIC, it's not here anymore on newer version, just change config.txt)
    - enable I2C (needed for PCA9685)
    - enable SPI (needed for LED Control)
    - Serial Port (needed for LD2410 sensor):
        - disable ligin shell over serial
        - enable serial hardware

- MIC Configuration setup:
    I. ```sudo nano /boot/firmware/config.txt```
    II. Add/uncomment:
        - ```dtparam=i2s=on```
        - ```dtoverlay=googlevoicehat-soundcard```
    III. sudo reboot
    IV. Check if MIC is recognised and record test audio:
        - ```arecord -l```
        - ```arecord -D hw:3,0 -f S32_LE -r 48000 -c2 test.wav``` (if it doesn't work change hw:...,... -> maybe different port, can be seen by using func above)
        - speak into MIC
        - Ctrl+C -> stop recording
        - test.wav is created in /pi

- Speaker Configuration setup (JBL Flip):
    I. ```sudo systemctl start bluetooth```
    II. ```sudo systemctl enable bluetooth```
    III. ```bluetoothctl``` -> inside it:
        ```power on```
        ```agent on```
        ```default-agent```
        ```scan on``` -> search for XX:XX:XX:XX:XX:XX JBL Flip 3 (don't forget to start bluetooth on speaker aswell)
        ```pair XX:XX:XX:XX:XX:XX``` -> after success, should also see a popup: connection successful
        ```trust XX:XX:XX:XX:XX:XX```
        ```connect XX:XX:XX:XX:XX:XX```
        ```exit```
        
    IV. ```pactl list short sinks or wpctl status``` (depends on system) -> verify audio output, should be already default audio output (has * before name)
    V. ```speaker-test -c2``` or ```pw-play /usr/share/sounds/alsa/Front_Center.wav``` -> you should a stomp sound

    - if there is an error in bluetoothctl, check:
        - ```sudo systemctl status bluetooth --no-pager```
        - ```rfkill list```
        - ```hciconfig -a```
        - ```sudo rfkill unblock bluetooth```
        - ```sudo systemctl restart bluetooth```
        -> try running ```bluetoothctl``` and its commands again, should work now

---

## Installing new packages:

- WRONG APPROACH: sudo pip install --break-system-packages ... -> can break system
- CORRECT APPROACH: .venv (more stable) -> pip install ...