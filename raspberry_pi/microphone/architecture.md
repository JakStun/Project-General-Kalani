# Microphone Architecture

Used MIC: Adafruit I2S MEMS microphone breakout - SPH0645LM4H

```
microphone_control.py   -> main Controller for mic
recorder.py             -> captures PCM frames
vad.py                  -> voice activity detection
wakeword.py             -> OpenWakeWord wrapper
audio_buffer.py         -> ring buffer
```

Idea: 
First detect speech -> wake word - for that to work, I need MIC to record from the moment the robot is 'up'
    - this will be handled by recorder.py
    