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
    - this will be handled by recorder.py -> returns PCM frames (~ 33 frames per second)

Because of my beginner skills I needed to mostly rely on AI...
Nevertheless I read the docs to each library I used in MIC architecture

docs:
- https://python-sounddevice.readthedocs.io/en/0.5.3


0 vc4-hdmi-0: MAI PCM i2s-hifi-0 (hw:0,0), ALSA (0 in, 2 out)
1 bcm2835 Headphones: - (hw:2,0), ALSA (0 in, 8 out)
2 snd_rpi_googlevoicehat_soundcar: Google voiceHAT SoundCard HiFi voicehat-hifi-0 (hw:3,0), ALSA (2 in, 2 out)
3 sysdefault, ALSA (0 in, 128 out)
4 hdmi, ALSA (0 in, 2 out)
5 default, ALSA (0 in, 128 out)  