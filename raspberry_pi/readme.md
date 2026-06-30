###Before launching code intended for servo movement start pigpiod in cmd (stops servo jittering):
-> No need anymore, created service: sudo nano /etc/systemd/system/pigpiod.service
-> sudo pigpiod
-> pigs hwver (check if running -> if number appears -> OK)

###When Installing new packages:
-> sudo pip install --break-system-packages ...