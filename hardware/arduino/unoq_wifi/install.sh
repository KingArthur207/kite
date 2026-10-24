#!/bin/sh
# Run ON the UNO Q (over adb shell or ssh), from the folder holding these files:
#     sh install.sh
# Copies hello_sender.py to ~/kite and installs a service that starts it at every power-on.
set -e
mkdir -p /home/arduino/kite
cp hello_sender.py /home/arduino/kite/hello_sender.py
sudo cp kite-hello.service /etc/systemd/system/kite-hello.service
sudo systemctl daemon-reload
sudo systemctl enable --now kite-hello.service
echo "installed. live log: journalctl -u kite-hello -f   stop: sudo systemctl disable --now kite-hello"
