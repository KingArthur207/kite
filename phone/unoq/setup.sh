#!/usr/bin/env bash
# Runs ON the UNO Q (install.sh calls it): BlueZ, a venv with bless, and the kite-phone service.
set -euo pipefail
MAIN="${1:?main computer IP}"
DIR="$HOME/kite-phone"

sudo apt-get install -y -q bluez python3-venv python3-numpy python3-scipy >/dev/null
sudo rfkill unblock bluetooth || true
sudo systemctl enable --now bluetooth
cd "$DIR"
[ -d .venv ] || python3 -m venv --system-site-packages .venv
.venv/bin/pip install -q bless

# root: BlueZ only lets privileged users register a GATT server and advertise
sudo tee /etc/systemd/system/kite-phone.service >/dev/null <<UNIT
[Unit]
Description=Kite phone receiver (iPhone GPS over Bluetooth to the main computer)
After=bluetooth.target network-online.target
Wants=network-online.target bluetooth.target

[Service]
WorkingDirectory=$DIR
Environment=PYTHONUNBUFFERED=1
ExecStart=$DIR/.venv/bin/python -m phone.ble --ear-host $MAIN --no-watch
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable kite-phone
sudo systemctl restart kite-phone
sleep 5
systemctl --no-pager --lines=10 status kite-phone || true
echo "kite-phone sends to $MAIN:5010. Log: journalctl -u kite-phone -f"
