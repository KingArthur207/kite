#!/usr/bin/env bash
# Runs ON the UNO Q (install.sh calls it): BlueZ, a venv with bless, and the kite-phone service.
set -euo pipefail
MAIN="${1:?main computer IP}"
DIR="$HOME/kite-phone"

sudo DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a apt-get install -y -q bluez python3-venv python3-numpy python3-scipy >/dev/null
sudo rfkill unblock bluetooth || true
sudo systemctl enable --now bluetooth
cd "$DIR"
[ -d .venv ] || python3 -m venv --system-site-packages .venv
sudo chown -R "$USER:$USER" .venv
PYTHONNOUSERSITE=1 .venv/bin/python -m pip install -q bless httpx

# root: BlueZ only lets privileged users register a GATT server and advertise
sudo tee /etc/systemd/system/kite-phone.service >/dev/null <<UNIT
[Unit]
Description=Kite phone receiver (iPhone GPS over Bluetooth to the main computer)
After=bluetooth.target network-online.target
Wants=network-online.target bluetooth.target

[Service]
WorkingDirectory=$DIR
Environment=PYTHONUNBUFFERED=1
# GPS goes out through SpacetimeDB, the same internet path as the board's hello.
# --no-send: eduroam often blocks UDP straight to the laptop.
ExecStart=$DIR/.venv/bin/python -m phone.ble --no-send --spacetime
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
echo "kite-phone publishes GPS to SpacetimeDB. On the laptop: python -m phone.latest --watch"
echo "Board log: journalctl -u kite-phone -f"
