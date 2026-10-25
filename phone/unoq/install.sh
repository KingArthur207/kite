#!/usr/bin/env bash
# Put the phone receiver on the Arduino UNO Q, so the walker's iPhone connects to the board on their body
# (Bluetooth) and the board forwards each position to the main computer's earpiece over Wi-Fi (UDP 5010).
#
#   phone/unoq/install.sh <main-computer-ip> [user@board]      the board defaults to arduino@kite-unoq.local
#
# Run from the repo root on a computer that can ssh to the board (it asks for the board password).
# On the main computer, start the earpiece with --bind 0.0.0.0 so it accepts the board's UDP.
# Board log: ssh <board> journalctl -u kite-phone -f
set -euo pipefail

MAIN="${1:?usage: phone/unoq/install.sh <main-computer-ip> [user@board]}"
BOARD="${2:-arduino@kite-unoq.local}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

echo "copying the receiver to $BOARD:~/kite-phone"
tar -C "$ROOT" --exclude='__pycache__' --exclude='.state' --exclude='tests' -czf - phone earpiece/earpiece \
  | ssh "$BOARD" "mkdir -p ~/kite-phone && tar -C ~/kite-phone -xzf -"
echo "installing on the board (sudo there asks for the board password)"
ssh -t "$BOARD" "bash ~/kite-phone/phone/unoq/setup.sh '$MAIN'"
