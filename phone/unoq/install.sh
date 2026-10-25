#!/usr/bin/env bash
# Put the phone receiver on the Arduino UNO Q, so the walker's iPhone connects to the board on their body
# (Bluetooth) and the board forwards each position to the main computer's earpiece over Wi-Fi (UDP 5010).
#
#   phone/unoq/install.sh                                      run it ON the main computer: it uses this computer's IP
#   phone/unoq/install.sh <main-computer-ip> [user@board]      the board defaults to arduino@kite-unoq.local
#
# Run from the repo root on a computer that can ssh to the board (it asks for the board password).
# On the main computer, start the earpiece with --bind 0.0.0.0 so it accepts the board's UDP.
# Board log: ssh <board> journalctl -u kite-phone -f
set -euo pipefail

BOARD="${2:-arduino@kite-unoq.local}"
MAIN="${1:-}"
if [ -z "$MAIN" ]; then
  # this computer's address on the network that reaches the board (no packet is sent: UDP connect only)
  MAIN=$(python3 -c "import socket,sys; s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.connect((socket.gethostbyname(sys.argv[1]),9)); print(s.getsockname()[0])" "${BOARD#*@}") \
    || { echo "could not find this computer's IP toward ${BOARD#*@}; pass it: phone/unoq/install.sh <main-computer-ip>"; exit 1; }
  echo "main computer = this computer, $MAIN"
fi
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

echo "copying the receiver to $BOARD:~/kite-phone"
tar -C "$ROOT" --exclude='__pycache__' --exclude='.state' --exclude='tests' -czf - phone earpiece/earpiece \
  | ssh "$BOARD" "mkdir -p ~/kite-phone && tar -C ~/kite-phone -xzf -"
echo "installing on the board (sudo there asks for the board password)"
ssh -t "$BOARD" "bash ~/kite-phone/phone/unoq/setup.sh '$MAIN'"
