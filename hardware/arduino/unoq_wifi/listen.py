"""Runs on the computer (Windows, macOS or Linux): print every hello the UNO Q sends over Wi-Fi.

    python listen.py               # no installs needed (standard library only)

Listens on UDP port 5099 and prints each message with the time it arrived and who sent it, plus a
warning when the board goes quiet. On Windows, allow the port through the firewall once (PowerShell as admin):

    New-NetFirewallRule -DisplayName "Kite UNO Q hello" -Direction Inbound -Protocol UDP -LocalPort 5099 -Action Allow
"""

import json
import socket
import sys
import time
from datetime import datetime

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 5099
QUIET_S = 5.0


def main() -> None:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", PORT))
    s.settimeout(1.0)
    print(f"listening for the UNO Q on UDP {PORT} (Ctrl+C to stop)")
    last = None
    warned = False
    while True:
        try:
            data, (ip, _) = s.recvfrom(2048)
        except socket.timeout:
            if last and time.monotonic() - last > QUIET_S and not warned:
                print(f"{datetime.now():%H:%M:%S}  ... nothing for {QUIET_S:.0f} s (board off, or left the Wi-Fi?)")
                warned = True
            continue
        last, warned = time.monotonic(), False
        text = data.decode("utf-8", errors="replace").strip()
        try:
            m = json.loads(text)
            text = f"hello {m.get('n')} from {m.get('host')}"
        except ValueError:
            pass
        print(f"{datetime.now():%H:%M:%S}  {ip:15s}  {text}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
