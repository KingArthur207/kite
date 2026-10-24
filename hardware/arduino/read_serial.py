"""Print what the Arduino sends over USB serial (Windows, macOS or Linux).

    pip install pyserial
    python read_serial.py              # finds the Arduino's port by itself
    python read_serial.py COM5         # or name the port (Windows: COM3, COM5...; macOS: /dev/cu.usbmodem...)
    python read_serial.py --list       # show every serial port and what it looks like

Each line is printed with the time it arrived. Unplugging the board is survived: the script waits and reconnects.
Ctrl+C stops it.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    sys.exit("pyserial is not installed: run  pip install pyserial")

BAUD = 115200
# USB chips Arduino boards use: Arduino's own (2341, 2A03), CH340 clones (1A86), FTDI (0403),
# Silicon Labs CP210x on most ESP32s (10C4), Espressif native USB (303A).
ARDUINO_VIDS = {0x2341, 0x2A03, 0x1A86, 0x0403, 0x10C4, 0x303A}


def guess_port() -> str | None:
    ports = list(list_ports.comports())
    for p in ports:
        if p.vid in ARDUINO_VIDS:
            return p.device
    for p in ports:                       # fall back to anything that says Arduino or USB serial
        text = f"{p.description} {p.manufacturer or ''}".lower()
        if "arduino" in text or "usb serial" in text or "ch340" in text or "cp210" in text:
            return p.device
    return None


def show_ports() -> None:
    ports = list(list_ports.comports())
    if not ports:
        print("no serial ports found: is the board plugged in, and is its USB driver installed?")
    for p in ports:
        vid = f"{p.vid:04X}:{p.pid:04X}" if p.vid is not None else "----:----"
        mark = "  <- looks like an Arduino" if p.vid in ARDUINO_VIDS else ""
        print(f"{p.device:22s} {vid}  {p.description}{mark}")


def read_forever(port: str | None, baud: int) -> None:
    while True:
        name = port or guess_port()
        if not name:
            print("waiting for an Arduino to be plugged in... (python read_serial.py --list to see ports)")
            time.sleep(2)
            continue
        try:
            with serial.Serial(name, baud, timeout=1) as ser:
                print(f"connected to {name} at {baud} baud")
                while True:
                    raw = ser.readline()
                    if raw:
                        line = raw.decode("utf-8", errors="replace").rstrip()
                        print(f"{datetime.now():%H:%M:%S}  {line}", flush=True)
        except serial.SerialException as e:
            print(f"lost {name} ({e}); retrying in 2 s")
            time.sleep(2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("port", nargs="?", help="COM5, /dev/cu.usbmodem1101, ... (default: find it)")
    ap.add_argument("--baud", type=int, default=BAUD)
    ap.add_argument("--list", action="store_true", help="list serial ports and exit")
    a = ap.parse_args()
    if a.list:
        show_ports()
        return
    try:
        read_forever(a.port, a.baud)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
