"""How many datagrams of a back-to-back burst does a UDP socket on this machine keep while its reader is busy?

    python scripts/udp_burst_check.py                 # loopback, SO_RCVBUF default / 64 KB / 384 KB / 8 MB
    python scripts/udp_burst_check.py --ffmpeg        # also: the SO_RCVBUF FFmpeg's udp:// really gets (netstat)

Models one Tello IDR frame: N x 1460-byte datagrams sent with no pacing while the receiving thread is not reading
(busy decoding the previous frame), then drained. Prints kept/sent per buffer size and burst length.
"""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time

DGRAM = 1460


def burst(rcvbuf: int | None, n: int, port: int) -> tuple[int, int]:
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    if rcvbuf:
        rx.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf)
    rx.bind(("127.0.0.1", port))
    got_buf = rx.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    tx.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8 * 1024 * 1024)
    payload = b"\x00" * DGRAM
    for _ in range(n):  # the reader is "busy": nothing reads during the burst
        try:
            tx.sendto(payload, ("127.0.0.1", port))
        except OSError:
            pass
    time.sleep(0.05)
    rx.setblocking(False)
    kept = 0
    while True:
        try:
            rx.recv(65536)
            kept += 1
        except BlockingIOError:
            break
    rx.close()
    tx.close()
    return kept, got_buf


def ffmpeg_rcvbuf(port: int) -> str:
    """Open udp:// with OpenCV's and PyAV's FFmpeg and read the socket's receive high-water mark from netstat."""
    code = ("import sys, time, threading, av\n"
            f"c = threading.Thread(target=lambda: av.open('udp://@0.0.0.0:{port}', format='h264', timeout=(4, 4)), daemon=True)\n"
            "c.start(); time.sleep(1.5)\n"
            "import subprocess; print(subprocess.run(['netstat', '-anv', '-p', 'udp'], capture_output=True, text=True).stdout)\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30).stdout
    lines = out.splitlines()
    head = next((ln for ln in lines if "rhiwat" in ln), "")
    row = [ln for ln in lines if f".{port} " in ln]
    return head + "\n" + "\n".join(row)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=11195)
    ap.add_argument("--ffmpeg", action="store_true")
    a = ap.parse_args(argv)
    print(f"{'SO_RCVBUF asked':>16s} {'got':>9s}  " + "  ".join(f"burst {n:4d}" for n in (17, 60, 200, 600)))
    for buf in (None, 65536, 393216, 8 * 1024 * 1024):
        cells, got = [], 0
        for n in (17, 60, 200, 600):
            k, got = burst(buf, n, a.port)
            cells.append(f"{k:4d}/{n:<5d}")
        print(f"{str(buf or 'default'):>16s} {got:>9d}  " + "  ".join(cells))
    if a.ffmpeg:
        print(ffmpeg_rcvbuf(a.port))
    return 0


if __name__ == "__main__":
    sys.exit(main())
