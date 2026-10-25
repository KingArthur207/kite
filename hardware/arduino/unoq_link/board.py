#!/usr/bin/env python3
"""Kite link, board side: runs on the Arduino UNO Q's Linux side (Debian) as the kite-link service.

One HTTP port (8765) on every interface, protected by the token in /etc/kite/token:

    GET  /health          no token: proves this is a Kite board (for kite_link.py doctor)
    GET  /hello           {"reply": "hello from kite-unoq", ...}
    GET  /ping            {"pong": true}               (round-trip time)
    GET  /status          uptime, IPs, Wi-Fi, CPU temperature, camera
    POST /send            {"text": "..."}  -> the board logs it and replies {"reply": "got \\"...\\""}
    GET  /events          Server-Sent Events: the recurring hello, once a second
    GET  /video.mjpg      MJPEG stream (USB camera, or a labelled test pattern when there is none)
    GET  /snapshot.jpg    one frame
    GET  /view            a page that shows the stream (what `kite_link.py watch` opens)

The token goes in an `Authorization: Bearer <token>` header or a `?token=` query. Wrong or missing: 401.

Through the internet (works even where eduroam blocks device-to-device traffic), via Kite's SpacetimeDB:
  * every 2 s a signed heartbeat (type=heartbeat, source=unoq) carrying the board's IPs and port, so the
    computer finds the board without typing an IP (same payload shape as unoq_wifi/hello_sender.py);
  * it reads signed text commands (source=unoq.link) and answers them (source=unoq.board), the fallback
    for text when the direct path is blocked. Video has no fallback.

Standard library only; OpenCV (apt: python3-opencv) only for a real camera. Settings come from the
environment (see Config); install.sh writes the systemd unit.
"""

from __future__ import annotations

import argparse
import glob
import html
import json
import os
import queue
import re
import shutil
import socket
import subprocess  # fixed commands (ip, iwgetid, nmcli), no shell, no user input  # nosec B404
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kite_common import (  # noqa: E402
    DEFAULT_PORT, DEFAULT_SOURCE, FRESH_S, SPACETIME_DB, SPACETIME_HOST, Spacetime, check_source,
    clean_text, link_source, reply_source, row_time, sign, verify,
)

VERSION = 1
BOUNDARY = "kiteframe"
MAX_VIEWERS = 4
IDLE_S = 5.0                 # the camera stops capturing this long after the last viewer leaves
RESCAN_S = 3.0               # while showing "no camera" to a viewer, look for a newly plugged camera this often


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


# --------------------------------------------------------------------------------------------- JPEG
# A tiny baseline JPEG encoder for the synthetic test pattern, so it needs no OpenCV or Pillow.
# Every 8x8 block is one flat colour, so each block is only a DC coefficient and an end-of-block:
# fast in pure Python (a 640x480 frame is 4800 blocks), and any JPEG decoder reads it.

_DC_BITS = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]       # the standard luminance DC table
_DC_VALS = list(range(12))
_AC_BITS = [1] + [0] * 15                                          # one AC symbol: end-of-block, code "0"
_AC_VALS = [0x00]


def _huff_codes(bits: list[int], vals: list[int]) -> dict[int, str]:
    codes, code, k = {}, 0, 0
    for length, count in enumerate(bits, start=1):
        for _ in range(count):
            codes[vals[k]] = format(code, f"0{length}b")
            code, k = code + 1, k + 1
        code <<= 1
    return codes


_DC_CODES = _huff_codes(_DC_BITS, _DC_VALS)
_EOB = _huff_codes(_AC_BITS, _AC_VALS)[0x00]


def _dc_bits(diff: int) -> str:
    cat = abs(diff).bit_length()
    if cat == 0:
        return _DC_CODES[0] + _EOB
    mag = diff if diff > 0 else diff + (1 << cat) - 1
    return _DC_CODES[cat] + format(mag, f"0{cat}b") + _EOB


_DIFF = {d: _dc_bits(d) for d in range(-2047, 2048)}


# Most rows of the test pattern repeat from frame to frame, so their entropy-coded bits are cached (a row's
# bits depend only on its blocks and the DC values carried in from the previous block).
_ROW_CACHE: dict = {}


def _segment(marker: int, payload: bytes) -> bytes:
    return bytes([0xFF, marker]) + (len(payload) + 2).to_bytes(2, "big") + payload


def encode_blocks(grid: list[list[tuple[int, int, int]]]) -> bytes:
    """A JPEG of a grid of flat 8x8 blocks: grid[row][col] = (Y, Cb, Cr), each 0..255."""
    rows, cols = len(grid), len(grid[0])
    h, w = rows * 8, cols * 8
    out = [b"\xff\xd8", _segment(0xE0, b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"),
           _segment(0xDB, b"\x00" + bytes([8] * 64))]          # quant 8: DC value = block mean - 128
    sof = bytes([8]) + h.to_bytes(2, "big") + w.to_bytes(2, "big") + bytes([3])
    sof += b"".join(bytes([c, 0x11, 0]) for c in (1, 2, 3))
    out.append(_segment(0xC0, sof))
    out.append(_segment(0xC4, bytes([0x00]) + bytes(_DC_BITS) + bytes(_DC_VALS)))
    out.append(_segment(0xC4, bytes([0x10]) + bytes(_AC_BITS) + bytes(_AC_VALS)))
    out.append(_segment(0xDA, bytes([3, 1, 0x00, 2, 0x00, 3, 0x00, 0, 63, 0])))
    pred = (0, 0, 0)
    parts = []
    for row in grid:
        key = (pred, tuple(row))
        hit = _ROW_CACHE.get(key)
        if hit is None:
            p = list(pred)
            bits = []
            for px in row:
                for c in range(3):
                    v = px[c] - 128
                    bits.append(_DIFF[v - p[c]])
                    p[c] = v
            if len(_ROW_CACHE) > 4000:
                _ROW_CACHE.clear()
            hit = _ROW_CACHE[key] = ("".join(bits), tuple(p))
        parts.append(hit[0])
        pred = hit[1]
    bits = "".join(parts)
    bits += "1" * (-len(bits) % 8)                              # pad the last byte with 1s
    data = int(bits, 2).to_bytes(len(bits) // 8, "big") if bits else b""
    out.append(data.replace(b"\xff", b"\xff\x00"))              # byte stuffing
    out.append(b"\xff\xd9")
    return b"".join(out)


def _ycc(r: int, g: int, b: int) -> tuple[int, int, int]:
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return tuple(max(0, min(255, round(v))) for v in (y, cb, cr))


_FONT = {  # 3x5 pixel font, rows top to bottom
    "0": "111101101101111", "1": "010110010010111", "2": "111001111100111", "3": "111001111001111",
    "4": "101101111001001", "5": "111100111001111", "6": "111100111101111", "7": "111001010010010",
    "8": "111101111101111", "9": "111101111001111", ":": "000010000010000", " ": "000000000000000",
    "A": "010101111101101", "C": "111100100100111", "E": "111100110100111", "I": "111010010010111",
    "K": "101101110101101", "M": "101111111101101", "N": "110101101101101", "O": "111101101101111",
    "R": "110101110101101", "S": "111100111001111", "T": "111010010010010", "#": "101111101111101",
    "P": "111101111100100", "F": "111100110100100", "-": "000000111000000", "L": "100100100100111",
}
_BASE: dict = {}
_BARS = [(255, 255, 255), (255, 255, 0), (0, 255, 255), (0, 255, 0), (255, 0, 255), (255, 0, 0), (0, 0, 255),
         (0, 0, 0)]


def _text(grid, x: int, y: int, s: str, color) -> None:
    for i, ch in enumerate(s.upper()):
        glyph = _FONT.get(ch, _FONT[" "])
        for k, bit in enumerate(glyph):
            gx, gy = x + i * 4 + k % 3, y + k // 3
            if bit == "1" and 0 <= gy < len(grid) and 0 <= gx < len(grid[0]):
                grid[gy][gx] = color


def test_pattern(n: int, width: int = 640, height: int = 480, label: str = "NO CAMERA") -> bytes:
    """Frame n of the synthetic stream: colour bars, a moving square, the frame number and time, and a
    label saying it is a test pattern (so nobody mistakes it for a camera)."""
    cols, rows = max(width // 8, 40), max(height // 8, 24)
    bar_h = rows // 3
    white, amber = _ycc(255, 255, 255), _ycc(255, 176, 0)
    key = (cols, rows, label)
    if key not in _BASE:                       # the parts that never change, drawn once
        base = [[_ycc(24, 24, 32)] * cols for _ in range(rows)]
        for y in range(bar_h):
            base[y] = [_ycc(*_BARS[x * len(_BARS) // cols]) for x in range(cols)]
        _text(base, 2, bar_h + 2, "KITE TEST PATTERN", white)
        _text(base, 2, bar_h + 8, label[:(cols - 4) // 4], amber)
        _BASE[key] = base
    grid = [list(r) for r in _BASE[key]]
    _text(grid, 2, bar_h + 14, f"#{n}", white)
    _text(grid, 2, bar_h + 20, time.strftime("%H:%M:%S"), white)
    span = cols - 6
    pos = n % (2 * span)
    sx = pos if pos < span else 2 * span - pos                # bounce left and right
    for y in range(rows - 7, rows - 2):
        for x in range(sx, sx + 5):
            grid[y][x] = amber
    return encode_blocks(grid)


# -------------------------------------------------------------------------------------------- camera

def usb_cameras(sys_root: str = "/") -> list[str]:
    """/dev/videoN nodes that belong to a USB (UVC) camera, lowest first. The UNO Q's Qualcomm chip has
    its own video nodes (hardware codecs), so "any /dev/video*" would pick the wrong one."""
    found = []
    for d in glob.glob(os.path.join(sys_root, "sys/class/video4linux/video*")):
        driver = os.path.join(d, "device", "driver")
        if os.path.basename(os.path.realpath(driver)) == "uvcvideo":
            found.append(os.path.basename(d))
    found.sort(key=lambda v: int(re.sub(r"\D", "", v) or 0))
    return [f"/dev/{v}" for v in found]


class SyntheticCamera:
    name = "synthetic"

    def __init__(self, width: int, height: int, label: str = "NO CAMERA") -> None:
        self.width, self.height, self.label, self.n = width, height, label, 0

    def read(self) -> bytes | None:
        self.n += 1
        return test_pattern(self.n, self.width, self.height, self.label)

    def close(self) -> None:
        pass


class OpenCVCamera:
    """A V4L2 camera through OpenCV, JPEG-encoded at the given quality."""

    def __init__(self, device: str, width: int, height: int, fps: float, quality: int = 70) -> None:
        import cv2  # apt: python3-opencv
        self.cv2, self.device, self.quality = cv2, device, quality
        self.name = f"usb camera {device}"
        target = int(device) if device.isdigit() else device
        self.cap = cv2.VideoCapture(target, cv2.CAP_V4L2) if hasattr(cv2, "CAP_V4L2") else cv2.VideoCapture(target)
        if not self.cap.isOpened():
            raise OSError(f"can't open {device}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.size = (width, height)

    def read(self) -> bytes | None:
        ok, img = self.cap.read()
        if not ok or img is None:
            return None
        if (img.shape[1], img.shape[0]) != self.size:
            img = self.cv2.resize(img, self.size)
        ok, buf = self.cv2.imencode(".jpg", img, [int(self.cv2.IMWRITE_JPEG_QUALITY), self.quality])
        return buf.tobytes() if ok else None

    def close(self) -> None:
        self.cap.release()


def open_camera(choice: str, width: int, height: int, fps: float, sys_root: str = "/"):
    """(camera, reason). choice: auto (a USB camera if there is one, else the test pattern),
    synthetic, or a device (/dev/video2 or 2). Falls back to the test pattern, never fails."""
    if choice == "synthetic":
        return SyntheticCamera(width, height, "NO CAMERA"), "synthetic by request"
    if choice == "auto":
        devices = usb_cameras(sys_root)
        if not devices:
            return SyntheticCamera(width, height, "NO CAMERA"), "no USB camera found"
        choice = devices[0]
    try:
        return OpenCVCamera(choice, width, height, fps), "ok"
    except ImportError:
        return SyntheticCamera(width, height, "NO OPENCV"), "OpenCV missing (sudo apt install python3-opencv)"
    except OSError as e:
        return SyntheticCamera(width, height, "CAMERA ERROR"), str(e)


class FrameHub:
    """Captures only while someone watches; every viewer gets the newest frame."""

    def __init__(self, camera, reason: str, fps: float, opener=None) -> None:
        self.camera, self.reason, self.fps = camera, reason, max(0.5, fps)
        self.opener = opener                  # finds a camera plugged in (or switched to host mode) after start
        self.next_scan = 0.0
        self.cond = threading.Condition()
        self.frame: bytes | None = None
        self.seq = 0
        self.demand = 0.0
        self.viewers = 0
        self.failures = 0
        self.stop = threading.Event()

    def wait(self, after_seq: int, timeout: float = 5.0) -> tuple[int, bytes] | None:
        deadline = time.monotonic() + timeout
        with self.cond:
            self.demand = time.monotonic()
            self.cond.notify_all()
            while self.seq <= after_seq or self.frame is None:
                left = deadline - time.monotonic()
                if left <= 0 or self.stop.is_set():
                    return None
                self.cond.wait(left)
                self.demand = time.monotonic()
            return self.seq, self.frame

    def run(self) -> None:
        period = 1.0 / self.fps
        next_t = time.monotonic()
        while not self.stop.is_set():
            with self.cond:
                if time.monotonic() - self.demand > IDLE_S:
                    self.cond.wait(0.5)
                    next_t = time.monotonic()
                    continue
            if self.opener and isinstance(self.camera, SyntheticCamera) and time.monotonic() >= self.next_scan:
                self.next_scan = time.monotonic() + RESCAN_S
                camera, reason = self.opener()
                if isinstance(camera, SyntheticCamera):
                    camera.close()
                else:
                    log(f"camera: found {camera.name}; switching from the test pattern")
                    self.camera.close()
                    self.camera, self.reason, self.failures = camera, reason, 0
            jpeg = self.camera.read()
            if jpeg is None:
                self.failures += 1
                if self.failures == 10 and not isinstance(self.camera, SyntheticCamera):
                    log(f"camera: {self.camera.name} stopped giving frames; showing the test pattern")
                    self.camera.close()
                    w, h = getattr(self.camera, "size", (640, 480))
                    self.camera, self.reason = SyntheticCamera(w, h, "CAMERA LOST"), "camera stopped giving frames"
                time.sleep(0.05)
                continue
            self.failures = 0
            with self.cond:
                self.frame, self.seq = jpeg, self.seq + 1
                self.cond.notify_all()
            next_t += period
            delay = next_t - time.monotonic()
            if delay > 0:
                self.stop.wait(delay)
            else:
                next_t = time.monotonic()         # running late: don't try to catch up with a burst


# -------------------------------------------------------------------------------------------- status

def _read(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def _run(cmd: list[str]) -> str | None:
    if not shutil.which(cmd[0]):
        return None
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=3, check=False).stdout  # nosec B603
    except (OSError, subprocess.SubprocessError):
        return None


def primary_ip() -> str | None:
    """The address this machine uses to reach the internet (a UDP connect sends no packets)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.0.2.1", 9))           # TEST-NET-1: only picks a route
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


def _is_tailscale(ip: str) -> bool:
    try:
        a, b = (int(x) for x in ip.split(".")[:2])
    except ValueError:
        return False
    return a == 100 and 64 <= b <= 127          # 100.64.0.0/10, Tailscale's CGNAT range


def addresses(run=_run, advertise: str = "") -> tuple[list[str], str | None]:
    """(LAN IPv4 addresses, Tailscale IPv4 or None). advertise ("ip1,ip2", KITE_ADVERTISE) overrides the LAN list."""
    lan, ts = [], None
    out = run(["ip", "-4", "-o", "addr", "show"]) or ""
    for line in out.splitlines():
        m = re.match(r"\d+:\s+(\S+)\s+inet\s+([\d.]+)/", line)
        if not m:
            continue
        iface, ip = m.groups()
        if iface == "lo" or ip.startswith("127."):
            continue
        if iface.startswith("tailscale") or _is_tailscale(ip):
            ts = ts or ip
        elif not iface.startswith(("docker", "veth", "br-")):
            lan.append(ip)
    if not lan:
        p = primary_ip()
        if p and not _is_tailscale(p):
            lan.append(p)
    if advertise.strip():
        lan = [x.strip() for x in advertise.split(",") if x.strip()]
    return lan, ts


def wifi(root: str = "/", run=_run) -> dict:
    """SSID and signal (dBm) from /proc/net/wireless and iwgetid or nmcli; None where unknown."""
    out = {"ssid": None, "iface": None, "signal_dbm": None}
    text = _read(os.path.join(root, "proc/net/wireless")) or ""
    for line in text.splitlines()[2:]:
        parts = line.split()
        if len(parts) >= 4 and parts[0].endswith(":"):
            out["iface"] = parts[0][:-1]
            try:
                out["signal_dbm"] = int(float(parts[3].rstrip(".")))
            except ValueError:
                pass
            break
    ssid = (run(["iwgetid", "-r"]) or "").strip()
    if not ssid:
        for line in (run(["nmcli", "-t", "-f", "active,ssid", "dev", "wifi"]) or "").splitlines():
            if line.startswith("yes:"):
                ssid = line[4:].replace("\\:", ":")
                break
    out["ssid"] = ssid or None
    return out


def cpu_temp_c(root: str = "/") -> float | None:
    temps = []
    for p in glob.glob(os.path.join(root, "sys/class/thermal/thermal_zone*/temp")):
        try:
            temps.append(int((_read(p) or "").strip()) / 1000)
        except ValueError:
            pass
    return round(max(temps), 1) if temps else None


def battery_v(root: str = "/") -> float | None:
    for p in sorted(glob.glob(os.path.join(root, "sys/class/power_supply/*/voltage_now"))):
        try:
            return round(int((_read(p) or "").strip()) / 1e6, 2)
        except ValueError:
            continue
    return None


def uptime_s(root: str = "/") -> int | None:
    try:
        return int(float((_read(os.path.join(root, "proc/uptime")) or "").split()[0]))
    except (IndexError, ValueError):
        return None


# -------------------------------------------------------------------------------------------- board

class Config:
    def __init__(self, env=os.environ) -> None:
        self.port = int(env.get("KITE_PORT", DEFAULT_PORT))
        self.bind = env.get("KITE_BIND", "0.0.0.0")  # reachable on its Wi-Fi IP  # nosec B104
        self.token_file = env.get("KITE_TOKEN_FILE", "/etc/kite/token")
        self.token = env.get("KITE_TOKEN", "")
        self.source = check_source(env.get("KITE_SOURCE", DEFAULT_SOURCE))
        self.camera = env.get("KITE_CAMERA", "auto")
        size = env.get("KITE_VIDEO_SIZE", "640x480").lower().split("x")
        self.width, self.height = int(size[0]), int(size[1])
        self.fps = float(env.get("KITE_VIDEO_FPS", "10"))
        self.hello_every = float(env.get("KITE_HELLO_EVERY_S", "1"))
        self.cloud = env.get("KITE_CLOUD", "1") not in ("0", "no", "off", "")
        self.cloud_every = float(env.get("KITE_CLOUD_EVERY_S", "2"))
        self.cloud_poll = max(1.0, float(env.get("KITE_CLOUD_POLL_S", "2")))
        self.st_host = env.get("SPACETIME_HOST", SPACETIME_HOST)
        self.st_db = env.get("SPACETIME_DB", SPACETIME_DB)
        self.sys_root = env.get("KITE_SYS_ROOT", "/")
        self.advertise = env.get("KITE_ADVERTISE", "")

    def load_token(self) -> str:
        if not self.token:
            self.token = (_read(self.token_file) or "").strip()
        if len(self.token) < 16:
            raise SystemExit(f"board: no token (need 16+ characters in {self.token_file} or KITE_TOKEN); "
                             "install.sh makes one")
        return self.token


class Board:
    def __init__(self, cfg: Config, run=_run) -> None:
        self.cfg, self.run_cmd = cfg, run
        self.token = cfg.load_token()
        self.host = socket.gethostname()
        self.started = time.monotonic()
        self.received = 0
        self.n = 0
        self.lock = threading.Lock()
        self.subscribers: list[queue.Queue] = []
        camera, reason = open_camera(cfg.camera, cfg.width, cfg.height, cfg.fps, cfg.sys_root)
        opener = None
        if (cfg.camera or "auto") == "auto":      # hot-plug: a camera plugged in after start is picked up
            def opener():
                return open_camera("auto", cfg.width, cfg.height, cfg.fps, cfg.sys_root)
        self.hub = FrameHub(camera, reason, cfg.fps, opener=opener)
        self.cloud = Spacetime(cfg.st_host, cfg.st_db) if cfg.cloud else None
        self.stop = threading.Event()
        self._addr_cache: tuple[float, tuple] = (0.0, ([], None))

    # ---- shared commands (direct HTTP and the SpacetimeDB fallback both land here)
    def addresses(self) -> tuple[list[str], str | None]:
        t, val = self._addr_cache
        if time.monotonic() - t > 10:
            val = addresses(self.run_cmd, self.cfg.advertise)
            self._addr_cache = (time.monotonic(), val)
        return val

    def status(self) -> dict:
        lan, ts = self.addresses()
        r = self.cfg.sys_root
        return {
            "host": self.host, "version": VERSION, "uptime_s": uptime_s(r),
            "service_uptime_s": int(time.monotonic() - self.started), "ips": lan, "tailscale_ip": ts,
            "port": self.cfg.port, "wifi": wifi(r, self.run_cmd), "cpu_temp_c": cpu_temp_c(r),
            "battery_v": battery_v(r), "messages_received": self.received, "hello_n": self.n,
            "camera": {"source": self.hub.camera.name, "reason": self.hub.reason, "width": self.cfg.width,
                       "height": self.cfg.height, "fps": self.cfg.fps, "viewers": self.hub.viewers},
        }

    def command(self, cmd: str, text=None, via: str = "direct") -> dict:
        if cmd == "hello":
            return {"reply": f"hello from {self.host}", "host": self.host,
                    "uptime_s": int(time.monotonic() - self.started)}
        if cmd == "ping":
            return {"pong": True, "host": self.host}
        if cmd == "status":
            return {"status": self.status(), "host": self.host}
        if cmd == "send":
            text = clean_text(text)
            with self.lock:
                self.received += 1
                n = self.received
            log(f"message {n} ({via}): {text}")
            self.publish({"kind": "got", "n": n, "text": text, "via": via})
            return {"reply": f'got "{text}"', "n": n, "host": self.host}
        raise ValueError(f"unknown command {str(cmd)[:20]!r} (hello, ping, status, send)")

    # ---- the recurring hello: to direct listeners (SSE) every second, to SpacetimeDB every 2 s
    def publish(self, event: dict) -> None:
        with self.lock:
            subs = list(self.subscribers)
        for q in subs:
            try:
                q.put_nowait(event)
            except queue.Full:
                pass

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=100)
        with self.lock:
            self.subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self.lock:
            if q in self.subscribers:
                self.subscribers.remove(q)

    def heartbeat(self) -> dict:
        lan, ts = self.addresses()
        return sign(self.token, {"msg": "hello", "n": self.n, "host": self.host, "ips": lan, "tailscale": ts,
                                 "port": self.cfg.port, "v": VERSION, "ts": int(time.time() * 1000)})

    def hello_loop(self) -> None:
        last_cloud, fails, quiet_until = 0.0, 0, 0.0
        while not self.stop.is_set():
            self.n += 1
            self.publish({"kind": "hello", "msg": "hello", "n": self.n, "host": self.host,
                          "t": int(time.time() * 1000)})
            now = time.monotonic()
            if self.cloud and now - last_cloud >= self.cfg.cloud_every and now >= quiet_until:
                last_cloud = now
                try:
                    self.cloud.push("heartbeat", self.cfg.source, self.heartbeat())
                    if fails:
                        log(f"cloud: heartbeat sent again after {fails} failures")
                    fails = 0
                except (OSError, ValueError, KeyError) as e:
                    fails += 1
                    quiet_until = now + min(30.0, 2.0 * fails)
                    if fails in (1, 5) or fails % 30 == 0:
                        log(f"cloud: heartbeat failed ({type(e).__name__}: {e}); retrying")
            self.stop.wait(self.cfg.hello_every)

    # ---- text commands through SpacetimeDB, for when the direct path is blocked
    def inbox_loop(self) -> None:
        if not self.cloud:
            return
        last_id, seen, backoff = None, {}, self.cfg.cloud_poll
        src = link_source(self.cfg.source)
        while not self.stop.is_set():
            try:
                if last_id is None:              # startup: earlier commands are history, not news
                    last_id = self.cloud.last_id("control", src)
                    log(f"cloud: reading text commands from {src} (after row {last_id})")
                for row in self.cloud.rows("control", src, last_id):
                    last_id = max(last_id, int(row["id"]))
                    self.handle_cloud_row(row, seen)
                backoff = self.cfg.cloud_poll
            except (OSError, ValueError, KeyError) as e:
                backoff = min(30.0, backoff * 2)
                log(f"cloud: inbox read failed ({type(e).__name__}: {e}); next try in {backoff:.0f} s")
            self.stop.wait(backoff)

    def handle_cloud_row(self, row: dict, seen: dict) -> dict | None:
        """Verify one command row and answer it. Returns the reply pushed, or None if it was ignored."""
        try:
            d = json.loads(row["payload"])
        except (ValueError, KeyError, TypeError):
            return None
        cid, ts = d.get("id") if isinstance(d, dict) else None, d.get("ts") if isinstance(d, dict) else None
        if not verify(self.token, d) or not isinstance(cid, str) or len(cid) > 40:
            log(f"cloud: ignored row {row.get('id')}: not signed with this board's token")
            return None
        server_t = row_time(row.get("t"))
        if isinstance(ts, bool) or not isinstance(ts, int) or server_t is None or abs(ts / 1000 - server_t) > FRESH_S:
            log(f"cloud: ignored row {row.get('id')}: stale or replayed")
            return None
        if d.get("to") != self.cfg.source or cid in seen:
            return None
        now = time.time()
        for k in [k for k, t in seen.items() if now - t > 4 * FRESH_S]:
            del seen[k]
        seen[cid] = now
        try:
            result = {"ok": True, **self.command(str(d.get("cmd")), d.get("text"), via="internet")}
        except ValueError as e:
            result = {"ok": False, "error": str(e)}
        reply = sign(self.token, {"reply_to": cid, "ts": int(time.time() * 1000), **result})
        self.cloud.push("control", reply_source(self.cfg.source), reply)
        return reply


# ---------------------------------------------------------------------------------------------- HTTP

def _redact(path: str) -> str:
    return re.sub(r"(token=)[^&\s]*", r"\1***", path)


VIEW_PAGE = """<!doctype html><meta charset="utf-8"><title>Kite: {host}</title>
<style>body{{margin:0;background:#111;color:#eee;font:15px system-ui;display:grid;place-items:center;
min-height:100vh}}img{{max-width:100vw;max-height:90vh;background:#000}}p{{opacity:.7}}</style>
<img src="/video.mjpg?token={token}" alt="Kite board camera"><p>{host}: {camera}</p>"""


class Handler(BaseHTTPRequestHandler):
    board: Board = None  # set by make_server
    server_version = "kite-link/1"

    def log_message(self, fmt, *args) -> None:  # never log the token from a ?token= query
        log("http " + _redact(fmt % args))

    def _json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _authed(self, query: dict) -> bool:
        import hmac
        got = ""
        auth = self.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            got = auth[7:].strip()
        elif "token" in query:
            got = query["token"][0]
        if got and hmac.compare_digest(got.encode(), self.board.token.encode()):
            return True
        self._json(401, {"error": "bad or missing token"})
        return False

    def do_GET(self) -> None:  # noqa: N802
        url = urllib.parse.urlsplit(self.path)
        query = urllib.parse.parse_qs(url.query)
        path = url.path.rstrip("/") or "/"
        b = self.board
        if path == "/health":
            return self._json(200, {"kite": "unoq-link", "host": b.host, "version": VERSION})
        if not self._authed(query):
            return None
        if path in ("/hello", "/ping", "/status"):
            return self._json(200, {"ok": True, **b.command(path[1:])})
        if path == "/snapshot.jpg":
            got = b.hub.wait(b.hub.seq, timeout=5)      # a fresh frame, not one left from an idle camera
            if not got:
                return self._json(503, {"error": "no frame from the camera"})
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(got[1])))
            self.send_header("X-Kite-Camera", b.hub.camera.name)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(got[1])
            return None
        if path == "/video.mjpg":
            return self._video()
        if path == "/events":
            return self._events()
        if path in ("/", "/view"):
            token = urllib.parse.quote(b.token, safe="")
            body = VIEW_PAGE.format(host=html.escape(b.host), token=token,
                                    camera=html.escape(b.hub.camera.name)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(body)
            return None
        return self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        url = urllib.parse.urlsplit(self.path)
        if not self._authed(urllib.parse.parse_qs(url.query)):
            return None
        if url.path.rstrip("/") not in ("/send", "/cmd"):
            return self._json(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = -1
        if not 0 < n <= 8192:
            return self._json(400, {"error": "send a JSON body under 8 KB"})
        try:
            d = json.loads(self.rfile.read(n).decode("utf-8"))
            if not isinstance(d, dict):
                raise ValueError("body must be a JSON object")
            cmd = "send" if url.path.rstrip("/") == "/send" else str(d.get("cmd"))
            return self._json(200, {"ok": True, **self.board.command(cmd, d.get("text"))})
        except (ValueError, UnicodeDecodeError) as e:
            return self._json(400, {"ok": False, "error": str(e)})

    def _video(self) -> None:
        hub = self.board.hub
        with hub.cond:
            if hub.viewers >= MAX_VIEWERS:
                busy = True
            else:
                busy = False
                hub.viewers += 1
        if busy:
            return self._json(503, {"error": f"already {MAX_VIEWERS} viewers"})
        try:
            self.send_response(200)
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Kite-Camera", hub.camera.name)
            self.end_headers()
            seq = 0
            while not self.board.stop.is_set():
                got = hub.wait(seq, timeout=5)
                if not got:
                    continue
                seq, jpeg = got
                self.wfile.write(mjpeg_part(jpeg))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with hub.cond:
                hub.viewers -= 1

    def _events(self) -> None:
        q = self.board.subscribe()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b"retry: 2000\n\n")
            self.wfile.flush()
            last = time.monotonic()
            while not self.board.stop.is_set():
                try:
                    ev = q.get(timeout=1)
                except queue.Empty:
                    if time.monotonic() - last < 10:
                        continue
                    self.wfile.write(b": keepalive\n\n")
                else:
                    self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                self.wfile.flush()
                last = time.monotonic()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            self.board.unsubscribe(q)


def mjpeg_part(jpeg: bytes) -> bytes:
    head = f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n\r\n".encode()
    return head + jpeg + b"\r\n"


def make_server(board: Board) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"board": board})
    ThreadingHTTPServer.allow_reuse_address = True
    srv = ThreadingHTTPServer((board.cfg.bind, board.cfg.port), handler)
    srv.daemon_threads = True
    return srv


def start(board: Board) -> ThreadingHTTPServer:
    """Start every thread and the server (serving in a thread); returns the server."""
    srv = make_server(board)
    board.cfg.port = srv.server_address[1]          # the real port when asked for port 0
    for target, name in ((board.hub.run, "camera"), (board.hello_loop, "hello"), (board.inbox_loop, "inbox"),
                         (srv.serve_forever, "http")):
        threading.Thread(target=target, name=name, daemon=True).start()
    return srv


def shutdown(board: Board, srv: ThreadingHTTPServer) -> None:
    board.stop.set()
    board.hub.stop.set()
    with board.hub.cond:
        board.hub.cond.notify_all()           # wake viewers so their connections close now
    srv.shutdown()
    srv.server_close()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, help="override KITE_PORT (default 8765)")
    ap.add_argument("--camera", help="auto | synthetic | /dev/videoN (overrides KITE_CAMERA)")
    ap.add_argument("--no-cloud", action="store_true", help="no SpacetimeDB heartbeat or text fallback")
    a = ap.parse_args(argv)
    cfg = Config()
    if a.port:
        cfg.port = a.port
    if a.camera:
        cfg.camera = a.camera
    if a.no_cloud:
        cfg.cloud = False
    board = Board(cfg)
    srv = start(board)
    lan, ts = board.addresses()
    log(f"kite-link on port {cfg.port}: {board.hub.camera.name} ({board.hub.reason}), "
        f"{cfg.width}x{cfg.height} at {cfg.fps:g} fps; LAN {', '.join(lan) or 'none'}"
        f"{'; tailscale ' + ts if ts else ''}; cloud {'on as ' + cfg.source if cfg.cloud else 'off'}")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown(board, srv)


if __name__ == "__main__":
    main()
