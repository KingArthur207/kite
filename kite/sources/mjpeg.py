"""Serve any kite.sources frame source as an MJPEG stream for the dashboard's video panes.

    python -m kite.sources.mjpeg --source recordings/<take> --port 8090        # a recorded take, looped
    python -m kite.sources.mjpeg --source tello --port 8090                    # live Tello (only when run/app.py isn't flying it)
    python -m kite.sources.mjpeg --source webcam:0 --port 8090 --fps 15 --width 640

GET /video.mjpg is multipart/x-mixed-replace JPEG, so a plain <img src> shows it.
Files and recordings play at their own frame rate and loop at the end.

It is a live camera with no password, so it listens on this computer only (127.0.0.1) and sends no CORS header:
another web page can show the stream in an <img> but can't read its pixels, and a page that rebinds its own
domain to 127.0.0.1 is refused by the Host check. To watch from another machine, pass --host 0.0.0.0, and only
on a network you trust: anyone on it can then watch.
"""

from __future__ import annotations

import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

from .open import open_source, parse_spec

BOUNDARY = "kiteframe"
LOOPED_KINDS = ("file", "recording")
LOOPBACK = ("127.0.0.1", "localhost", "::1")


def host_ok(host_header: str | None, bound: str) -> bool:
    """Bound to loopback, only requests that name this computer are served (DNS rebinding sends its own name)."""
    if bound not in LOOPBACK:
        return True                           # --host 0.0.0.0 or a LAN address: shared on purpose
    name = (host_header or "").strip().lower()
    if name.startswith("["):
        name = name[1:].split("]", 1)[0]
    else:
        name = name.rsplit(":", 1)[0] if name.count(":") == 1 else name
    return name in LOOPBACK


class Latest:
    """The newest JPEG; clients wait on it so each gets every new frame once."""

    def __init__(self):
        self.cond = threading.Condition()
        self.jpeg: bytes | None = None
        self.seq = 0

    def put(self, jpeg: bytes) -> None:
        with self.cond:
            self.jpeg, self.seq = jpeg, self.seq + 1
            self.cond.notify_all()

    def wait_newer(self, seq: int, timeout: float = 5.0) -> tuple[int, bytes | None]:
        with self.cond:
            self.cond.wait_for(lambda: self.seq != seq, timeout)
            return self.seq, self.jpeg


def produce(spec: str, latest: Latest, fps: float, width: int, quality: int) -> None:
    kind, _ = parse_spec(spec)
    looped = kind in LOOPED_KINDS
    opts = {"realtime": True} if looped else {}
    period = 1.0 / fps if fps > 0 else 0.0
    while True:
        try:
            with open_source(spec, **opts) as src:
                print(f"mjpeg: reading {src.name}", flush=True)
                last = 0.0
                while True:
                    ok, bgr = src.read()
                    if not ok:
                        break
                    now = time.monotonic()
                    if now - last < period:
                        continue
                    last = now
                    h, w = bgr.shape[:2]
                    if width and w > width:
                        bgr = cv2.resize(bgr, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
                    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
                    if ok:
                        latest.put(buf.tobytes())
        except Exception as e:  # a live source that drops out: say why, then reopen
            print(f"mjpeg: source error: {e}", flush=True)
        if not looped:
            print("mjpeg: no frames, reopening in 2 s", flush=True)
            time.sleep(2)


def make_handler(latest: Latest, bound: str = "127.0.0.1") -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: object) -> None:
            pass

        def do_GET(self) -> None:
            if self.path.split("?")[0] != "/video.mjpg":
                self.send_error(404, "try /video.mjpg")
                return
            if not host_ok(self.headers.get("Host"), bound):
                self.send_error(403, "this stream is for this computer only")
                return
            self.send_response(200)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
            self.end_headers()
            seq = 0
            try:
                while True:
                    seq, jpeg = latest.wait_newer(seq)
                    if jpeg is None:
                        continue
                    self.wfile.write(f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n\r\n".encode())
                    self.wfile.write(jpeg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="any kite.sources spec: tello | webcam:N | file.mp4 | recordings/<take>")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--host", default="127.0.0.1",
                    help="where to listen (default this computer only; 0.0.0.0 shares the camera with the network)")
    ap.add_argument("--fps", type=float, default=15)
    ap.add_argument("--width", type=int, default=640, help="downscale wider frames to this width (0: never)")
    ap.add_argument("--quality", type=int, default=75)
    return ap


def main() -> None:
    a = parser().parse_args()
    latest = Latest()
    threading.Thread(target=produce, args=(a.source, latest, a.fps, a.width, a.quality), daemon=True).start()
    server = ThreadingHTTPServer((a.host, a.port), make_handler(latest, a.host))
    server.daemon_threads = True
    print(f"mjpeg: http://localhost:{a.port}/video.mjpg" + ("" if a.host in LOOPBACK else
          f" (listening on {a.host}: anyone on this network can watch)"), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
