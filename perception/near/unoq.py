"""The UNO Q body camera as a live source: the board's MJPEG stream, over whatever network both are on.

    python -m perception.near --source unoq                    # find the board (heartbeat, Tailscale), as kite_link does
    python -m perception.near --source unoq:10.31.226.210      # or name it: HOST[:PORT]

The board (hardware/arduino/unoq_link/board.py) serves its USB webcam at /video.mjpg behind a token. This reuses
kite_link.py's discovery and token (saved once with `python kite_link.py token <token>`, or KITE_TOKEN), so
there is nothing new to set up.

What it guarantees to the near loop:
  * only the newest frame: a reader thread drains the stream and keeps the latest JPEG, so a slow model never
    builds a backlog (a beep about a frame from 3 s ago would be worse than no beep);
  * no fake distances: while the board shows its "NO CAMERA" test pattern, read() returns nothing;
  * it survives Wi-Fi drops: the stream is reopened with back-off, and the board found again if it moved.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LINK_DIR = ROOT / "hardware" / "arduino" / "unoq_link"
STATUS_EVERY_S = 3.0          # how often the camera's state (real camera or test pattern) is re-read
STALE_S = 2.0                 # read() waits this long for a new frame before reporting none


def _kite_link():
    if str(LINK_DIR) not in sys.path:
        sys.path.insert(0, str(LINK_DIR))
    import kite_link  # noqa: PLC0415  (standard library only; lives with the board code)
    return kite_link


def parse_spec(spec: str) -> str | None:
    """'unoq' -> None (discover); 'unoq:HOST[:PORT]' -> 'HOST[:PORT]'."""
    low = spec.strip()
    if low.lower() == "unoq":
        return None
    if low.lower().startswith("unoq:") and low[5:]:
        return low[5:]
    raise ValueError(f"not a UNO Q source: {spec!r} (use unoq or unoq:HOST[:PORT])")


def is_unoq(spec: str | None) -> bool:
    return bool(spec) and (spec.lower() == "unoq" or spec.lower().startswith("unoq:"))


class UnoqSource:
    """cv2.VideoCapture-like (read, release, name) over the board's MJPEG stream."""

    def __init__(self, spec: str, out=print, connect_timeout: float = 3.0) -> None:
        import cv2
        import numpy as np
        self._cv2, self._np, self.out = cv2, np, out
        kl = self.kl = _kite_link()
        board = parse_spec(spec)
        args = kl.build_parser().parse_args(["status"])
        args.board, args.timeout = board, connect_timeout
        if os.environ.get("KITE_NO_CLOUD") == "1":
            args.no_cloud = True
        self.link = kl.Link(args, out=lambda *_: None)
        self.link.need_token()
        self.direct = self._find()
        st = self._status()
        cam = (st or {}).get("camera") or {}
        self.camera = str(cam.get("source") or "unknown")
        self.size = (int(cam.get("width") or 640), int(cam.get("height") or 480))
        self.test_pattern = self.camera == "synthetic"
        # the board reports "usb camera videoN", not the model; the team's body camera is a Logitech C270
        self.name = f"unoq {self.direct} {self.camera} (Logitech)"
        self._cond = threading.Condition()
        self._jpeg: bytes | None = None
        self._seq = self._taken = 0
        self._stop = threading.Event()
        self.frames_in = self.frames_dropped = self.reconnects = 0
        self._warned_pattern = False
        self._started = False                 # the stream opens on the first read(): nothing flows while models load
        threading.Thread(target=self._watch_status, name="unoq-status", daemon=True).start()
        out(f"[near] UNO Q at {self.direct} ({self.direct.label}): camera {self.camera} "
            f"{self.size[0]}x{self.size[1]}" + (" - showing its test pattern, waiting for a camera"
                                                  if self.test_pattern else ""), flush=True)

    # ------------------------------------------------------------------ discovery and status
    def _find(self):
        d = self.link.direct()
        if d is None:
            tried = ", ".join(f"{label} {host}:{port} ({why})" for label, host, port, _, why in self.link.tried) \
                or "nothing to try (no heartbeat in SpacetimeDB and no Tailscale)"
            raise SystemExit(f"cannot reach the UNO Q for video: tried {tried}. "
                             "Run `python hardware/arduino/unoq_link/kite_link.py doctor` to see why, "
                             "or pass --source unoq:HOST.")
        return d

    def _status(self) -> dict | None:
        try:
            code, body = self.direct.call("/status")
        except OSError:
            return None
        if code == 401:
            raise SystemExit("the UNO Q rejected this computer's token (401): save it again with "
                             "`python hardware/arduino/unoq_link/kite_link.py token <token>`")
        if code != 200 or not isinstance(body, dict):
            return None
        return body.get("status") if isinstance(body.get("status"), dict) else body   # board: {"ok", "status", "host"}

    def _watch_status(self) -> None:
        while not self._stop.wait(STATUS_EVERY_S):
            st = self._status()
            if not st:
                continue
            cam = st.get("camera") or {}
            pattern = cam.get("source") == "synthetic"
            if pattern != self.test_pattern:
                self.test_pattern = pattern
                self.camera = str(cam.get("source") or self.camera)
                self.out("[near] UNO Q: " + ("camera lost, test pattern: no distances until it is back"
                                             if pattern else f"camera {self.camera} is back"), flush=True)

    # ------------------------------------------------------------------ the stream
    def _reader(self) -> None:
        backoff = 0.5
        while not self._stop.is_set():
            try:
                with self.direct.open("/video.mjpg", timeout=10) as r:
                    boundary = self.kl.boundary_of(r.headers.get("Content-Type", ""))
                    backoff = 0.5
                    for jpeg in self.kl.iter_mjpeg(r, boundary):
                        if self._stop.is_set():
                            return
                        if not self.kl.is_jpeg(jpeg):
                            continue
                        with self._cond:
                            if self._seq > self._taken:
                                self.frames_dropped += 1          # the loop never asked for the previous one
                            self._jpeg, self._seq = jpeg, self._seq + 1
                            self.frames_in += 1
                            self._cond.notify_all()
            except Exception as e:  # noqa: BLE001  (any network error: reconnect, never crash the loop)
                if self._stop.is_set():
                    return
                self.reconnects += 1
                if self.reconnects in (1, 5) or self.reconnects % 30 == 0:
                    self.out(f"[near] UNO Q video: {type(e).__name__}: {e}; reconnecting", flush=True)
            if self._stop.wait(backoff):
                return
            backoff = min(5.0, backoff * 2)
            if self.reconnects and self.reconnects % 5 == 0 and not self.link.a.board:
                try:                                   # it may have moved networks: look for it again
                    self.direct = self._find()
                except SystemExit:
                    pass

    def read(self):
        """(True, BGR image) for the newest frame not yet returned; (False, None) after STALE_S without one,
        or while the board shows its test pattern."""
        if not self._started:
            self._started = True
            threading.Thread(target=self._reader, name="unoq-video", daemon=True).start()
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq > self._taken or self._stop.is_set(), STALE_S):
                return False, None
            if self._stop.is_set():
                return False, None
            jpeg, self._taken = self._jpeg, self._seq
        if self.test_pattern:
            if not self._warned_pattern:
                self._warned_pattern = True
                self.out("[near] the UNO Q is sending its test pattern (no camera): skipping frames", flush=True)
            return False, None
        self._warned_pattern = False
        img = self._cv2.imdecode(self._np.frombuffer(jpeg, self._np.uint8), self._cv2.IMREAD_COLOR)
        return (img is not None), img

    def release(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
