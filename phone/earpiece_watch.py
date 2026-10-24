"""A background WebSocket client to the local earpiece (ws://127.0.0.1:5011): is it up, what does it know about the
drone, and a way to hand it chest-camera frames (`scene_image`, which is WebSocket only).

The earpiece's Origin allow-list (earpiece/earpiece/transport.py) lets non-browser clients in: this one sends no
Origin header. Positions still go over UDP with EarpieceClient, like every other Kite sender.
"""

from __future__ import annotations

import base64
import json
import threading
import time

STATE_FRESH_S = 2.0         # the earpiece pushes `state` at 5 Hz
MAX_SCENE_JPEG = 180_000    # events.py refuses base64 over 250 000 characters


class EarpieceWatch:
    def __init__(self, url: str = "ws://127.0.0.1:5011", log=None):
        self.url = url
        self.log = log or (lambda s: None)
        self.last_state = 0.0
        self.drone: dict = {}
        self._ws = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.sent_frames = 0

    @property
    def up(self) -> bool:
        return time.monotonic() - self.last_state <= STATE_FRESH_S

    def start(self) -> EarpieceWatch:
        self._thread = threading.Thread(target=self._run, name="earpiece-watch", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            ws = self._ws
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass

    def _run(self) -> None:
        from websockets.sync.client import connect
        delay = 0.5
        while not self._stop.is_set():
            try:
                with connect(self.url, open_timeout=2, max_size=4 * 1024 * 1024) as ws:
                    with self._lock:
                        self._ws = ws
                    delay = 0.5
                    for raw in ws:
                        if self._stop.is_set():
                            break
                        try:
                            m = json.loads(raw)
                        except ValueError:
                            continue
                        if isinstance(m, dict) and m.get("type") == "state":
                            self.last_state = time.monotonic()
                            d = m.get("drone")
                            self.drone = d if isinstance(d, dict) else {}
            except Exception:
                pass
            with self._lock:
                self._ws = None
            self._stop.wait(delay)
            delay = min(delay * 2, 5.0)

    def send_chest_frame(self, jpeg: bytes) -> bool:
        """Give the earpiece the newest chest frame, for "what's in front of me?" (app.py keeps the newest one)."""
        if len(jpeg) > MAX_SCENE_JPEG or not jpeg.startswith(b"\xff\xd8"):
            return False
        with self._lock:
            ws = self._ws
        if ws is None:
            return False
        msg = {"type": "scene_image", "source": "phone", "camera": "chest", "jpeg": base64.b64encode(jpeg).decode()}
        try:
            ws.send(json.dumps(msg))
            self.sent_frames += 1
            return True
        except Exception:
            return False

    def summary(self) -> dict:
        d = {"ear": self.up}
        if self.drone:
            d["drone"] = {k: self.drone.get(k) for k in ("battery", "state") if self.drone.get(k) is not None}
        return d
