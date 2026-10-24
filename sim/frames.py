"""Sim camera frames for perception, shaped like cv2.VideoCapture.

    from frames import SimFrameSource           # run from sim/, or add sim/ to sys.path
    src = SimFrameSource("drone")               # or "chest"
    ok, bgr = src.read()                        # 960x720 BGR ndarray, blocks for the next frame
    src.pose  # {"x":..,"y":..,"heading":..,"cam_height":..,"pitch":0} (meters east/north, compass deg)

The drone frames go through the Tello camera model (tello_cam.py, sim/public/tello_camera_model.json): lens,
noise, exposure, colour, compression, plus 100-200 ms of latency and the odd dropped or repeated frame. The sim
page's ?cam=raw (sent as "cam_model": "raw" in its pose messages) or SimFrameSource(raw=True) turns it off.
.pose is not delayed, like the Tello's state stream, which runs ahead of its video.

Needs websockets, numpy and opencv-python.
"""

from __future__ import annotations

import json
import threading
import time

import cv2
import numpy as np
from websockets.sync.client import connect

from tello_cam import TelloCam, load_model

CAMERAS = {"drone": 0, "chest": 1}
MAX_QUEUED = 8          # frames in flight on the simulated link


class SimFrameSource:
    def __init__(self, camera: str = "drone", url: str = "ws://127.0.0.1:5020/sub", timeout: float = 2.0,
                 raw: bool = False):
        if camera not in CAMERAS:
            raise ValueError(f"camera must be one of {list(CAMERAS)}")
        self.camera, self.url, self.timeout = camera, url, timeout
        self.pose: dict | None = None
        model = None if raw else load_model()
        self.tello = TelloCam(model) if model and camera in model["cameras"] else None
        self._yaw_rate = 0.0                       # deg/s, from successive pose headings
        self._last_pose_t: float | None = None
        self._queue: list = []                     # (due time, jpeg or None = repeat the last frame, yaw rate)
        self._last_img: np.ndarray | None = None
        self._cond = threading.Condition()
        self._running = True
        threading.Thread(target=self._loop, name=f"sim-{camera}", daemon=True).start()

    @property
    def modelled(self) -> bool:
        return self.tello is not None and (self.pose or {}).get("cam_model") != "raw"

    def _on_pose(self, d: dict) -> None:
        now = time.monotonic()
        if self.pose and self._last_pose_t is not None and 0 < now - self._last_pose_t < 1.0:
            dh = (d["heading"] - self.pose["heading"] + 180) % 360 - 180
            self._yaw_rate = dh / (now - self._last_pose_t)
        self.pose, self._last_pose_t = d, now

    def _on_frame(self, jpeg: bytes) -> None:
        now = time.monotonic()
        if not self.modelled:
            entry = (now, jpeg, 0.0)
        else:
            kind, latency = self.tello.stream()
            if kind == "drop":
                return
            entry = (now + latency, None if kind == "dup" else jpeg, self._yaw_rate)
        with self._cond:
            self._queue = (self._queue + [entry])[-MAX_QUEUED:]
            self._cond.notify_all()

    def _loop(self) -> None:
        cid = CAMERAS[self.camera]
        while self._running:
            try:
                with connect(self.url, max_size=8 * 1024 * 1024) as ws:
                    for msg in ws:
                        if not self._running:
                            return
                        if isinstance(msg, bytes):
                            if msg[:1] == bytes([cid]):
                                self._on_frame(msg[1:])
                        else:
                            d = json.loads(msg)
                            if d.get("type") == "pose" and d.get("camera", self.camera) == self.camera:
                                self._on_pose(d)
            except Exception:
                time.sleep(1.0)            # relay not up yet, or it restarted

    def _take_due(self):
        """The newest queued frame whose time has come (older due ones are skipped, as before), or None."""
        now = time.monotonic()
        due = [e for e in self._queue if e[0] <= now]
        if not due:
            return None
        self._queue = [e for e in self._queue if e[0] > now]
        return due[-1]

    def read(self) -> tuple[bool, np.ndarray | None]:
        deadline = time.monotonic() + self.timeout
        with self._cond:
            while (entry := self._take_due()) is None:
                left = deadline - time.monotonic()
                if left <= 0:
                    return False, None
                nxt = min((e[0] for e in self._queue), default=None)
                self._cond.wait(left if nxt is None else min(left, max(nxt - time.monotonic(), 0.001)))
        t, jpeg, yaw_rate = entry
        if jpeg is None:                   # the link repeated the previous frame
            if self._last_img is None:
                return False, None
            return True, self._last_img.copy()
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if img is not None and self.modelled:
            img = self.tello.process(img, t, yaw_rate)
        self._last_img = img
        return img is not None, img

    def isOpened(self) -> bool:
        return self._running

    def release(self) -> None:
        self._running = False
