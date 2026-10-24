"""Sim camera frames for perception, shaped like cv2.VideoCapture.

    from frames import SimFrameSource           # run from sim/, or add sim/ to sys.path
    src = SimFrameSource("drone")               # or "chest"
    ok, bgr = src.read()                        # 960x720 BGR ndarray, blocks for the next frame
    src.pose  # {"x":..,"y":..,"heading":..,"cam_height":..,"pitch":0} (meters east/north, compass deg)

Needs websockets, numpy and opencv-python.
"""

from __future__ import annotations

import json
import threading
import time

import cv2
import numpy as np
from websockets.sync.client import connect

CAMERAS = {"drone": 0, "chest": 1}


class SimFrameSource:
    def __init__(self, camera: str = "drone", url: str = "ws://127.0.0.1:5020/sub", timeout: float = 2.0):
        if camera not in CAMERAS:
            raise ValueError(f"camera must be one of {list(CAMERAS)}")
        self.camera, self.url, self.timeout = camera, url, timeout
        self.pose: dict | None = None
        self._jpeg: bytes | None = None
        self._seq = 0
        self._read_seq = 0
        self._cond = threading.Condition()
        self._running = True
        threading.Thread(target=self._loop, name=f"sim-{camera}", daemon=True).start()

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
                                with self._cond:
                                    self._jpeg, self._seq = msg[1:], self._seq + 1
                                    self._cond.notify_all()
                        else:
                            d = json.loads(msg)
                            if d.get("type") == "pose" and d.get("camera", self.camera) == self.camera:
                                self.pose = d
            except Exception:
                time.sleep(1.0)            # relay not up yet, or it restarted

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self._cond:
            if not self._cond.wait_for(lambda: self._seq > self._read_seq, self.timeout):
                return False, None
            jpeg, self._read_seq = self._jpeg, self._seq
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        return img is not None, img

    def isOpened(self) -> bool:
        return self._running

    def release(self) -> None:
        self._running = False
