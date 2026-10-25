"""YOLO26n (COCO) obstacle boxes for the drone's camera pane: ``python run/app.py --yolo``.

Show only. The boxes are drawn on the camera and nothing here reaches the drone, the follow or the console.
The model loads with the person detector, before the control loop starts; then detection runs on its own thread
on the newest frame, at most ``MAX_HZ`` times a second, so the 20 Hz control loop never waits on it.
"""

from __future__ import annotations

import threading
import time

MAX_HZ = 10.0


def load_detector(device: str = ""):
    """YOLO26n from third_party/videotesting/weights, keeping the obstacle classes the webcam demo keeps."""
    from demo import _vendor_path

    _vendor_path()
    from streetseg.models import YoloDetector

    detector = YoloDetector("yolo26n.pt", imgsz=640, device=device or None)
    detector.load()
    return detector


class ObstacleView:
    """The newest frame in, the newest boxes out (``.last``, ``.last_t``)."""

    def __init__(self, detector):
        self._detector = detector
        self._frame = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="obstacles", daemon=True)
        self.last: list = []
        self.last_t = 0.0

    def start(self) -> ObstacleView:
        self._thread.start()
        return self

    def offer(self, image) -> None:
        """Keep only the newest frame. Returns at once."""
        with self._lock:
            self._frame = image

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                image, self._frame = self._frame, None
            if image is None:
                self._stop.wait(0.01)
                continue
            t0 = time.monotonic()
            try:
                boxes = self._detector.predict(image)
            except Exception:  # a bad frame only means no boxes this time
                boxes = []
            self.last, self.last_t = boxes, time.time()
            self._stop.wait(max(0.0, 1.0 / MAX_HZ - (time.monotonic() - t0)))


def overlaps(a, b, iou: float = 0.5) -> bool:
    """True when boxes a and b (x1, y1, x2, y2) are mostly the same box."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return union > 0 and inter / union >= iou
