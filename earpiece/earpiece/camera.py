"""The body camera, on demand.

Kite has two ways of seeing:
  * outdoors, the drone looks ahead and the body camera covers the near field;
  * indoors (or anywhere a drone can't fly), the body camera alone is enough for
    obstacles, "what's in front of me?" and reading text aloud.

Frames usually arrive from the perception process as `scene_image` events
(camera "chest"). When nothing is sending them (indoor mode, or perception not
running) the earpiece grabs one frame itself from a local camera, only when asked:
the camera is never kept open, so it costs nothing and holds no privacy-sensitive
stream.
"""

from __future__ import annotations

import os
import threading
import time


class BodyCamera:
    def __init__(self, index: int | None = None, log=print):
        env = os.environ.get("KITE_CAMERA")
        self.index = index if index is not None else (int(env) if env and env.isdigit() else 0)
        self.enabled = os.environ.get("KITE_CAMERA", "0") != "off"
        self.log = log
        self._lock = threading.Lock()
        self.last_error: str | None = None

    def capture_jpeg(self, width: int = 1280, quality: int = 85, warmup_frames: int = 6) -> bytes | None:
        """Open the camera, let exposure settle for a few frames, return one JPEG, close it.
        None if OpenCV or the camera is unavailable (e.g. macOS camera permission not granted)."""
        if not self.enabled:
            return None
        try:
            import cv2
        except ImportError:
            self.last_error = "opencv not installed (pip install 'kite-earpiece[camera]')"
            return None
        with self._lock:
            cap = cv2.VideoCapture(self.index)
            try:
                if not cap.isOpened():
                    self.last_error = "camera not available (check the camera permission for this terminal)"
                    return None
                frame = None
                t_end = time.monotonic() + 2.5
                n = 0
                while time.monotonic() < t_end and n < warmup_frames:
                    ok, f = cap.read()
                    if ok and f is not None:
                        frame = f
                        n += 1
                if frame is None:
                    self.last_error = "camera returned no frames"
                    return None
                h, w = frame.shape[:2]
                if w > width:
                    frame = cv2.resize(frame, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
                ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                return buf.tobytes() if ok else None
            finally:
                cap.release()
