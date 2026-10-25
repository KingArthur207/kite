"""Save the app's camera, with the person boxes, without touching the flight loop.

The control thread only copies a new frame into a short queue. A writer thread draws the boxes and writes
``video.mkv`` plus one JPEG per frame. A full queue drops the frame: a slow disk must not stall takeoff.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Same colours as the window overlay (run/ui.py).
OTHER_BGR = (130, 128, 125)
ACCENT_BGR = (87, 119, 217)
LOOSE_BGR = (94, 178, 214)


@dataclass
class CaptureSummary:
    directory: Path
    written: int = 0
    dropped: int = 0
    error: str | None = None


def _fresh_dir(root: Path) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = root / f"{stamp}_app"
    n = 2
    while path.exists():
        path = root / f"{stamp}_app_{n}"
        n += 1
    path.mkdir(parents=True)
    (path / "frames").mkdir()
    return path


def _boxes(seen) -> list[tuple[int, int, int, int]]:
    if seen is None or time.time() - seen.last_t >= 1.0:
        return []
    out = []
    for det in seen.last:
        out.append((int(det.x1), int(det.y1), int(det.x2), int(det.y2)))
    return out


def _track(follow) -> tuple[tuple[int, int, int, int], bool] | None:
    if follow is None:
        return None
    snap = follow.snapshot()
    if snap is None or not snap.confirmed or time.time() - snap.t >= 1.0:
        return None
    box = tuple(int(v) for v in snap.box)
    return box, bool(follow.active)


def _draw(img: np.ndarray, boxes, track) -> None:
    import cv2

    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), OTHER_BGR, 1, cv2.LINE_AA)
    if track is None:
        return
    (x1, y1, x2, y2), locked = track
    col = ACCENT_BGR if locked else LOOSE_BGR
    cv2.rectangle(img, (x1, y1), (x2, y2), col, 3, cv2.LINE_AA)
    label = "following" if locked else "person in front"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    y = max(y1 - 8, th + 8)
    cv2.rectangle(img, (x1, y - th - 8), (x1 + tw + 12, y + 4), col, -1)
    cv2.putText(img, label, (x1 + 6, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (24, 24, 26), 1, cv2.LINE_AA)


class SessionCapture:
    """One flight's ``video.mkv`` and ``frames/NNNNNN.jpg`` under ``recordings/<time>_app``."""

    def __init__(self, root: Path, queue_frames: int = 15):
        self.dir = _fresh_dir(Path(root))
        self.written = 0
        self.dropped = 0
        self.error: str | None = None
        self._video_failed = False
        self._last_seq: int | None = None
        self._q: queue.Queue = queue.Queue(maxsize=queue_frames)
        self._thread = threading.Thread(target=self._writer, name="capture", daemon=True)
        self._started = False

    def start(self) -> SessionCapture:
        if not self._started:
            self._thread.start()
            self._started = True
        return self

    def offer(self, image: np.ndarray, seq: int, seen, follow) -> None:
        """Copy this frame for the writer. Returns immediately; drops the frame when the queue is full."""
        if seq == self._last_seq:
            return
        self._last_seq = seq
        if self._q.full():
            self.dropped += 1
            return
        boxes, track = _boxes(seen), _track(follow)
        self._q.put_nowait((image.copy(), boxes, track))

    def close(self, timeout_s: float = 5.0) -> CaptureSummary:
        """Finish the file. Safe to call once, from the control thread as it exits."""
        if not self._started:
            self.start()
        try:
            self._q.put_nowait(None)
        except queue.Full:
            try:
                self._q.get_nowait()
                self.dropped += 1
            except queue.Empty:
                pass
            self._q.put_nowait(None)
        self._thread.join(timeout=timeout_s)
        return CaptureSummary(self.dir, self.written, self.dropped, self.error)

    def _writer(self) -> None:
        import cv2

        vw = None
        try:
            while True:
                item = self._q.get()
                if item is None:
                    break
                img, boxes, track = item
                _draw(img, boxes, track)
                if vw is None and not self._video_failed:
                    h, w = img.shape[:2]
                    vw = cv2.VideoWriter(str(self.dir / "video.mkv"), cv2.CAP_FFMPEG,
                                         cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (w, h))
                    if not vw.isOpened():
                        vw.release()
                        vw = None
                        self._video_failed = True
                        self.error = self.error or "video.mkv writer unavailable"
                if vw is not None:
                    vw.write(img)
                cv2.imwrite(str(self.dir / "frames" / f"{self.written:06d}.jpg"), img)
                self.written += 1
        except Exception as e:
            self.error = f"{type(e).__name__}: {e}"
        finally:
            if vw is not None:
                vw.release()
