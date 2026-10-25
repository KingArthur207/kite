"""The app's session capture: a video file and one JPEG per frame, dropping instead of blocking."""

import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "run"))

from capture import ACCENT_BGR, SessionCapture  # noqa: E402


class _Det:
    def __init__(self, box):
        self.x1, self.y1, self.x2, self.y2 = box


class _Seen:
    def __init__(self, box):
        self.last = [_Det(box)]
        self.last_t = time.time()


class _Snap:
    def __init__(self, box):
        self.box = box
        self.t = time.time()
        self.confirmed = True


class _Follow:
    active = True

    def __init__(self, box):
        self._snap = _Snap(box)

    def snapshot(self):
        return self._snap


def test_writes_video_and_frames_with_the_person_box(tmp_path):
    cap = SessionCapture(tmp_path, queue_frames=8).start()
    box = (8, 8, 40, 50)
    for seq in range(3):
        img = np.full((60, 80, 3), 30, np.uint8)
        cap.offer(img, seq, _Seen(box), _Follow(box))
    summary = cap.close()

    assert summary.error is None
    assert summary.written == 3
    assert summary.dropped == 0
    jpgs = sorted((summary.directory / "frames").glob("*.jpg"))
    assert [p.name for p in jpgs] == ["000000.jpg", "000001.jpg", "000002.jpg"]
    saved = cv2.imread(str(jpgs[0]))
    # The accent box is inside the rectangle; JPEG keeps it far from the grey background.
    pixel = saved[20, 20].astype(int)
    assert np.abs(pixel - np.array(ACCENT_BGR)).max() < 40
    assert np.abs(pixel - np.array([30, 30, 30])).max() > 40

    video = cv2.VideoCapture(str(summary.directory / "video.mkv"))
    n = 0
    while True:
        ok, _ = video.read()
        if not ok:
            break
        n += 1
    video.release()
    assert n == 3


def test_a_full_queue_drops_the_frame_and_returns(tmp_path):
    cap = SessionCapture(tmp_path, queue_frames=1)  # writer not started: the queue stays full
    img = np.full((60, 80, 3), 30, np.uint8)
    cap.offer(img, 1, None, None)
    t0 = time.perf_counter()
    cap.offer(img, 2, None, None)
    assert time.perf_counter() - t0 < 0.2
    assert cap.dropped == 1
    cap.close()


def test_the_same_frame_is_saved_once(tmp_path):
    cap = SessionCapture(tmp_path).start()
    img = np.full((60, 80, 3), 30, np.uint8)
    cap.offer(img, 7, None, None)
    cap.offer(img, 7, None, None)
    summary = cap.close()
    assert summary.written == 1
    assert len(list((summary.directory / "frames").glob("*.jpg"))) == 1
