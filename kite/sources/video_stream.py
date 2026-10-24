"""Low-latency OpenCV/FFmpeg reader for the Tello's UDP video (the team's `video_stream.py`).

Kept as written by the team except (marked CHANGED):
  * one comment correction: with the FFmpeg backend grab() does decode each H.264 frame (it has to,
    frames depend on each other); retrieve() only converts the last decoded frame to BGR.
  * the capture is opened with open/read timeouts, and stop() never releases the capture while the
    reader thread may still be inside grab(): releasing under a blocked grab() on a stalled stream
    segfaults the process (found in review; reproduced by killing the UDP sender).
  * the unused copying read() is gone: read_seq() is the only accessor.
  * the open timeout always outlasts FFmpeg's probe (analyzeduration). The probe has to read up to a keyframe (SPS/PPS
    + IDR: nothing decodes before one) and was allowed 12 s / 10 MB, but drone_test opened with a 5 s timeout, so
    the open was killed mid-probe ("Stream timeout triggered after 5015 ms", after "non-existing PPS 0 referenced"
    from the packets before the first keyframe). Kept the large probe (a small one can return without the frame
    size when the first keyframe is late) and raised the open timeout instead: at least analyzeduration + 3 s
    (15 s), whatever the caller asks for.
"""

import os
import re
import threading
import time
from contextlib import contextmanager

import cv2

# Tells ffmpeg's demuxer not to build up its own internal buffer and to drop frames it can't keep up
# with, rather than queueing them -- this is what actually stops latency from growing the longer the
# stream runs. CHANGED: applied only while this reader opens its capture (OpenCV reads the variable at
# open time), never process-wide: set globally, "nobuffer" made every later cv2 read of a video FILE
# silently lose its first frame. A value the user exported themselves still wins.
STREAM_OPTIONS = "fflags;nobuffer|flags;low_delay|framedrop;1|probesize;10000000|analyzeduration;12000000"
_ENV = "OPENCV_FFMPEG_CAPTURE_OPTIONS"
_env_lock = threading.Lock()


def stream_options() -> str:
    return os.environ.get(_ENV, STREAM_OPTIONS)


@contextmanager
def _stream_capture_options():
    with _env_lock:
        user = _ENV in os.environ
        if not user:
            os.environ[_ENV] = STREAM_OPTIONS
        try:
            yield
        finally:
            if not user:
                os.environ.pop(_ENV, None)


PROBE_MARGIN_MS = 3000  # after analyzeduration, time to open the decoder and hand back the first frame


def min_open_timeout_ms() -> int:
    """Open timeout that lets FFmpeg's probe finish: the analyzeduration actually in effect + PROBE_MARGIN_MS."""
    m = re.search(r"analyzeduration;(\d+)", stream_options())
    analyze_us = int(m.group(1)) if m else 5_000_000  # FFmpeg's default analyzeduration is 5 s
    return analyze_us // 1000 + PROBE_MARGIN_MS


class VideoStream:
    """
    Reads the Tello's UDP video feed directly via OpenCV's FFmpeg backend,
    instead of djitellopy's PyAV-based reader.

    A background thread continuously grab()s frames to keep the stream drained
    in real time. It only pays the cost of retrieve() (colour conversion) once
    per target interval, so consumers see at most `fps` new frames per second
    regardless of how fast frames arrive.
    """

    def __init__(self, url, fps=30, open_timeout_ms=15000, read_timeout_ms=3000):
        # CHANGED: bounded open/read so a stalled stream cannot block grab() forever, and an open timeout that never
        # cuts FFmpeg's probe short (it reads up to the first keyframe, up to analyzeduration)
        open_timeout_ms = max(open_timeout_ms, min_open_timeout_ms())
        with _stream_capture_options():
            self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, open_timeout_ms,
                                                              cv2.CAP_PROP_READ_TIMEOUT_MSEC, read_timeout_ms])
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video stream at {url}")

        self._read_timeout_s = read_timeout_ms / 1000.0
        self._interval = 1 / fps
        self._frame = None
        self._seq = 0      # bumped per decoded frame so consumers can tell "new" from "same"
        self._ts = 0.0     # time.time() at retrieve(), i.e. when the frame became available
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._update, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def _update(self):
        last_retrieve = 0.0
        while not self._stopped.is_set():
            if not self.cap.grab():
                time.sleep(0.005)
                continue

            now = time.time()
            if now - last_retrieve >= self._interval:
                ok, frame = self.cap.retrieve()
                if ok:
                    with self._lock:
                        self._frame = frame
                        self._seq += 1
                        self._ts = time.time()
                last_retrieve = now

    def read_seq(self, copy=True):
        """Returns (seq, frame, timestamp). seq only changes when a new frame was decoded.

        copy=False hands back the internal array to skip a ~2.7 MB memcpy per frame; the
        caller must then treat it as read-only. _update never mutates a frame in place, it
        rebinds self._frame to whatever retrieve() allocated, so a stale reference stays valid.
        """
        with self._lock:
            if self._frame is None:
                return -1, None, 0.0
            return self._seq, (self._frame.copy() if copy else self._frame), self._ts

    def stop(self):
        self._stopped.set()
        if self._thread.is_alive():
            self._thread.join(timeout=self._read_timeout_s + 0.5)  # a blocked grab() returns within the read timeout
        # CHANGED: only release once the reader thread is out of grab(); otherwise leak it (daemon)
        if not self._thread.is_alive():
            self.cap.release()
