"""Frame sources: the Tello's UDP video, a webcam (the walker's body camera), video files, kite.record takes.

open_source(spec) opens any of them (and the sim) with one shape for perception: see kite/sources/open.py."""

from .base import FrameSource
from .opencv_sources import (
    TELLO_UDP_URL,
    StaticImageSource,
    SteppedVideoSource,
    TelloVideoSource,
    VideoFileSource,
    WebcamSource,
)
from .pose import POSE_KEYS, empty_pose
from .recording import RecordingFrameSource, RecordingSource

__all__ = [
    "FrameSource", "TelloVideoSource", "WebcamSource", "VideoFileSource", "SteppedVideoSource",
    "StaticImageSource", "TELLO_UDP_URL",
    "RecordingSource", "RecordingFrameSource",
    "open_source", "parse_spec", "POSE_KEYS", "empty_pose",
]


def __getattr__(name: str):
    # open_source / parse_spec load kite.sources.open on first use, so `python -m kite.sources.open` runs it fresh
    if name in ("open_source", "parse_spec"):
        from . import open as _open

        return getattr(_open, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
