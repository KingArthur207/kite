"""Frame sources: the Tello's UDP video, a webcam (the walker's body camera), video files, kite.record takes."""

from .base import FrameSource
from .opencv_sources import (
    TELLO_UDP_URL,
    StaticImageSource,
    SteppedVideoSource,
    TelloVideoSource,
    VideoFileSource,
    WebcamSource,
)
from .recording import RecordingFrameSource, RecordingSource

__all__ = [
    "FrameSource", "TelloVideoSource", "WebcamSource", "VideoFileSource", "SteppedVideoSource",
    "StaticImageSource", "TELLO_UDP_URL",
    "RecordingSource", "RecordingFrameSource",
]
