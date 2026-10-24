"""Configuration: nested dataclasses with defaults. The drone and safety sections are jerkgt13's reachglass/config.py
(github.com/nathanwuzhao/jerkgt13); kite additions are marked "kite:".

    cfg = Config()
    cfg = Config(drone=DroneCfg(kind="dry_run"), safety=SafetyCfg(max_rc=70))

Each module reads only its own section.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

# kite: the Tello H.264 reader is a setting (every drone command's --video-backend choices come from here).
#   pyav    FFmpeg's udp:// + h264 demuxer, djitellopy's decoder (jerkgt13's default; works on Windows)
#   opencv  the team's OpenCV VideoStream (tests/drone_test.py's original reader)
#   raw     kite's own UDP socket + PyAV's bare H.264 decoder (kite/sources/tello_raw.py)
#   auto    raw on macOS, pyav elsewhere
VIDEO_BACKENDS = ("auto", "raw", "pyav", "opencv")
DEFAULT_VIDEO_BACKEND = "pyav"


def resolve_video_backend(backend: str | None = None, platform: str | None = None) -> str:
    """The reader that will actually run: None -> DEFAULT_VIDEO_BACKEND, "auto" -> this platform's. ValueError if
    unknown."""
    b = (backend or DEFAULT_VIDEO_BACKEND).lower()
    if b not in VIDEO_BACKENDS:
        raise ValueError(f"video backend {backend!r}: use one of {', '.join(VIDEO_BACKENDS)}")
    if b == "auto":
        return "raw" if (platform or sys.platform) == "darwin" else "pyav"
    return b


@dataclass
class SafetyCfg:
    max_rc: int = 50  # clamp on every rc channel
    min_altitude_m: float = 0.5
    max_altitude_m: float = 2.3
    min_battery_pct: float = 20.0
    max_flight_s: float = 420.0
    frame_timeout_s: float = 1.0  # hover if no new frame for this long
    telemetry_timeout_s: float = 1.0


@dataclass
class DroneCfg:
    kind: str = "tello"  # tello | dry_run
    host: str = "192.168.10.1"  # kite: the drone's address (the tests point it at kite.testing's emulator)
    move_speed_cm_s: int = 50  # speed for discrete moves ('speed' command)
    yaw_sign: int = 1  # +1 if telemetry yaw grows with `cw`
    video_fps: int = 60  # reader retrieve cap: must sit above the ~30 fps stream (equal drops ~40 % of frames)
    video_backend: str = DEFAULT_VIDEO_BACKEND  # kite: see VIDEO_BACKENDS above
    # camera-to-laptop video delay (s). After a discrete move/rotation, only frames arriving later than
    # "command done + video_lag_s" show the new view. Measure it with tests/drone_test.py.
    video_lag_s: float = 0.25


@dataclass
class Config:
    safety: SafetyCfg = field(default_factory=SafetyCfg)
    drone: DroneCfg = field(default_factory=DroneCfg)
