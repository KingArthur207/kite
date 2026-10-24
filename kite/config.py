"""Configuration: nested dataclasses with defaults (ported from vigil's reachglass/config.py: drone + safety).

    cfg = Config()
    cfg = Config(drone=DroneCfg(kind="dry_run"), safety=SafetyCfg(max_rc=70))

Each module reads only its own section.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

VIDEO_BACKENDS = ("auto", "raw", "pyav", "opencv")


def resolve_video_backend(backend: str | None = "auto", platform: str | None = None) -> str:
    """The Tello H.264 reader to use. "auto" is the per-platform profile: "raw" on macOS (kite/sources/tello_raw.py:
    kite's own UDP socket + PyAV's bare H.264 decoder; on a Mac both FFmpeg udp:// readers got the packets but never
    a frame), "pyav" elsewhere (unchanged). raw / pyav / opencv force one reader on any platform."""
    b = (backend or "auto").lower()
    if b not in VIDEO_BACKENDS:
        raise ValueError(f"video backend {backend!r}: use one of {', '.join(VIDEO_BACKENDS)}")
    if b == "auto":
        return "raw" if (platform or sys.platform) == "darwin" else "pyav"
    return b


@dataclass
class SafetyCfg:
    max_rc: int = 50  # clamp on every rc channel (stick 50 ~ 0.5 m/s forward: raise it to keep up with a walk)
    min_altitude_m: float = 0.5
    max_altitude_m: float = 2.3
    min_battery_pct: float = 20.0
    max_flight_s: float = 420.0
    frame_timeout_s: float = 1.0  # hover if no new frame for this long
    telemetry_timeout_s: float = 1.0


@dataclass
class DroneCfg:
    kind: str = "tello"  # tello | dry_run
    move_speed_cm_s: int = 50  # speed for discrete moves ('speed' command)
    yaw_sign: int = 1  # +1 if telemetry yaw grows with `cw`
    video_fps: int = 60  # reader retrieve cap: must sit above the ~30 fps stream (equal drops ~40 % of frames)
    # Tello H.264 reader: auto (raw on macOS, pyav elsewhere), raw (kite's UDP socket + bare decoder, tello_raw.py),
    # pyav (FFmpeg udp:// + h264 demuxer, djitellopy's decoder) or opencv (the team's VideoStream). See resolve_video_backend.
    video_backend: str = "auto"
    # camera-to-laptop video delay (s). After a discrete move/rotation, only frames arriving later than
    # "command done + video_lag_s" show the new view. Measure it with tests/drone_test.py.
    # vigil 2026-09-26 run: video 6 +/- 78 ms (yaw) and 21 +/- 40 ms (forward move) behind telemetry, worst 85 ms
    video_lag_s: float = 0.25


@dataclass
class Config:
    safety: SafetyCfg = field(default_factory=SafetyCfg)
    drone: DroneCfg = field(default_factory=DroneCfg)
