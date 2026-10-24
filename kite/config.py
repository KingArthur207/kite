"""Configuration: nested dataclasses with defaults (ported from vigil's reachglass/config.py: drone + safety).

    cfg = Config()
    cfg = Config(drone=DroneCfg(kind="dry_run"), safety=SafetyCfg(max_rc=70))

Each module reads only its own section.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

VIDEO_BACKENDS = ("auto", "raw", "socket", "pyav", "opencv")
DEFAULT_VIDEO_BACKEND = "pyav"  # kite's own UDP readers (raw, socket, auto) are opt-in until they run on a real Tello
# Tello SDK 2.0+ video link settings (kite.drone.tello.apply_video_link): None sends nothing, the drone keeps its own
VIDEO_BITRATES = ("auto", "1", "2", "3", "4", "5")  # setbitrate 0..5, Mbps (0 = auto)
VIDEO_RESOLUTIONS = ("high", "low")  # setresolution: 720p / 480p
VIDEO_STREAM_FPS = ("high", "middle", "low")  # setfps


def resolve_video_backend(backend: str | None = None, platform: str | None = None) -> str:
    """The Tello H.264 reader to use. None: DEFAULT_VIDEO_BACKEND, "pyav" (FFmpeg's udp:// + h264 demuxer,
    djitellopy's way). "opencv" is the team's VideoStream. Opt-in until they have run against a real Tello: "raw"
    (kite/sources/tello_raw.py: kite's own UDP socket + PyAV's bare H.264 decoder), "socket"
    (kite/sources/socket_stream.py: kite's UDP socket + PyAV's parser, big receive buffer) and "auto", the
    per-platform profile: "raw" on macOS (on a Mac both FFmpeg udp:// readers got the packets but never a frame),
    "pyav" elsewhere."""
    b = (backend or DEFAULT_VIDEO_BACKEND).lower()
    if b not in VIDEO_BACKENDS:
        raise ValueError(f"video backend {backend!r}: use one of {', '.join(VIDEO_BACKENDS)}")
    if b == "auto":
        return "raw" if (platform or sys.platform) == "darwin" else "pyav"
    return b


def add_video_link_args(p) -> None:
    """--video-bitrate / --video-resolution / --video-fps on an argparse parser (default: send nothing)."""
    p.add_argument("--video-bitrate", choices=VIDEO_BITRATES, default=None,
                   help="Tello stream bitrate in Mbps (setbitrate; auto = the drone picks). Lower survives a weak link")
    p.add_argument("--video-resolution", choices=VIDEO_RESOLUTIONS, default=None,
                   help="Tello stream resolution (setresolution: high 720p, low 480p)")
    p.add_argument("--video-fps", choices=VIDEO_STREAM_FPS, default=None, help="Tello stream frame rate (setfps)")


def video_link_args(drone: DroneCfg, args) -> None:
    """Copy the --video-bitrate / --video-resolution / --video-fps flags that were given into DroneCfg."""
    for flag, field_name in (("video_bitrate", "video_bitrate"), ("video_resolution", "video_resolution"),
                             ("video_fps", "video_stream_fps")):
        if getattr(args, flag, None) is not None:
            setattr(drone, field_name, getattr(args, flag))


def video_link_argv(args) -> str:
    """The same flags as a command-line suffix (' --video-bitrate 2 ...'), to hand on to the next command."""
    return "".join(f" --{f.replace('_', '-')} {getattr(args, f)}" for f in ("video_bitrate", "video_resolution",
                                                                            "video_fps") if getattr(args, f, None))


@dataclass
class SafetyCfg:
    max_rc: int = 50  # clamp on every rc channel (stick 50 ~ 0.5 m/s forward: raise it to keep up with a walk)
    min_altitude_m: float = 0.5
    max_altitude_m: float = 2.3
    min_battery_pct: float = 20.0
    max_flight_s: float = 420.0
    frame_timeout_s: float = 1.0  # hover if no new frame for this long
    telemetry_timeout_s: float = 1.0
    # escalation: a hover on stale data does not last for ever (the rc 0 keep-alive stops the Tello's own 15 s
    # auto-land, so without these it would hover blind until max_flight_s)
    telemetry_land_s: float = 5.0  # no state packet this long -> land (battery, height and temperature unknown)
    video_land_s: float = 15.0  # no frame this long -> land
    max_temp_c: float = 90.0  # 'temph' at or above this -> land (idle Tellos report 90-96 C before shutting down)
    land_retry_s: float = 2.0  # a land asked for that has not happened is re-sent this often
    # geofence: horizontal distance from the takeoff point, dead-reckoned from the state stream's vgx / vgy, above
    # which the governor lands. The magnitude of the integrated velocity does not depend on the (unverified) axis
    # signs; readings within geofence_db_m_s of 0 are ignored so hover noise does not add up. The Moss St route is
    # ~75 m from its start at most and the Tello's rated range is 100 m. None: off.
    geofence_m: float | None = 100.0
    geofence_db_m_s: float = 0.11


@dataclass
class DroneCfg:
    kind: str = "tello"  # tello | dry_run
    move_speed_cm_s: int = 50  # speed for discrete moves ('speed' command)
    yaw_sign: int = 1  # +1 if telemetry yaw grows with `cw`
    video_fps: int = 60  # reader retrieve cap: must sit above the ~30 fps stream (equal drops ~40 % of frames)
    # Tello H.264 reader (kite.config.resolve_video_backend): pyav (FFmpeg udp:// + h264 demuxer, djitellopy's
    # decoder) or opencv (the team's VideoStream); kite's own UDP readers are opt-in until they have run against a
    # real Tello: raw (tello_raw.py, bare H.264 decoder), socket (socket_stream.py, PyAV's parser, big receive
    # buffer) and auto (raw on macOS, where the FFmpeg udp:// readers showed no image, pyav elsewhere).
    video_backend: str = DEFAULT_VIDEO_BACKEND
    # Tello video link, sent after `command` and before `streamon` (None: not sent, the drone keeps its default).
    # Older firmware answers "error": logged once, never fatal. A weak or crowded link: video_bitrate "2" or "1",
    # video_resolution "low" (docs/internal/windows-drone.md, "If the drone video is corrupted").
    video_bitrate: str | None = None  # VIDEO_BITRATES
    video_resolution: str | None = None  # VIDEO_RESOLUTIONS
    video_stream_fps: str | None = None  # VIDEO_STREAM_FPS (video_fps above is the laptop-side reader cap)
    # camera-to-laptop video delay (s). After a discrete move/rotation, only frames arriving later than
    # "command done + video_lag_s" show the new view. Measure it with tests/drone_test.py.
    # vigil 2026-09-26 run: video 6 +/- 78 ms (yaw) and 21 +/- 40 ms (forward move) behind telemetry, worst 85 ms
    video_lag_s: float = 0.25


@dataclass
class Config:
    safety: SafetyCfg = field(default_factory=SafetyCfg)
    drone: DroneCfg = field(default_factory=DroneCfg)
