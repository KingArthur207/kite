"""The camera pose every frame source reports (perception's `pose`), and Tello telemetry -> pose.

One dict shape for the sim, a kite.record take, the live Tello, a webcam and a video file:
    x, y, heading, cam_height, pitch, roll, t, battery, flying, tel_dt, heading_ref
None where the source does not know a value. Meanings: kite/sources/recording.py's docstring.

TelemetryTrack turns converted state packets (Telemetry, from TelloDrone.telemetry()) into those fields at any time t:
interpolated between the packets just before and after t when both are within max_gap_s, else the nearest packet
within max_gap_s, else nothing. RecordingSource uses it for every recorded frame, the live Tello source for the newest
frame, so both report the same numbers for the same packets.
"""

from __future__ import annotations

import math

import numpy as np

from ..types import Telemetry

POSE_KEYS = ("x", "y", "heading", "cam_height", "pitch", "roll", "t", "battery", "flying", "tel_dt", "heading_ref")
MAX_GAP_S = 0.2  # a frame further than this from every state packet has no pose (the Tello sends ~10 per second)
PITCH_SIGN = 1  # raw Tello 'pitch' -> pose pitch (+ = nose up). ASSUMED: +1 if the Tello reports nose-up as positive


def empty_pose(t: float | None = None, **known) -> dict:
    """A pose with every key, None except t and the given ones."""
    pose = dict.fromkeys(POSE_KEYS)
    pose["t"] = t
    pose.update(known)
    return pose


def _f(v) -> float:
    return math.nan if v is None else float(v)


class TelemetryTrack:
    """Telemetry packets sorted by .t, queried for the pose fields at a time on the same clock."""

    def __init__(self, tel: list[Telemetry], pitch_sign: int = PITCH_SIGN):
        self.tel = tel
        self.tt = np.array([t.t for t in tel], dtype=float)
        self.yaw = np.array([_f(t.yaw_deg) for t in tel])
        self.cols = {
            "cam_height": np.array([_f(t.floor_altitude_m()) for t in tel]),
            "pitch": np.array([_f(t.pitch_deg) * pitch_sign if t.pitch_deg is not None else math.nan for t in tel]),
            "roll": np.array([_f(t.roll_deg) for t in tel]),
        }
        self.bat = [t.battery_pct for t in tel]

    def __len__(self) -> int:
        return len(self.tel)

    def fill(self, pose: dict, t: float, max_gap_s: float = MAX_GAP_S) -> int | None:
        """Set heading, cam_height, pitch, roll, battery and tel_dt in `pose` for time t. Returns the index of the
        nearest packet used, or None (pose untouched) when no packet is within max_gap_s."""
        tt = self.tt
        if not len(tt):
            return None
        j = int(np.searchsorted(tt, t))  # tt[j-1] <= t < tt[j]
        lo, hi = j - 1, j
        d_lo = t - tt[lo] if lo >= 0 else math.inf
        d_hi = tt[hi] - t if hi < len(tt) else math.inf
        near = lo if d_lo <= d_hi else hi
        if min(d_lo, d_hi) > max_gap_s:
            return None
        both = d_lo <= max_gap_s and d_hi <= max_gap_s and tt[hi] > tt[lo]
        w = (t - tt[lo]) / (tt[hi] - tt[lo]) if both else None
        for k, arr in self.cols.items():
            if both and math.isfinite(arr[lo]) and math.isfinite(arr[hi]):
                v = arr[lo] + w * (arr[hi] - arr[lo])
            else:
                v = arr[near]
            pose[k] = float(v) if math.isfinite(v) else None
        yaw = self.yaw
        if both and math.isfinite(yaw[lo]) and math.isfinite(yaw[hi]):  # shortest way round +-180
            y = yaw[lo] + w * (((yaw[hi] - yaw[lo] + 180.0) % 360.0) - 180.0)
        else:
            y = yaw[near]
        pose["heading"] = float(y % 360.0) if math.isfinite(y) else None
        pose["battery"] = self.bat[near]
        pose["tel_dt"] = float(t - tt[near])
        return near
