"""Shared data types (ported from vigil's reachglass/types.py: the drone and video parts).

Conventions used by every module
--------------------------------
Heading in degrees, CLOCKWISE positive (like a compass and like the Tello's `cw` command).
Times: time.time() seconds. Distances: metres. Angles: degrees unless a name ends in _rad.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def wrap_deg(a: float) -> float:
    """Wrap an angle to (-180, 180]."""
    a = (a + 180.0) % 360.0 - 180.0
    return 180.0 if a == -180.0 else a


@dataclass
class Frame:
    image: np.ndarray  # H x W x 3, uint8, BGR
    t: float  # time the frame became available on the laptop
    seq: int  # increases by one for every new frame from the source
    source: str = ""

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])


@dataclass
class Telemetry:
    """One state packet from the drone, already converted to SI units."""

    t: float
    yaw_deg: float | None = None  # IMU yaw as reported by the drone
    pitch_deg: float | None = None
    roll_deg: float | None = None
    height_m: float | None = None  # 'h': height above the takeoff point
    tof_m: float | None = None  # 'tof': downward range to whatever is below
    vx: float | None = None  # state-stream speeds, m/s (frame as reported by the drone; vigil's lag log read forward as NEGATIVE vgx)
    vy: float | None = None
    vz: float | None = None
    battery_pct: float | None = None
    flight_time_s: float | None = None

    def altitude_m(self, prefer_tof: bool = True, tof_max_m: float = 3.0) -> float | None:
        """Clearance above whatever is BELOW (downward ToF when in range, else 'h'). Over a table this is
        the height above the table: right for descending, wrong for 'how high above the floor am I'."""
        if prefer_tof and self.tof_m is not None and 0.1 <= self.tof_m <= tof_max_m:
            return self.tof_m
        return self.height_m

    def floor_altitude_m(self, agree_m: float = 0.2) -> float | None:
        """Height above the FLOOR: the ToF only when it agrees with 'h' (i.e. there is floor below),
        otherwise 'h' (height above the takeoff point, which was on the floor)."""
        if self.height_m is None:
            return self.altitude_m()
        if self.tof_m is not None and abs(self.tof_m - self.height_m) <= agree_m:
            return self.tof_m
        return self.height_m
