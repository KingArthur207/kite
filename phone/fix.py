"""The phone's wire format, shared by the Bluetooth and the USB transports.

One fix is one compact JSON object, terminated by a newline on Bluetooth (writes are chunked to fit the ATT MTU):

    {"t":1761330000123,"lat":37.778352,"lng":-122.408768,"acc":4.8,"alt":12.3,"speed":1.25,
     "course":271.4,"heading":268.9,"heading_acc":10}

    t            phone clock, ms since 1970, of the GPS fix (null while there is none)
    lat, lng     degrees (null while there is no fix: the message is then a heartbeat with a heading)
    acc          horizontal accuracy, m          alt   altitude, m
    speed        m/s from GPS                    course  direction of travel from GPS, deg (null when standing)
    heading      compass, deg clockwise from north, the way the phone's back camera faces
    heading_acc  compass accuracy, +/- deg (iOS webkitCompassAccuracy; null if unknown)

Every number goes through the earpiece's own validators (earpiece/earpiece/events.py), and the `position` built here
is checked with events.parse() before it is sent, so the earpiece never gets something it would reject.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field

from . import _earpiece_on_path  # noqa: F401  (the earpiece package, from the repo root)

from earpiece.events import BadMessage, _coord, _num, parse  # noqa: E402

MAX_LINE = 1024              # one fix is ~150 bytes; anything longer is garbage
MAX_BUFFER = 4096            # Bluetooth reassembly buffer
HEADING_ACC_MAX = 45.0       # a compass worse than +/-45 deg is not used for the walker's heading
COURSE_MIN_SPEED = 0.6       # m/s: below this the GPS course is noise (walking is ~1.2 m/s)
CARDINALS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


@dataclass
class Fix:
    t: float | None = None           # phone ms epoch of the GPS fix
    lat: float | None = None
    lng: float | None = None
    acc: float | None = None
    alt: float | None = None
    speed: float | None = None
    course: float | None = None
    heading: float | None = None
    heading_acc: float | None = None
    received: float = field(default_factory=time.monotonic)   # laptop clock (monotonic) when it arrived

    @property
    def has_position(self) -> bool:
        return self.lat is not None and self.lng is not None

    def walker_heading(self) -> float | None:
        """The heading to give the earpiece: the compass (the way the chest faces) when it is trustworthy, else
        the GPS course while walking, else None (route.py then uses the track, and sends no beacon rather than a
        wrong one)."""
        if self.heading is not None and (self.heading_acc is None or self.heading_acc <= HEADING_ACC_MAX):
            return self.heading
        if self.course is not None and self.speed is not None and self.speed >= COURSE_MIN_SPEED:
            return self.course
        return None


def _angle(d: dict, name: str) -> float | None:
    v = _num(d, name)
    return None if v is None else v % 360.0


def parse_fix(raw: bytes | str | dict, now: float | None = None) -> Fix:
    """One message from the phone -> Fix. Raises BadMessage (and only that) for anything malformed."""
    if isinstance(raw, (bytes, bytearray)):
        if len(raw) > MAX_LINE:
            raise BadMessage("fix too long")
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError as e:
            raise BadMessage("not utf-8") from e
    if isinstance(raw, str):
        if len(raw) > MAX_LINE:
            raise BadMessage("fix too long")
        try:
            raw = json.loads(raw)
        except (ValueError, RecursionError) as e:
            raise BadMessage("bad json") from e
    if not isinstance(raw, dict):
        raise BadMessage("fix must be a JSON object")
    d = raw
    lat, lng = _coord(d, "lat", 90.0), _coord(d, "lng", 180.0)
    if (lat is None) != (lng is None):
        lat = lng = None                     # half a coordinate is no coordinate
    acc = _num(d, "acc", lo=0.0, hi=1e5)
    heading_acc = _num(d, "heading_acc", hi=360.0)
    if heading_acc is not None and heading_acc < 0:
        heading_acc = None                   # iOS reports -1 for "unknown"
    return Fix(t=_num(d, "t", lo=0.0), lat=lat, lng=lng, acc=acc, alt=_num(d, "alt", lo=-1e4, hi=1e5),
               speed=_num(d, "speed", lo=0.0, hi=100.0), course=_angle(d, "course"), heading=_angle(d, "heading"),
               heading_acc=heading_acc, received=time.monotonic() if now is None else now)


def to_position(fix: Fix, source: str = "phone") -> dict | None:
    """Fix -> an earpiece `position` event (validated by events.parse), or None when the fix has no coordinates."""
    if not fix.has_position:
        return None
    msg = {"type": "position", "source": source, "lat": round(fix.lat, 7), "lng": round(fix.lng, 7)}
    h = fix.walker_heading()
    if h is not None:
        msg["heading"] = round(h, 1)
    if fix.acc is not None:
        msg["accuracy"] = round(fix.acc, 1)
    parse(msg)                               # raises BadMessage if the earpiece would refuse it
    return msg


def cardinal(deg: float | None) -> str:
    if deg is None or not math.isfinite(deg):
        return "-"
    return CARDINALS[int(((deg % 360.0) + 22.5) // 45.0) % 8]


def _age(s: float | None) -> str:
    if s is None:
        return "never"
    if s < 10:
        return f"{s:.1f} s ago"
    if s < 120:
        return f"{s:.0f} s ago"
    return f"{s / 60:.0f} min ago"


def format_line(fix: Fix | None, fix_age: float | None, link_age: float | None, link_ok: bool,
                sent: int = 0, extra: str = "") -> str:
    """The live terminal line: position, accuracy, compass with a cardinal (only if the phone sends one), speed,
    fix age, link state."""
    if fix is None:
        return f"waiting for the phone ...  {extra}".rstrip()
    if fix.has_position:
        pos = f"{fix.lat:.6f}, {fix.lng:.6f}"
        pos += f"  +/-{fix.acc:.0f} m" if fix.acc is not None else ""
    else:
        pos = "no GPS fix yet"
    hd = None                                # the app sends GPS only for now: no heading, no column
    if fix.heading is not None:
        hd = f"heading {fix.heading:5.1f} {cardinal(fix.heading):<2}"
        hd += f" +/-{fix.heading_acc:.0f}" if fix.heading_acc is not None else ""
    spd = f"{fix.speed:.1f} m/s" if fix.speed is not None else "- m/s"
    if fix.course is not None and fix.speed is not None and fix.speed >= COURSE_MIN_SPEED:
        spd += f" toward {cardinal(fix.course)}"
    link = "phone ok" if link_ok else f"PHONE LOST ({_age(link_age)})"
    parts = [p for p in (pos, hd, spd, f"fix {_age(fix_age)}", link, f"sent {sent}") if p]
    if extra:
        parts.append(extra)
    return "  |  ".join(parts)


class LineAssembler:
    """Reassembles newline-framed messages from Bluetooth writes of any size (a write may hold part of a message,
    one message, or several). Thread-safe enough for one writer: CoreBluetooth calls it from its own queue."""

    def __init__(self, max_buffer: int = MAX_BUFFER):
        self.max_buffer = max_buffer
        self._buf = bytearray()
        self.overflows = 0

    def feed(self, chunk: bytes) -> list[bytes]:
        self._buf.extend(chunk)
        out: list[bytes] = []
        while True:
            i = self._buf.find(b"\n")
            if i < 0:
                break
            line = bytes(self._buf[:i]).strip()
            del self._buf[: i + 1]
            if line:
                out.append(line)
        if len(self._buf) > self.max_buffer:   # no newline in 4 KB: a lost terminator; start over
            self._buf.clear()
            self.overflows += 1
        return out
