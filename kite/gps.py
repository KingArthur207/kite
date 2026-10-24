"""The walker's phone GPS log from the shoot -> earpiece `position` events, per-frame walker pose and sim metres.

    python -m kite.gps walk.gpx --recording recordings/20261024_070312_tello     # align with the Tello take
    python -m kite.gps walk.csv --recording <dir> --offset-s 2.5 --tz America/Los_Angeles
    python -m kite.gps walk.gpx                                                 # no video: the whole track
    python -m kite.gps --route --out data                                       # the planned route (spec section 8)

Input: a GPX file (trkpt or rtept with lat, lon, time; ele, speed, course, hdop and an accuracy field in any
namespace or <extensions> are picked up) or a CSV with a header naming time, lat and lon columns (latitude /
longitude / lng also work, accuracy optional; ',', ';' or tab separated). Times are ISO 8601 with Z or an offset,
or epoch seconds / ms / us / ns. Times with no zone are taken in --tz (default UTC, which is what GPX requires;
many CSV loggers write local time instead).

Clocks: GPS times are true UTC. The recording's frames.csv t_wall is the laptop's time.time(), and the laptop sits
on the Tello Wi-Fi without internet, so it can be off by seconds. A frame's GPS time is t_wall + --offset-s
(offset = phone clock - laptop clock). The overlap of the two spans is printed so a mismatch is obvious; a
whole-hour gap means a time-zone mistake. To measure the offset, film the phone showing a seconds clock with the
Tello at the start of the take: offset = clock on screen - that frame's t_wall.

Track: points are projected to sim metres, one outlier pass drops fixes far from the smoothed track, positions are
smoothed with a Gaussian-weighted local linear fit (--smooth-s) at the fix times and linearly interpolated between
fixes (gaps longer than --max-gap-s are left empty). Speed and heading (course over ground, compass degrees,
0 = north, 90 = east) come from the displacement over --heading-window-s. Heading is None while the walker stands
still (speed over a 4 s window below --stop-speed); heading_hold keeps the last walking heading, which is the best
guess for which way a stopped walker faces.

Outputs in --out (default <recording>/gps, or <track>_gps next to the track):
    positions.jsonl  `position` events (earpiece/earpiece/events.py) at --rate Hz with "t" = seconds from the first
                     video frame, so scripts/replay.py plays them in step with the video. Extra fields like the sim's:
                     x, y (sim metres), speed, moving, frame. heading is heading_hold (absent before the first step).
                     A heartbeat at t=0 keeps replay's clock on the video's when the GPS starts later.
    frames_gps.csv   one row per video frame (index = frame number in video.mkv): walker lat, lng, x, y, heading,
                     heading_hold, speed, moving, gps_gap_s. Empty where the GPS doesn't cover the frame.
    track.csv        every kept fix: UTC time, raw and smoothed lat/lng, sim x/y, speed, heading, accuracy.
    summary.json     duration, distance, mean speed, overlap with the video, closest approach to the crossing node,
                     distance from the planned route.

Sim metres: x east, y north of the Moss St zebra crossing node, on the same equirectangular tangent plane as
sim/src/config.ts toLatLng (111320 m per degree of latitude, times cos(origin latitude) for longitude). Under
1 cm of error over the few hundred metres of the shoot. latlng_to_sim() / sim_to_latlng() are the helpers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

CROSSING = (37.778352, -122.408768)  # OSM node of the zebra crossing across Moss St at Howard (spec section 8)
M_PER_DEG_LAT = 111320.0  # same constant as sim/src/config.ts toLatLng
MOSS_BEARING_DEG = 315.0  # APPROXIMATE: Moss St towards Howard, NW (the SoMa grid is ~45 deg off north)
STOP_WINDOW_S = 4.0  # speed window for the standing-still test (longer than the heading window: steadier)
GRID_HZ = 10.0  # internal sampling of the smoothed track
PLACE = "Moss St at Howard St"


# ------------------------------------------------------------------ sim projection
def latlng_to_sim(lat, lng, origin: tuple[float, float] = CROSSING):
    """lat/lng (degrees; floats or numpy arrays) -> (x metres east, y metres north) of origin."""
    x = (np.asarray(lng, dtype=float) - origin[1]) * M_PER_DEG_LAT * math.cos(math.radians(origin[0]))
    y = (np.asarray(lat, dtype=float) - origin[0]) * M_PER_DEG_LAT
    return (float(x), float(y)) if np.ndim(x) == 0 else (x, y)


def sim_to_latlng(x, y, origin: tuple[float, float] = CROSSING):
    """(x metres east, y metres north) of origin -> (lat, lng). Inverse of latlng_to_sim."""
    lat = origin[0] + np.asarray(y, dtype=float) / M_PER_DEG_LAT
    lng = origin[1] + np.asarray(x, dtype=float) / (M_PER_DEG_LAT * math.cos(math.radians(origin[0])))
    return (float(lat), float(lng)) if np.ndim(lat) == 0 else (lat, lng)


def compass(dx, dy):
    """Displacement (east, north) -> compass heading in [0, 360)."""
    return np.degrees(np.arctan2(dx, dy)) % 360.0


# ------------------------------------------------------------------ reading tracks
@dataclass
class Fix:
    t: float  # UTC epoch seconds
    lat: float
    lng: float
    ele: float | None = None
    accuracy: float | None = None  # metres (horizontal)
    speed: float | None = None  # m/s as logged (not used for the outputs: walking-speed Doppler is poor)
    course: float | None = None  # degrees as logged (same)


def parse_tz(name: str) -> tzinfo:
    """'UTC', '-07:00' / '+0530', or an IANA name like 'America/Los_Angeles'."""
    if name.upper() in ("UTC", "Z", "GMT"):
        return timezone.utc
    m = re.fullmatch(r"([+-])(\d{1,2}):?(\d{2})?", name.strip())
    if m:
        delta = timedelta(hours=int(m[2]), minutes=int(m[3] or 0))
        return timezone(-delta if m[1] == "-" else delta)
    return ZoneInfo(name)


def parse_time(s: str, tz: tzinfo = timezone.utc) -> float:
    """ISO 8601 (Z, +hh:mm, +hhmm, or no zone -> tz) or an epoch number in s / ms / us / ns -> UTC epoch seconds."""
    s = s.strip()
    if re.fullmatch(r"\d+(\.\d+)?", s):
        v = float(s)
        for scale in (1e18, 1e15, 1e12):  # ns, us, ms: epoch seconds stay below 1e10 until the year 2286
            if v > scale / 100:
                return v / (scale / 1e9)
        return v
    s = re.sub(r"\s*(UTC|GMT)$", "Z", s)
    s = re.sub(r"^(\d{4})/(\d{2})/(\d{2})", r"\1-\2-\3", s)
    try:
        dt = datetime.fromisoformat(s)
    except ValueError as e:
        raise ValueError(f"unreadable time {s!r}") from e
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    return dt.timestamp()


def _num(v) -> float | None:
    try:
        f = float(str(v).strip())
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


ACCURACY_KEYS = ("accuracy", "horizontalaccuracy", "hacc", "haccuracy", "acc", "horizontal_accuracy")


def read_gpx(path: str | Path, tz: tzinfo = timezone.utc) -> list[Fix]:
    root = ET.parse(path).getroot()
    fixes = []
    for el in root.iter():
        if _local(el.tag) not in ("trkpt", "rtept"):
            continue
        vals: dict[str, str] = {}
        for ch in el.iter():
            if ch is not el and ch.text and ch.text.strip():
                vals.setdefault(re.sub(r"[^a-z_]", "", _local(ch.tag)), ch.text.strip())
        lat, lng = _num(el.get("lat")), _num(el.get("lon"))
        if lat is None or lng is None or "time" not in vals:
            continue
        acc = next((_num(vals[k]) for k in ACCURACY_KEYS if k in vals), None)
        if acc is None and "hdop" in vals and _num(vals["hdop"]) is not None:
            acc = 5.0 * _num(vals["hdop"])  # rough: HDOP x ~5 m user range error
        fixes.append(Fix(parse_time(vals["time"], tz), lat, lng, _num(vals.get("ele")), acc,
                         _num(vals.get("speed")), _num(vals.get("course"))))
    return fixes


def _key(h: str) -> str:
    return re.sub(r"[^a-z]", "", re.sub(r"[(\[].*?[)\]]", "", h.lower()))


CSV_COLS = {
    "time": ("timeutc", "utctime", "datetimeutc", "time", "datetime", "timestamp", "gpstime", "loggingtime",
             "locationtimestamp", "date"),
    "lat": ("lat", "latitude"),
    "lng": ("lon", "lng", "long", "longitude"),
    "accuracy": ("accuracy", "horizontalaccuracy", "hacc", "haccuracy", "acc", "locationhorizontalaccuracy"),
    "ele": ("ele", "elevation", "altitude", "alt"),
    "speed": ("speed",),
    "course": ("course", "bearing"),
}


def read_csv(path: str | Path, tz: tzinfo = timezone.utc) -> list[Fix]:
    text = Path(path).read_text(encoding="utf-8-sig")
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    rows = list(csv.reader(text.splitlines(), dialect))
    if not rows:
        return []
    keys = [_key(h) for h in rows[0]]
    col = {}
    for field, names in CSV_COLS.items():
        col[field] = next((keys.index(n) for n in names if n in keys), None)
    if col["time"] is None or col["lat"] is None or col["lng"] is None:
        raise ValueError(f"{path}: need time, lat and lon columns, got {rows[0]}")
    split_date = keys[col["time"]] == "time" and "date" in keys  # separate date and time-of-day columns
    fixes = []
    for r in rows[1:]:
        if len(r) < len(keys):
            continue
        get = lambda f: r[col[f]] if col[f] is not None else None  # noqa: E731
        lat, lng, ts = _num(get("lat")), _num(get("lng")), get("time")
        if lat is None or lng is None or not ts or not ts.strip():
            continue
        if split_date and not re.search(r"\d{4}-\d{2}-\d{2}|^\d+(\.\d+)?$", ts):
            ts = f"{r[keys.index('date')].strip()} {ts.strip()}"
        fixes.append(Fix(parse_time(ts, tz), lat, lng, _num(get("ele")), _num(get("accuracy")),
                         _num(get("speed")), _num(get("course"))))
    return fixes


def read_track(path: str | Path, tz: tzinfo = timezone.utc) -> list[Fix]:
    """GPX or CSV, by content. Sorted by time, one fix per timestamp, impossible coordinates dropped."""
    head = Path(path).read_text(encoding="utf-8-sig", errors="replace")[:512].lstrip()
    fixes = read_gpx(path, tz) if head.startswith("<") else read_csv(path, tz)
    by_t = {}
    for f in fixes:
        if -90 <= f.lat <= 90 and -180 <= f.lng <= 180 and not (f.lat == 0 and f.lng == 0) and math.isfinite(f.t):
            by_t[f.t] = f
    return [by_t[t] for t in sorted(by_t)]


# ------------------------------------------------------------------ smoothing, speed, heading
def _kernel_fit(t, x, y, w, tq, sigma):
    """Gaussian-weighted local linear fit of x(t), y(t) evaluated at tq."""
    out_x, out_y = np.empty(len(tq)), np.empty(len(tq))
    for i in range(0, len(tq), 1000):
        q = tq[i:i + 1000, None]
        d = t[None, :] - q
        k = np.exp(-0.5 * (d / sigma) ** 2) * w[None, :]
        s0, s1, s2 = k.sum(1), (k * d).sum(1), (k * d * d).sum(1)
        det = s0 * s2 - s1 * s1
        ok = det > 1e-9 * np.maximum(s0 * s2, 1e-300)
        for vals, out in ((x, out_x), (y, out_y)):
            t0, t1 = (k * vals).sum(1), (k * d * vals).sum(1)
            mean = t0 / np.maximum(s0, 1e-300)
            lin = (s2 * t0 - s1 * t1) / np.where(ok, det, 1.0)
            out[i:i + 1000] = np.where(ok, lin, mean)
    return out_x, out_y


class Track:
    """A cleaned, smoothed walker track in sim metres. Query any UTC time with .at()."""

    def __init__(self, fixes: list[Fix], smooth_s: float = 3.0, heading_window_s: float = 2.0,
                 stop_speed: float = 0.25, max_gap_s: float = 15.0, max_accuracy: float = 50.0,
                 origin: tuple[float, float] = CROSSING):
        self.origin = origin
        self.n_read = len(fixes)
        fixes = [f for f in fixes if f.accuracy is None or f.accuracy <= max_accuracy]
        self.n_bad_accuracy = self.n_read - len(fixes)
        if len(fixes) < 2:
            raise ValueError(f"need at least 2 usable GPS fixes, have {len(fixes)}")
        t = np.array([f.t for f in fixes])
        x, y = latlng_to_sim([f.lat for f in fixes], [f.lng for f in fixes], origin)
        acc = np.array([f.accuracy if f.accuracy is not None else np.nan for f in fixes])
        w = 1.0 / np.maximum(np.where(np.isfinite(acc), acc, 5.0), 2.0) ** 2
        keep = np.ones(len(t), bool)
        if len(t) >= 5:  # one outlier pass: a fix far from the smoothed track of its neighbours is a glitch
            sx, sy = _kernel_fit(t, x, y, w, t, smooth_s)
            r = np.hypot(x - sx, y - sy)
            keep = r <= max(15.0, 4.0 * float(np.median(r)))
        self.n_outliers = int((~keep).sum())
        self.fixes = [f for f, k in zip(fixes, keep) if k]
        self.t, self.raw_x, self.raw_y, self.acc = t[keep], x[keep], y[keep], acc[keep]
        self.x, self.y = _kernel_fit(self.t, self.raw_x, self.raw_y, w[keep], self.t, smooth_s)
        self.t0, self.t1 = float(self.t[0]), float(self.t[-1])
        self.max_gap_s, self.heading_window_s, self.stop_speed = max_gap_s, heading_window_s, stop_speed
        # regular grid: interpolated smoothed positions, then speed / heading / moving / held heading
        n = int(math.floor((self.t1 - self.t0) * GRID_HZ)) + 1
        self.gt = self.t0 + np.arange(n) / GRID_HZ
        self.gx, self.gy = np.interp(self.gt, self.t, self.x), np.interp(self.gt, self.t, self.y)
        idx = np.clip(np.searchsorted(self.t, self.gt), 1, len(self.t) - 1)
        before, after = self.t[idx - 1], self.t[idx]
        inside = (self.gt >= before) & (self.gt <= after)
        self.ggap = np.where(inside, after - before, 0.0)  # length of the fix gap a grid point sits in
        vx, vy = self._diff(self.gx, heading_window_s), self._diff(self.gy, heading_window_s)
        self.gspeed = np.hypot(vx, vy)
        sx, sy = self._diff(self.gx, STOP_WINDOW_S), self._diff(self.gy, STOP_WINDOW_S)
        self.gmoving = (np.hypot(sx, sy) >= stop_speed) & (self.gspeed >= 0.5 * stop_speed)
        self.gheading = np.where(self.gmoving, compass(vx, vy), np.nan)
        held, last = np.full(n, np.nan), np.nan
        for i in range(n):
            if self.gmoving[i]:
                last = self.gheading[i]
            held[i] = last
        self.gheld = held

    def _diff(self, v: np.ndarray, window_s: float) -> np.ndarray:
        """Velocity over a centred window (shrunk at the ends of the track)."""
        h = max(1, int(round(window_s * GRID_HZ / 2)))
        n = len(v)
        i = np.arange(n)
        lo, hi = np.clip(i - h, 0, n - 1), np.clip(i + h, 0, n - 1)
        dt = np.maximum((hi - lo) / GRID_HZ, 1e-9)
        return (v[hi] - v[lo]) / dt

    def at(self, tq) -> dict[str, np.ndarray]:
        """Walker state at UTC times tq: x, y, lat, lng, speed, heading (nan when still), heading_hold, moving,
        gap (s between the fixes around tq), valid (inside the track and no gap longer than max_gap_s)."""
        tq = np.atleast_1d(np.asarray(tq, dtype=float))
        gi = np.clip(np.round((tq - self.t0) * GRID_HZ).astype(int), 0, len(self.gt) - 1)
        valid = (tq >= self.t0) & (tq <= self.t1) & (self.ggap[gi] <= self.max_gap_s)
        x = np.where(valid, np.interp(tq, self.t, self.x), np.nan)
        y = np.where(valid, np.interp(tq, self.t, self.y), np.nan)
        lat, lng = sim_to_latlng(x, y, self.origin)
        nan = np.full(len(tq), np.nan)
        return {"x": x, "y": y, "lat": lat, "lng": lng, "valid": valid,
                "speed": np.where(valid, self.gspeed[gi], nan), "heading": np.where(valid, self.gheading[gi], nan),
                "heading_hold": np.where(valid, self.gheld[gi], nan), "moving": valid & self.gmoving[gi],
                "gap": np.where(valid, self.ggap[gi], nan)}

    def stats(self) -> dict:
        step = np.hypot(np.diff(self.gx), np.diff(self.gy))
        moving = self.gmoving[1:] & self.gmoving[:-1] & (self.ggap[1:] <= self.max_gap_s)
        dist = float(step[moving].sum())
        moving_s = float(moving.sum() / GRID_HZ)
        dur = self.t1 - self.t0
        d = np.hypot(self.x, self.y)  # to the origin (the crossing node)
        i = int(np.argmin(d))
        acc = self.acc[np.isfinite(self.acc)]
        return {
            "fixes_read": self.n_read, "fixes_used": len(self.t), "dropped_accuracy": self.n_bad_accuracy,
            "dropped_outliers": self.n_outliers,
            "start_utc": _iso(self.t0), "end_utc": _iso(self.t1), "duration_s": round(dur, 1),
            "median_fix_interval_s": round(float(np.median(np.diff(self.t))), 2),
            "longest_fix_gap_s": round(float(np.max(np.diff(self.t))), 1),
            "median_accuracy_m": round(float(np.median(acc)), 1) if len(acc) else None,
            "distance_m": round(dist, 1), "moving_s": round(moving_s, 1),
            "mean_walking_speed_mps": round(dist / moving_s, 2) if moving_s > 0 else None,
            "mean_speed_overall_mps": round(dist / dur, 2) if dur > 0 else None,
            "closest_to_crossing": {"distance_m": round(float(d[i]), 1), "utc": _iso(float(self.t[i])),
                                    "t_epoch": round(float(self.t[i]), 3), "x": round(float(self.x[i]), 2),
                                    "y": round(float(self.y[i]), 2)},
        }


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


# ------------------------------------------------------------------ planned route (spec section 8)
def planned_route(moss_bearing: float = MOSS_BEARING_DEG, before_m: float = 70.0, along_howard_m: float = 20.0,
                  sidewalk_offset_m: float = 5.0) -> list[tuple[str, float, float]]:
    """APPROXIMATE shoot route in sim metres: (label, x, y). Start on Moss's NE sidewalk before_m SE of the zebra,
    walk NW to the corner, turn left across the zebra, continue along_howard_m SW on Howard's south sidewalk.
    Assumes the crosswalk sits in line with Howard's south sidewalk and the sidewalk centrelines are
    sidewalk_offset_m either side of the crossing node (Moss's centreline). Check it against the real track."""
    b = math.radians(moss_bearing)
    u = (math.sin(b), math.cos(b))  # along Moss towards Howard
    ne = (math.sin(b + math.pi / 2), math.cos(b + math.pi / 2))  # perpendicular, to the walker's right going NW
    p = lambda along, side: (along * u[0] + side * ne[0], along * u[1] + side * ne[1])  # noqa: E731
    return [("start, NE sidewalk of Moss", *p(-before_m, sidewalk_offset_m)),
            ("corner, NE side of the zebra", *p(0, sidewalk_offset_m)),
            ("SW side of the zebra", *p(0, -sidewalk_offset_m)),
            ("stop, Howard south sidewalk", *p(0, -sidewalk_offset_m - along_howard_m))]


def densify(pts: list[tuple[float, float]], step_m: float = 1.0) -> np.ndarray:
    out = [pts[0]]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        n = max(1, int(math.ceil(math.hypot(x1 - x0, y1 - y0) / step_m)))
        out += [(x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n) for k in range(1, n + 1)]
    return np.array(out)


def distance_to_polyline(x, y, poly: np.ndarray) -> np.ndarray:
    """Distance (m) from each point to the nearest point of a polyline (sim metres)."""
    p = np.stack([np.atleast_1d(x), np.atleast_1d(y)], 1)[:, None, :]
    a, b = poly[None, :-1], poly[None, 1:]
    ab = b - a
    s = np.clip(((p - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12), 0, 1)
    return np.linalg.norm(p - (a + s[..., None] * ab), axis=-1).min(1)


def write_route(out: Path, moss_bearing: float = MOSS_BEARING_DEG, walk_speed: float = 0.6,
                start_utc: str = "2026-10-24T14:00:00Z", source: str = "route-plan") -> dict:
    """data/shoot_route.gpx (timed track at walk_speed, so it also runs through `python -m kite.gps`) and
    shoot_route.jsonl (`position` events, replayable as a planned walk)."""
    out.mkdir(parents=True, exist_ok=True)
    corners = planned_route(moss_bearing)
    line = densify([(x, y) for _, x, y in corners])
    seg = np.hypot(*np.diff(line, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    t0 = parse_time(start_utc)
    lat, lng = sim_to_latlng(line[:, 0], line[:, 1])
    hdg = compass(*np.diff(line, axis=0).T)
    hdg = np.append(hdg, hdg[-1])
    gpx = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<gpx version="1.1" creator="kite.gps --route" xmlns="http://www.topografix.com/GPX/1/1">',
           f'<metadata><name>Kite shoot route (APPROXIMATE, Moss bearing {moss_bearing:g} deg)</name>'
           '<desc>Derived from spec section 8 and the crossing node; not surveyed.</desc></metadata>']
    clat, clng = CROSSING
    gpx.append(f'<wpt lat="{clat}" lon="{clng}"><name>Moss St zebra crossing (sim origin)</name></wpt>')
    for label, x, y in corners:
        la, ln = sim_to_latlng(x, y)
        gpx.append(f'<wpt lat="{la:.7f}" lon="{ln:.7f}"><name>{label}</name></wpt>')
    gpx.append("<trk><name>planned walk</name><trkseg>")
    for i in range(len(line)):
        gpx.append(f'<trkpt lat="{lat[i]:.7f}" lon="{lng[i]:.7f}"><time>{_iso(t0 + s[i] / walk_speed)}</time>'
                   f"<course>{hdg[i]:.1f}</course><speed>{walk_speed:g}</speed></trkpt>")
    gpx += ["</trkseg></trk>", "</gpx>"]
    (out / "shoot_route.gpx").write_text("\n".join(gpx) + "\n")
    with open(out / "shoot_route.jsonl", "w") as fh:
        for i in range(len(line)):
            fh.write(json.dumps({"type": "position", "source": source, "lat": round(float(lat[i]), 7),
                                 "lng": round(float(lng[i]), 7), "heading": round(float(hdg[i]), 1),
                                 "x": round(float(line[i, 0]), 2), "y": round(float(line[i, 1]), 2),
                                 "place": PLACE, "t": round(float(s[i] / walk_speed), 2)}) + "\n")
    return {"length_m": round(float(s[-1]), 1), "duration_s": round(float(s[-1] / walk_speed), 1),
            "points": len(line), "corners": [(lbl, round(x, 2), round(y, 2)) for lbl, x, y in corners]}


# ------------------------------------------------------------------ recording alignment
def read_recording(rec: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """(frame index, t_wall, meta) of a kite.record take."""
    rec = Path(rec)
    meta = json.loads((rec / "meta.json").read_text()) if (rec / "meta.json").exists() else {}
    idx, tw = [], []
    if (rec / "frames.csv").exists():
        with open(rec / "frames.csv", newline="") as fh:
            for r in csv.DictReader(fh):
                if _num(r.get("t_wall")) is not None:
                    idx.append(int(r["index"]))
                    tw.append(float(r["t_wall"]))
    if not tw and "started_wall" in meta:  # no frames: at least place the start
        idx, tw = [0], [float(meta["started_wall"])]
    if not tw:
        raise ValueError(f"{rec}: no frames.csv rows and no started_wall in meta.json")
    return np.array(idx, int), np.array(tw), meta


def overlap_report(track: Track, v0: float, v1: float) -> dict:
    """Overlap of the GPS span with the video span [v0, v1] (both UTC, offset already applied)."""
    ov = max(0.0, min(v1, track.t1) - max(v0, track.t0))
    vid = max(v1 - v0, 1e-9)
    d = track.t0 - v0
    rep = {"video_start_utc": _iso(v0), "video_end_utc": _iso(v1), "video_duration_s": round(v1 - v0, 1),
           "overlap_s": round(ov, 1), "overlap_frac_of_video": round(min(1.0, ov / vid), 3),
           "gps_start_minus_video_start_s": round(d, 1), "warnings": []}
    hours = round(d / 3600)
    if hours and abs(d - 3600 * hours) < 600:
        rep["warnings"].append(f"GPS and video starts differ by ~{hours:+d} h: a time-zone mistake? "
                               f"Check --tz for times without a zone (or the phone app's export setting).")
    if ov <= 0:
        rep["warnings"].append(f"NO OVERLAP between GPS and video. If they started together, try --offset-s "
                               f"{d:+.1f} (shifts the video onto the GPS start).")
    elif ov < 0.5 * vid:
        rep["warnings"].append("GPS covers less than half of the video: check --offset-s and the phone's log.")
    return rep


def _fmt(v, nd):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return ""
    return f"{v:.{nd}f}"


def position_event(s: dict, i: int, t_rel: float, source: str, place: str | None, frame: int | None) -> dict:
    ev = {"type": "position", "source": source, "lat": round(float(s["lat"][i]), 7),
          "lng": round(float(s["lng"][i]), 7)}
    if math.isfinite(s["heading_hold"][i]):
        ev["heading"] = round(float(s["heading_hold"][i]), 1)
    ev.update(x=round(float(s["x"][i]), 2), y=round(float(s["y"][i]), 2), speed=round(float(s["speed"][i]), 2),
              moving=bool(s["moving"][i]))
    if place:
        ev["place"] = place
    if frame is not None:
        ev["frame"] = int(frame)
    ev["t"] = round(t_rel, 3)
    return ev


def process(track_path: str | Path, recording: str | Path | None = None, out: str | Path | None = None,
            offset_s: float = 0.0, tz: str = "UTC", rate: float = 5.0, smooth_s: float = 3.0,
            heading_window_s: float = 2.0, stop_speed: float = 0.25, max_gap_s: float = 15.0,
            max_accuracy: float = 50.0, source: str = "gps", place: str | None = PLACE,
            moss_bearing: float = MOSS_BEARING_DEG) -> dict:
    """Read, smooth, align and write every output. Returns the summary (also written to summary.json)."""
    track_path = Path(track_path)
    fixes = read_track(track_path, parse_tz(tz))
    track = Track(fixes, smooth_s, heading_window_s, stop_speed, max_gap_s, max_accuracy)
    if out is None:
        out = Path(recording) / "gps" if recording else track_path.with_name(track_path.stem + "_gps")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"track_file": str(track_path), "tz_for_naive_times": tz, "offset_s": offset_s,
               "smoothing": {"smooth_s": smooth_s, "heading_window_s": heading_window_s, "stop_speed_mps": stop_speed,
                             "max_gap_s": max_gap_s, "max_accuracy_m": max_accuracy},
               "origin": {"lat": CROSSING[0], "lng": CROSSING[1], "what": "Moss St zebra crossing node; x east, "
                          "y north in metres (sim/src/config.ts toLatLng convention)"},
               "track": track.stats()}

    # (3) every fix in sim metres
    st = track.at(track.t)
    with open(out / "track.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["utc", "t_epoch", "lat_raw", "lng_raw", "x_raw", "y_raw", "lat", "lng", "x", "y", "speed",
                    "heading", "heading_hold", "moving", "accuracy"])
        for i, f in enumerate(track.fixes):
            w.writerow([_iso(f.t), f"{f.t:.3f}", f"{f.lat:.7f}", f"{f.lng:.7f}", _fmt(track.raw_x[i], 2),
                        _fmt(track.raw_y[i], 2), _fmt(st["lat"][i], 7), _fmt(st["lng"][i], 7), _fmt(st["x"][i], 2),
                        _fmt(st["y"][i], 2), _fmt(st["speed"][i], 2), _fmt(st["heading"][i], 1),
                        _fmt(st["heading_hold"][i], 1), int(st["moving"][i]), _fmt(f.accuracy, 1)])

    # planned-route sanity check
    route = densify([(x, y) for _, x, y in planned_route(moss_bearing)])
    rd = distance_to_polyline(track.x, track.y, route)
    summary["vs_planned_route"] = {"median_m": round(float(np.median(rd)), 1), "max_m": round(float(np.max(rd)), 1),
                                   "note": "planned route is approximate (Moss bearing assumed)"}

    # (1) + (2): events and per-frame pose
    frame_idx = None
    if recording:
        frame_idx, t_wall, meta = read_recording(recording)
        v0, v1 = float(t_wall[0]), float(t_wall[-1])
        ov = overlap_report(track, v0 + offset_s, v1 + offset_s)
        summary["recording"] = {"dir": str(recording), "frames": int(len(frame_idx)),
                                "started_iso_laptop": meta.get("started_iso"), **ov}
        fs = track.at(t_wall + offset_s)
        summary["recording"]["frames_with_gps"] = int(fs["valid"].sum())
        with open(out / "frames_gps.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["index", "t_wall", "t_gps", "video_s", "lat", "lng", "x", "y", "heading", "heading_hold",
                        "speed", "moving", "gps_gap_s"])
            for i in range(len(frame_idx)):
                w.writerow([int(frame_idx[i]), f"{t_wall[i]:.6f}", f"{t_wall[i] + offset_s:.3f}",
                            f"{t_wall[i] - v0:.3f}", _fmt(fs["lat"][i], 7), _fmt(fs["lng"][i], 7), _fmt(fs["x"][i], 2),
                            _fmt(fs["y"][i], 2), _fmt(fs["heading"][i], 1), _fmt(fs["heading_hold"][i], 1),
                            _fmt(fs["speed"][i], 2), int(fs["moving"][i]) if fs["valid"][i] else "",
                            _fmt(fs["gap"][i], 1)])
        rel = np.arange(0.0, v1 - v0 + 1e-9, 1.0 / rate)
        tq = v0 + offset_s + rel
        frames_at = frame_idx[np.clip(np.searchsorted(t_wall, v0 + rel), 0, len(frame_idx) - 1)]
        c = summary["track"]["closest_to_crossing"]
        c["video_s"] = round(c["t_epoch"] - offset_s - v0, 2)
    else:
        rel = np.arange(0.0, track.t1 - track.t0 + 1e-9, 1.0 / rate)
        tq = track.t0 + rel
    s = track.at(tq)
    n_ev = 0
    with open(out / "positions.jsonl", "w") as fh:
        first = np.flatnonzero(s["valid"])
        if recording and len(first) and rel[first[0]] > 0:
            fh.write(json.dumps({"type": "heartbeat", "source": source, "t": 0.0}) + "\n")
        for i in first:
            fh.write(json.dumps(position_event(s, i, float(rel[i]), source, place,
                                               None if frame_idx is None else frames_at[i])) + "\n")
            n_ev += 1
    summary["positions_jsonl"] = {"events": n_ev, "rate_hz": rate,
                                  "t_zero": "first video frame" if recording else "first GPS fix"}
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    summary["out"] = str(out)
    return summary


def print_summary(s: dict) -> None:
    tr = s["track"]
    print(f"GPS  {tr['start_utc']} -> {tr['end_utc']}  ({tr['duration_s']:.0f} s, {tr['fixes_used']} fixes used of "
          f"{tr['fixes_read']}; {tr['dropped_accuracy']} poor accuracy, {tr['dropped_outliers']} outliers; "
          f"fix every {tr['median_fix_interval_s']} s, longest gap {tr['longest_fix_gap_s']} s, "
          f"median accuracy {'-' if tr['median_accuracy_m'] is None else tr['median_accuracy_m']} m)")
    sp = tr["mean_walking_speed_mps"]
    print(f"walk {tr['distance_m']:.0f} m in {tr['moving_s']:.0f} s moving, mean walking speed "
          f"{sp if sp is not None else '-'} m/s (overall {tr['mean_speed_overall_mps']} m/s)")
    c = tr["closest_to_crossing"]
    print(f"closest to the crossing node: {c['distance_m']} m at {c['utc']}"
          + (f" (video {c['video_s']:.1f} s)" if "video_s" in c else ""))
    r = s["vs_planned_route"]
    print(f"distance from the planned route (approximate): median {r['median_m']} m, max {r['max_m']} m")
    if "recording" in s:
        v = s["recording"]
        print(f"video {v['video_start_utc']} -> {v['video_end_utc']} (laptop clock + offset {s['offset_s']:+g} s, "
              f"{v['frames']} frames)")
        print(f"OVERLAP {v['overlap_s']:.1f} s = {100 * v['overlap_frac_of_video']:.0f}% of the video; "
              f"{v['frames_with_gps']} frames have a walker pose; GPS starts {v['gps_start_minus_video_start_s']:+.1f} s "
              f"after the video")
        for wmsg in v["warnings"]:
            print(f"WARNING: {wmsg}")
    print(f"{s['positions_jsonl']['events']} position events -> {s['out']}")


# ------------------------------------------------------------------ CLI
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m kite.gps", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("track", nargs="?", help="phone GPS log: .gpx or .csv")
    p.add_argument("--recording", help="a kite.record take directory (frames.csv, meta.json)")
    p.add_argument("--offset-s", type=float, default=0.0,
                   help="phone clock minus laptop clock, s: a frame's GPS time is t_wall + offset (default 0)")
    p.add_argument("--tz", default="UTC", help="zone for times written without one: UTC (default), -07:00, "
                   "America/Los_Angeles")
    p.add_argument("--out", help="output folder (default <recording>/gps or <track>_gps)")
    p.add_argument("--rate", type=float, default=5.0, help="position events per second (default 5, like the sim)")
    p.add_argument("--smooth-s", type=float, default=3.0, help="Gaussian smoothing sigma, s (default 3)")
    p.add_argument("--heading-window-s", type=float, default=2.0, help="course-over-ground window, s (default 2)")
    p.add_argument("--stop-speed", type=float, default=0.25, help="below this (m/s) the walker is standing still")
    p.add_argument("--max-gap-s", type=float, default=15.0, help="no pose inside longer gaps between fixes")
    p.add_argument("--max-accuracy", type=float, default=50.0, help="drop fixes with worse accuracy (m)")
    p.add_argument("--source", default="gps", help="event source (default gps)")
    p.add_argument("--place", default=PLACE, help="event place text ('' for none)")
    p.add_argument("--route", action="store_true", help="write the planned shoot route (GPX + JSONL) to --out "
                   "(default data/) and exit")
    p.add_argument("--moss-bearing", type=float, default=MOSS_BEARING_DEG,
                   help=f"compass bearing of Moss St towards Howard (APPROXIMATE, default {MOSS_BEARING_DEG:g})")
    p.add_argument("--walk-speed", type=float, default=0.6, help="--route pacing, m/s (default 0.6)")
    a = p.parse_args(argv)

    if a.route:
        out = Path(a.out or "data")
        info = write_route(out, a.moss_bearing, a.walk_speed)
        print(f"planned route (APPROXIMATE, Moss bearing {a.moss_bearing:g} deg): {info['length_m']} m, "
              f"{info['duration_s']:.0f} s at {a.walk_speed:g} m/s -> {out}/shoot_route.gpx, shoot_route.jsonl")
        for label, x, y in info["corners"]:
            la, ln = sim_to_latlng(x, y)
            print(f"  {label:32s} x {x:7.2f}  y {y:7.2f}   {la:.6f}, {ln:.6f}")
        return 0
    if not a.track:
        p.error("give a GPS track (or --route)")
    if a.rate <= 0 or a.smooth_s <= 0 or a.heading_window_s <= 0:
        p.error("--rate, --smooth-s and --heading-window-s must be > 0")
    try:
        s = process(a.track, a.recording, a.out, a.offset_s, a.tz, a.rate, a.smooth_s, a.heading_window_s,
                    a.stop_speed, a.max_gap_s, a.max_accuracy, a.source, a.place or None, a.moss_bearing)
    except (ValueError, OSError, ET.ParseError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print_summary(s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
