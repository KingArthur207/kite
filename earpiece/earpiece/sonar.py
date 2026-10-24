"""Proximity sonar: turns the perception feed into spatial warning beeps.

For every tracked object we estimate how soon the wearer would reach it and
how much it is in their way, then sonify at most two of them:

  where   -> the beep comes from the object's direction (binaural rendering)
  what    -> timbre (knock = person, buzz = vehicle, pip = obstacle, glide = drop)
  urgency -> beep rate (1.0 s apart far away -> 0.11 s close) and pitch (rising)
  danger  -> inside ~0.75 m or < 1 s to contact: alarm bus + spoken "Stop."

Numbers follow published travel-aid work: WatchOut (Presti et al. 2021: 50 ->
280 pulses/min, 0.5 m walkable corridor), parking sensors (continuous at
~0.3 m), BBeep (CHI 2019: warn on predicted collision, three stages), and
Brungart et al. 2005 (localisation degrades past two concurrent sources).

Robustness:
  * urgency uses time-to-contact when the object is approaching, distance otherwise;
  * closing speed is measured from the track history when the sender does not supply it;
  * entry/exit hysteresis and bearing smoothing stop flicker between frames;
  * a track that stops being reported fades out after `stale_track_s`;
  * if the whole perception feed goes quiet, the watchdog disables sonar
    entirely (stale beeps are worse than none) and says so.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable

import numpy as np

from . import sounds, spatial
from .config import SonarCfg
from .events import Obstacle, clock_phrase
from .mixer import Engine

_SPOKEN_KIND = {"person": "Person", "vehicle": "Car", "bike": "Bike", "curb": "Curb",
                "stairs_down": "Step down"}


@dataclass
class Track:
    id: str
    kind: str
    source: str
    bearing: float
    distance: float
    confidence: float = 1.0
    given_speed: float | None = None
    est_speed: float | None = None
    last_seen: float = 0.0
    ttl: float | None = None
    hist: deque = field(default_factory=lambda: deque(maxlen=12))
    urgency: float = 0.0
    ttc: float = math.inf
    active: bool = False
    critical: bool = False
    next_beep: int | None = None       # engine sample of the next scheduled beep
    last_beep: int | None = None       # engine sample of the last scheduled beep
    last_label_t: float = -1e9
    last_stop_t: float = -1e9
    first_seen: float = 0.0

    @property
    def closing_speed(self) -> float | None:
        return self.given_speed if self.given_speed is not None else self.est_speed


@lru_cache(maxsize=2048)
def _cue(kind: str, pitch_q: int, bearing_q: int, rate: int) -> np.ndarray:
    """Rendered stereo beep, cached on (kind, quantised pitch, 5-degree bearing)."""
    mono = sounds.proximity_cue(kind, float(pitch_q))
    if rate != sounds.RATE:
        from scipy.signal import resample_poly
        g = math.gcd(rate, sounds.RATE)
        mono = resample_poly(mono, rate // g, sounds.RATE // g).astype(np.float32)
    return spatial.render(mono, float(bearing_q), rate)


class Sonar:
    def __init__(self, engine: Engine, cfg: SonarCfg,
                 speak: Callable[..., object] | None = None, log=print):
        self.engine = engine
        self.cfg = cfg
        self.speak = speak                 # speak(text, priority, key=, ttl=)
        self.log = log
        self.tracks: dict[str, Track] = {}
        self.enabled = True                # watchdog switches this off when perception is stale
        self.muted = False                 # wearer asked for quiet (critical still sounds)
        self.labels = True                 # speak "person, left" on new urgent objects
        self._last_global_label = -1e9
        self._last_seq: dict[str, int] = {}
        self._last_seq_t: dict[str, float] = {}
        self.stats = {"frames": 0, "objects": 0, "out_of_order": 0, "beeps": 0, "criticals": 0}

    # ------------------------------------------------------------------ input
    def update_frame(self, source: str, objects: list[Obstacle], seq: int | None = None,
                     now: float | None = None) -> None:
        """A full perception frame from `source`: ids missing from it are gone."""
        now = time.monotonic() if now is None else now
        if seq is not None:
            last = self._last_seq.get(source)
            last_t = self._last_seq_t.get(source, -1e9)
            # A late UDP datagram (a few frames old) must never overwrite newer data. A big
            # backward jump, or any seq after a pause, means the sender restarted: accept it.
            if (last is not None and seq <= last and last - seq <= 5 and now - last_t < 0.5):
                self.stats["out_of_order"] += 1
                return
            self._last_seq[source] = seq
            self._last_seq_t[source] = now
        self.stats["frames"] += 1
        self.stats["objects"] += len(objects)
        seen = set()
        for o in objects:
            key = f"{source}:{o.id}"
            seen.add(key)
            self._upsert(key, source, o, now, ttl=None)
        for key, t in list(self.tracks.items()):
            if t.source == source and key not in seen and t.ttl is None:
                # gone from this frame: let it fade over stale_track_s instead of
                # vanishing (detectors drop objects for a frame all the time)
                t.last_seen = min(t.last_seen, now - 1e-3)

    def update_one(self, source: str, o: Obstacle, ttl: float | None, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        self._upsert(f"{source}:{o.id}", source, o, now, ttl=ttl or 0.5)

    def _upsert(self, key: str, source: str, o: Obstacle, now: float, ttl: float | None) -> None:
        t = self.tracks.get(key)
        if t is None:
            t = Track(key, o.kind, source, o.bearing, o.distance, o.confidence, o.closing_speed,
                      last_seen=now, ttl=ttl, first_seen=now)
            self.tracks[key] = t
        else:
            # wrap-aware bearing smoothing; distance mostly trusted (it is what matters)
            t.bearing = spatial.wrap_deg(t.bearing + 0.6 * spatial.wrap_deg(o.bearing - t.bearing))
            t.distance = 0.3 * t.distance + 0.7 * o.distance
            t.kind = o.kind
            t.confidence = o.confidence
            t.given_speed = o.closing_speed
            t.last_seen = now
            t.ttl = ttl
        t.hist.append((now, o.distance))
        t.est_speed = self._estimate_speed(t)

    @staticmethod
    def _estimate_speed(t: Track) -> float | None:
        """Least-squares slope of distance over the last ~0.8 s: positive = approaching."""
        pts = [(tt, d) for tt, d in t.hist if t.hist[-1][0] - tt <= 0.8]
        if len(pts) < 3 or pts[-1][0] - pts[0][0] < 0.2:
            return None
        ts = np.array([p[0] for p in pts])
        ds = np.array([p[1] for p in pts])
        ts -= ts.mean()
        den = float(np.sum(ts * ts))
        if den <= 0:
            return None
        slope = float(np.sum(ts * (ds - ds.mean())) / den)
        return max(-10.0, min(15.0, -slope))

    def clear(self, source: str | None = None) -> None:
        for k in [k for k, t in self.tracks.items() if source is None or t.source == source]:
            del self.tracks[k]

    # ------------------------------------------------------------------ urgency
    def _score(self, t: Track) -> None:
        c = self.cfg
        b = math.radians(t.bearing)
        fwd = t.distance * math.cos(b)
        lat = abs(t.distance * math.sin(b))
        half = c.corridor_half_width_m + (0.25 if t.distance < 1.5 else 0.0)  # close = wider
        if lat <= half:
            corridor = 1.0
        else:
            corridor = max(0.0, 1.0 - (lat - half) / c.corridor_margin_m)
        if fwd <= 0.0:
            # beside or behind: only matters when very close (someone right next to you)
            corridor = 1.0 if t.distance < 0.8 else 0.0
            fwd = t.distance
        v = t.closing_speed
        approach = c.walk_speed_mps if v is None else max(v, 0.0)
        approach = max(approach, 0.25)
        ttc = fwd / approach
        warn_range = min(c.max_range_m, c.walk_speed_mps * c.ttc_start_s + 0.5)
        span_t = max(1e-3, c.ttc_start_s - c.ttc_critical_s)
        u_ttc = min(1.0, max(0.0, (c.ttc_start_s - ttc) / span_t))
        u_dist = min(1.0, max(0.0, (warn_range - t.distance) / max(1e-3, warn_range - c.critical_dist_m)))
        conf = 0.5 + 0.5 * t.confidence
        u = max(u_ttc, 0.85 * u_dist) * corridor * conf
        if t.kind == "vehicle":
            u = min(1.0, u * 1.25)
        h = 1.0 - c.hysteresis if t.critical else 1.0
        critical = corridor > 0.5 and (t.distance <= c.critical_dist_m / h or
                                        (v is not None and v > 0.3 and ttc <= c.ttc_critical_s / h))
        enter, exit_ = 0.08, 0.08 * (1.0 - c.hysteresis) * 0.5
        t.active = u > enter or (t.active and u > exit_)
        t.urgency = u
        t.ttc = ttc
        if critical and not t.critical:
            self.stats["criticals"] += 1
        t.critical = critical

    # ------------------------------------------------------------------ tick
    def tick(self, now: float | None = None) -> list[Track]:
        """Call ~50 Hz. Updates urgency, prunes stale tracks, schedules beeps.
        Returns the tracks being sonified (for the dashboard and tests)."""
        now = time.monotonic() if now is None else now
        c = self.cfg
        for k, t in list(self.tracks.items()):
            age = now - t.last_seen
            limit = t.ttl if t.ttl is not None else c.stale_track_s
            if age > limit:
                del self.tracks[k]
        for t in self.tracks.values():
            self._score(t)
        if not self.enabled:
            for t in self.tracks.values():
                t.next_beep = None
            return []
        ranked = sorted((t for t in self.tracks.values() if t.active),
                        key=lambda t: (not t.critical, -t.urgency))
        chosen = ranked[: c.max_sources]
        chosen_ids = {t.id for t in chosen}
        for t in self.tracks.values():
            if t.id not in chosen_ids:
                t.next_beep = None
        lookahead = self.engine.now + self.engine.samples(0.06)
        for rank, t in enumerate(chosen):
            if self.muted and not t.critical:
                t.next_beep = None
                continue
            u = t.urgency if not t.critical else 1.0
            interval = c.interval_slow_s - (c.interval_slow_s - c.interval_fast_s) * (u ** 0.8)
            pitch = c.pitch_low_hz + (c.pitch_high_hz - c.pitch_low_hz) * u
            pitch_q = int(round(pitch / 25.0) * 25)
            bear_q = int(round(t.bearing / 5.0) * 5)
            fade = min(1.0, max(0.0, 1.0 - (now - t.last_seen) / c.stale_track_s))
            gain = (1.0 if rank == 0 else c.second_source_gain) * spatial.distance_gain(t.distance) * fade
            step = self.engine.samples(interval)
            if t.next_beep is None:
                t.next_beep = self.engine.now + self.engine.samples(0.02)
                if t.last_beep is not None:      # resuming: keep at least one gap after the last beep
                    t.next_beep = max(t.next_beep, t.last_beep + step)
            elif t.last_beep is not None and t.next_beep > t.last_beep + step:
                t.next_beep = max(self.engine.now, t.last_beep + step)   # speeding up: pull the beat in
            while t.next_beep <= lookahead:
                self.engine.play(_cue(t.kind, pitch_q, bear_q, self.engine.rate), at=t.next_beep,
                                 bus="alert" if t.critical else "cue", gain=gain, tag=f"sonar:{t.id}")
                self.stats["beeps"] += 1
                t.last_beep = t.next_beep
                t.next_beep += step
            self._maybe_speak(t, now)
        return chosen

    def _maybe_speak(self, t: Track, now: float) -> None:
        if self.speak is None:
            return
        c = self.cfg
        name = _SPOKEN_KIND.get(t.kind, "Obstacle")
        if t.critical and now - t.last_stop_t > 4.0:
            t.last_stop_t = now
            t.last_label_t = now
            self._last_global_label = now
            ahead = abs(t.bearing) <= 25
            text = f"Stop. {name} ahead." if ahead and name in ("Person", "Obstacle") else (
                "Stop. Car." if name == "Car" else "Stop.")
            self.speak(text, "critical", key=f"stop:{t.id}", ttl=1.0)
            return
        if not self.labels or self.muted:
            return
        # Label early (urgency 0.3) so the words finish before any "Stop"; skip the label
        # entirely when contact is so near that "Stop" would cut it off mid-word.
        v = t.closing_speed
        imminent = v is not None and v > 0.3 and t.ttc < c.ttc_critical_s + 1.2
        if (t.urgency >= 0.3 and not imminent and now - t.last_label_t > c.speak_label_cooldown_s
                and now - self._last_global_label > c.speak_global_gap_s):
            t.last_label_t = now
            self._last_global_label = now
            self.speak(f"{name}, {clock_phrase(t.bearing)}.", "hazard", key=f"label:{t.id}", ttl=1.5)

    def summary(self) -> list[dict]:
        return [{"id": t.id, "kind": t.kind, "bearing": round(t.bearing, 1), "distance": round(t.distance, 2),
                 "urgency": round(t.urgency, 2), "ttc": None if math.isinf(t.ttc) else round(t.ttc, 2),
                 "critical": t.critical, "active": t.active,
                 "closing_speed": None if t.closing_speed is None else round(t.closing_speed, 2)}
                for t in sorted(self.tracks.values(), key=lambda t: -t.urgency)]
