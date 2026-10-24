"""Audio beacon toward the next waypoint (after Microsoft Soundscape's design).

The navigation side sends the waypoint's bearing relative to the wearer and its
distance. The beacon ticks from that direction, Soundscape's 4-region scheme:
    |bearing| <= 15 deg  bright double tick  ("on course")
    15 - 55 deg          tick from that side
    55 - 125 deg         lower tick from that side ("turn")
    > 125 deg            distinct two-tone, "behind you"
Distance is encoded by tempo (slower far away, faster when close), never by
loudness, so the beacon stays audible. It ducks under sonar warnings, goes quiet
while speech plays, and announces arrival once (with hysteresis so walking back
and forth across the threshold does not repeat it).
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Callable

import numpy as np

from . import sounds, spatial
from .config import BeaconCfg
from .mixer import Engine


REGIONS = ("on", "near", "side", "behind")      # ordered by |bearing|
_WARM: dict[tuple[str, int, int], np.ndarray] = {}   # pre-rendered at startup, never evicted


def _mono_sound(region: str, rate: int) -> np.ndarray:
    mono = {"on": sounds.beacon_on_course, "near": sounds.beacon_off_course,
            "side": lambda: sounds.knock(880, 0.08, 0.4), "behind": sounds.beacon_behind}[region]()
    if rate != sounds.RATE:
        from scipy.signal import resample_poly
        import math
        g = math.gcd(rate, sounds.RATE)
        mono = resample_poly(mono, rate // g, sounds.RATE // g).astype(np.float32)
    return mono


def _beacon_sound(region: str, bearing_q: int, rate: int) -> np.ndarray:
    hit = _WARM.get((region, bearing_q, rate))
    return hit if hit is not None else _beacon_render(region, bearing_q, rate)


@lru_cache(maxsize=512)
def _beacon_render(region: str, bearing_q: int, rate: int) -> np.ndarray:
    return spatial.render(_mono_sound(region, rate), float(bearing_q), rate)


def region_of(bearing: float, cfg: BeaconCfg) -> str:
    """Soundscape's 4-region scheme on |bearing| (see the module docstring)."""
    a = abs(spatial.wrap_deg(bearing))
    if a <= cfg.on_course_deg:
        return "on"
    if a <= cfg.near_axis_deg:
        return "near"
    if a <= cfg.behind_deg:
        return "side"
    return "behind"


def prewarm(cfg: BeaconCfg, rate: int) -> int:
    """Pre-render every (region, 5-degree bearing) pair tick() can ask for: a bearing
    rounds to bq from within +-2.5 deg, so each bq gets the regions spanned by that
    interval. One batched native render per region. Returns the number of cues."""
    want: dict[str, list[int]] = {r: [] for r in REGIONS}
    for bq in spatial.BEARINGS_5DEG:
        lo = REGIONS.index(region_of(max(0.0, abs(bq) - 2.5), cfg))
        hi = REGIONS.index(region_of(min(180.0, abs(bq) + 2.5), cfg))
        for r in REGIONS[lo:hi + 1]:
            want[r].append(bq)
    n = 0
    for region, bqs in want.items():
        if not bqs:
            continue
        for bq, st in zip(bqs, spatial.render_many(_mono_sound(region, rate), bqs, rate)):
            _WARM[(region, bq, rate)] = st
            n += 1
    return n


class Beacon:
    def __init__(self, engine: Engine, cfg: BeaconCfg, speak: Callable[..., object] | None = None,
                 log=print, stale_s: float = 10.0):
        self.engine = engine
        self.cfg = cfg
        self.speak = speak
        self.log = log
        self.stale_s = stale_s
        self.enabled = True              # wearer can switch the beacon off
        self.bearing: float | None = None
        self.distance: float | None = None
        self.label: str | None = None
        self.updated = 0.0
        self._next: int | None = None
        self._arrived = False
        self._stale_warned = False
        self.duck = 1.0                  # set by the app while sonar warns
        self.region: str | None = None

    @property
    def active(self) -> bool:
        return self.bearing is not None

    def set(self, bearing: float, distance: float | None, label: str | None, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        new_target = label is not None and label != self.label
        self.bearing = spatial.wrap_deg(bearing)
        self.distance = distance
        if label:
            self.label = label
        self.updated = now
        self._stale_warned = False
        if new_target:
            self._arrived = False
        if self._arrived and distance is not None and distance > self.cfg.arrive_dist_m * 2.5:
            self._arrived = False         # walked away again: re-arm arrival

    def clear(self) -> None:
        self.bearing = self.distance = None
        self.label = None
        self._next = None
        self.region = None

    def _period(self) -> float:
        c = self.cfg
        d = self.distance
        if d is None:
            return (c.period_far_s + c.period_near_s) / 2
        x = min(1.0, max(0.0, (d - c.near_dist_m) / max(1e-3, c.far_dist_m - c.near_dist_m)))
        return c.period_near_s + (c.period_far_s - c.period_near_s) * x

    def classify(self, bearing: float) -> str:
        return region_of(bearing, self.cfg)

    def tick(self, now: float | None = None, speech_playing: bool = False) -> None:
        now = time.monotonic() if now is None else now
        if self.bearing is None:
            return
        if now - self.updated > self.stale_s:
            if not self._stale_warned:
                self._stale_warned = True
                self.log("[beacon] no waypoint update: pausing the beacon")
            self._next = None
            return
        if self.distance is not None and self.distance <= self.cfg.arrive_dist_m and not self._arrived:
            self._arrived = True
            self.engine.play(np.repeat(sounds.arrived()[:, None], 2, axis=1), bus="cue", gain=1.0)
            if self.speak:
                self.speak(f"You have arrived{' at ' + self.label if self.label else ''}.", "nav",
                           key="arrived", ttl=10)
        if not self.enabled or self._arrived or speech_playing:
            self._next = None
            return
        region = self.classify(self.bearing)
        self.region = region
        bq = int(round(self.bearing / 5.0) * 5)
        snd = _beacon_sound(region, bq, self.engine.rate)
        step = self.engine.samples(self._period())
        if self._next is None:
            self._next = self.engine.now + self.engine.samples(0.05)
        horizon = self.engine.now + self.engine.samples(0.06)
        while self._next <= horizon:
            self.engine.play(snd, at=self._next, bus="cue", gain=0.8 * self.duck, tag="beacon")
            self._next += step
