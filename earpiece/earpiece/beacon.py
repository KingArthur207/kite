"""Audio beacon toward the next waypoint (after Microsoft Soundscape's design).

The navigation side sends the waypoint's bearing relative to the wearer and its
distance. The beacon pings from that direction, Soundscape's 4-region scheme
(apps/ios/GuideDogs/Code/Audio/Protocols/DynamicAudioEngineAsset.swift, the
4-asset selector: +-15 / 55 / 125 deg):
    |bearing| <= 15 deg  bright same-pitch double ping, C7 C7       ("on course")
    15 - 55 deg          rising pair D6 -> G6 if right, falling G6 -> D6 if left
    55 - 125 deg         lower, dull pair A5 -> D6 right / D6 -> A5 left ("turn")
    > 125 deg            three low, dull G5 pulses                    ("behind you")
Positive bearing is right (spatial.py). Every cue differs by pitch contour, rhythm
and timbre alone, so they stay distinct in mono (table speakers, one earbud),
where left/right panning collapses; the spatialisation is a redundant cue on top.
Soundscape switches assets abruptly at the region edges. Here the ping's
brightness also follows a raised-cosine curve of |bearing| (sounds.
beacon_brightness: flat on axis, 0 at 55 deg), so turning toward the waypoint
is heard as a continuous "warmer" gradient with the double ping as the on-target
accent: continuous feedback plus an on-target ping beat a lone on-target ping
for blind walkers turning (Ahmetovic et al., IJHCS 177, 2023).
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

from . import hrtf, sounds, spatial
from .config import BeaconCfg
from .mixer import Engine


REGIONS = ("on", "near", "side", "behind")      # ordered by |bearing|
BEHIND_MIN_PERIOD_S = 0.9
_WARM: dict[tuple[str, int, int], np.ndarray] = {}   # pre-rendered at startup, never evicted


def _mono_sound(region: str, bearing_q: int, rate: int) -> np.ndarray:
    """The beacon cue for (region, 5-degree bearing): brightness depends on |bearing|, the
    contour (rising right / falling left) on its sign."""
    mono = sounds.beacon_tick(region, float(bearing_q))
    if rate != sounds.RATE:
        from scipy.signal import resample_poly
        import math
        g = math.gcd(rate, sounds.RATE)
        mono = resample_poly(mono, rate // g, sounds.RATE // g).astype(np.float32)
    return mono


def _beacon_sound(region: str, bearing_q: int, rate: int) -> np.ndarray:
    hit = _WARM.get((region, bearing_q, rate))
    return hit if hit is not None else _beacon_render(region, bearing_q, rate)


def _use_hrtf(rate: int) -> bool:
    """KITE_HRTF=kemar and the KEMAR set is already loaded (prewarm loads it, off the tick)."""
    return hrtf.enabled() and hrtf.ready(rate)


def _place_many(mono: np.ndarray, bqs: list[int], rate: int) -> list[np.ndarray]:
    if _use_hrtf(rate):
        return [hrtf.render(mono, float(bq), rate) for bq in bqs]
    return spatial.render_many(mono, bqs, rate)


@lru_cache(maxsize=512)
def _beacon_render(region: str, bearing_q: int, rate: int) -> np.ndarray:
    mono = _mono_sound(region, bearing_q, rate)
    if _use_hrtf(rate):
        return hrtf.render(mono, float(bearing_q), rate)
    return spatial.render(mono, float(bearing_q), rate)


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


def prewarm(cfg: BeaconCfg, rate: int, log=print) -> int:
    """Pre-render every (region, 5-degree bearing) pair tick() can ask for: a bearing
    rounds to bq from within +-2.5 deg, so each bq gets the regions spanned by that
    interval. One batched native render per (region, |bearing|), since the mono cue
    depends only on those. With KITE_HRTF=kemar this is also where the KEMAR set is
    loaded (downloaded on first use), never on the audio tick. Returns the cue count."""
    if hrtf.enabled() and not hrtf.ready(rate) and hrtf.ensure(rate, log=log):
        _WARM.clear()
        _beacon_render.cache_clear()
    want: dict[str, list[int]] = {r: [] for r in REGIONS}
    for bq in spatial.BEARINGS_5DEG:
        lo = REGIONS.index(region_of(max(0.0, abs(bq) - 2.5), cfg))
        hi = REGIONS.index(region_of(min(180.0, abs(bq) + 2.5), cfg))
        for r in REGIONS[lo:hi + 1]:
            want[r].append(bq)
    n = 0
    for region, bqs in want.items():
        for mag, right in sorted({(abs(bq), bq >= 0) for bq in bqs}):
            group = [bq for bq in bqs if abs(bq) == mag and (bq >= 0) == right]
            for bq, st in zip(group, _place_many(_mono_sound(region, group[0], rate), group, rate)):
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
        period = self._period()
        if region == "behind":
            # "turn around" carries no distance; repeating a 275 ms cue every 0.45 s would be
            # 60 % duty cycle and pure nagging (BBeep, CHI 2019: denser alerts annoyed more
            # without changing behaviour), so it never repeats faster than BEHIND_MIN_PERIOD_S.
            period = max(period, BEHIND_MIN_PERIOD_S)
        step = self.engine.samples(period)
        if self._next is None:
            self._next = self.engine.now + self.engine.samples(0.05)
        horizon = self.engine.now + self.engine.samples(0.06)
        while self._next <= horizon:
            self.engine.play(snd, at=self._next, bus="cue", gain=0.8 * self.duck, tag="beacon")
            self._next += step
