"""Lag-compensated yaw sources for kite.follow: FlySteer (the fruit-fly pursuit circuit, DESIGN.md section 3) and
PredPD (the classical PRED-PD ablation around exactly the same filter and stick conditioning, DESIGN.md R7).

    cfg = FollowCfg(behind_m=1.0, above_m=0.30, max_yaw=70)
    fly = make_fly_steer("FLY-YAW-PREV", cfg=cfg)     # or FlySteer.from_file(path, cfg=cfg, size_blind=True)
    pd = PredPD.from_x(x, cfg=cfg)                    # or PredPD(cfg=cfg): the follow law's gains
    ctrl = FollowController(cfg, camera, max_rc, yaw_source=fly)
    ... FollowController.update() calls fly.yaw(t, track.t, bearing_deg, range_m, fallback) while following

The geometry (behind_m, the standoff where s = 1) and the believed video lag come from the controller's own
FollowCfg instance (`cfg`), so the source and the controller can never disagree; explicit behind_m / video_lag_s
must match it (they are for callers without a cfg).

Per call (kite.record runs the controller at 10 Hz, but any rate works):
1. Measure. A new frame (a new `frame_t`) adds its bearing and range to a constant-velocity Kalman filter at the
   frame's capture time, frame_t - video_lag_s. Outliers (> 5 sigma) are gated; three in a row restart the filter.
2. Tick in DT = 0.05 s steps of the caller's clock up to `t`, at most MAX_TICKS_PER_CALL per call. After a longer
   gap (the controller was coasting) the backlog is dropped, not caught up. Each tick sees the filter predicted to
   that tick's time, and the last one of the call to `t` itself, so the source steers on where the walker is now,
   not where the late frame showed it. No frame for more than HOLD_S: no target.
3. The raw stick of the tick. FlySteer: encode s = behind_m / range (or 1 when size_blind), run 50 ms of the LIF
   network, read the yaw out (with no target the DNs relax and the readout returns to its bias). PredPD:
   kp * deadband(theta_hat) + kd * dtheta_hat (0 with no target).
4. Condition the stick per tick: slew limit SLEW per second, soft deadband DEADBAND, then hysteresis (the output
   moves only when the new value is more than HYSTERESIS away, or 0): this removes the spike-noise dither of a
   one-cell-per-side readout without adding lag to real turns.
5. Fail safe: any exception (or a non-finite stick) -> `fallback`, kite's own law, for the rest of the episode
   (until the next reset); it is logged once. reset() never raises, and it is cheap: FollowController calls it on
   every lost -> following resume.

Signs: bearing + = walker right of the image centre; the returned stick + = clockwise = turn right (Tello rc yaw).
Deterministic for a given seed: the only randomness is the brain's Poisson input, seeded by reset(seed) (the
simulator passes the episode seed). The seed is remembered, so the argument-less reset() of FollowController keeps it.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections import deque
from pathlib import Path
from typing import Callable

import numpy as np

from ..follow import FollowCfg
from .brain import PursuitBrain
from .decoder import YawReadout, soft_deadzone
from .encoder import TargetEncoder
from .interfaces import (BRAIN_FILE, BRAIN_TICK_MS, DT, NO_TARGET, OUTPUT_GROUPS, PREV_PARAMS, TRAINED_FILE,
                         TargetFeatures)
from .params import HAND_FILE, PD_SPACE, SPACE, load_params, norm_array

log = logging.getLogger("kite.fly")

ARM_FILES = {"FLY-YAW-HAND": HAND_FILE, "FLY-YAW-PREV": PREV_PARAMS, "FLY-YAW": TRAINED_FILE}
KITE_BEHIND_M = 1.0  # the kite spec's standoff, for callers without a FollowCfg
MAX_TICKS_PER_CALL = 3
WARMUP_S = 1.0  # brain run with no target at every reset
HOLD_S = 0.5  # no new measurement for longer than this: no target input
DEADBAND = 4.0  # stick units, soft (no jump at its edge)
HYSTERESIS = 3.0
SLEW = 300.0  # stick units per second
MAX_PRED_DEG = 60.0  # the predicted bearing is clamped here (a wild velocity estimate cannot run away)
MIN_RANGE_M = 0.2


def follow_geometry(cfg: FollowCfg | None, behind_m: float | None, video_lag_s: float | None) -> tuple[float, float]:
    """(behind_m, video_lag_s) from the controller's FollowCfg; explicit values must agree with it."""
    if cfg is None:
        return (KITE_BEHIND_M if behind_m is None else float(behind_m),
                FollowCfg.video_lag_s if video_lag_s is None else float(video_lag_s))
    for name, v in (("behind_m", behind_m), ("video_lag_s", video_lag_s)):
        if v is not None and not math.isclose(float(v), getattr(cfg, name)):
            raise ValueError(f"{name}={v} disagrees with the FollowCfg's {getattr(cfg, name)}: pass cfg only")
    return float(cfg.behind_m), float(cfg.video_lag_s)


class CVFilter:
    """Independent constant-velocity Kalman filters on a few scalars measured together at capture times.

    q: white-acceleration spectral density per axis (unit^2 / s^3); r_abs, r_rel: measurement noise sigma per axis,
    absolute plus relative to the value; v0: initial velocity sigma. A measurement more than `gate` sigma off on any
    axis is rejected; `reinit` rejections in a row restart the filter there (the tracker switched walkers)."""

    def __init__(self, q: tuple[float, ...], r_abs: tuple[float, ...], r_rel: tuple[float, ...],
                 v0: tuple[float, ...], gate: float = 5.0, reinit: int = 3):
        self.q, self.r_abs, self.r_rel, self.v0 = q, r_abs, r_rel, v0
        self.gate2, self.reinit = gate * gate, reinit
        self.n = len(q)
        self.reset()

    def reset(self) -> None:
        self.ready = False
        self.t = 0.0  # capture time the state refers to
        self.x = [0.0] * (2 * self.n)  # per axis: value, rate
        self.P = [0.0] * (3 * self.n)  # per axis: var value, cov, var rate
        self.rejected_in_row = 0
        self.fused = 0
        self.rejected = 0

    def _r2(self, j: int, value: float) -> float:
        return (self.r_abs[j] + self.r_rel[j] * abs(value)) ** 2

    def start(self, tc: float, z: tuple[float, ...]) -> None:
        for j, zj in enumerate(z):
            self.x[2 * j:2 * j + 2] = [zj, 0.0]
            self.P[3 * j:3 * j + 3] = [self._r2(j, zj), 0.0, self.v0[j] ** 2]
        self.t, self.ready, self.rejected_in_row = tc, True, 0
        self.fused += 1

    def update(self, tc: float, z: tuple[float, ...]) -> bool:
        """Fuse one measurement taken at capture time tc. False if it was gated out."""
        if not self.ready:
            self.start(tc, z)
            return True
        dt = max(0.0, tc - self.t)
        pred, outlier = [], False
        for j, zj in enumerate(z):
            p, v = self.x[2 * j], self.x[2 * j + 1]
            a, b, c = self.P[3 * j:3 * j + 3]
            q = self.q[j]
            p += v * dt
            a += 2.0 * b * dt + c * dt * dt + q * dt ** 3 / 3.0
            b += c * dt + q * dt * dt / 2.0
            c += q * dt
            s = a + self._r2(j, p)
            y = zj - p
            outlier |= y * y > self.gate2 * s
            pred.append((p, v, a, b, c, s, y))
        if outlier:
            self.rejected += 1
            self.rejected_in_row += 1
            if self.rejected_in_row >= self.reinit:
                self.start(tc, z)
                return True
            return False
        for j, (p, v, a, b, c, s, y) in enumerate(pred):
            k1, k2 = a / s, b / s
            self.x[2 * j:2 * j + 2] = [p + k1 * y, v + k2 * y]
            self.P[3 * j:3 * j + 3] = [(1.0 - k1) * a, (1.0 - k1) * b, c - k2 * b]
        self.t, self.rejected_in_row = tc, 0
        self.fused += 1
        return True

    def predict(self, t: float) -> list[tuple[float, float]]:
        """(value, rate) per axis at time t."""
        dt = t - self.t
        return [(self.x[2 * j] + self.x[2 * j + 1] * dt, self.x[2 * j + 1]) for j in range(self.n)]


def lag_filter() -> CVFilter:
    """Bearing (deg) and range (m). Bearing: about 3 px of box-centre noise at fx 920 and the drone's own yaw
    rate as the fast dynamics; range: 3 % noise and walker accelerations of about 1 m/s^2."""
    return CVFilter(q=(400.0, 1.0), r_abs=(0.2, 0.01), r_rel=(0.0, 0.03), v0=(25.0, 1.0))


class LagCompensatedYaw:
    """kite.follow.YawSource skeleton shared by FlySteer and PredPD: the latency filter, the tick schedule, the stick
    conditioning, the fail-safe and the stats. Subclasses give _raw_yaw(features) per tick and _reset_inner()."""

    arm: str | None = None

    def __init__(self, *, cfg: FollowCfg | None = None, behind_m: float | None = None,
                 video_lag_s: float | None = None, size_blind: bool = True, seed: int = 0, arm: str | None = None,
                 source: str | None = None, log_fn: Callable[[str], None] | None = None):
        self.cfg = cfg
        self.behind_m, self.video_lag_s = follow_geometry(cfg, behind_m, video_lag_s)
        self.size_blind = bool(size_blind)
        self.seed = int(seed)
        self.arm = arm or self.arm
        self.source = source  # the parameter file, for stats
        self._log = log_fn or log.warning
        self.filter = lag_filter()

    # ------------------------------------------------------------------ YawSource
    def reset(self, seed: int | None = None) -> None:
        """New episode (or a resume after a loss): fresh state (seed None keeps the last seed), empty filter,
        conditioning at 0, fallback cleared. Never raises: a failure here puts the episode on the fallback."""
        self.failed: str | None = None
        try:
            self._frame_t: float | None = None
            self._meas_t: float | None = None  # arrival time (frame_t) of the last frame with a measurement
            self._t_next: float | None = None
            self._t_last: float | None = None
            self._ticks = 0
            self._resyncs = 0
            self._tick_ms: deque[float] = deque(maxlen=500)
            self._yaw_raw = self._yaw_slew = self._yaw = 0.0
            self._valid = False
            self._features = NO_TARGET
            if seed is not None:
                self.seed = int(seed)
            self.filter.reset()
            self._reset_inner()
        except Exception as e:  # noqa: BLE001 (the yaw source must never take follow down)
            self._fail(e)

    def yaw(self, t: float, frame_t: float, bearing_deg: float, range_m: float, fallback: float) -> float:
        """The yaw stick for now (+ = turn right). `fallback` (the law's value) after any failure."""
        if self.failed is not None:
            return fallback
        try:
            t = float(t)
            if not math.isfinite(t):
                raise ValueError(f"non-finite time {t}")
            if frame_t is not None and math.isfinite(frame_t) and frame_t != self._frame_t:
                self._measure(float(frame_t), float(bearing_deg), float(range_m))
            ticks = self._due_ticks(t)
            for k, tick_t in enumerate(ticks):
                # the network runs on its 50 ms grid, but the stick returned now is for now: the last tick of the
                # call sees the walker predicted to t, so the caller's period jitter adds no lag
                self._tick(t if k == len(ticks) - 1 else tick_t)
            if not math.isfinite(self._yaw):
                raise FloatingPointError(f"non-finite yaw stick {self._yaw}")
            return self._yaw
        except Exception as e:  # noqa: BLE001
            self._fail(e)
            return fallback

    # ------------------------------------------------------------------ for subclasses
    def _reset_inner(self) -> None:
        pass

    def _raw_yaw(self, f: TargetFeatures) -> float:
        raise NotImplementedError

    def _extra_stats(self) -> dict:
        return {}

    # ------------------------------------------------------------------ inside
    def _fail(self, e: BaseException) -> None:
        self.failed = f"{type(e).__name__}: {e}"
        self._log(f"kite.fly: {type(self).__name__} ({self.arm}) failed ({self.failed}); "
                  f"using the follow law's yaw for the rest of the episode")

    def _measure(self, frame_t: float, bearing_deg: float, range_m: float) -> None:
        if not (math.isfinite(bearing_deg) and math.isfinite(range_m)):
            raise ValueError(f"non-finite measurement: bearing {bearing_deg}, range {range_m}")
        self._frame_t = frame_t
        if self._meas_t is not None and frame_t - self._meas_t > HOLD_S:
            self.filter.ready = False  # the old estimate is stale: start over rather than bridge the gap
        self.filter.update(frame_t - self.video_lag_s, (bearing_deg, max(MIN_RANGE_M, range_m)))
        self._meas_t = frame_t  # a gated outlier still means the walker is in view: it refreshes the hold

    def _due_ticks(self, t: float) -> list[float]:
        """Times of the 50 ms ticks due by t (at most MAX_TICKS_PER_CALL, the most recent ones)."""
        if self._t_next is None or (self._t_last is not None and t < self._t_last):
            self._t_next = t  # first call, or the clock jumped back: start the schedule here
        self._t_last = t
        due = math.floor((t - self._t_next) / DT + 1e-6) + 1 if t >= self._t_next - 1e-9 else 0
        if due > MAX_TICKS_PER_CALL:  # a long gap: drop the backlog instead of catching up
            self._t_next += (due - MAX_TICKS_PER_CALL) * DT
            self._resyncs += 1
            due = MAX_TICKS_PER_CALL
        times = [self._t_next + k * DT for k in range(due)]
        self._t_next += due * DT
        return times

    def features(self, t: float) -> TargetFeatures:
        """What the controller sees at time t: the filter predicted to t, or no target."""
        if not self.filter.ready or self._meas_t is None or t - self._meas_t > HOLD_S:
            return NO_TARGET
        (b, vb), (r, vr) = self.filter.predict(t)
        b = max(-MAX_PRED_DEG, min(MAX_PRED_DEG, b))
        if self.size_blind:
            s, ds = 1.0, 0.0
        else:
            r = max(MIN_RANGE_M, r)
            s, ds = self.behind_m / r, -self.behind_m * vr / (r * r)
        return TargetFeatures(True, math.radians(b), s, math.radians(vb), ds)

    def _tick(self, tick_t: float) -> None:
        t0 = time.perf_counter()
        self._features = f = self.features(tick_t)
        self._valid = f.valid
        y = self._raw_yaw(f)
        if not math.isfinite(y):
            raise FloatingPointError(f"non-finite raw yaw {y}")
        self._yaw_raw = y
        step = SLEW * DT
        self._yaw_slew += max(-step, min(step, y - self._yaw_slew))
        y = soft_deadzone(self._yaw_slew, DEADBAND)
        if y == 0.0 or abs(y - self._yaw) > HYSTERESIS:
            self._yaw = y
        self._ticks += 1
        self._tick_ms.append(1000.0 * (time.perf_counter() - t0))

    @property
    def stats(self) -> dict:
        ms = np.asarray(self._tick_ms, float)
        f = self._features
        return {
            "arm": self.arm, "source": self.source, "size_blind": self.size_blind, "seed": self.seed,
            "behind_m": self.behind_m, "video_lag_s": self.video_lag_s,
            "ticks": self._ticks,
            "tick_ms_mean": round(float(ms.mean()), 3) if ms.size else None,
            "tick_ms_p95": round(float(np.percentile(ms, 95)), 3) if ms.size else None,
            "resyncs": self._resyncs,
            "fused": self.filter.fused, "rejected": self.filter.rejected,
            "target_valid": self._valid,
            "bearing_pred_deg": round(math.degrees(f.theta), 2) if f.valid else None,
            "s": round(f.s, 3) if f.valid else None,
            "yaw_raw": round(self._yaw_raw, 2), "yaw": round(self._yaw, 2),
            **self._extra_stats(),
            "failed": self.failed,
        }


class FlySteer(LagCompensatedYaw):
    """The fly brain as the yaw source. See the module docstring for the pipeline, units and signs."""

    def __init__(self, params: dict[str, float], norm: dict | np.ndarray, *, cfg: FollowCfg | None = None,
                 size_blind: bool = True, behind_m: float | None = None, video_lag_s: float | None = None,
                 seed: int = 0, arm: str | None = None, brain_file: str | Path = BRAIN_FILE, source: str | None = None,
                 log_fn: Callable[[str], None] | None = None):
        """params: the 34 values (clipped into their bounds); norm: the readout normalization they belong to
        ({DN: Hz} or 10 rates); cfg: the controller's FollowCfg (behind_m and video_lag_s come from it)."""
        super().__init__(cfg=cfg, behind_m=behind_m, video_lag_s=video_lag_s, size_blind=size_blind, seed=seed,
                         arm=arm, source=source, log_fn=log_fn)
        self.params = SPACE.decode(SPACE.encode(params))
        self.encoder = TargetEncoder(self.params)
        self.readout = YawReadout(self.params, norm_array(norm))
        self.brain = PursuitBrain(brain_file, seed=self.seed)
        self._rest = self.encoder.rest()
        self.reset()

    @classmethod
    def from_file(cls, path: str | Path, cfg: FollowCfg | None = None, size_blind: bool | None = None,
                  behind_m: float | None = None, video_lag_s: float | None = None, seed: int = 0, **kw) -> FlySteer:
        """From a parameter file (kite.fly.params.load_params). size_blind None: the file's own mode, else True."""
        pf = load_params(path)
        sb = pf.size_blind if size_blind is None else size_blind
        kw.setdefault("arm", pf.arm)
        return cls(pf.params, pf.norm, cfg=cfg, size_blind=True if sb is None else sb, behind_m=behind_m,
                   video_lag_s=video_lag_s, seed=seed, source=str(pf.path), **kw)

    def _reset_inner(self) -> None:
        self.last_channels = self._rest
        self.last_rates = np.zeros(len(OUTPUT_GROUPS))
        self.brain.reset(self.seed)
        self.readout.reset()
        for _ in range(round(WARMUP_S / DT)):  # 1 s with no target (free: the network sits exactly at rest)
            self.last_rates = self.brain.tick(self._rest, BRAIN_TICK_MS)
            self.readout.filter(self.last_rates, DT)

    def _raw_yaw(self, f: TargetFeatures) -> float:
        self.last_channels = self.encoder.channels(f)
        self.last_rates = self.brain.tick(self.last_channels, BRAIN_TICK_MS)
        return self.readout.update(self.last_rates, DT)

    @property
    def dn_rates(self) -> dict[str, float]:
        """The last tick's DN rates (Hz) by OUTPUT_GROUPS name."""
        return dict(zip(OUTPUT_GROUPS, self.last_rates.tolist()))

    def _extra_stats(self) -> dict:
        return {"dn_hz": {k: round(v, 1) for k, v in self.dn_rates.items()}}


class PredPD(LagCompensatedYaw):
    """PRED-PD: yaw = kp * deadband(theta_hat_deg, deadband_deg) + kd * dtheta_hat_deg_s, where theta_hat and its rate
    come from FlySteer's latency filter predicted to the tick, then FlySteer's stick conditioning. The ablation that
    separates "the fly helps" from "prediction helps". Parameters: params.PD_SPACE. Its defaults borrow the law's kp
    (1.5) and deadband (2 deg) with kd 0.1, but they do NOT reproduce the law: the shared conditioning adds a 4-stick
    deadband and 3-stick hysteresis on top, so the untuned PD under-steers small errors. It is meant to be CMA-tuned."""

    arm = "PRED-PD"

    def __init__(self, params: dict[str, float] | None = None, *, cfg: FollowCfg | None = None,
                 behind_m: float | None = None, video_lag_s: float | None = None, seed: int = 0,
                 arm: str | None = None, source: str | None = None, log_fn: Callable[[str], None] | None = None):
        super().__init__(cfg=cfg, behind_m=behind_m, video_lag_s=video_lag_s, seed=seed, arm=arm, source=source,
                         log_fn=log_fn)
        self.params = PD_SPACE.decode(PD_SPACE.encode(params or PD_SPACE.defaults()))
        self.kp, self.kd, self.deadband_deg = (self.params[k] for k in ("kp", "kd", "deadband_deg"))
        self.reset()

    @classmethod
    def from_x(cls, x: np.ndarray, **kw) -> PredPD:
        """From a point of PD_SPACE (x in [0, 1]^3, clipped)."""
        return cls(PD_SPACE.decode(x), **kw)

    @classmethod
    def from_file(cls, path: str | Path, **kw) -> PredPD:
        """From a JSON with `params` ({kp, kd, deadband_deg}) or `x`."""
        rec = json.loads(Path(path).read_text(encoding="utf-8"))
        params = rec["params"] if isinstance(rec.get("params"), dict) else PD_SPACE.decode(rec["x"])
        return cls(params, source=str(path), **kw)

    def _raw_yaw(self, f: TargetFeatures) -> float:
        if not f.valid:
            return 0.0
        return self.kp * soft_deadzone(math.degrees(f.theta), self.deadband_deg) + self.kd * math.degrees(f.dtheta)

    def _extra_stats(self) -> dict:
        return {"pd": {k: round(v, 4) for k, v in self.params.items()}}


def make_fly_steer(arm: str, **kw) -> FlySteer:
    """FlySteer for one of the fly arms: FLY-YAW-HAND, FLY-YAW-PREV or FLY-YAW (kw: FlySteer.from_file's, e.g. cfg)."""
    if arm not in ARM_FILES:
        raise ValueError(f"not a fly arm: {arm!r} (expected one of {', '.join(ARM_FILES)})")
    path = Path(ARM_FILES[arm])
    if not path.is_file():
        hint = "python -m kite.fly.calibrate" if arm == "FLY-YAW-HAND" else "kite.fly.train (only if it passes G3)"
        raise FileNotFoundError(f"{arm}: no parameter file at {path} (written by {hint})")
    return FlySteer.from_file(path, arm=arm, **kw)


def make_hand() -> FlySteer:
    """Zero-argument factories ("kite.fly.steer:make_prev") for callers that build one yaw source per episode; they
    use the kite spec geometry (behind_m 1.0, FollowCfg's video_lag_s)."""
    return make_fly_steer("FLY-YAW-HAND")


def make_prev() -> FlySteer:
    return make_fly_steer("FLY-YAW-PREV")


def make_trained() -> FlySteer:
    return make_fly_steer("FLY-YAW")


def make_pd() -> PredPD:
    return PredPD()
