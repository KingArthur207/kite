"""Tuned follow behind: kite.follow_behind's FollowBehind with the knobs kite.fly.behind trains (its DESIGN.md,
section 2).
The flown follow (kite/follow_behind.py) stays exactly as it is; this module only adds to it.

    cfg = load_model("kite/fly/data/behind/v3-joint.json")      # a trained model file -> TunedCfg
    mission = TunedFollowMission(drone, Perception(cfg, PoseDetector(), camera), cfg, log)

Why the seen branch of FollowBehind.step is copied and not wrapped: the new knobs sit INSIDE the forward law (the cap,
the feed-forward, the slew before the too-close override), so they cannot be bolted on from outside. That branch is
kept line for line with the flown one. The not-seen branches have no new knob: they are FollowBehind.step itself (only
_climb_rc is overridden), so a change to them in the flown follow is followed here without a copy to update. With
TunedCfg() defaults it sends bit-identical rc to FollowBehind on any input (tests/test_follow_tuned.py drives both
through the same 60 s of scripted perception and compares every rc: it fails the day the two seen branches drift).

What the knobs add (all off at their defaults):
    speed only when needed  the flown forward stick is capped at max_rc_forward (35 ~ 0.34 m/s): a person walking
                            faster outwalks the drone. Here the cap rises above max_rc_forward ONLY through a gate: the
                            person measured walking away faster than fwd_cap_v0_mps WITH the range not shrinking (the
                            drone at its present speed is not gaining on them), both for fwd_cap_dwell_s on end. Then
                            it is max_rc_forward + fwd_cap_per_mps per m/s above fwd_cap_v0_mps, never above
                            min(fwd_cap_fast, FWD_CAP_LIMIT = 63 ~ 0.6 m/s). The gate shuts the moment the estimate is
                            at or below fwd_cap_v0_mps (or unknown): speeding up needs evidence, slowing down does not.
                            A slow or stopped person keeps the flown cap, and a drone closing in on a slow walker can
                            never open the gate, however noisy the estimate.
    person speed            v_away = range rate + own speed (+ = walking away). The range rate is the slope of the
                            filtered range over ~1 s of frames that are video_lag_s + their YOLO age + the median's own
                            delay old, so the own speed is taken over the SAME stretch of the past, as the drone really
                            moved then: the forward stick that reached the drone (clamped to the governor's max_rc when
                            the drone is a SafetyGovernor, 0 while it holds a safety hover) through a nominal Tello
                            response (dead time own_dead_s, first-order lag own_tau_s, RC_PER_MPS), averaged with the
                            weights a least-squares slope gives its window. (The flown approach feed-forward's own
                            "mean stick 0.3-1.3 s ago" leads the range by ~1 s: the stick rises long before the range
                            shows the drone moving, which reads a slow walker as fast and feeds itself.) Low-passed
                            with walker_speed_tau_s from 0 (unknown = standing). None when the range rate is None (a
                            dropout, a fresh lock): then the flown cap and no feed-forward.
    feed-forward            too far (e > 0, the flown approach gates hold): + fwd_ff_gain * max(0, v_away) * RC_PER_MPS,
                            so the drone matches a walking person's speed without a big distance error, within the cap
                            above. At or inside the distance (e <= 0) it is exactly the flown law: nothing pushes it
                            closer.
    fwd_slew                speed-ups of the forward stick are limited to this many stick per second. Slowing down,
                            braking for a person who stops and backing away are never slowed (the too-close back-off is
                            applied after it in any case).
    approach_model_w        the flown approach feed-forward (back away at their speed when they come closer faster
                            than approach_rate_mps) judges "their speed" with the misaligned mean stick above; this
                            blends in the person-speed model's (1 = only it), so a fast drone braking for a person who
                            stopped is not read as the person walking at it.
    approach_tau_s          that closing speed low-passed before the comparison. The camera tilts with the drone and
                            the app corrects the 0.3 s old frame with the current telemetry pitch; at 0.2 m above the
                            head 1 deg is ~16 % of range, so a back-off's own pitch reads as the person coming closer
                            and the flown feed-forward can sustain a ~1 Hz fore-aft cycle. The too-close override is
                            never filtered.
    backoff_climb_m         the back-off climbs (too close, lost while close) stop at altitude_m + this (flown 0.25):
                            a smaller value keeps the drone off the ceiling.

The governor. A raised cap only means something if the SafetyGovernor passes it: kite.config.SafetyCfg's max_rc is
50 (~0.48 m/s), so a model that wants 0.6 m/s needs SafetyCfg(max_rc >= its "governor"."needs_max_rc"), the number
save_model writes (required_max_rc: the largest stick the follow can ask for on any axis). kite.fly.behind trains
under max_rc 65 ("trained_max_rc"). The flown follow never asks for more than 50.

Model files (kite/fly/data/behind/<name>.json): {"name", "arm", "x", "brain", "follow", "steering", "governor",
"trained", "signoff"}. "arm" / "x" / "brain" make a fly model a valid FlySteer params file, so the app's brain loader
reads the brain straight from it (cfg.fly_params_path = the model file). A law model has no brain and steers with the
yaw law.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections import deque
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .follow_behind import FLY_PARAMS, RUNNING, BehindCfg, Ctx, FollowBehind, FollowMission, clamp, deadband

RC_PER_MPS = 104.0  # stick per m/s at the nominal Tello forward gain (0.96 m/s per 100)
FWD_CAP_LIMIT = 63  # ~0.6 m/s: never above this, whatever a model file says
SMOOTH2_PATH: Path = FLY_PARAMS  # the flown brain (smooth2): a follow-only model copies its x
BRAIN_ARM = "FLY-YAW"  # the 34-number yaw-only fly brain the models train (flyfollow.rl.params.param_space)
DEFAULT_BRAIN = "pursuit_core1.npz"
MODEL_DIR = Path(__file__).resolve().parent / "fly" / "data" / "behind"
NOMINAL_TICK_S = 0.05  # run/pilot.py's control tick: the first slew / speed-filter step assumes it
RANGE_MEDIAN_S = 0.1  # the delay of FollowBehind._range (median of the last 5 estimates, ~2 control ticks)
RATE_WINDOW_S = 1.0  # FollowBehind._range_rate's window
OWN_HIST = 80  # samples of the own-speed model kept (4 s at 20 Hz: the rate window + its lag + the dead time)
STEERINGS = ("fly", "pid")
_LOADER_FIELDS = ("steering", "fly_params_path", "model_name")  # set by load_model, never by a model's "follow"
GOVERNOR_KEY = "governor"  # a model file's governor requirement: {"needs_max_rc": ..., "trained_max_rc": ...}


@dataclass
class TunedCfg(BehindCfg):
    """BehindCfg plus the trained forward and climb knobs. Every BehindCfg field keeps its meaning and default; the
    defaults below reproduce the flown follow exactly (the person-speed model fields only feed the knobs, which are
    off)."""

    fwd_cap_fast: int = 35  # forward stick cap when the person walks fast (hard limit FWD_CAP_LIMIT ~ 0.6 m/s)
    fwd_cap_v0_mps: float = 0.35  # person speed (m/s away) at which the cap starts to rise above max_rc_forward
    fwd_cap_per_mps: float = 0.0  # cap rise (stick) per m/s of person speed above fwd_cap_v0_mps; 0 = flown cap
    fwd_ff_gain: float = 0.0  # feed-forward when too far: + fwd_ff_gain * person speed away (m/s) * RC_PER_MPS
    walker_speed_tau_s: float = 0.5  # time constant of the person-speed low-pass
    fwd_slew: float = 0.0  # max forward stick INCREASE per second, 0 = off (slowing down is never limited)
    backoff_climb_m: float = 0.25  # how far above altitude_m the back-off climbs may go (flown: 0.25)
    # the gate of the raised cap: faster than fwd_cap_v0_mps with the range not shrinking, this long on end
    fwd_cap_dwell_s: float = 1.0
    # the approach feed-forward's closing speed: 0 = flown (range rate + the mean stick 0.3-1.3 s ago / 100), 1 = the
    # person-speed model's (range rate + the own speed over the same past), in between a blend
    approach_model_w: float = 0.0
    # ... low-passed with this time constant before it is compared with approach_rate_mps (0 = off, flown)
    approach_tau_s: float = 0.0
    # the own-speed model of the person-speed estimate: the Tello's forward response (lag-test medians)
    own_dead_s: float = 0.47
    own_tau_s: float = 0.45
    model_name: str = ""  # set by load_model

    def check(self) -> None:
        super().check()
        if not self.walker_speed_tau_s > 0:
            raise ValueError(f"walker_speed_tau_s must be > 0, got {self.walker_speed_tau_s}")
        if not self.own_tau_s > 0:
            raise ValueError(f"own_tau_s must be > 0, got {self.own_tau_s}")
        for name in ("fwd_slew", "backoff_climb_m", "fwd_cap_per_mps", "fwd_ff_gain", "fwd_cap_dwell_s",
                     "own_dead_s", "approach_model_w", "approach_tau_s"):
            v = getattr(self, name)
            if not (math.isfinite(v) and v >= 0):
                raise ValueError(f"{name} must be a finite value >= 0, got {v}")
        for name in ("max_rc_forward", "fwd_cap_fast"):
            if getattr(self, name) > FWD_CAP_LIMIT:
                raise ValueError(f"{name} must be <= FWD_CAP_LIMIT ({FWD_CAP_LIMIT}), got {getattr(self, name)}")

    def forward_cap(self, v_away: float | None) -> float:
        """The forward stick cap for a person walking away at v_away m/s (None = unknown, or the gate is shut).
        Exactly max_rc_forward (at most FWD_CAP_LIMIT) unless they are faster than fwd_cap_v0_mps; then at most
        min(fwd_cap_fast, FWD_CAP_LIMIT), and never below max_rc_forward (a fwd_cap_fast under it cannot make the
        follow slower than flown)."""
        base = min(self.max_rc_forward, FWD_CAP_LIMIT)
        if v_away is None or self.fwd_cap_per_mps <= 0 or v_away <= self.fwd_cap_v0_mps:
            return base
        rise = base + self.fwd_cap_per_mps * (v_away - self.fwd_cap_v0_mps)
        return float(min(FWD_CAP_LIMIT, max(base, min(self.fwd_cap_fast, rise))))


def as_tuned(cfg: BehindCfg) -> TunedCfg:
    """cfg itself when it is a TunedCfg, else a TunedCfg with the same BehindCfg values and the flown defaults."""
    if isinstance(cfg, TunedCfg):
        return cfg
    return TunedCfg(**{f.name: getattr(cfg, f.name) for f in fields(BehindCfg)})


def required_max_rc(cfg: BehindCfg) -> int:
    """The largest stick this follow can ask for on any axis: the SafetyGovernor's max_rc must be at least this
    for it to fly as it was trained (the forward cap it can raise to, the back-off and its side step, the climb,
    the lateral orbit, the yaw)."""
    c = as_tuned(cfg)
    fwd = c.forward_cap(math.inf) if c.fwd_cap_per_mps > 0 else min(c.max_rc_forward, FWD_CAP_LIMIT)
    yaw = max(c.max_rc_yaw, c.fly_max_rc_yaw if c.steering == "fly" else 0, c.search_yaw_rc)
    return int(math.ceil(max(fwd, c.max_rc_backoff, c.max_rc_up, c.max_rc_lateral, yaw, c.creep_rc)))


# ------------------------------------------------------------------ the behaviour
class TunedFollowBehind(FollowBehind):
    """FollowBehind with TunedCfg's knobs. step(): FollowBehind.step's seen branch line for line plus the speed
    estimate, the gated raised cap, the feed-forward and the slew in the forward law; the not-seen branches are
    FollowBehind.step's own. _climb_rc uses backoff_climb_m (in both). Extra state for logs and the simulator:
    walker_speed_mps (latest person-speed estimate, + = away, None when unknown), fast_gate (the raised cap's gate
    is open) and fwd_cap_rc (the forward cap of the last forward law)."""

    def __init__(self, cfg: BehindCfg, yaw_source=None, warn: Callable[[str], None] | None = None):
        super().__init__(as_tuned(cfg), yaw_source, warn)
        self._reset_tuned()

    def _reset_tuned(self) -> None:
        self.walker_speed_mps: float | None = None
        self.fast_gate = False
        self.fwd_cap_rc: float = float(min(self.c.max_rc_forward, FWD_CAP_LIMIT))
        self._v_t: float | None = None  # time of the last speed-filter update
        self._fast_since: float | None = None  # the gate's condition has held since then
        self._fwd_f = 0.0  # forward stick the slew limits from (float: a slow slew must not stall on int())
        self._fwd_t: float | None = None  # time it was sent
        self._closing_f: float | None = None  # the low-passed closing speed (approach_tau_s)
        self._closing_t = 0.0
        # the own-speed model: forward sticks that reached the drone, and the modeled forward speed they gave
        self._cmd: deque = deque(maxlen=OWN_HIST)  # (t, stick passed on)
        self._own: deque = deque(maxlen=OWN_HIST)  # (t, modeled speed m/s)
        self._own_v = 0.0
        self._own_t: float | None = None

    def start(self, ctx: Ctx) -> None:
        super().start(ctx)
        self._reset_tuned()

    def _climb_rc(self, ctx: Ctx) -> int:
        """Climb to get clear of a person, at most backoff_climb_m above the follow altitude (the governor also
        caps)."""
        alt = ctx.altitude
        return self.c.max_rc_up if (alt is None or alt < self.c.altitude_m + self.c.backoff_climb_m) else \
            self._altitude_rc(ctx)

    # ---- the person-speed estimate
    def _model_own(self, now: float) -> None:
        """Advance the own-speed model to `now`: the stick that reached the drone own_dead_s ago through a
        first-order lag own_tau_s (nominal gain RC_PER_MPS). Every step calls it, seen or not."""
        if self._own_t is not None and now > self._own_t:
            t_in = now - self.c.own_dead_s
            u = 0.0
            for t, f in reversed(self._cmd):
                if t <= t_in:
                    u = f
                    break
            k = 1.0 - math.exp(-(now - self._own_t) / self.c.own_tau_s)
            self._own_v += k * (u / RC_PER_MPS - self._own_v)
        self._own_t = now
        self._own.append((now, self._own_v))

    def _passed(self, ctx: Ctx, fb: float) -> None:
        """Remember the forward stick that reaches the drone: clamped to the governor's max_rc (a SafetyGovernor has
        cfg.max_rc), nothing while it holds a safety hover or lands (it drops the rc)."""
        gov = ctx.drone
        if getattr(gov, "hovering_for_safety", False) or getattr(gov, "land_requested", False):
            fb = 0.0
        m = getattr(getattr(gov, "cfg", None), "max_rc", None)
        if isinstance(m, (int, float)) and m >= 0:
            fb = max(-float(m), min(float(m), fb))
        self._cmd.append((ctx.now, float(fb)))

    def _own_speed(self, ctx: Ctx) -> float:
        """The modeled own speed over the past the range rate describes: the rate window's sample times minus the
        video lag, the frame's YOLO age and the range median's delay, weighted (t - a) (b - t) as a least-squares
        slope weights the derivative across its window. 0 for a past before this follow started (it hovered)."""
        h = [t for t, _ in self.range_hist if ctx.now - t <= RATE_WINDOW_S]
        if not h:
            return 0.0
        age = 0.0 if ctx.frame is None else max(0.0, ctx.now - ctx.frame.t)
        lag = self.c.video_lag_s + age + RANGE_MEDIAN_S
        a, b = h[0] - lag, h[-1] - lag
        if b - a <= 1e-6:
            return self._own_at(b)
        num = den = 0.0
        for t, v in self._own:
            if a <= t <= b:
                w = (t - a) * (b - t)
                num += w * v
                den += w
        return num / den if den > 0 else self._own_at(0.5 * (a + b))

    def _own_at(self, t: float) -> float:
        for tt, v in reversed(self._own):
            if tt <= t:
                return v
        return 0.0

    def _walker_speed(self, now: float, rate: float | None, own: float | None) -> float | None:
        """Low-passed person speed away from the drone (m/s): rate + own. Starts from 0 after every gap: unknown is
        treated as standing, so the cap never rises on a guess."""
        if rate is None or own is None:
            self.walker_speed_mps, self._v_t = None, None
            return None
        raw = rate + own
        prev, dt = (0.0, NOMINAL_TICK_S) if self.walker_speed_mps is None else \
            (self.walker_speed_mps, max(0.0, now - self._v_t))
        v = prev + (1.0 - math.exp(-dt / self.c.walker_speed_tau_s)) * (raw - prev)
        self.walker_speed_mps, self._v_t = v, now
        return v

    def _gate(self, now: float, v_away: float | None, rate: float | None) -> bool:
        """The raised cap's gate. Opens once v_away > fwd_cap_v0_mps with a range rate >= 0 (the drone is not gaining
        on them) has held fwd_cap_dwell_s on end; stays open while v_away > fwd_cap_v0_mps; shuts at once otherwise."""
        c = self.c
        if v_away is None or v_away <= c.fwd_cap_v0_mps:
            self.fast_gate, self._fast_since = False, None
        elif not self.fast_gate:
            if rate is None or rate < 0.0:
                self._fast_since = None
            else:
                if self._fast_since is None:
                    self._fast_since = now
                self.fast_gate = now - self._fast_since >= c.fwd_cap_dwell_s - 1e-9
        return self.fast_gate

    def _closing_lp(self, now: float, closing: float) -> float:
        """The approach feed-forward's closing speed through a first-order low-pass (approach_tau_s), from the raw
        value after every gap in the range rate."""
        if self._closing_f is None:
            self._closing_f = closing
        else:
            k = 1.0 - math.exp(-max(0.0, now - self._closing_t) / self.c.approach_tau_s)
            self._closing_f += k * (closing - self._closing_f)
        self._closing_t = now
        return self._closing_f

    def _slew(self, now: float, fwd: float) -> float:
        """Speed-ups at most fwd_slew stick per second; a lower stick (slowing, braking, backing away) at once."""
        dt = NOMINAL_TICK_S if self._fwd_t is None else max(0.0, now - self._fwd_t)
        return min(fwd, self._fwd_f + self.c.fwd_slew * dt)

    def _sent_fwd(self, now: float, fwd: float) -> None:
        self._fwd_f, self._fwd_t = float(fwd), now

    def step(self, ctx: Ctx) -> str:
        self._model_own(ctx.now)
        c = self.c
        ud = self._altitude_rc(ctx)
        p = ctx.res.person if (ctx.res is not None and ctx.new_frame) else None
        if p is not None:  # FollowBehind.step's seen branch, line for line, plus the knobs
            self.last_seen_t, self.last_bearing = ctx.now, p.bearing_deg
            if p.range_m is not None:
                self.ranges.append((ctx.now, p.range_m))
            r = self._range(ctx.now)
            if r is not None:
                self.range_hist.append((ctx.now, r))
            r_lo = min(v for v in (p.range_lo_m, r) if v is not None) if (p.range_lo_m or r) else None
            yaw = clamp(deadband(p.bearing_deg, c.yaw_deadband_deg) * c.yaw_gain, c.max_rc_yaw)
            if self.fly is not None:  # fly steering: replaces only this yaw value (as flown)
                rng = r if self.fly.fwd_enabled else None
                yaw = clamp(self.fly.yaw(ctx.now, p.bearing_deg, rng, ctx.frame.t, fallback=yaw), c.max_rc_yaw)
            else:
                yaw = clamp(self._source_yaw(ctx, p.bearing_deg, r, yaw), c.max_rc_yaw)
            fwd, approaching = 0.0, False
            width, height = ctx.res.image_size
            too_big = p.det.w > c.max_box_width_frac * width
            if r is not None:
                # the same history the flown approach feed-forward reads (computing it first changes nothing)
                rate = self._range_rate(ctx.now)
                own = [f for t, f in self.fwd_hist if 0.3 <= ctx.now - t <= 1.3]
                own_m = None if rate is None else self._own_speed(ctx)
                v_away = self._walker_speed(ctx.now, rate, own_m)
                fast = self._gate(ctx.now, v_away, rate)
                e = deadband(r - c.distance_m, c.dist_deadband_m)
                if e > 0:  # too far: only approach if even the most conservative cue says we are not too close
                    if r_lo is not None and r_lo > c.min_range_m and not too_big:
                        fwd = e * c.dist_gain
                        if c.fwd_ff_gain > 0 and v_away is not None and v_away > 0:
                            fwd += c.fwd_ff_gain * v_away * RC_PER_MPS  # match their walking speed
                        self.fwd_cap_rc = c.forward_cap(v_away if fast else None)
                        fwd = clamp(fwd, self.fwd_cap_rc)
                else:  # at or inside the distance: exactly the flown law
                    fwd = clamp(e * c.dist_gain, c.max_rc_forward)
                # feed-forward as flown: they walk toward us -> back away at their speed now (rc 100 ~ 1 m/s)
                if rate is not None and own:
                    closing = rate + (sum(own) / len(own)) / 100.0
                    if c.approach_model_w > 0 and own_m is not None:  # the aligned own speed instead (tuned)
                        closing += c.approach_model_w * (rate + own_m - closing)
                    if c.approach_tau_s > 0:
                        closing = self._closing_lp(ctx.now, closing)
                    approaching = closing < -c.approach_rate_mps
                    if approaching:
                        fwd = min(fwd, clamp(100.0 * closing - 10, c.max_rc_backoff))
                else:
                    self._closing_f = None
            else:
                self.walker_speed_mps, self._v_t = None, None
                self.fast_gate, self._fast_since = False, None
                self._closing_f = None
            if self.fly is not None and self.fly.fwd_enabled:  # full fly following, gated as flown
                f = self.fly.forward(fallback=fwd)
                if f > 0 and not (r_lo is not None and r_lo > c.min_range_m and not too_big):
                    f = 0.0
                if approaching:
                    f = min(f, fwd)
                fwd = max(-float(c.max_rc_backoff), min(float(c.max_rc_forward), f))
            if c.fwd_slew > 0:  # last, except the too-close back-off below: safety is never slowed
                fwd = self._slew(ctx.now, fwd)
            if (r_lo is not None and r_lo < c.min_range_m) or too_big:
                fwd = -c.max_rc_backoff  # too close: back off fast and gain a little height
                ud = max(ud, self._climb_rc(ctx) // 2)
            self.last_close = (approaching or (r is not None and r < c.min_range_m + 0.25)
                               or p.det.bbox[1] >= 0.88 * height)
            lat = 0.0
            if p.facing_deg is not None and p.facing_conf >= 0.5:
                self.facings.append(math.radians(p.facing_deg))
            if len(self.facings) >= 3:
                f = math.degrees(math.atan2(sum(math.sin(a) for a in self.facings),
                                            sum(math.cos(a) for a in self.facings)))
                lat = clamp(-c.orbit_gain * deadband(f, c.orbit_deadband_deg), c.max_rc_lateral)
            self.cmd = (int(lat), int(fwd), ud, int(yaw))
            self.fwd_hist.append((ctx.now, int(fwd)))
            self._sent_fwd(ctx.now, fwd)
            rng = "?" if p.range_m is None else f"{p.range_m:.1f}"
            lo = "?" if r_lo is None else f"{r_lo:.1f}"
            face = "?" if p.facing_deg is None else f"{p.facing_deg:+.0f}"
            self.status = f"person {rng} m (lo {lo}) bearing {p.bearing_deg:+.0f} facing {face}"
            self.last_range, self.last_range_lo = r, r_lo
            ctx.drone.rc(*self.cmd)
            self._passed(ctx, self.cmd[1])
            return RUNNING
        return self._step_unseen(ctx)

    def _step_unseen(self, ctx: Ctx) -> str:
        """Not seen on this tick: exactly the flown branches (keep the last command, missing for a moment, lost while
        close, search turn, hover), run by FollowBehind.step itself so a change there is followed here too; the
        overridden _climb_rc applies inside it. The forward stick it sends is tapped for the slew and the own-speed
        model."""
        tap = _RcTap(ctx.drone)
        ctx.drone = tap
        try:
            out = FollowBehind.step(self, ctx)
        finally:
            ctx.drone = tap.inner
        if tap.fb is not None:
            if tap.fb == int(self._fwd_f):  # the same stick again (between frames): keep its exact value
                self._fwd_t = ctx.now
            else:
                self._sent_fwd(ctx.now, tap.fb)
        self._passed(ctx, 0.0 if tap.fb is None else float(tap.fb))
        return out


class _RcTap:
    """The drone, with the forward stick of the last rc() remembered (everything else passes through)."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.fb = None

    def rc(self, lr, fb, ud, yaw) -> None:
        self.fb = fb
        self.inner.rc(lr, fb, ud, yaw)

    def __getattr__(self, name: str):
        return getattr(self.inner, name)


class TunedFollowMission(FollowMission):
    """FollowMission (CLIMB -> FOLLOW -> REACQUIRE -> HOLD, unchanged) whose FOLLOW runs TunedFollowBehind."""

    def _follow(self, why: str) -> None:
        self._go("FOLLOW", why, TunedFollowBehind(self.cfg, self.yaw_source, warn=lambda m: self.log(m, "warn")))


# ------------------------------------------------------------------ model files
def _field_defaults() -> dict:
    return {f.name: f.default for f in fields(TunedCfg)}


def _coerce(name: str, v, default):
    """A model value as the field's type (integer knobs rounded, as params.make_cfg does)."""
    if isinstance(default, bool):
        return bool(v)
    if isinstance(default, int):
        f = float(v)
        if not math.isfinite(f):
            raise ValueError(f"follow.{name} must be finite, got {v!r}")
        return int(round(f))
    if isinstance(default, float) or (default is None and v is not None):
        f = float(v)
        if not math.isfinite(f):
            raise ValueError(f"follow.{name} must be finite, got {v!r}")
        return f
    if default is None:
        return None
    return str(v)


def clean_follow(follow: dict, where: str = "follow") -> dict:
    """Validated "follow" overrides: TunedCfg field names only (the loader's own fields refused), values as the
    field's type, the forward caps clamped to FWD_CAP_LIMIT. A typo must fail loudly, not fly the default."""
    defaults = _field_defaults()
    out = {}
    for k, v in (follow or {}).items():
        if k in _LOADER_FIELDS:
            raise ValueError(f"{where}: '{k}' is set by the model loader, not in 'follow'")
        if k not in defaults:
            raise ValueError(f"{where}: unknown follow setting '{k}'")
        out[k] = _coerce(k, v, defaults[k])
    for k in ("fwd_cap_fast", "max_rc_forward"):
        if k in out:
            out[k] = max(0, min(FWD_CAP_LIMIT, out[k]))
    return out


def _brain_dim() -> int:
    from .flybrain.flyfollow.rl.params import param_space

    return param_space(BRAIN_ARM).dim


def load_model(path: str | Path, **overrides) -> TunedCfg:
    """A model file (see the module docstring) -> TunedCfg: its "follow" overrides on the flown defaults, steering
    "fly" with fly_params_path = the file (the brain is read from it) or "pid" for a law model, max_rc_yaw =
    fly_max_rc_yaw when a fly model sets that, model_name = its name. fwd_cap_fast and max_rc_forward are clamped to
    FWD_CAP_LIMIT whatever the file says. `overrides` (TunedCfg fields, e.g. person_height_m) are applied last."""
    p = Path(path)
    rec = json.loads(p.read_text(encoding="utf-8"))
    follow = clean_follow(rec.get("follow") or {}, str(p))
    has_brain = rec.get("x") is not None
    steering = rec.get("steering") or ("fly" if has_brain else "pid")
    if steering not in STEERINGS:
        raise ValueError(f"{p}: steering must be one of {STEERINGS}, got {steering!r}")
    if steering == "fly" and not has_brain:
        raise ValueError(f"{p}: a fly model needs the brain's 'x'")
    cfg = TunedCfg(**follow)
    cfg.steering = steering
    cfg.model_name = str(rec.get("name") or p.stem)
    if steering == "fly":
        cfg.fly_params_path = str(p.resolve())
        if "fly_max_rc_yaw" in follow:
            cfg.max_rc_yaw = cfg.fly_max_rc_yaw
    defaults = _field_defaults()
    for k, v in overrides.items():
        if k not in defaults:
            raise ValueError(f"unknown TunedCfg field '{k}'")
        setattr(cfg, k, v)
    return cfg


def model_governor(path: str | Path) -> dict:
    """A model file's governor requirement ({} when it has none): "needs_max_rc", the SafetyCfg.max_rc it needs to
    fly as trained (required_max_rc), and "trained_max_rc", the governor's max_rc in training."""
    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    return dict(rec.get(GOVERNOR_KEY) or {})


def save_model(path: str | Path, *, name: str, follow: dict, brain_x: list[float] | None, steering: str,
               trained: dict, signoff: dict | None = None, brain: str = DEFAULT_BRAIN,
               governor: dict | None = None) -> Path:
    """Write a model file atomically and return its path. steering "fly": the brain is brain_x (34 FLY-YAW numbers
    in [0, 1]) or, when None, smooth2's x, arm and brain (a follow-only model), so the file is a valid FlySteer
    params file. steering "pid": a law model, no brain (brain_x must be None). trained gets a "time" if it has
    none. "governor" always records needs_max_rc (required_max_rc of the follow), plus whatever `governor` adds
    (the trainer: trained_max_rc)."""
    if steering not in STEERINGS:
        raise ValueError(f"steering must be one of {STEERINGS}, got {steering!r}")
    p = Path(path)
    rec: dict = {"name": str(name)}
    if steering == "fly":
        if brain_x is None:
            s2 = json.loads(SMOOTH2_PATH.read_text(encoding="utf-8"))
            rec.update(arm=s2.get("arm") or BRAIN_ARM, x=[float(v) for v in s2["x"]], brain=s2.get("brain") or brain)
        else:
            x = [float(v) for v in brain_x]
            dim = _brain_dim()
            if len(x) != dim:
                raise ValueError(f"brain_x must have {dim} numbers ({BRAIN_ARM}), got {len(x)}")
            if not all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in x):
                raise ValueError("brain_x must be finite numbers in [0, 1]")
            rec.update(arm=BRAIN_ARM, x=x, brain=str(brain))
    elif brain_x is not None:
        raise ValueError("a law model (steering 'pid') has no brain: brain_x must be None")
    rec["follow"] = clean_follow(follow)
    rec["steering"] = steering
    cfg = TunedCfg(**rec["follow"])
    cfg.steering = steering
    rec[GOVERNOR_KEY] = {"needs_max_rc": required_max_rc(cfg), **dict(governor or {})}
    rec["trained"] = dict(trained or {})
    rec["trained"].setdefault("time", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    if signoff is not None:
        rec["signoff"] = signoff
    text = json.dumps(rec, indent=2, allow_nan=False) + "\n"
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{p.name}.", suffix=".tmp", dir=p.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, p)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return p
