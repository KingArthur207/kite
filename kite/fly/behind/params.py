"""The arms of kite.fly.behind and the parameter spaces CMA-ES searches (DESIGN section 7).

    sp = space("V3-JOINT")                   # 18 follow + 5 fly conditioning + 34 brain dims = 57
    x0 = init_x("V3-JOINT")                  # the flown follow values + smooth2's brain x: the app as it flies
    cfg, brain_x = make_cfg("V3-JOINT", x0)  # TunedCfg (steering "fly") + the 34 brain numbers for the episode
    model_follow("V3-JOINT", x0)             # the "follow" overrides a model file (DESIGN section 6) stores
    cma_stds("V3-JOINT", sigma0=0.15)        # per-dimension step multipliers: wide follow search, brain fine-tune

Arms:
  FLOWN      FollowMission + BehindCfg() + smooth2: the app today, the baseline and the fitness normalizer (no knobs)
  V3-FOLLOW  TunedFollowMission, follow knobs + fly conditioning (23); smooth2's brain frozen (fly_params_path)
  V3-JOINT   the same 23 + the 34 FLY-YAW brain numbers (57), trained together from flown + smooth2
  V3-LAW     follow knobs + the yaw law (steering "pid", 21): what the follow can do without the fly

Every dimension lives on [0, 1] (pycma's bounded box). A knob maps linearly (lo + x (hi - lo)) or on a log scale
(lo (hi / lo)^x), reusing kite.fly.params.ParamSpace; the brain dims are identity rows ("brain.<name>", lo 0, hi 1):
the 34 FLY-YAW numbers are already normalized by kite.flybrain's own param_space("FLY-YAW"), so they pass through
bit-exact and smooth2's x is the joint init as is. Changing a bound changes what an existing x means: never edit them
between runs (state.pkl and best.json store x).

Decoding snaps knob values to 12 significant digits (a log scale turns 40 into 39.999999999999996), rounds the
integer knobs (stick limits), and maps the top of fwd_slew (2000 stick/s, the flown value, where a 0.05 s control
tick could change the stick by 100: never binding) to TunedCfg's 0 = off. So init_x(arm) decodes to EXACTLY the
flown settings and the tuned controller starts bit-identical to the flown one; CMA explores from there.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from kite.fly.params import ParamSpace
from kite.follow_behind import FLY_PARAMS, BehindCfg

ARMS = ("FLOWN", "V3-FOLLOW", "V3-JOINT", "V3-LAW", "V3B-FOLLOW", "V3B-JOINT")
TUNED_ARMS = ARMS[1:]
FLY_ARMS = ("FLOWN", "V3-FOLLOW", "V3-JOINT", "V3B-FOLLOW", "V3B-JOINT")  # the fly brain steers yaw
BRAIN_PREFIX = "brain."
FWD_SLEW_OFF = 2000.0  # the top of fwd_slew = no slew limit (TunedCfg.fwd_slew 0)
SNAP_DIGITS = 12

# (name, lo, hi, scale, flown). The flown column is BehindCfg's / TunedCfg's default (tests check it).
FOLLOW_KNOBS: tuple[tuple[str, float, float, str, float], ...] = (
    ("dist_gain", 15.0, 90.0, "log", 40.0),  # rc forward per metre of distance error
    ("dist_deadband_m", 0.0, 0.3, "lin", 0.1),
    # the cap while the person is not measured faster (~0.34 m/s at 35). Its top is the flown 35: any speed above
    # the flown follow's comes only through the person-speed gate below, never whatever the person does
    ("max_rc_forward", 20.0, 35.0, "lin", 35.0),
    ("fwd_cap_fast", 35.0, 63.0, "lin", 35.0),  # the cap when they walk fast (63 ~ 0.6 m/s, FWD_CAP_LIMIT)
    ("fwd_cap_v0_mps", 0.3, 0.5, "lin", 0.35),  # person speed where the cap starts to rise (>= a slow walk)
    ("fwd_cap_per_mps", 0.0, 150.0, "lin", 0.0),  # cap rise per m/s above fwd_cap_v0_mps (0 = never faster)
    ("fwd_ff_gain", 0.0, 1.2, "lin", 0.0),  # forward feed-forward on the person's measured speed away
    ("walker_speed_tau_s", 0.2, 1.5, "log", 0.5),  # low-pass of that speed estimate
    ("fwd_slew", 60.0, FWD_SLEW_OFF, "log", FWD_SLEW_OFF),  # stick/s; the top = off (flown)
    ("max_rc_backoff", 30.0, 63.0, "lin", 50.0),
    # at 0.5 the back-away from a person walking at the drone almost never fires: capped at 0.3 so training cannot
    # switch off a safety reaction (approach_model_w gives it a better closing speed instead)
    ("approach_rate_mps", 0.05, 0.3, "lin", 0.2),
    ("approach_model_w", 0.0, 1.0, "lin", 0.0),  # that closing speed from the person-speed model (0 = flown)
    ("approach_tau_s", 0.0, 1.0, "lin", 0.0),  # ... low-passed (0 = flown): damps the pitch-driven fore-aft cycle
    ("min_range_m", 0.55, 1.2, "lin", 0.55),
    ("alt_gain", 20.0, 100.0, "log", 60.0),
    ("alt_deadband_m", 0.02, 0.08, "lin", 0.05),  # 0.15 let it sag a free 0.15 m toward head height
    ("max_rc_up", 10.0, 40.0, "lin", 30.0),
    ("backoff_climb_m", 0.0, 0.25, "lin", 0.25),  # back-off climb ceiling above altitude_m (no ceiling rush)
    # the person missing for a moment (BehindCfg, "ride out a missed frame"): how long to keep turning and creeping
    # before the search starts (at most lost_after_s, 1.5), and the creep stick (~0.1 m/s at 10)
    ("missing_s", 0.5, 1.5, "lin", 1.5),
    ("creep_rc", 0.0, 25.0, "lin", 10.0),
)
FLY_KNOBS: tuple[tuple[str, float, float, str, float], ...] = (  # FlySteer's output conditioning
    ("fly_deadband", 0.0, 8.0, "lin", 4.0),
    ("fly_hysteresis", 0.0, 6.0, "lin", 3.0),
    ("fly_slew", 60.0, 600.0, "log", 300.0),
    ("fly_smoothing_ms", 0.0, 250.0, "lin", 0.0),
    ("fly_max_rc_yaw", 20.0, 60.0, "lin", 40.0),  # max_rc_yaw is set to the same value
)
LAW_KNOBS: tuple[tuple[str, float, float, str, float], ...] = (  # FollowBehind's own yaw law (steering "pid")
    ("yaw_gain", 0.5, 3.0, "log", 1.2),
    ("yaw_deadband_deg", 0.0, 6.0, "lin", 4.0),
    ("max_rc_yaw", 20.0, 60.0, "lin", 40.0),
)
INT_KNOBS = frozenset({"creep_rc", "max_rc_forward", "fwd_cap_fast", "max_rc_backoff", "max_rc_up", "fly_max_rc_yaw",
                       "max_rc_yaw"})  # int fields of BehindCfg / TunedCfg: rounded when decoded
# Round 2 (V3B, 2026-10-25): round 1 never used the speed-up (every run settled fwd_cap_fast at 36-41, ~0.35-0.39
# m/s), so a person walking faster than the flown 0.34 m/s still outwalked it. The V3B arms keep the capacity (a cap
# of 50-63 when the person-speed gate opens, rising at least 40 stick per m/s) and tune only WHEN the gate opens
# (fwd_cap_v0_mps, walker_speed_tau_s, ...), not whether there is a speed-up at all. Below the gate the cap is still
# max_rc_forward (<= the flown 35), so a slow or stopped person is matched as before.
SPEED_BOUNDS = {"fwd_cap_fast": (50.0, 63.0), "fwd_cap_per_mps": (40.0, 150.0)}


def _with_bounds(rows: tuple, bounds: dict[str, tuple[float, float]]) -> tuple:
    """rows with the named knobs' bounds replaced (their flown value clipped into the new range)."""
    out = []
    for n, lo, hi, scale, flown in rows:
        if n in bounds:
            lo, hi = bounds[n]
            flown = min(max(flown, lo), hi)
        out.append((n, lo, hi, scale, flown))
    return tuple(out)


SPEED_KNOBS = _with_bounds(FOLLOW_KNOBS, SPEED_BOUNDS)
ARM_KNOBS = {"FLOWN": (), "V3-FOLLOW": FOLLOW_KNOBS + FLY_KNOBS, "V3-JOINT": FOLLOW_KNOBS + FLY_KNOBS,
             "V3-LAW": FOLLOW_KNOBS + LAW_KNOBS, "V3B-FOLLOW": SPEED_KNOBS + FLY_KNOBS,
             "V3B-JOINT": SPEED_KNOBS + FLY_KNOBS}
JOINT_ARMS = ("V3-JOINT", "V3B-JOINT")  # the brain's 34 numbers are trained too

# Per-dimension CMA step multipliers (pycma CMA_stds). The knobs search widely (sigma0 0.15 of their range); the
# brain only fine-tunes: its base multiplier is BRAIN_STEP / sigma0 (0.2 at sigma0 0.15, i.e. kite.fly's FLY-YAW
# step of 0.03), and the readout weights / bias get kite.fly FLY-YAW's own extra factors on top (they live on +-15 /
# +-5 and act on O(1) rate differences, so the same normalized step that is small for the encoder flips the readout).
DEFAULT_SIGMA0 = 0.15
BRAIN_STEP = 0.03
KNOB_STD = 1.0


def _brain_extra() -> dict[str, float]:
    from kite.fly.train.cma_train import ARM_DEFAULTS

    return dict(ARM_DEFAULTS["FLY-YAW"]["stds"])  # {"dec_w_": 0.08, "dec_b_": 0.2}


@lru_cache(maxsize=1)
def brain_space():
    """kite.flybrain's FLY-YAW space (34): the names and order of a brain x (FlySteer params files store this x)."""
    from kite.flybrain.flyfollow.rl.params import param_space

    return param_space("FLY-YAW")


@lru_cache(maxsize=1)
def _smooth2_x() -> tuple[float, ...]:
    from kite.flybrain.flyfollow.steer import load_params

    x, _arm, _brain = load_params(FLY_PARAMS)
    return tuple(float(v) for v in np.asarray(x, np.float64))


def smooth2_x() -> list[float]:
    """smooth2's 34 normalized brain numbers (kite.follow_behind.FLY_PARAMS, the brain the app flies)."""
    return list(_smooth2_x())


def _snap(v: float) -> float:
    return float(f"{v:.{SNAP_DIGITS}g}")


@dataclass(frozen=True, eq=False)
class BehindSpace:
    """One arm's search space: the knob rows then (V3-JOINT) the 34 brain rows, all on [0, 1]."""

    arm: str
    base: ParamSpace  # every dimension; brain rows are identity (lo 0, hi 1, lin, default smooth2's x)
    n_knobs: int  # the first n_knobs dimensions are knobs, the rest brain

    @property
    def names(self) -> tuple[str, ...]:
        return self.base.names

    @property
    def dim(self) -> int:
        return self.base.dim

    @property
    def knob_names(self) -> tuple[str, ...]:
        return self.base.names[: self.n_knobs]

    @property
    def brain_names(self) -> tuple[str, ...]:
        return self.base.names[self.n_knobs:]

    def _x(self, x) -> np.ndarray:
        x = np.asarray(x, np.float64).ravel()
        if x.shape != (self.dim,):
            raise ValueError(f"{self.arm}: expected {self.dim} numbers, got {x.shape[0]}")
        if not np.all(np.isfinite(x)):
            raise ValueError(f"{self.arm}: non-finite x")
        return np.clip(x, 0.0, 1.0)

    def decode(self, x) -> dict[str, float]:
        """x (clipped to [0, 1]) -> {name: value}: knobs as the cfg gets them (snapped, ints rounded, fwd_slew top =
        0 off), brain dims as the normalized numbers themselves."""
        x = self._x(x)
        v = self.base.to_values(x)
        out: dict[str, float] = {}
        for i, n in enumerate(self.names):
            if i >= self.n_knobs:
                out[n] = float(x[i])  # bit-exact pass-through
                continue
            val = _snap(float(v[i]))
            if n == "fwd_slew" and val >= FWD_SLEW_OFF * (1.0 - 1e-9):
                val = 0.0
            out[n] = int(round(val)) if n in INT_KNOBS else val
        return out

    def encode(self, values: dict[str, float]) -> np.ndarray:
        """{name: value} -> x. A missing name takes the arm's flown default; names outside the space (e.g. max_rc_yaw
        of a fly model) are ignored. fwd_slew 0 (off) encodes as the top of its range."""
        v = np.array([float(values.get(n, d)) for n, d in zip(self.names, self.base.default)], np.float64)
        for i, n in enumerate(self.names[: self.n_knobs]):
            if n == "fwd_slew" and v[i] <= 0.0:
                v[i] = FWD_SLEW_OFF
        x = self.base.from_values(v)
        x[self.n_knobs:] = np.clip(v[self.n_knobs:], 0.0, 1.0)  # identity rows, exact
        return x

    def default_x(self) -> np.ndarray:
        """The flown values (and smooth2's brain x): the init of every tuned arm."""
        x = self.base.default_x()
        x[self.n_knobs:] = self.base.default[self.n_knobs:]
        return x

    def knobs(self, x) -> dict[str, float]:
        d = self.decode(x)
        return {n: d[n] for n in self.knob_names}

    def brain_x(self, x) -> list[float] | None:
        """The 34 brain numbers of x (V3-JOINT), else None."""
        if not self.brain_names:
            return None
        x = self._x(x)
        return [float(v) for v in x[self.n_knobs:]]


@lru_cache(maxsize=None)
def space(arm: str) -> BehindSpace:
    """The arm's BehindSpace (FLOWN: an empty one)."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; one of {ARMS}")
    rows = list(ARM_KNOBS[arm])
    n_knobs = len(rows)
    if arm in JOINT_ARMS:
        bs = brain_space()
        rows += [(BRAIN_PREFIX + n, 0.0, 1.0, "lin", v) for n, v in zip(bs.names, _smooth2_x())]
    if rows:
        base = ParamSpace.from_rows(rows)
    else:
        base = ParamSpace((), np.zeros(0), np.zeros(0), np.zeros(0, bool), np.zeros(0))
    return BehindSpace(arm, base, n_knobs)


def init_x(arm: str) -> np.ndarray:
    """CMA's starting point: the flown follow (V3-JOINT: + smooth2's x). FLOWN has no dimensions."""
    return space(arm).default_x()


def cma_stds(arm: str, sigma0: float = DEFAULT_SIGMA0) -> list[float]:
    """pycma CMA_stds for the arm: knobs KNOB_STD, brain dims BRAIN_STEP / sigma0 (x 0.08 dec_w_, x 0.2 dec_b_)."""
    sp = space(arm)
    base = BRAIN_STEP / float(sigma0)
    extra = _brain_extra()
    out = [KNOB_STD] * sp.n_knobs
    for n in sp.brain_names:
        short = n[len(BRAIN_PREFIX):]
        out.append(base * next((float(m) for p, m in extra.items() if short.startswith(p)), 1.0))
    return out


def _check_arm(arm: str) -> None:
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; one of {ARMS}")


def model_follow(arm: str, x=None) -> dict:
    """The follow overrides of the arm at x (None: the init) as a model file stores them (DESIGN section 6): every
    knob of the arm, and max_rc_yaw = fly_max_rc_yaw for the fly arms. FLOWN: {}."""
    _check_arm(arm)
    if arm == "FLOWN":
        return {}
    sp = space(arm)
    out = sp.knobs(sp.default_x() if x is None else x)
    if arm in FLY_ARMS:
        out["max_rc_yaw"] = int(out["fly_max_rc_yaw"])
    return out


def steering(arm: str) -> str:
    _check_arm(arm)
    return "pid" if arm == "V3-LAW" else "fly"


def make_cfg(arm: str, x=None) -> tuple[BehindCfg, list[float] | None]:
    """(cfg, brain_x) for an arm at x (None: the init). FLOWN -> (BehindCfg(), None): the real FollowMission flies it.
    Tuned arms -> a TunedCfg (TunedFollowMission) with the knobs, steering "fly" (V3-FOLLOW / V3-JOINT; smooth2 stays
    the fly_params_path) or "pid" (V3-LAW); brain_x is V3-JOINT's 34 brain numbers for run_behind_episode, else
    None. One new cfg per call."""
    _check_arm(arm)
    if arm == "FLOWN":
        if x is not None and np.asarray(x).size:
            raise ValueError("FLOWN has no parameters")
        return BehindCfg(), None
    from kite.follow_tuned import TunedCfg

    sp = space(arm)
    x = sp.default_x() if x is None else x
    cfg = TunedCfg(**model_follow(arm, x), steering=steering(arm))
    cfg.check()
    return cfg, sp.brain_x(x)


def encode_model(arm: str, rec: dict) -> np.ndarray:
    """A model file's record (follow overrides, and for V3-JOINT its brain x) -> x in the arm's space (a warm start,
    or re-scoring a saved model as an arm candidate)."""
    sp = space(arm)
    values = dict(rec.get("follow") or {})
    if sp.brain_names:
        bx = rec.get("x")
        if bx is None or len(bx) != len(sp.brain_names):
            raise ValueError(f"{arm} needs the model's {len(sp.brain_names)} brain numbers in 'x'")
        values.update(zip(sp.brain_names, (float(v) for v in bx)))
    return sp.encode(values)


def describe(arm: str) -> list[tuple[str, float, float, str, float]]:
    """(name, lo, hi, scale, flown) per dimension (brain rows: 0, 1, lin, smooth2's value)."""
    sp = space(arm).base
    return [(n, float(a), float(b), "log" if g else "lin", float(d))
            for n, a, b, g, d in zip(sp.names, sp.lo, sp.hi, sp.log, sp.default)]

