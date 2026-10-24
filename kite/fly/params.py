"""The fly's 34 tunable yaw parameters (DESIGN.md section 3), their bounds, and the parameter files; also the 3 of
the PRED-PD ablation (PD_SPACE, DESIGN.md R7).

CMA-ES searches x in [0, 1]^34; each x maps to a value linearly (v = lo + x (hi - lo)) or on a log scale
(v = lo (hi / lo)^x). The names, order and bounds match data/reference/, so the previously trained values load
unchanged. Changing a bound changes what an existing x means: never edit them between runs.

A parameter file is JSON with `params` ({name: value}) and/or `x` (34 numbers, this order), plus optionally `norm`
({DN: Hz}, the readout normalization the weights expect), `arm` and `size_blind`. A file without `norm` uses the
previous calibration's (data/reference/fly_calibration_prev.json), which is what the previously trained values and
anything fine-tuned from them expect.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .decoder import NORM_FLOOR_HZ, READOUT_NAMES
from .encoder import ENCODER_NAMES
from .interfaces import DATA_DIR, OUTPUT_GROUPS, PREV_CALIBRATION

NAMES = ENCODER_NAMES + READOUT_NAMES
HAND_FILE = DATA_DIR / "init" / "fly_yaw_hand.json"  # FLY-YAW-HAND, written by python -m kite.fly.calibrate

# name or shared prefix -> (lo, hi, scale, default)
SPECS: dict[str, tuple[float, float, str, float]] = {
    "enc_r_max_hz": (20.0, 400.0, "log", 160.0),  # peak Poisson rate into one LC10a bin
    "enc_width_deg": (3.0, 40.0, "log", 8.0),  # tuning sigma at s = 1 (bins are about 5.75 deg apart)
    "enc_overlap_deg": (0.0, 20.0, "lin", 6.0),  # each side's bins reach this far past the midline
    "enc_size_width_exp": (0.0, 2.0, "lin", 1.0),  # sigma * s^a: the spot widens when closer
    "enc_size_amp_exp": (-1.0, 2.0, "lin", 0.5),  # amplitude * s^b
    "enc_vel_gain": (0.0, 3.0, "lin", 0.5),  # * (1 + k_v |dtheta|), dtheta in rad/s
    "enc_bin_gain": (0.5, 2.0, "log", 1.0),  # 16 per-bin multipliers
    "enc_arousal": (0.5, 2.0, "log", 1.0),  # LC10a gain ("gain" arousal mode)
    "enc_lc9_hz": (0.1, 10.0, "log", 2.0),
    "enc_lc11_hz": (0.1, 10.0, "log", 2.0),
    "dec_w_yaw": (-15.0, 15.0, "lin", 0.0),  # 5 weights on (n_R - n_L), one per DN type
    "dec_b_yaw": (-5.0, 5.0, "lin", 0.0),  # cancels the circuit's right bias
    "dec_g_yaw": (0.0, 1.0, "lin", 0.8),
    "dec_tau_yaw_ms": (10.0, 150.0, "log", 80.0),
    "dec_deadzone": (0.0, 10.0, "lin", 0.5),  # stick units
}


def _spec(name: str) -> tuple[float, float, str, float]:
    for prefix in ("enc_bin_gain", "dec_w_yaw"):
        if name.startswith(prefix + "_"):
            return SPECS[prefix]
    return SPECS[name]


@dataclass(frozen=True, eq=False)  # holds arrays: compare by identity
class ParamSpace:
    names: tuple[str, ...]
    lo: np.ndarray
    hi: np.ndarray
    log: np.ndarray  # bool per parameter
    default: np.ndarray

    @classmethod
    def from_rows(cls, rows: list[tuple[str, float, float, str, float]]) -> ParamSpace:
        """rows of (name, lo, hi, "lin" | "log", default)."""
        col = lambda i: np.array([r[i] for r in rows], float)  # noqa: E731
        return cls(tuple(r[0] for r in rows), col(1), col(2), np.array([r[3] == "log" for r in rows]), col(4))

    @classmethod
    def yaw(cls) -> ParamSpace:
        return cls.from_rows([(n, *_spec(n)) for n in NAMES])

    @property
    def dim(self) -> int:
        return len(self.names)

    def to_values(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, np.float64)
        if x.shape != (self.dim,):
            raise ValueError(f"expected x of shape ({self.dim},), got {x.shape}")
        if not np.all(np.isfinite(x)):
            raise ValueError(f"non-finite x: {x}")
        x = np.clip(x, 0.0, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            lg = self.lo * (self.hi / self.lo) ** x
        return np.where(self.log, lg, self.lo + x * (self.hi - self.lo))

    def from_values(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, np.float64)
        if not np.all(np.isfinite(v)):
            bad = [n for n, ok in zip(self.names, np.isfinite(v)) if not ok]
            raise ValueError(f"non-finite parameter values: {', '.join(bad)}")
        v = np.clip(v, self.lo, self.hi)
        with np.errstate(divide="ignore", invalid="ignore"):
            lg = np.log(v / self.lo) / np.log(self.hi / self.lo)
        return np.clip(np.where(self.log, lg, (v - self.lo) / (self.hi - self.lo)), 0.0, 1.0)

    def decode(self, x: np.ndarray) -> dict[str, float]:
        """x in [0, 1]^34 (clipped) -> {name: value}."""
        return dict(zip(self.names, self.to_values(x).tolist()))

    def encode(self, params: dict[str, float]) -> np.ndarray:
        """{name: value} -> x (values clipped to the bounds). Every name is required; extra names are ignored."""
        missing = [n for n in self.names if n not in params]
        if missing:
            raise KeyError(f"parameters missing: {', '.join(missing)}")
        return self.from_values(np.array([float(params[n]) for n in self.names]))

    def default_x(self) -> np.ndarray:
        return self.from_values(self.default)

    def defaults(self) -> dict[str, float]:
        return dict(zip(self.names, self.default.tolist()))


SPACE = ParamSpace.yaw()
assert SPACE.dim == 34

# PRED-PD (DESIGN.md R7): the classical ablation inside FlySteer's latency filter and stick conditioning.
# yaw = kp * deadband(theta_hat_deg, deadband_deg) + kd * dtheta_hat_deg_s. Defaults: the follow law's gains.
PD_SPACE = ParamSpace.from_rows([
    ("kp", 0.5, 4.0, "log", 1.5),  # stick per degree of predicted bearing
    ("kd", 0.0, 1.0, "lin", 0.1),  # stick per deg/s of bearing rate (off a bound, so CMA does not start on it)
    ("deadband_deg", 0.0, 4.0, "lin", 2.0),  # soft, on the predicted bearing
])


def norm_array(norm: dict | list | np.ndarray) -> np.ndarray:
    """{DN: Hz} (or 10 numbers in OUTPUT_GROUPS order) -> array, floored at NORM_FLOOR_HZ."""
    v = np.array([norm[k] for k in OUTPUT_GROUPS] if isinstance(norm, dict) else norm, np.float64)
    if v.shape != (len(OUTPUT_GROUPS),) or not np.all(np.isfinite(v)):
        raise ValueError(f"norm must have {len(OUTPUT_GROUPS)} finite rates, got {norm!r}")
    return np.maximum(v, NORM_FLOOR_HZ)


def load_norm(path: str | Path = PREV_CALIBRATION) -> np.ndarray:
    """The `norm` of a calibration (or parameter) file."""
    return norm_array(json.loads(Path(path).read_text(encoding="utf-8"))["norm"])


@dataclass(frozen=True, eq=False)  # holds arrays: compare by identity
class ParamFile:
    path: Path
    arm: str | None
    params: dict[str, float]  # the 34 values (within bounds)
    x: np.ndarray
    norm: np.ndarray  # readout normalization, OUTPUT_GROUPS order
    norm_from: Path  # where the norm came from (the file itself or the previous calibration)
    size_blind: bool | None  # the size mode the parameters were made for, if the file says


def load_params(path: str | Path, default_norm: str | Path = PREV_CALIBRATION) -> ParamFile:
    """Read a parameter file: `params` if present (by name, robust to order), else `x`."""
    path = Path(path)
    rec = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(rec.get("params"), dict):
        x = SPACE.encode(rec["params"])
    elif rec.get("x") is not None:
        x = np.asarray(rec["x"], np.float64)
        if x.shape != (SPACE.dim,) or not np.all(np.isfinite(x)):
            raise ValueError(f"{path}: 'x' must be {SPACE.dim} finite numbers")
        x = np.clip(x, 0.0, 1.0)
    else:
        raise ValueError(f"{path}: expected 'params' or 'x'")
    if rec.get("norm") is not None:
        norm, norm_from = norm_array(rec["norm"]), path
    else:
        norm, norm_from = load_norm(default_norm), Path(default_norm)
    sb = rec.get("size_blind")
    return ParamFile(path, rec.get("arm"), SPACE.decode(x), x, norm, norm_from, None if sb is None else bool(sb))


def params_record(params: dict[str, float], norm: np.ndarray | None = None, **extra) -> dict:
    """A JSON-ready parameter record (params, x and norm), the format load_params reads."""
    x = SPACE.encode(params)
    rec = {**extra, "params": SPACE.decode(x), "x": x.tolist()}
    if norm is not None:
        rec["norm"] = dict(zip(OUTPUT_GROUPS, np.asarray(norm, np.float64).tolist()))
    return rec
