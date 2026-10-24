"""Hand calibration of the fly's yaw readout: the FLY-YAW-HAND arm (DESIGN.md section 3).

    python -m kite.fly.calibrate                 # writes kite/fly/data/init/fly_yaw_hand.json (a few seconds)
    python -m kite.fly.calibrate --repeats 1 --out /tmp/hand.json

1. The encoder runs at its hand-set defaults (params.SPECS); the readout's gain, time constant and deadzone too.
2. The brain warms up 1 s with no target. Then a static spot is shown at each of BEARINGS_DEG x SIZES, in a fresh
   shuffled order per repeat: 200 ms to settle, then the DN rates are averaged over 1 s. Three repeats.
3. Each DN is normalized by its mean rate over all stimuli (floored at 2 Hz): that is the file's `norm`.
4. The yaw weights come from a ridge regression of the tanh argument z = 0.04 per degree of bearing on the five
   features n_R - n_L (standardized ridge, lambda 1, intercept free), solved inside the parameter bounds: clipping
   an unbounded fit afterwards would wreck it, because a nearly saturated DN has a tiny spread and a huge weight.
   R^2 is reported in-sample and on the last repeat held out.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

from .brain import PursuitBrain
from .decoder import NORM_FLOOR_HZ
from .encoder import TargetEncoder
from .interfaces import BRAIN_FILE, BRAIN_TICK_MS, DN_TYPES, OUTPUT_GROUPS, TargetFeatures
from .params import HAND_FILE, SPACE, params_record

BEARINGS_DEG = (-30.0, -18.0, -10.0, -4.0, 0.0, 4.0, 10.0, 18.0, 30.0)  # denser near 0, where following spends time
SIZES = (0.6, 1.0, 1.5)  # around the standoff: DN activity saturates in s, wide sizes bias the zero crossing
WARMUP_MS, SETTLE_MS, MEASURE_MS, REPEATS = 1000.0, 200.0, 1000.0, 3
RIDGE = 1.0
Z_PER_DEG = 0.04  # tanh argument per degree: 1.2 at 30 deg, about 40 stick at G 0.8


def record(params: dict[str, float], seed: int = 0, repeats: int = REPEATS, settle_ms: float = SETTLE_MS,
           measure_ms: float = MEASURE_MS, warmup_ms: float = WARMUP_MS,
           brain_file: str | Path = BRAIN_FILE) -> tuple[np.ndarray, np.ndarray, dict]:
    """Mean DN rates per (repeat, stimulus): X (n, 10) in OUTPUT_GROUPS order, rows (n, 3) = bearing_deg, s,
    repeat; plus run info."""
    brain = PursuitBrain(brain_file, seed=seed)
    enc = TargetEncoder(params)
    rest = enc.rest()
    n_settle, n_meas = max(1, round(settle_ms / BRAIN_TICK_MS)), max(1, round(measure_ms / BRAIN_TICK_MS))
    for _ in range(round(warmup_ms / BRAIN_TICK_MS)):
        brain.tick(rest)
    rest_rates = np.mean([brain.tick(rest) for _ in range(n_meas)], axis=0)
    stim = [(b, s) for s in SIZES for b in BEARINGS_DEG]
    rng = np.random.default_rng(seed)
    X, rows = [], []
    t0 = time.perf_counter()
    for r in range(repeats):
        for i in rng.permutation(len(stim)):
            b, s = stim[i]
            ch = enc.channels(TargetFeatures(True, math.radians(b), s, 0.0, 0.0))
            for _ in range(n_settle):
                brain.tick(ch)
            X.append(np.mean([brain.tick(ch) for _ in range(n_meas)], axis=0))
            rows.append((b, s, r))
    info = {"seed": seed, "repeats": repeats, "warmup_ms": warmup_ms, "settle_ms": settle_ms, "measure_ms": measure_ms,
            "rest_rates": dict(zip(OUTPUT_GROUPS, np.round(rest_rates, 3).tolist())),
            "brain_ms_per_tick": round(1000.0 * brain.brain_s / max(1, brain.ticks), 3),
            "wall_s": round(time.perf_counter() - t0, 2)}
    return np.asarray(X), np.asarray(rows, np.float64), info


def bounded_ridge(F: np.ndarray, z: np.ndarray, lam: float, w_bounds: tuple[float, float],
                  b_bounds: tuple[float, float]) -> tuple[np.ndarray, float]:
    """argmin |F w + b - z|^2 + lam |sd * w|^2 with w and b inside their bounds (b unpenalized)."""
    k = F.shape[1]
    sd = F.std(axis=0)
    A = np.vstack([np.hstack([F, np.ones((len(F), 1))]), np.hstack([math.sqrt(lam) * np.diag(sd), np.zeros((k, 1))])])
    t = np.concatenate([z, np.zeros(k)])
    lo = np.array([w_bounds[0]] * k + [b_bounds[0]])
    hi = np.array([w_bounds[1]] * k + [b_bounds[1]])
    res = lsq_linear(A, t, bounds=(lo, hi), method="bvls")
    return res.x[:k], float(res.x[k])


def _r2(z: np.ndarray, pred: np.ndarray) -> float:
    ss = float(((z - z.mean()) ** 2).sum())
    return 1.0 - float(((z - pred) ** 2).sum()) / ss if ss > 0 else float("nan")


def fit_yaw(X: np.ndarray, rows: np.ndarray, lam: float = RIDGE,
            z_per_deg: float = Z_PER_DEG) -> tuple[dict[str, float], np.ndarray, dict]:
    """(yaw weights and bias, norm, fit report) from recorded rates."""
    norm = np.maximum(X.mean(axis=0), NORM_FLOOR_HZ)
    N = X / norm
    F = N[:, 1::2] - N[:, 0::2]  # n_R - n_L per DN type
    z = z_per_deg * rows[:, 0]
    i_w, i_b = SPACE.names.index(f"dec_w_yaw_{DN_TYPES[0]}"), SPACE.names.index("dec_b_yaw")
    wb, bb = (SPACE.lo[i_w], SPACE.hi[i_w]), (SPACE.lo[i_b], SPACE.hi[i_b])
    w, b = bounded_ridge(F, z, lam, wb, bb)
    report: dict = {"r2": _r2(z, F @ w + b)}
    rep = rows[:, 2]
    if np.unique(rep).size > 1:
        train, test = rep < rep.max(), rep == rep.max()
        wh, bh = bounded_ridge(F[train], z[train], lam, wb, bb)
        report["r2_holdout"] = _r2(z[test], F[test] @ wh + bh)
    D = X[:, 1::2] - X[:, 0::2]
    report["corr_bearing_vs_R_minus_L"] = {
        t: float(np.corrcoef(rows[:, 0], D[:, k])[0, 1]) if D[:, k].std() > 0 else 0.0 for k, t in enumerate(DN_TYPES)}
    report["at_bound"] = int(np.isclose(w, wb[0]).sum() + np.isclose(w, wb[1]).sum() + np.isclose(b, bb).sum())
    # the static response this readout gives (tanh argument, mean over sizes and repeats)
    report["z_by_bearing"] = {f"{bd:+g}": float(np.mean((F @ w + b)[rows[:, 0] == bd])) for bd in BEARINGS_DEG}
    readout = {f"dec_w_yaw_{t}": float(w[k]) for k, t in enumerate(DN_TYPES)} | {"dec_b_yaw": b}
    return readout, norm, report


def calibrate(out: str | Path | None = HAND_FILE, seed: int = 0, repeats: int = REPEATS, settle_ms: float = SETTLE_MS,
              measure_ms: float = MEASURE_MS, brain_file: str | Path = BRAIN_FILE) -> dict:
    """Record, fit, and write the FLY-YAW-HAND parameter file (out None: do not write). Returns the record."""
    params = SPACE.defaults()
    X, rows, info = record(params, seed, repeats, settle_ms, measure_ms, brain_file=brain_file)
    readout, norm, report = fit_yaw(X, rows)
    params.update(readout)
    rec = params_record(params, norm, arm="FLY-YAW-HAND", brain=Path(brain_file).name,
                        created=time.strftime("%Y-%m-%dT%H:%M:%S"))
    rec |= {"fit": report, **info,
            "stimuli": {"bearings_deg": list(BEARINGS_DEG), "sizes": list(SIZES), "ridge": RIDGE,
                        "z_per_deg": Z_PER_DEG},
            "rates": {"columns": list(OUTPUT_GROUPS), "rows": rows.tolist(), "X": np.round(X, 3).tolist()}}
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m kite.fly.calibrate", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=HAND_FILE, help=f"output JSON (default {HAND_FILE})")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=REPEATS)
    a = ap.parse_args(argv)
    rec = calibrate(a.out, a.seed, a.repeats)
    fit = rec["fit"]
    print(f"FLY-YAW-HAND -> {a.out}")
    print(f"  R^2 {fit['r2']:.3f}" + (f" (holdout {fit['r2_holdout']:.3f})" if "r2_holdout" in fit else "")
          + f", {fit['at_bound']} at a bound; brain {rec['brain_ms_per_tick']:.2f} ms/tick, {rec['wall_s']:.1f} s")
    print("  weights " + ", ".join(f"{t} {rec['params'][f'dec_w_yaw_{t}']:+.3f}" for t in DN_TYPES)
          + f"; bias {rec['params']['dec_b_yaw']:+.3f}")
    print("  corr(bearing, R-L) " + ", ".join(f"{k} {v:+.2f}" for k, v in fit["corr_bearing_vs_R_minus_L"].items()))
    print("  tanh argument by bearing " + ", ".join(f"{k}: {v:+.2f}" for k, v in fit["z_by_bearing"].items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
