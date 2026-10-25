"""Sign-off of the behind follow models (DESIGN section 9): every model file against FLOWN, the app as it flies today
(FollowMission + BehindCfg() + smooth2), on the test seeds, paired per seed, and the gates that decide whether a model
is recommended.

    python -m kite.fly.behind.signoff --models kite/fly/data/behind/*.json results/behind/runs/*/best.json \\
        --profiles train,demo,fast,slow,stress --seeds test --n 200 --backend modal
    python -m kite.fly.behind.signoff --models ... --cache-only            # the report again, no episode re-run
    python -m kite.fly.behind.signoff --from results/behind/report/signoff.json   # recompute tables and gates only
    python -m kite.fly.behind.signoff --models ... --init-arms V3-FOLLOW    # + a tuned arm at its init (= FLOWN)

Models. A model file (DESIGN section 6: kite.follow_tuned.save_model, a run's best.json) is evaluated as its arm
(trained.behind_arm) at x = kite.fly.behind.params.encode_model(arm, file): the very items training ran, on any
backend (a Modal container has no copy of the file). Before anything runs, make_cfg(arm, x) is checked field by field
against load_model(file), what the app loads, and the arm's brain against the file's x: a model whose reconstruction
would fly something else is refused, never silently evaluated.

Pairing. Every entry runs on the same seeds x profiles (common random numbers: the same person, video and dynamics),
so every comparison is per seed against FLOWN: the score (kite.fly.behind.fitness.episode_score: FLOWN = 1, lower is
better, a collision or an episode that raised = 6, a failed start excluded) and each metric, with the mean and a
95 % percentile bootstrap CI of the per-seed difference (kite.fly.train.report.bootstrap_ci). Metric comparisons use
the seeds where both ran, neither start failed and neither collided (a collision ends the episode, so its metrics
stop early); collisions are compared seed by seed instead.

Gates (DESIGN section 9, criteria numbered as there). A model is RECOMMENDED only if every check holds:
  1. train and demo: mean score <= 0.9 x FLOWN's mean score on the same seeds, and the 95 % CI of the paired
     difference lies below 0;
  2. every profile (train, demo, fast, slow, stress): #seeds where the model collides and FLOWN does not
     <= #seeds where FLOWN collides and the model does not + 1, and no episode of the model errored on a seed where
     FLOWN's ran (a seed where both fail is the simulator's error; it is counted in the detail, not against the model);
  3. train and demo: the upper CI of (model - FLOWN) <= max(10 % of FLOWN's mean, an absolute margin) on too_close_s,
     very_close_s, overshoot_m, ceiling_s, losses and wrong_lock_s;
  4. train and demo: smoother: the mean of the four stick changes, each normalized by FitnessCfg's s_fb / s_yaw /
     s_ud / s_lr (the fitness's own stick term, so no axis dominates by its units), <= FLOWN's on the same seeds;
  5. slow: excess_speed and fast_unneeded_s pass 3's non-inferiority test (speed only when needed); fast:
     rms_range_err_m and far_s below FLOWN's, paired mean difference < 0 (it keeps up).
Missing data fails its criterion. Absolute margins: NONINF_ABS and SLOW_ABS below.

Cache. Every finished episode is appended to <out>/cache/episodes.jsonl, keyed by what defines it (arm, x, seed,
profile, episode length, overrides, stub) and stamped with a digest of the episode path's sources (the sim, the follow,
the brain and its data): a re-run takes every episode it can from the cache and runs only the rest, and a changed
simulator never reuses stale episodes (--reuse-stale overrides that). Failed episodes are not cached.

Writes <out>/<name>.json (config, entries, every episode's metrics, the analysis) and <name>.md (default
results/behind/report/signoff.*); --docs also copies the markdown to docs/follow-v3-results.md. Simulation only: a
recommended model stays opt-in until a real flight next to smooth2.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import datetime as _dt
import hashlib
import json
import math
import random
import shutil
import sys
import time
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from kite.fly.behind.fitness import FitnessCfg
from kite.fly.train.report import bootstrap_ci

REPO = Path(__file__).resolve().parents[3]
FLOWN = "FLOWN"
PROFILES = ("train", "demo", "fast", "slow", "stress")
MAIN = ("train", "demo")
N_BOOT = 10_000
SCORE_GAIN = 0.10  # 1: mean score at least 10 % below FLOWN's
COLLISION_SLACK = 1  # 2: seed by seed, at most one more collision-only seed than FLOWN has
NONINF_REL = 0.10  # 3 and 5 (slow): upper CI <= max(10 % of FLOWN's mean, the absolute margin)
NONINF_ABS = {"too_close_s": 0.2, "very_close_s": 0.1, "overshoot_m": 0.05, "ceiling_s": 0.2, "losses": 0.1,
              "wrong_lock_s": 0.2}
SLOW_ABS = {"excess_speed": 0.02, "fast_unneeded_s": 0.5}
FAST_BELOW = ("rms_range_err_m", "far_s")
STICKS = ("stick_fb", "stick_yaw", "stick_ud", "stick_lr")
STICK_SCALES = ("s_fb", "s_yaw", "s_ud", "s_lr")
# FitnessCfg's calibrated stick scales, for a cfg-like object (tests) without them
_STICK_SCALE_DEFAULTS = tuple(float(getattr(FitnessCfg(), s)) for s in STICK_SCALES)
STICK_NORM = "stick_norm"  # derived per episode: mean(stick_<axis> / s_<axis>), criterion 4
TABLE_METRICS = ("rms_range_err_m", "in_band", "too_close_s", "very_close_s", "min_range_m", "overshoot_m", "far_s",
                 "rms_bearing_deg", "in_view", "losses", "time_lost_s", "wrong_lock_s", STICK_NORM, "accel_rms",
                 "alt_max_m", "ceiling_s", "speed_track_err", "excess_speed", "fast_unneeded_s", "max_speed_mps",
                 "governor_events")
PRINT_METRICS = ("rms_range_err_m", "too_close_s", "overshoot_m", "far_s", "excess_speed", "fast_unneeded_s",
                 "max_speed_mps", STICK_NORM)
SEED_SETS = ("test", "selection")
# The episode path's sources: kite.fly.behind.digest (the files the import probe sees, kite/config.py's SafetyCfg
# and kite/camera.py included, plus the brain data), shared with the trainer's FLOWN cache. A change to any of them
# invalidates cached episodes; the modules that only drive or score episodes are left out.


def report_dir() -> Path:
    return REPO / "results" / "behind" / "report"


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")


def _json_default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    if is_dataclass(o):
        return asdict(o)
    return str(o)


# ---------------------------------------------------------------------------------------------------------------
# Episode records: {seed, ok, metrics, ...}; recs[label][profile] = records sorted by seed
# ---------------------------------------------------------------------------------------------------------------


def _flag(m: dict, key: str) -> bool:
    v = (m or {}).get(key, 0.0)
    return v is not None and float(v) > 0.0


def is_ok(r: dict) -> bool:
    """The episode ran (kite.fly.behind.evaluate.is_error's complement)."""
    return bool(r.get("ok", False)) and not _flag(r.get("metrics") or {}, "error")


def start_failed(r: dict) -> bool:
    return is_ok(r) and _flag(r.get("metrics"), "start_failed")


def collided(r: dict) -> bool:
    return is_ok(r) and _flag(r.get("metrics"), "collided")


def usable(r: dict) -> bool:
    """Ran and started (a failed start is excluded identically for every arm)."""
    return is_ok(r) and not _flag(r.get("metrics"), "start_failed")


def by_seed(recs: list[dict]) -> dict[int, dict]:
    return {int(r["seed"]): r for r in recs}


def stick_norm(m: dict, fc: Any) -> float | None:
    """mean over the four axes of stick_<axis> / FitnessCfg.s_<axis> (None if any axis is missing)."""
    try:
        return float(np.mean([float(m[k]) / float(getattr(fc, s, d))
                              for k, s, d in zip(STICKS, STICK_SCALES, _STICK_SCALE_DEFAULTS)]))
    except (KeyError, TypeError, ValueError):
        return None


def with_derived(recs: dict, fc: Any) -> dict:
    """A copy of recs whose usable episodes carry STICK_NORM."""
    out: dict = {}
    for lab, per in recs.items():
        for p, rs in per.items():
            new = []
            for r in rs:
                r = dict(r)
                if is_ok(r):
                    m = dict(r.get("metrics") or {})
                    s = stick_norm(m, fc)
                    if s is not None:
                        m[STICK_NORM] = s
                    r["metrics"] = m
                new.append(r)
            out.setdefault(lab, {})[p] = new
    return out


def paired(a: list[dict], b: list[dict], metric: str, n_boot: int = N_BOOT) -> dict:
    """Mean and 95 % CI of (a - b) per seed over the seeds where both ran, neither start failed, neither collided and
    both have the metric; n_excluded counts the other seeds either side has. mean_a / mean_b: each side's mean over
    the same kept seeds."""
    A, B = by_seed(a), by_seed(b)
    keep = [s for s in sorted(set(A) & set(B)) if usable(A[s]) and usable(B[s]) and not collided(A[s])
            and not collided(B[s]) and metric in A[s]["metrics"] and metric in B[s]["metrics"]]
    va = np.array([float(A[s]["metrics"][metric]) for s in keep])
    vb = np.array([float(B[s]["metrics"][metric]) for s in keep])
    m, lo, hi = bootstrap_ci((va - vb).tolist(), n_boot)
    return {"n": len(keep), "n_excluded": len(set(A) | set(B)) - len(keep), "mean": m, "lo": lo, "hi": hi,
            "mean_a": float(va.mean()) if keep else math.nan, "mean_b": float(vb.mean()) if keep else math.nan}


def collision_pairs(a: list[dict], b: list[dict]) -> dict:
    """Seed by seed over the seeds where both ran and neither start failed: a_only = #(a collides, b does not),
    b_only = #(b collides, a does not), both."""
    A, B = by_seed(a), by_seed(b)
    keep = [s for s in sorted(set(A) & set(B)) if usable(A[s]) and usable(B[s])]
    return {"n": len(keep), "a_only": sum(collided(A[s]) and not collided(B[s]) for s in keep),
            "b_only": sum(collided(B[s]) and not collided(A[s]) for s in keep),
            "both": sum(collided(A[s]) and collided(B[s]) for s in keep)}


def _metrics_for_score(r: dict) -> dict:
    m = dict(r.get("metrics") or {})
    if not is_ok(r):
        m["error"] = 1.0  # fitness charges an errored arm episode and excludes an errored FLOWN one
    return m


def score_pairs(a: list[dict], b: list[dict], score_fn: Callable[[dict, dict], float | None],
                n_boot: int = N_BOOT) -> dict:
    """Per seed: score_fn(a, FLOWN) and score_fn(FLOWN, FLOWN) (1.0 with kite.fly.behind.fitness), over the seeds both
    sides have and the score does not exclude (a failed start, a failed FLOWN episode). Mean of each and the paired
    difference with its 95 % CI; rel = mean_model / mean_flown - 1."""
    A, B = by_seed(a), by_seed(b)
    pairs = []
    for s in sorted(set(A) & set(B)):
        ma, mb = _metrics_for_score(A[s]), _metrics_for_score(B[s])
        sa, sb = score_fn(ma, mb), score_fn(mb, mb)
        if sa is None or sb is None or not math.isfinite(sa) or not math.isfinite(sb):
            continue
        pairs.append((s, float(sa), float(sb)))
    d, lo, hi = bootstrap_ci([sa - sb for _, sa, sb in pairs], n_boot)
    mm = float(np.mean([p[1] for p in pairs])) if pairs else math.nan
    mf = float(np.mean([p[2] for p in pairs])) if pairs else math.nan
    return {"n": len(pairs), "n_excluded": len(set(A) | set(B)) - len(pairs), "mean_model": mm, "mean_flown": mf,
            "mean": d, "lo": lo, "hi": hi, "rel": mm / mf - 1.0 if pairs and mf else math.nan}


def summarize(recs: list[dict]) -> dict:
    ok = [r for r in recs if usable(r)]
    keys = sorted({k for r in ok for k, v in (r.get("metrics") or {}).items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool)})
    means = {}
    for k in keys:
        v = [float(r["metrics"][k]) for r in ok if isinstance(r["metrics"].get(k), (int, float))
             and math.isfinite(float(r["metrics"][k]))]
        if v:
            means[k] = float(np.mean(v))
    return {"n": len(recs), "n_errors": sum(not is_ok(r) for r in recs), "n_start_failed": sum(map(start_failed, recs)),
            "collisions": sum(map(collided, recs)), "metrics": means}


class Comparer:
    """Every comparison of an entry with FLOWN on a profile, computed once (the gates and the tables share them)."""

    def __init__(self, recs: dict, score_fn: Callable[[dict, dict], float | None], n_boot: int = N_BOOT):
        self.recs, self.score_fn, self.n_boot = recs, score_fn, int(n_boot)
        self._memo: dict = {}

    def has(self, lab: str, p: str) -> bool:
        return bool(self.recs.get(lab, {}).get(p)) and bool(self.recs.get(FLOWN, {}).get(p))

    def _get(self, key: tuple, fn: Callable[[], dict]) -> dict:
        if key not in self._memo:
            self._memo[key] = fn()
        return self._memo[key]

    def pair(self, lab: str, p: str, metric: str) -> dict:
        return self._get(("pair", lab, p, metric),
                         lambda: paired(self.recs[lab][p], self.recs[FLOWN][p], metric, self.n_boot))

    def scores(self, lab: str, p: str) -> dict:
        return self._get(("score", lab, p),
                         lambda: score_pairs(self.recs[lab][p], self.recs[FLOWN][p], self.score_fn, self.n_boot))

    def coll(self, lab: str, p: str) -> dict:
        return self._get(("coll", lab, p), lambda: collision_pairs(self.recs[lab][p], self.recs[FLOWN][p]))

    def errors(self, lab: str, p: str) -> tuple[int, int]:
        """(own, shared): the entry's failed episodes on seeds where FLOWN ran, and on seeds where FLOWN failed too
        (the simulator's error, not the model's: the fitness excludes those seeds as well)."""
        F = by_seed(self.recs.get(FLOWN, {}).get(p, []))
        bad = [int(r["seed"]) for r in self.recs.get(lab, {}).get(p, []) if not is_ok(r)]
        shared = sum(s in F and not is_ok(F[s]) for s in bad)
        return len(bad) - shared, shared


# ---------------------------------------------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------------------------------------------


def _check(criterion: int, name: str, ok: bool, detail: str) -> dict:
    return {"criterion": criterion, "check": name, "pass": bool(ok), "detail": detail}


def _ci(q: dict, unit: str = "") -> str:
    return (f"paired diff {q['mean']:+.3f}{unit}, 95% CI [{q['lo']:+.3f}, {q['hi']:+.3f}], n {q['n']} "
            f"({q['n_excluded']} seeds excluded)")


def gate(cmp: Comparer, lab: str) -> dict:
    """DESIGN section 9's criteria 1-5 for one entry against FLOWN (module docstring). {"pass", "failed_criteria",
    "checks": [{criterion, check, pass, detail}]}."""
    checks: list[dict] = []

    def need(crit: int, p: str, name: str) -> bool:
        if cmp.has(lab, p):
            return True
        checks.append(_check(crit, f"{p}: {name}", False, f"{lab} or {FLOWN} not evaluated on {p}"))
        return False

    def noninf(crit: int, p: str, m: str, margin_abs: float) -> None:
        name = f"{m} not worse than {FLOWN}"
        if not need(crit, p, name):
            return
        q = cmp.pair(lab, p, m)
        margin = max(NONINF_REL * abs(q["mean_b"]), margin_abs) if q["n"] else margin_abs
        ok = q["n"] > 1 and q["hi"] <= margin
        checks.append(_check(crit, f"{p}: {name}", ok, f"{q['mean_a']:.3f} vs {q['mean_b']:.3f}; {_ci(q)}; upper CI "
                                                       f"must be <= {margin:.3f}"))

    for p in MAIN:  # 1
        name = f"mean score >= {SCORE_GAIN:.0%} below {FLOWN}, CI below 0"
        if need(1, p, name):
            q = cmp.scores(lab, p)
            ok = q["n"] > 1 and q["mean_model"] <= (1.0 - SCORE_GAIN) * q["mean_flown"] and q["hi"] < 0
            checks.append(_check(1, f"{p}: {name}", ok, f"{q['mean_model']:.3f} vs {q['mean_flown']:.3f} "
                                                        f"({q['rel']:+.1%}); {_ci(q)}"))
    for p in PROFILES:  # 2
        name = f"collisions seed by seed <= {FLOWN}'s + {COLLISION_SLACK}, no failed episode"
        if need(2, p, name):
            q, (n_err, n_shared) = cmp.coll(lab, p), cmp.errors(lab, p)
            ok = q["a_only"] <= q["b_only"] + COLLISION_SLACK and n_err == 0
            checks.append(_check(2, f"{p}: {name}", ok, f"collides alone on {q['a_only']} seeds, {FLOWN} alone on "
                                                        f"{q['b_only']} (both {q['both']}, n {q['n']}); {n_err} "
                                                        f"failed episodes ({n_shared} more on seeds where {FLOWN} "
                                                        "failed too: excluded)"))
    for p in MAIN:  # 3
        for m, margin_abs in NONINF_ABS.items():
            noninf(3, p, m, margin_abs)
    for p in MAIN:  # 4
        name = f"smoother: mean normalized stick change <= {FLOWN}'s"
        if need(4, p, name):
            q = cmp.pair(lab, p, STICK_NORM)
            ok = q["n"] > 0 and q["mean_a"] <= q["mean_b"]
            checks.append(_check(4, f"{p}: {name}", ok, f"{q['mean_a']:.3f} vs {q['mean_b']:.3f}; {_ci(q)}"))
    for m, margin_abs in SLOW_ABS.items():  # 5: speed only when needed
        noninf(5, "slow", m, margin_abs)
    for m in FAST_BELOW:  # 5: it keeps up
        name = f"{m} below {FLOWN}'s (keeps up)"
        if need(5, "fast", name):
            q = cmp.pair(lab, "fast", m)
            ok = q["n"] > 0 and (q["mean"] < 0 or (q["mean_a"] == 0.0 and q["mean_b"] == 0.0))
            checks.append(_check(5, f"fast: {name}", ok, f"{q['mean_a']:.3f} vs {q['mean_b']:.3f}; {_ci(q)}"))
    failed = sorted({c["criterion"] for c in checks if not c["pass"]})
    return {"pass": not failed, "failed_criteria": failed, "checks": checks}


def identity_check(recs: dict, lab: str, tol: float = 1e-9) -> dict:
    """A tuned fly arm at its init flies exactly FLOWN's controller (TunedCfg defaults, smooth2): every metric of every
    seed should match FLOWN's. Counts the seeds that do not and the largest difference."""
    n = differ = 0
    worst: tuple[float, str | None] = (0.0, None)
    for p, rs in recs.get(lab, {}).items():
        F = by_seed(recs.get(FLOWN, {}).get(p, []))
        for r in rs:
            f = F.get(int(r["seed"]))
            if f is None or not is_ok(r) or not is_ok(f):
                continue
            n += 1
            d = [(abs(float(v) - float(f["metrics"][k])), k) for k, v in r["metrics"].items()
                 if k in f["metrics"] and isinstance(v, (int, float)) and math.isfinite(float(v))
                 and math.isfinite(float(f["metrics"][k]))]
            top = max(d, default=(0.0, None))
            differ += top[0] > tol
            worst = max(worst, top, key=lambda t: t[0])
    return {"n": n, "differ": differ, "max_abs_diff": worst[0], "worst_metric": worst[1]}


def default_score_fn(fc: Any) -> Callable[[dict, dict], float | None]:
    from kite.fly.behind.fitness import episode_score

    return lambda ma, mf: episode_score(ma, mf, fc)


def analyse(recs: dict, entries: list[dict], profiles: list[str] | None = None, fc: Any = None,
            n_boot: int = N_BOOT, score_fn: Callable[[dict, dict], float | None] | None = None) -> dict:
    """Summaries, paired comparisons against FLOWN, the gates of every gated entry and the verdict. fc: the
    FitnessCfg (default kite.fly.behind.fitness.FitnessCfg()); score_fn(m_arm, m_flown) default its episode_score."""
    if fc is None:
        fc = FitnessCfg()
    score_fn = score_fn or default_score_fn(fc)
    present = {p for per in recs.values() for p in per}
    profiles = list(profiles) if profiles else [p for p in PROFILES if p in present] + sorted(present - set(PROFILES))
    recs = with_derived(recs, fc)
    cmp = Comparer(recs, score_fn, n_boot)
    out: dict = {"profiles": profiles, "summary": {}, "compare": {}, "gates": {}, "identity": {}}
    if FLOWN not in recs:
        out["verdict"] = f"NO VERDICT: {FLOWN} was not evaluated"
        return out
    for e in entries:
        lab = e["label"]
        for p in profiles:
            if not recs.get(lab, {}).get(p):
                continue
            out["summary"].setdefault(lab, {})[p] = summarize(recs[lab][p])
            if lab == FLOWN or not cmp.has(lab, p):
                continue
            mets = [m for m in TABLE_METRICS if any(m in (r.get("metrics") or {}) for r in recs[lab][p])]
            out["compare"].setdefault(lab, {})[p] = {
                "score": cmp.scores(lab, p), "collisions": cmp.coll(lab, p),
                "metrics": {m: cmp.pair(lab, p, m) for m in mets}}
        if lab != FLOWN and e.get("gated", True):
            out["gates"][lab] = gate(cmp, lab)
        if e.get("kind") == "init" and e.get("arm") != "V3-LAW":
            out["identity"][lab] = identity_check(recs, lab)
    out["verdict"] = verdict(out, cmp)
    return out


def verdict(res: dict, cmp: Comparer) -> str:
    gates = res["gates"]
    if not gates:
        return "NO VERDICT: no model was evaluated"
    ok = [lab for lab, g in gates.items() if g["pass"]]

    def main_score(lab: str) -> float:
        v = [cmp.scores(lab, p)["mean_model"] for p in MAIN if cmp.has(lab, p)]
        return float(np.mean(v)) if v else math.inf

    if ok:
        best = min(ok, key=main_score)
        s = ", ".join(f"{p} {cmp.scores(best, p)['mean_model']:.3f}" for p in MAIN)
        others = [lab for lab in ok if lab != best]
        return (f"RECOMMENDED: {best} (mean score {s}; {FLOWN} 1.0)" + (f"; also passing: {', '.join(others)}"
                                                                         if others else "")
                + ". It stays opt-in until a real flight next to smooth2.")
    parts = [f"{lab} fails criteria {g['failed_criteria']}" for lab, g in gates.items()]
    return f"NO MODEL RECOMMENDED: {FLOWN} (smooth2) stays; " + "; ".join(parts)


def signoff_record(res: dict, lab: str, config: dict | None = None) -> dict:
    """What kite.follow_tuned.save_model(signoff=...) can store for one model: its verdict and the headline numbers."""
    g = res["analysis"]["gates"].get(lab) or {}
    comp = res["analysis"]["compare"].get(lab, {})
    return {"recommended": bool(g.get("pass")), "failed_criteria": g.get("failed_criteria"), "time": res.get("time"),
            "seeds": (config or res.get("config", {})).get("seeds"), "n": (config or res.get("config", {})).get("n"),
            "score": {p: comp[p]["score"]["mean_model"] for p in comp},
            "collisions": {p: comp[p]["collisions"] for p in comp}, "verdict": res["analysis"].get("verdict")}


# ---------------------------------------------------------------------------------------------------------------
# Entries (FLOWN, model files, tuned arms at their init)
# ---------------------------------------------------------------------------------------------------------------


def _same(a: Any, b: Any) -> bool:
    num = (int, float)
    if isinstance(a, num) and isinstance(b, num) and not isinstance(a, bool) and not isinstance(b, bool):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def reconstruction_diffs(arm: str, x, rec: dict, path: Path) -> dict:
    """Where make_cfg(arm, x) would fly differently from load_model(path) (every TunedCfg field but the loader's own
    fly_params_path / model_name), and whether the arm's brain is the file's x. {} = identical."""
    from kite.fly.behind import params as bp
    from kite.follow_tuned import load_model

    cfg_a, brain_a = bp.make_cfg(arm, x)
    cfg_b = load_model(path)
    diffs = {f.name: (getattr(cfg_a, f.name, None), getattr(cfg_b, f.name)) for f in fields(cfg_b)
             if f.name not in ("fly_params_path", "model_name")
             and not _same(getattr(cfg_a, f.name, None), getattr(cfg_b, f.name))}
    file_x = rec.get("x")
    if cfg_b.steering == "fly":
        want = brain_a if brain_a is not None else bp.smooth2_x()  # V3-FOLLOW flies smooth2 (fly_params_path)
        if file_x is None or len(file_x) != len(want) or any(abs(float(u) - float(v)) > 1e-12
                                                             for u, v in zip(file_x, want)):
            diffs["brain_x"] = "the file's brain is not the one the arm flies"
    elif brain_a is not None:
        diffs["brain_x"] = "a law model, but the arm has a brain"
    return diffs


def _label(rec: dict, path: Path) -> str:
    name = rec.get("name") or path.stem
    return path.parent.name if name == "best" else str(name)


def load_entry(path: str | Path) -> dict:
    """A model file -> an entry {label, arm, x, source, ...}; SystemExit if it cannot be evaluated faithfully."""
    from kite.fly.behind import params as bp

    p = Path(path)
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(f"{p}: cannot read the model file ({e})")
    trained = rec.get("trained") or {}
    arm = trained.get("behind_arm")
    if arm not in bp.TUNED_ARMS:
        raise SystemExit(f"{p}: trained.behind_arm must be one of {bp.TUNED_ARMS}, got {arm!r}")
    try:
        x = bp.encode_model(arm, rec)
        diffs = reconstruction_diffs(arm, x, rec, p)
    except (ValueError, KeyError, TypeError) as e:
        raise SystemExit(f"{p}: cannot reconstruct {arm} from the file ({type(e).__name__}: {e})")
    if diffs:
        raise SystemExit(f"{p}: {arm} at the reconstructed x would not fly what load_model loads: {diffs}")
    gov = governor_check(rec, p)
    keep = ("run", "cma_seed", "tag", "gen", "fitness", "time", "fitness_cfg")
    return {"label": _label(rec, p), "arm": arm, "x": [float(v) for v in np.asarray(x, float)], "source": str(p),
            "name": rec.get("name"), "steering": rec.get("steering"), "follow": rec.get("follow"),
            "governor": gov, "trained": {k: trained.get(k) for k in keep if k in trained}, "gated": True,
            "kind": "model"}


def governor_check(rec: dict, path: Path) -> dict:
    """The model's governor requirement against the simulator's SafetyGovernor (world.MAX_RC): a model trained under
    another max_rc, or needing more than the sim allows, would be signed off on something it does not fly. The app's
    SafetyCfg.max_rc is 50: the report says what a model needs."""
    from kite.fly.behind.world import MAX_RC
    from kite.follow_tuned import load_model, required_max_rc

    gov = dict(rec.get("governor") or {})
    needs = int(gov.get("needs_max_rc") or required_max_rc(load_model(path)))
    trained_at = gov.get("trained_max_rc")
    if trained_at is not None and int(trained_at) != MAX_RC:
        raise SystemExit(f"{path}: trained under a governor max_rc of {trained_at}, the simulator's is {MAX_RC}")
    if needs > MAX_RC:
        raise SystemExit(f"{path}: needs a governor max_rc of {needs}, above the simulator's {MAX_RC}")
    return {"needs_max_rc": needs, "trained_max_rc": trained_at, "sim_max_rc": MAX_RC}


def build_entries(models: list[str], init_arms: list[str]) -> list[dict]:
    from kite.fly.behind import params as bp

    out = [{"label": FLOWN, "arm": FLOWN, "x": None, "source": "FollowMission + BehindCfg() + smooth2", "gated": False,
            "kind": "baseline"}]
    for a in init_arms:
        if a not in bp.TUNED_ARMS:
            raise SystemExit(f"--init-arms: {a!r} is not one of {bp.TUNED_ARMS}")
        out.append({"label": f"{a}@init", "arm": a, "x": [float(v) for v in bp.init_x(a)], "source": "params.init_x",
                    "gated": False, "kind": "init"})
    for m in models:
        out.append(load_entry(m))
    seen: dict[str, int] = {}
    for e in out:
        n = seen[e["label"]] = seen.get(e["label"], 0) + 1
        if n > 1:
            e["label"] = f"{e['label']}#{n}"
    return out


# ---------------------------------------------------------------------------------------------------------------
# Episodes: keys, the cache, evaluation
# ---------------------------------------------------------------------------------------------------------------


def sim_digest(repo: Path = REPO) -> str:
    """kite.fly.behind.digest.episode_digest: what a cached episode's metrics depend on (iCloud copies ignored)."""
    from kite.fly.behind.digest import episode_digest

    return episode_digest(repo)


def episode_key(arm: str, x, seed: int, profile: str, episode_s: float, stub_s: float | None = None,
                overrides: dict | None = None) -> str:
    core = {"arm": arm, "x": None if x is None else [float(v) for v in x], "seed": int(seed), "profile": profile,
            "episode_s": float(episode_s), "stub_s": None if stub_s is None else float(stub_s),
            "overrides": overrides or None}
    return hashlib.sha256(json.dumps(core, sort_keys=True, default=_json_default).encode()).hexdigest()[:32]


def compact(r: dict, key: str, digest: str) -> dict:
    """The part of an episode result the report keeps (an error comes back with metrics["error"] = 1)."""
    ok = bool(r.get("ok", False)) and not _flag(r.get("metrics") or {}, "error")
    m = dict(r.get("metrics") or {})
    if not ok:
        m["error"] = 1.0
    return {"key": key, "digest": digest, "arm": r.get("arm"), "seed": int(r["seed"]), "profile": r.get("profile"),
            "ok": ok, "metrics": m, "wall_s": r.get("wall_s"), "brain_s": r.get("brain_s"),
            "error": None if ok else str(r.get("error"))[:500], "time": _now()}


class EpisodeCache:
    """Finished episodes as JSON lines {key, digest, ...}; the last line of a key wins. A torn line is skipped."""

    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self.recs: dict[str, dict] = {}
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(r, dict) and r.get("key"):
                    self.recs[r["key"]] = r

    def get(self, key: str, digest: str, reuse_stale: bool = False) -> dict | None:
        r = self.recs.get(key)
        if r is None or not r.get("ok") or (r.get("digest") != digest and not reuse_stale):
            return None
        return r

    def add(self, recs: list[dict]) -> None:
        ok = [r for r in recs if r.get("ok")]
        for r in ok:
            self.recs[r["key"]] = r
        if self.path and ok:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                for r in ok:
                    f.write(json.dumps(r, default=_json_default) + "\n")


def run_eval(entries: list[dict], seeds: list[int], profiles: list[str], backend: Any, cache: EpisodeCache,
             digest: str, *, episode_s: float = 60.0, stub_s: float | None = None, overrides: dict | None = None,
             cache_only: bool = False, reuse_stale: bool = False, batch: int = 1024,
             log: Callable[[str], None] = print) -> tuple[dict, dict]:
    """recs[label][profile] (sorted by seed) for every entry x profile x seed: from the cache when it has the
    episode, else run on the backend in batches (each batch cached as it lands, so an interrupted sign-off resumes).
    Identical episodes (same arm and x) run once and serve every entry that needs them."""
    recs: dict = {e["label"]: {p: [] for p in profiles} for e in entries}
    todo: dict[str, dict] = {}
    want: dict[str, list[tuple[str, str]]] = {}
    hits = 0
    for e in entries:
        for p in profiles:
            for s in seeds:
                key = episode_key(e["arm"], e["x"], s, p, episode_s, stub_s, overrides)
                hit = cache.get(key, digest, reuse_stale)
                if hit is not None:
                    recs[e["label"]][p].append(dict(hit))
                    hits += 1
                    continue
                want.setdefault(key, []).append((e["label"], p))
                if key not in todo:
                    todo[key] = (e, s, p)
    stats = {"cache_hits": hits, "episodes": 0, "errors": 0, "usd": 0.0, "wall_s": 0.0, "container_s": 0.0}
    if todo and cache_only:
        raise SystemExit(f"--cache-only: {len(todo)} episodes are not in the cache {cache.path}")
    if todo:
        from kite.fly.behind import evaluate as ev

        items = {k: ev.make_item(e["arm"], s, p, x=e["x"], episode_s=episode_s, overrides=overrides, stub_s=stub_s,
                                 tag={"key": k}) for k, (e, s, p) in todo.items()}
        keys = list(items)
        random.Random(0).shuffle(keys)  # mix cheap law and brain episodes so chunks last alike
        log(f"{hits} episodes from the cache, {len(keys)} to run")
        t0 = time.perf_counter()
        for i in range(0, len(keys), max(1, int(batch))):
            ks = keys[i:i + max(1, int(batch))]
            results, st = backend.evaluate([items[k] for k in ks])
            new = [compact(r, k, digest) for k, r in zip(ks, results)]
            for k, c in zip(ks, new):
                for lab, p in want[k]:
                    recs[lab][p].append(dict(c))
            cache.add(new)
            stats["episodes"] += len(new)
            stats["errors"] += sum(not c["ok"] for c in new)
            for f in ("usd", "container_s"):
                stats[f] += float(st.get(f) or 0.0)
            done = min(i + len(ks), len(keys))
            log(f"  {done}/{len(keys)} episodes, {stats['errors']} errors, {time.perf_counter() - t0:.0f} s, "
                f"~${stats['usd']:.2f}")
        stats["wall_s"] = time.perf_counter() - t0
    for per in recs.values():
        for p in per:
            per[p].sort(key=lambda r: int(r["seed"]))
    return recs, stats


# ---------------------------------------------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------------------------------------------


def _f(v, d: int = 3) -> str:
    return "-" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{d}f}"


def _diff(q: dict | None, d: int = 3) -> str:
    if not q or not q.get("n"):
        return "-"
    return f"{q['mean']:+.{d}f} [{_f(q['lo'], d)}, {_f(q['hi'], d)}]"


def markdown(res: dict) -> str:
    a, cfg = res["analysis"], res["config"]
    lines = [f"# Behind follow sign-off: {cfg['name']}", "",
             f"{cfg.get('n')} {cfg.get('seeds')} seeds x {', '.join(a['profiles'])}; episode {cfg.get('episode_s')} s; "
             f"{res['time']}; sim digest {res.get('digest')}. Baseline {FLOWN} = the app as it flies (FollowMission, "
             "BehindCfg() defaults, smooth2). Score = kite.fly.behind.fitness.episode_score against FLOWN on the same "
             "seed (FLOWN 1.0, lower is better: distance relative to FLOWN's, every other term at a fixed price; a "
             "collision 6; the entry's own failed start 6, FLOWN's excluded). Diff = paired per-seed mean "
             "(entry - FLOWN) with its 95 % bootstrap CI; metric diffs over the seeds where neither collided and the "
             "start did not fail. stick_norm = mean of the four stick changes, each by FitnessCfg's s_<axis>.", "",
             f"**Verdict: {a.get('verdict')}**", "", "Simulation only: a recommended model stays opt-in until a real "
             "flight next to smooth2.", "", "## Gates (DESIGN section 9)", ""]
    for lab, g in a["gates"].items():
        state = 'RECOMMENDED' if g['pass'] else f"not recommended, fails criteria {g['failed_criteria']}"
        lines.append(f"- **{lab}**: {state}")
        lines += [f"  - [{'x' if c['pass'] else ' '}] ({c['criterion']}) {c['check']}: {c['detail']}"
                  for c in g["checks"]]
    lines.append("")
    govs = [(e["label"], e["governor"]) for e in res.get("entries") or [] if e.get("governor")]
    if govs:
        lines += ["## Governor", "", "Each model flies as signed off only with the SafetyGovernor passing its sticks: "
                  "kite.config.SafetyCfg's max_rc is 50 (~0.48 m/s), the simulator's 65.", ""]
        lines += [f"- {lab}: needs SafetyCfg(max_rc >= {g['needs_max_rc']}) (trained under {g.get('trained_max_rc')})"
                  for lab, g in govs]
        lines.append("")
    if a.get("identity"):
        lines += ["## Tuned arms at their init (must fly exactly FLOWN)", ""]
        for lab, q in a["identity"].items():
            lines.append(f"- {lab}: {q['differ']} of {q['n']} seeds differ from {FLOWN} (largest |diff| "
                         f"{q['max_abs_diff']:.3g} on {q['worst_metric']})")
        lines.append("")
    for p in a["profiles"]:
        lines += [f"## {p}", "", "| entry | n | err | start failed | coll | score | score diff [CI] | "
                  + " | ".join(TABLE_METRICS) + " |", "|" + "---|" * (len(TABLE_METRICS) + 7)]
        for lab, per in a["summary"].items():
            if p not in per:
                continue
            s = per[p]
            comp = a["compare"].get(lab, {}).get(p, {})
            sc = comp.get("score")
            score = "1.000" if lab == FLOWN else _f(sc["mean_model"] if sc else None)
            lines.append(f"| {lab} | {s['n']} | {s['n_errors']} | {s['n_start_failed']} | {s['collisions']} | "
                         f"{score} | "
                         f"{_diff(sc)} | " + " | ".join(_f(s["metrics"].get(m)) for m in TABLE_METRICS) + " |")
        lines.append("")
        for lab, per in a["compare"].items():
            if p not in per:
                continue
            q = per[p]["collisions"]
            lines.append(f"- {lab}: collisions alone on {q['a_only']} seeds, {FLOWN} alone on {q['b_only']}, both "
                         f"{q['both']} (n {q['n']}); vs {FLOWN}: " + "; ".join(
                             f"{m} {_diff(d)}" for m, d in per[p]["metrics"].items() if d["n"] > 1))
        lines.append("")
    return "\n".join(lines) + "\n"


def summary_table(res: dict) -> str:
    """The short version printed to the terminal."""
    a = res["analysis"]
    head = ["profile", "entry", "n", "coll", "score", "diff [95% CI]"] + list(PRINT_METRICS)
    rows = []
    for p in a["profiles"]:
        for lab, per in a["summary"].items():
            if p not in per:
                continue
            s = per[p]
            sc = a["compare"].get(lab, {}).get(p, {}).get("score")
            rows.append([p, lab, str(s["n"]), str(s["collisions"]),
                         "1.000" if lab == FLOWN else _f(sc["mean_model"] if sc else None), _diff(sc)]
                        + [_f(s["metrics"].get(m)) for m in PRINT_METRICS])
    w = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(head)]
    out = ["  ".join(c.ljust(n) for c, n in zip(r, w)) for r in [head] + rows]
    for lab, g in a["gates"].items():
        out.append(f"{lab}: {'RECOMMENDED' if g['pass'] else 'fails criteria ' + str(g['failed_criteria'])}")
    out.append(f"Verdict: {a.get('verdict')}")
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------------


def seed_list(name: str, n: int | None) -> list[int]:
    from kite.fly.behind import SELECTION_SEEDS, TEST_SEEDS

    seeds = {"test": TEST_SEEDS, "selection": SELECTION_SEEDS}[name]
    if n is not None and n > len(seeds):
        raise SystemExit(f"--n {n} exceeds the {len(seeds)} {name} seeds")
    return list(seeds[: n or len(seeds)])


def load_fitness_cfg(spec: str | None, stored: dict | None = None):
    """--fitness-json (a JSON object or a path), else the stored one (--from), else FitnessCfg()."""
    if spec:
        text = Path(spec).read_text(encoding="utf-8") if not spec.lstrip().startswith("{") else spec
        return FitnessCfg(**json.loads(text))
    return FitnessCfg(**stored) if stored else FitnessCfg()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m kite.fly.behind.signoff", description=__doc__.split("\n\n")[0])
    p.add_argument("--models", nargs="*", default=[], help="model files (kite/fly/data/behind/*.json, best.json)")
    p.add_argument("--init-arms", default="", help="also evaluate these tuned arms at their init, ungated (V3-FOLLOW "
                                                   "and V3-JOINT must equal FLOWN exactly)")
    p.add_argument("--profiles", default=",".join(PROFILES))
    p.add_argument("--seeds", default="test", choices=SEED_SETS)
    p.add_argument("--n", type=int, default=None, help="first n seeds of the set (default all: 200 test seeds)")
    p.add_argument("--episode-s", type=float, default=60.0)
    p.add_argument("--stub-s", type=float, default=None, help="synthetic episodes (checks the pipeline, not models)")
    p.add_argument("--backend", choices=["local", "modal"], default="local")
    p.add_argument("--processes", type=int, help="local pool size (default: all cores)")
    p.add_argument("--chunk", type=int, help="items per Modal evaluate_chunk call (default 32)")
    p.add_argument("--batch", type=int, help="episodes per backend call, cached as each lands (default 4096 on "
                                             "Modal, 512 locally)")
    p.add_argument("--fitness-json", help="FitnessCfg overrides (a JSON object or a path); default FitnessCfg()")
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--name", default="signoff")
    p.add_argument("--out", default=None, help=f"output dir (default {report_dir()})")
    p.add_argument("--cache", default=None, help="episode cache (default <out>/cache/episodes.jsonl)")
    p.add_argument("--no-cache", action="store_true", help="neither read nor write the cache")
    p.add_argument("--cache-only", action="store_true", help="never run an episode: fail if one is not cached")
    p.add_argument("--reuse-stale", action="store_true", help="accept cached episodes of a different sim digest")
    p.add_argument("--from", dest="from_json", help="recompute the analysis of a saved report JSON (no episodes)")
    p.add_argument("--docs", action="store_true", help="also copy the markdown to docs/follow-v3-results.md")
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    out_dir = Path(a.out) if a.out else report_dir()
    if a.from_json:
        res = json.loads(Path(a.from_json).read_text(encoding="utf-8"))
        fc = load_fitness_cfg(a.fitness_json, res.get("fitness_cfg"))
        res["fitness_cfg"] = asdict(fc)
        res["analysis"] = analyse(res["records"], res["entries"], res["config"].get("profiles"), fc, a.n_boot)
        name = Path(a.from_json).stem
    else:
        fc = load_fitness_cfg(a.fitness_json)
        profiles = [p.strip() for p in a.profiles.split(",") if p.strip()]
        bad = [p for p in profiles if p not in PROFILES]
        if bad:
            raise SystemExit(f"unknown profiles {bad}; one of {PROFILES}")
        init_arms = [s.strip() for s in a.init_arms.split(",") if s.strip()]
        entries = build_entries(a.models, init_arms)
        for e in entries:
            stored = (e.get("trained") or {}).get("fitness_cfg")
            if stored and stored != asdict(fc):
                print(f"note: {e['label']} was trained with a different FitnessCfg; the score here uses the one in "
                      "the report (pass --fitness-json to change it)")
        seeds = seed_list(a.seeds, a.n)
        digest = sim_digest()
        cache = EpisodeCache(None if a.no_cache else (a.cache or out_dir / "cache" / "episodes.jsonl"))
        print(f"{len(entries) * len(seeds) * len(profiles)} episodes: {[e['label'] for e in entries]} x "
              f"{len(seeds)} {a.seeds} seeds x {profiles}; sim digest {digest}")
        t0 = time.perf_counter()
        kw = dict(episode_s=a.episode_s, stub_s=a.stub_s, cache_only=a.cache_only, reuse_stale=a.reuse_stale)
        if a.backend == "local" or a.cache_only:
            from kite.fly.behind import train as bt
            from kite.fly.train import cma_train as ct

            backend = bt.BehindLocalBackend(a.processes, ct.Prices())
            try:
                recs, stats = run_eval(entries, seeds, profiles, backend, cache, digest, batch=a.batch or 512, **kw)
            finally:
                backend.close()
        else:
            import modal

            from kite.fly.behind import modal_app
            from kite.fly.train import cma_train as ct

            prices = ct.load_prices()
            with modal.enable_output(), modal_app.app.run():
                backend = modal_app.modal_backend(a.chunk or modal_app.default_chunk(FLOWN), prices)
                recs, stats = run_eval(entries, seeds, profiles, backend, cache, digest, batch=a.batch or 4096, **kw)
                stats["usd"] += backend.finish_usd()
        name = a.name
        res = {"time": _now(),
               "config": {"name": name, "models": a.models, "init_arms": init_arms, "seeds": a.seeds, "n": len(seeds),
                          "profiles": profiles, "episode_s": a.episode_s, "stub_s": a.stub_s, "backend": a.backend,
                          "n_boot": a.n_boot, "wall_s": time.perf_counter() - t0, **stats},
               "digest": digest, "fitness_cfg": asdict(fc), "entries": entries, "records": recs}
        res["analysis"] = analyse(recs, entries, profiles, fc, a.n_boot)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(res, indent=1, default=_json_default) + "\n", encoding="utf-8")
    md = markdown(res)
    (out_dir / f"{name}.md").write_text(md, encoding="utf-8")
    if a.docs:
        dest = REPO / "docs" / "follow-v3-results.md"
        shutil.copyfile(out_dir / f"{name}.md", dest)
        print(f"copied to {dest}")
    print(summary_table(res))
    print(f"wrote {out_dir / name}.json and .md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
