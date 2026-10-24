"""Evaluate yaw controllers on fixed seeds, compare them seed by seed, and compute the go / no-go gates (DESIGN R6).

    # G1 / G2 on BASELINE_SEEDS (init and size mode only; never on the test seeds)
    python -m kite.fly.train.report --arms KITE-LAW,FLY-YAW-HAND,FLY-YAW-PREV@blind,FLY-YAW-PREV@size \\
        --seeds baseline --profiles train,demo --n 100 --name g2
    # G3, once, after the fine-tune: 200 TEST_SEEDS x 4 profiles
    python -m kite.fly.train.report --arms KITE-LAW,KITE-LAW-CMA,PRED-PD,FLY-YAW-PREV@blind,FLY-YAW \\
        --best results/fly/runs/FLY-YAW_s1_kite1/best.json --best results/fly/runs/KITE-LAW-CMA_s1_kite1/best.json \\
        --best results/fly/runs/PRED-PD_s1_kite1/best.json \\
        --seeds test --profiles train,demo,stress,fast --n 200 --backend modal --name g3
    python -m kite.fly.train.report --from results/fly/report/g3.json     # recompute tables and verdicts only

Side sets (--side-sets, not gated): G3 also reports every arm on `demo` at behind_m 1.2 and 1.5 (FollowCfg override, the
yaw sources built from that same config; DESIGN R9.5) and on `demo` with head-only detection p_head 0.3 (R10.3, a sim
override); G1 / G2 report the p_head set too. Paired comparisons skip seeds where either arm collided or the start
failed, and say how many (R11.1); collisions are compared seed by seed (R11.2).

Arms: interfaces.ARMS. FLY-YAW, KITE-LAW-CMA and PRED-PD come from a --best file (several of one arm: labels
<arm>#s<cma seed>). A fly arm takes @blind or @size for its size mode (default: its best.json's, else --size-blind).
Every arm runs on the same seeds (common random numbers), so differences are PAIRED per seed: mean and 95 % percentile
bootstrap CI of (arm - KITE-LAW) and of (arm - each --vs arm), per metric and profile. Fitness is the training one
(DESIGN R1). G2 is computed on --seeds baseline only, G3 on --seeds test only. Writes <out>/<name>.json (every
episode's metrics too) and <name>.md.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import datetime as _dt
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

from kite.fly.interfaces import ARMS, BASELINE_SEEDS, METRICS, SELECTION_SEEDS, TEST_SEEDS
from kite.fly.train import arms
from kite.fly.train import cma_train as ct
from kite.fly.train import evaluate as ev

N_BOOT = 10_000
LAW = "KITE-LAW"
TUNED = ("FLY-YAW", "KITE-LAW-CMA", "PRED-PD")  # arms that come from a best.json
REPORT_METRICS = METRICS + tuple(m for m in ("min_dist3d_m",) if m not in METRICS)  # + the sim's R10 extra
TABLE_METRICS = ("rms_bearing_deg", "p95_bearing_deg", "in_view", "following", "losses", "time_lost_s",
                 "rms_range_err_m", "in_band", "too_close_s", "pitch_exits", "corner_losses", "stick_change",
                 "min_dist3d_m")
SEED_SETS = {"baseline": BASELINE_SEEDS, "test": TEST_SEEDS, "selection": SELECTION_SEEDS}
# DESIGN R6 G3 (criteria numbered as there)
G3_MAIN, G3_EDGE = ("train", "demo"), ("stress", "fast")
RMS_GAIN = 0.10  # 1: mean rms_bearing_deg >= 10 % lower than each comparator ...
# 2: non-inferiority vs KITE-LAW: paired upper CI <= max(10 % of KITE-LAW's mean, absolute margin). R6 fixes only
# the losses margin (0.1 per episode); the others are this module's choice of "a small absolute margin".
NONINF_REL = 0.10
NONINF_ABS = {"losses": 0.1, "time_lost_s": 0.5, "rms_range_err_m": 0.02, "too_close_s": 0.2, "pitch_exits": 0.1}
VIEW_FLOOR = -0.02  # 3: in_view and following: paired lower CI >= -0.02
# 4 and 6 (R11.2): collisions seed by seed, #(arm collides, KITE-LAW doesn't) <= #(KITE-LAW collides, arm doesn't) + 2
COLLISION_SLACK = 2
# Paired metric comparisons (criteria 1, 2, 3, 5, 6 and the tables) use only the seeds where neither arm collided and
# the start did not fail (R11.1); the excluded count is printed with each.
STICK_X = 2.0  # 5: stick_change <= 2 x KITE-LAW
EDGE_RMS_X = 1.25  # 6: on stress and fast, rms_bearing <= 1.25 x KITE-LAW, collisions seed by seed as 4
# Side sets (not gated): demo at behind_m 1.2 and 1.5 (R9.5), and demo with the head-only detection probability at 0.3
# (R10.3, the sim's SimParams field P_HEAD_FIELD). G1 (baseline) reports the p_head set too.
P_HEAD_FIELD = "p_head"
SIDE_SETS = {"test": ("demo@1.2", "demo@1.5", f"demo@{P_HEAD_FIELD}0.3"), "baseline": (f"demo@{P_HEAD_FIELD}0.3",),
             "selection": ()}


def report_dir() -> Path:
    return ct.REPO / "results" / "fly" / "report"


# ---------------------------------------------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------------------------------------------


def bootstrap_ci(values, n_boot: int = N_BOOT, seed: int = 0, level: float = 0.95) -> tuple[float, float, float]:
    """(mean, lo, hi): percentile bootstrap of the mean. For paired differences pass the per-seed differences."""
    v = np.asarray([float(a) for a in values if a is not None and math.isfinite(float(a))])
    if v.size == 0:
        return (math.nan, math.nan, math.nan)
    if v.size == 1:
        return (float(v[0]), math.nan, math.nan)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    a = (1.0 - level) / 2.0
    return float(v.mean()), float(np.quantile(means, a)), float(np.quantile(means, 1.0 - a))


def by_seed(recs: list[dict]) -> dict[int, dict]:
    return {int(r["seed"]): r for r in recs if r.get("ok")}


def collided(r: dict) -> bool:
    return float((r.get("metrics") or {}).get("collision", 0.0)) > 0


def usable(r: dict) -> bool:
    """An episode that ran and whose start did not fail (R10.4)."""
    return bool(r.get("ok")) and not ct.start_failed(r)


def paired(a: list[dict], b: list[dict], metric: str, n_boot: int = N_BOOT) -> dict:
    """Mean and 95 % CI of (a - b) per seed, over the seeds where both episodes ran, neither start failed and neither
    arm collided (DESIGN R11.1); n_excluded counts the other seeds either arm has. mean_a / mean_b: the two arms'
    means over the same kept seeds."""
    A, B = by_seed(a), by_seed(b)
    keep = [s for s in sorted(set(A) & set(B)) if usable(A[s]) and usable(B[s]) and not collided(A[s])
            and not collided(B[s]) and metric in A[s]["metrics"] and metric in B[s]["metrics"]]
    seen = {int(r["seed"]) for r in a} | {int(r["seed"]) for r in b}
    va = np.array([float(A[s]["metrics"][metric]) for s in keep])
    vb = np.array([float(B[s]["metrics"][metric]) for s in keep])
    m, lo, hi = bootstrap_ci((va - vb).tolist(), n_boot)
    return {"n": len(keep), "n_excluded": len(seen) - len(keep), "mean": m, "lo": lo, "hi": hi,
            "mean_a": float(va.mean()) if keep else math.nan, "mean_b": float(vb.mean()) if keep else math.nan}


def collision_pairs(a: list[dict], b: list[dict]) -> dict:
    """Seed by seed (R11.2), over the seeds where both ran and neither start failed: a_only = #(a collides and b
    does not), b_only = #(b collides and a does not)."""
    A, B = by_seed(a), by_seed(b)
    keep = [s for s in sorted(set(A) & set(B)) if usable(A[s]) and usable(B[s])]
    return {"n": len(keep), "a_only": sum(collided(A[s]) and not collided(B[s]) for s in keep),
            "b_only": sum(collided(B[s]) and not collided(A[s]) for s in keep),
            "both": sum(collided(A[s]) and collided(B[s]) for s in keep)}


def mean_metric(recs: list[dict], metric: str) -> float:
    v = [float(r["metrics"][metric]) for r in recs if usable(r) and metric in r["metrics"]]
    return float(np.mean(v)) if v else math.nan


def count(recs: list[dict], metric: str) -> int:
    return int(sum(float(r["metrics"].get(metric, 0.0)) > 0 for r in recs if usable(r)))


def r1_fitness(recs: list[dict], law: list[dict]) -> float:
    """The training fitness (DESIGN R1 + the mean-based stick penalty) of episode records against KITE-LAW's records
    on the same seeds."""
    L = by_seed(law)
    return ct.fitness(recs, [ct.law_record(L[int(r["seed"])]) if int(r["seed"]) in L else None for r in recs])


def summarize(recs: list[dict], law: list[dict] | None) -> dict:
    rets = [r["ret"] for r in recs if usable(r)]
    return {"n": len(recs), "n_errors": sum(not r.get("ok") for r in recs),
            "n_start_failed": sum(bool(r.get("ok")) and ct.start_failed(r) for r in recs),
            "ret_mean": float(np.mean(rets)) if rets else None, "fitness": r1_fitness(recs, law) if law else None,
            "collisions": count(recs, "collision"), "metrics": {m: mean_metric(recs, m) for m in REPORT_METRICS}}


# ---------------------------------------------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------------------------------------------


def _check(criterion: int, name: str, ok: bool, detail: str) -> dict:
    return {"criterion": criterion, "check": name, "pass": bool(ok), "detail": detail}


def _ci(p: dict, unit: str = "") -> str:
    return (f"paired diff {p['mean']:+.3f}{unit}, 95% CI [{p['lo']:+.3f}, {p['hi']:+.3f}], n {p['n']} "
            f"({p['n_excluded']} seeds excluded: collision, failed start or failed episode)")


def _collision_check(criterion: int, prof: str, c: list[dict], lw: list[dict]) -> tuple[bool, str]:
    q = collision_pairs(c, lw)
    n_err = sum(not r.get("ok") for r in c)
    ok = q["a_only"] <= q["b_only"] + COLLISION_SLACK and n_err == 0
    return ok, (f"collides alone on {q['a_only']} seeds, {LAW} alone on {q['b_only']} (both {q['both']}, n {q['n']}; "
                f"allowed {LAW}'s + {COLLISION_SLACK}); {n_err} failed episodes")


def gate_checks(recs: dict, cand: str, comparators: list[str], n_boot: int = N_BOOT) -> list[dict]:
    """DESIGN R6 G3 criteria 1-6 for one candidate label (recs[label][profile] = episode records). Criterion 1 is
    checked against every comparator, the rest against KITE-LAW. Missing data fails its criterion."""
    out = []
    for prof in G3_MAIN:
        c, lw = recs.get(cand, {}).get(prof), recs.get(LAW, {}).get(prof)
        if not c or not lw:
            out.append(_check(0, f"{prof}: data", False, f"{cand} or {LAW} not evaluated on {prof}"))
            continue
        for comp in comparators:
            other = recs.get(comp, {}).get(prof)
            if not other:
                out.append(_check(1, f"{prof}: rms_bearing >= 10% below {comp}", False, f"{comp} not evaluated"))
                continue
            p = paired(c, other, "rms_bearing_deg", n_boot)
            mc, mo = p["mean_a"], p["mean_b"]
            ok = p["n"] > 1 and mc <= (1 - RMS_GAIN) * mo and p["hi"] < 0
            out.append(_check(1, f"{prof}: rms_bearing >= 10% below {comp}", ok,
                              f"{mc:.3f} vs {mo:.3f} deg ({mc / mo - 1:+.1%}); {_ci(p, ' deg')}"))
        for m, abs_margin in NONINF_ABS.items():
            p = paired(c, lw, m, n_boot)
            margin = max(NONINF_REL * abs(p["mean_b"]), abs_margin) if p["n"] else abs_margin
            ok = p["n"] > 1 and p["hi"] <= margin
            out.append(_check(2, f"{prof}: {m} non-inferior to {LAW}", ok,
                              f"{_ci(p)}; upper CI must be <= {margin:.3f}"))
        for m in ("in_view", "following"):
            p = paired(c, lw, m, n_boot)
            out.append(_check(3, f"{prof}: {m} lower CI >= {VIEW_FLOOR}", p["n"] > 1 and p["lo"] >= VIEW_FLOOR, _ci(p)))
        ok, detail = _collision_check(4, prof, c, lw)
        out.append(_check(4, f"{prof}: collisions seed by seed vs {LAW}", ok, detail))
        p = paired(c, lw, "stick_change", n_boot)
        sc, sl = p["mean_a"], p["mean_b"]
        out.append(_check(5, f"{prof}: stick_change <= {STICK_X:g} x {LAW}", p["n"] > 0 and sc <= STICK_X * sl,
                          f"{sc:.3f} vs {STICK_X:g} x {sl:.3f} over {p['n']} kept seeds"))
    for prof in G3_EDGE:
        c, lw = recs.get(cand, {}).get(prof), recs.get(LAW, {}).get(prof)
        name = f"{prof}: rms_bearing <= {EDGE_RMS_X} x {LAW}, collisions seed by seed"
        if not c or not lw:
            out.append(_check(6, name, False, f"{cand} or {LAW} not evaluated on {prof}"))
            continue
        p = paired(c, lw, "rms_bearing_deg", n_boot)
        rb, rl = p["mean_a"], p["mean_b"]
        ok_c, detail = _collision_check(6, prof, c, lw)
        out.append(_check(6, name, p["n"] > 0 and rb <= EDGE_RMS_X * rl and ok_c,
                          f"rms bearing {rb:.3f} vs {EDGE_RMS_X} x {rl:.3f} over {p['n']} kept seeds; {detail}"))
    return out


def _result(checks: list[dict]) -> dict:
    failed = sorted({c["criterion"] for c in checks if not c["pass"]})
    return {"pass": not failed, "failed_criteria": failed, "checks": checks}


def g3(recs: dict, entries: list[dict], n_boot: int = N_BOOT) -> dict:
    """DESIGN R6 G3. FLY-YAW passes if it meets criteria 1-6 against KITE-LAW, KITE-LAW-CMA, PRED-PD and FLY-YAW-PREV
    (a comparator that was not evaluated fails criterion 1). PRED-PD and KITE-LAW-CMA are checked against KITE-LAW
    alone, which decides the honest outcome when the fly fails."""
    labels = {e["label"]: e for e in entries}
    if LAW not in labels:
        return {"verdict": f"NO VERDICT: {LAW} was not evaluated", "candidates": {}}
    of = {a: [e["label"] for e in entries if e["arm"] == a] for a in ("KITE-LAW-CMA", "PRED-PD", "FLY-YAW-PREV")}
    out: dict = {"candidates": {}}
    for e in entries:
        if e["arm"] == "FLY-YAW":
            checks = gate_checks(recs, e["label"], [LAW] + of["KITE-LAW-CMA"] + of["PRED-PD"] + of["FLY-YAW-PREV"],
                                 n_boot)
            checks += [_check(1, f"comparator {a}", False, f"{a} was not evaluated") for a, v in of.items() if not v]
            out["candidates"][e["label"]] = _result(checks)
        elif e["arm"] in ("KITE-LAW-CMA", "PRED-PD"):
            out["candidates"][e["label"]] = _result(gate_checks(recs, e["label"], [LAW], n_boot))
    res = out["candidates"]
    fly = [lab for lab in res if labels[lab]["arm"] == "FLY-YAW"]
    fly_ok = [lab for lab in fly if res[lab]["pass"]]
    pred_ok = [lab for lab in of["PRED-PD"] if res[lab]["pass"]]
    cma_ok = [lab for lab in of["KITE-LAW-CMA"] if res[lab]["pass"]]
    if fly_ok:
        out["verdict"] = (f"FLY-YAW PASSES G3 ({fly_ok[0]}): it may ship opt-in; the default stays {LAW} until the "
                          "real flight check (static walker at +-10 and +-20 deg, then a straight walk and a corner)")
        return out
    parts = []
    for lab in fly:
        parts.append(f"{lab} fails G3 on criteria {res[lab]['failed_criteria']}")
    beats_pred = {lab: all(c["pass"] for c in res[lab]["checks"] if c["criterion"] == 1 and "PRED-PD" in c["check"])
                  for lab in fly}
    if pred_ok and fly and not any(beats_pred.values()):
        parts.insert(0, f"prediction helps, the fly does not: {pred_ok[0]} beats {LAW}, FLY-YAW does not beat it")
    elif pred_ok:
        parts.append(f"{pred_ok[0]} beats {LAW}")
    if cma_ok:
        p = labels[cma_ok[0]].get("params") or {}
        parts.append(f"{cma_ok[0]} beats {LAW} (k_yaw {p.get('k_yaw', math.nan):.3f}, db_yaw_deg "
                     f"{p.get('db_yaw_deg', math.nan):.3f})")
    if not pred_ok and not cma_ok:
        parts.append(f"nothing beats {LAW}: no change")
    out["verdict"] = "; ".join(parts)
    return out


def g2(recs: dict, entries: list[dict]) -> dict:
    """DESIGN R6 / 6 on BASELINE_SEEDS: the size mode (FLY-YAW-PREV's lower mean rms_bearing over train + demo), the
    fine-tune init (prev or hand: the higher mean R1 fitness over train + demo), and G1's KITE-LAW demo numbers."""
    profs = [p for p in G3_MAIN if p in recs.get(LAW, {})]
    if not profs:
        return {"note": f"{LAW} not evaluated on train / demo"}
    out: dict = {"profiles": profs}
    prev = [e for e in entries if e["arm"] == "FLY-YAW-PREV" and all(p in recs.get(e["label"], {}) for p in profs)]
    if prev:
        rms = {e["label"]: float(np.mean([mean_metric(recs[e["label"]][p], "rms_bearing_deg") for p in profs]))
               for e in prev}
        best = min(rms, key=rms.get)
        out["size_mode"] = {"rms_bearing_deg": rms, "choice": best,
                            "size_blind": next(e["size_blind"] for e in prev if e["label"] == best)}
    fit = {e["label"]: float(np.mean([r1_fitness(recs[e["label"]][p], recs[LAW][p]) for p in profs]))
           for e in entries if e["arm"] in ("FLY-YAW-PREV", "FLY-YAW-HAND")
           and all(p in recs.get(e["label"], {}) for p in profs)}
    if fit:
        top = max(fit, key=fit.get)
        out["init"] = {"fitness": fit, "choice": "hand" if top.startswith("FLY-YAW-HAND") else "prev"}
    for name in [k for k in recs.get(LAW, {}) if k == "demo" or k.startswith("demo@")]:
        d = recs[LAW][name]
        out.setdefault("g1_law", {})[name] = {
            "collisions": count(d, "collision"), "in_view": mean_metric(d, "in_view"),
            "following": mean_metric(d, "following"), "n": len(d),
            "start_failed": sum(bool(r.get("ok")) and ct.start_failed(r) for r in d)}
    return out


# ---------------------------------------------------------------------------------------------------------------
# Entries and evaluation
# ---------------------------------------------------------------------------------------------------------------


def parse_entries(arms_csv: str, best: list[str], size_blind_default: bool = True) -> list[dict]:
    """--arms and --best -> entries {label, arm, params, norm, x, size_blind, source}."""
    bests: dict[str, list[dict]] = {}
    for path in best:
        b = json.loads(Path(path).read_text())
        b["_path"] = str(path)
        bests.setdefault(b["arm"], []).append(b)
    out = []
    for spec in [s.strip() for s in arms_csv.split(",") if s.strip()]:
        arm, _, mode = spec.partition("@")
        if arm not in ARMS:
            raise SystemExit(f"unknown arm {arm!r}; one of {ARMS}")
        if mode not in ("", "blind", "size"):
            raise SystemExit(f"size mode must be @blind or @size, not @{mode}")
        if arm in TUNED:
            if arm not in bests:
                raise SystemExit(f"{arm} needs --best <best.json of a {arm} run>")
            for b in bests[arm]:
                lab = arm if len(bests[arm]) == 1 else f"{arm}#s{b.get('cma_seed')}"
                sb = None
                if arm == "FLY-YAW":
                    sb = (mode == "blind") if mode else b.get("size_blind")
                    sb = size_blind_default if sb is None else bool(sb)
                    lab += f"@{mode}" if mode else ""
                out.append({"label": lab, "arm": arm, "params": b.get("params"), "norm": b.get("norm"),
                            "x": b.get("x"), "size_blind": sb, "source": b["_path"], "gen": b.get("gen"),
                            "sel_fitness": b.get("sel_fitness")})
            continue
        e = {"label": spec, "arm": arm, "params": None, "norm": None, "x": None, "size_blind": None, "source": arm}
        if arm in arms.FLY_ARMS:
            e["size_blind"] = (mode == "blind") if mode else size_blind_default
        out.append(e)
    labels = [e["label"] for e in out]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"duplicate entries {labels}")
    return out


def parse_side_set(name: str) -> tuple[str, float | None, dict | None]:
    """"demo@1.2" -> (demo, behind_m 1.2, None); "demo@p_head0.3" -> (demo, None, {"p_head": 0.3})."""
    prof, sep, spec = name.partition("@")
    if not sep or not spec:
        raise SystemExit(f"side set {name!r} is not <profile>@<behind_m> or <profile>@<sim field><value>")
    try:
        return prof, float(spec), None
    except ValueError:
        m = re.fullmatch(r"([A-Za-z_]+?)(-?[0-9.]+)", spec)
        if not m:
            raise SystemExit(f"side set {name!r}: cannot read {spec!r}") from None
        return prof, None, {m.group(1): float(m.group(2))}


def entry_items(e: dict, seeds: list[int], profile: str, episode_s: float, stub_s: float | None,
                set_name: str | None = None) -> list[dict]:
    """One entry's items on a profile, or on a side set ("demo@1.2": FollowCfg behind_m; "demo@p_head0.3": a sim
    parameter override)."""
    kw: dict = {}
    if e["arm"] in TUNED:
        kw = {"params": e["params"], "x": None if e["params"] else e["x"]}
        if e["arm"] == "FLY-YAW":
            kw["norm"] = e["norm"]
    behind_m = overrides = None
    if set_name is not None:
        profile, behind_m, overrides = parse_side_set(set_name)
    return [ev.make_item(e["arm"], s, profile, size_blind=e["size_blind"], episode_s=episode_s, behind_m=behind_m,
                         overrides=overrides, stub_s=stub_s, tag={"label": e["label"], "set": set_name or profile},
                         **kw) for s in seeds]


def compact(r: dict) -> dict:
    return {"seed": r["seed"], "ok": bool(r.get("ok")), "ret": r.get("ret"), "terms": r.get("terms") or {},
            "metrics": r.get("metrics") or {}, "start_failed": ct.start_failed(r), "wall_s": r.get("wall_s"),
            "error": r.get("error")}


def run_eval(entries: list[dict], seeds: list[int], profiles: list[str], backend, episode_s: float = 60.0,
             stub_s: float | None = None, side_sets: tuple[str, ...] = ()) -> tuple[dict, dict]:
    items = [it for p in profiles for e in entries for it in entry_items(e, seeds, p, episode_s, stub_s)]
    items += [it for n in side_sets for e in entries for it in entry_items(e, seeds, "demo", episode_s, stub_s, n)]
    results, stats = backend.evaluate(items)
    recs: dict = {}
    for r in results:
        recs.setdefault(r["tag"]["label"], {}).setdefault(r["tag"]["set"], []).append(compact(r))
    for lab in recs:
        for p in recs[lab]:
            recs[lab][p].sort(key=lambda r: r["seed"])
    return recs, stats


def analyse(recs: dict, entries: list[dict], vs: list[str], seeds: str, n_boot: int = N_BOOT) -> dict:
    order = ("train", "demo", "stress", "fast")
    profiles = sorted({p for lab in recs for p in recs[lab]}, key=lambda p: (order.index(p) if p in order else 9, p))
    summary, diffs = {}, {}
    for e in entries:
        lab = e["label"]
        for p in profiles:
            if p not in recs.get(lab, {}):
                continue
            summary.setdefault(lab, {})[p] = summarize(recs[lab][p], recs.get(LAW, {}).get(p))
            for ref in [LAW] + [v for v in vs if v != LAW]:
                if ref == lab or p not in recs.get(ref, {}):
                    continue
                diffs.setdefault(lab, {}).setdefault(ref, {})[p] = {
                    m: paired(recs[lab][p], recs[ref][p], m, n_boot) for m in REPORT_METRICS}
    out = {"profiles": profiles, "seeds": seeds, "summary": summary, "paired": diffs}
    if seeds == "baseline":
        out["g2"] = g2(recs, entries)
    if seeds == "test":
        out["g3"] = g3(recs, entries, n_boot)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------------------------------------------


def _f(v, d: int = 3) -> str:
    return "-" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{d}f}"


def markdown(res: dict) -> str:
    a, cfg = res["analysis"], res["config"]
    lines = [f"# Yaw controller evaluation: {cfg['name']}", "",
             f"{cfg['n']} {cfg['seeds']} seeds x {', '.join(a['profiles'])}; episode {cfg['episode_s']} s; "
             f"{res['time']}. Fitness (DESIGN R1) = mean clip(r_yaw / max(|{LAW} r_yaw|, 5), -4, 0), collision -4 "
             f"(failed starts 0), minus min(4, max(0, S / max(S_{LAW}, 0.1) - 1.8)) on the following-weighted mean "
             f"stick_change ({LAW} about -1, higher is better). Diff = paired per-seed "
             f"mean (arm - {LAW}) with 95 % bootstrap CI over the seeds where neither collided and the start did not "
             "fail (n, excluded). Side sets (not gated): demo@<m> = demo at that behind_m, demo@<field><v> = a sim "
             "override.", ""]
    for p in a["profiles"]:
        lines += [f"## {p}", "", "| arm | n | err | start failed | fitness | coll | " + " | ".join(TABLE_METRICS)
                  + f" | rms_bearing diff vs {LAW} (n, excluded) |", "|" + "---|" * (len(TABLE_METRICS) + 7)]
        for lab, s in a["summary"].items():
            if p not in s:
                continue
            q = s[p]
            d = a["paired"].get(lab, {}).get(LAW, {}).get(p, {}).get("rms_bearing_deg")
            dtxt = f"{d['mean']:+.2f} [{d['lo']:+.2f}, {d['hi']:+.2f}] ({d['n']}, {d['n_excluded']})" if d else ""
            lines.append(f"| {lab} | {q['n']} | {q['n_errors']} | {q['n_start_failed']} | {_f(q['fitness'])} | "
                         f"{q['collisions']} | "
                         + " | ".join(_f(q["metrics"].get(m)) for m in TABLE_METRICS) + f" | {dtxt} |")
        lines.append("")
        for lab, refs in a["paired"].items():
            for ref, per in refs.items():
                if p in per and ref != LAW:
                    cells = [f"{m} {per[p][m]['mean']:+.3f} [{per[p][m]['lo']:+.3f}, {per[p][m]['hi']:+.3f}]"
                             for m in TABLE_METRICS if per[p][m]["n"] > 1]
                    lines.append(f"- {lab} - {ref}: " + "; ".join(cells))
        lines.append("")
    if "g2" in a:
        g = a["g2"]
        lines += ["## G1 / G2 (baseline seeds)", ""]
        if "size_mode" in g:
            sm = g["size_mode"]
            lines.append(f"- size mode: {sm['choice']} (size_blind {sm['size_blind']}); mean rms bearing "
                         + ", ".join(f"{k} {v:.3f}" for k, v in sm["rms_bearing_deg"].items()))
        if "init" in g:
            lines.append(f"- fine-tune init: {g['init']['choice']} (R1 fitness "
                         + ", ".join(f"{k} {v:.3f}" for k, v in g["init"]["fitness"].items()) + ")")
        for name, q in g.get("g1_law", {}).items():
            lines.append(f"- {LAW} on {name}: {q['collisions']} collisions, in_view {_f(q['in_view'])}, following "
                         f"{_f(q['following'])} (n {q['n']}, {q['start_failed']} failed starts excluded; G1 wants "
                         "in_view >= 0.7)")
        lines.append("")
    if "g3" in a:
        lines += ["## G3 (test seeds)", "", f"**Verdict: {a['g3'].get('verdict')}**", ""]
        for lab, c in a["g3"].get("candidates", {}).items():
            lines.append(f"- {lab}: {'PASS' if c['pass'] else 'FAIL on criteria ' + str(c['failed_criteria'])}")
            lines += [f"  - [{'x' if k['pass'] else ' '}] ({k['criterion']}) {k['check']}: {k['detail']}"
                      for k in c["checks"]]
    return "\n".join(lines) + "\n"


def seed_list(name: str, n: int) -> list[int]:
    if name not in SEED_SETS:
        raise SystemExit(f"--seeds must be one of {tuple(SEED_SETS)}")
    if n > len(SEED_SETS[name]):
        raise SystemExit(f"--n {n} exceeds the {len(SEED_SETS[name])} {name} seeds")
    return list(SEED_SETS[name][:n])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m kite.fly.train.report", description=__doc__.split("\n\n")[0])
    p.add_argument("--arms", default="KITE-LAW,FLY-YAW-HAND,FLY-YAW-PREV@blind,FLY-YAW-PREV@size")
    p.add_argument("--best", action="append", default=[], help="best.json of a tuned run (repeat)")
    p.add_argument("--seeds", default="baseline", choices=list(SEED_SETS), help="baseline: G1 / G2; test: G3")
    p.add_argument("--profiles", default="train,demo")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--vs", default="", help="also pair every arm against these labels (comma separated)")
    p.add_argument("--size-blind", choices=["on", "off"], default="on", help="fly arms without @blind / @size")
    p.add_argument("--episode-s", type=float, default=60.0)
    p.add_argument("--side-sets", default=None,
                   help="ungated side sets, e.g. demo@1.2,demo@p_head0.3 (default per --seeds: SIDE_SETS; 'none')")
    p.add_argument("--stub-s", type=float, default=None, help="synthetic episodes (checks the pipeline, not the arms)")
    p.add_argument("--backend", choices=["local", "modal"], default="local")
    p.add_argument("--processes", type=int)
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--name", default="report")
    p.add_argument("--out", default=None, help=f"output dir (default {report_dir()})")
    p.add_argument("--from", dest="from_json", help="recompute the analysis of a saved report JSON (no episodes)")
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    out_dir = Path(a.out) if a.out else report_dir()
    vs = [v.strip() for v in a.vs.split(",") if v.strip()]
    if a.from_json:
        res = json.loads(Path(a.from_json).read_text())
        c = res["config"]
        res["analysis"] = analyse(res["records"], res["entries"], vs or c.get("vs", []), c["seeds"], a.n_boot)
        name = Path(a.from_json).stem
    else:
        entries = parse_entries(a.arms, a.best, a.size_blind == "on")
        seeds = seed_list(a.seeds, a.n)
        profiles = [p.strip() for p in a.profiles.split(",") if p.strip()]
        ss = a.side_sets if a.side_sets is not None else ",".join(SIDE_SETS[a.seeds])
        side_sets = tuple(v.strip() for v in ss.split(",") if v.strip() and v.strip() != "none")
        for n in side_sets:
            parse_side_set(n)
        t0 = time.perf_counter()
        n_sets = len(profiles) + len(side_sets)
        print(f"{len(entries) * len(seeds) * n_sets} episodes: {[e['label'] for e in entries]} x {len(seeds)} "
              f"{a.seeds} seeds x {profiles} + side sets {list(side_sets)}")
        if a.backend == "local":
            backend = ct.LocalBackend(a.processes)
            try:
                recs, stats = run_eval(entries, seeds, profiles, backend, a.episode_s, a.stub_s, side_sets)
            finally:
                backend.close()
        else:
            import modal

            from kite.fly.train import modal_app

            prices = ct.load_prices()
            chunk = ct.default_chunk("FLY-YAW" if any(e["arm"] in arms.FLY_ARMS for e in entries) else LAW)
            with modal.enable_output(), modal_app.app.run():
                backend = ct.ModalBackend(modal_app.evaluate_chunk(), chunk, prices)
                recs, stats = run_eval(entries, seeds, profiles, backend, a.episode_s, a.stub_s, side_sets)
                stats["usd"] += backend.finish_usd()
        name = a.name
        res = {"time": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
               "config": {"name": name, "arms": a.arms, "best": a.best, "seeds": a.seeds, "n": a.n,
                          "profiles": profiles, "side_sets": list(side_sets), "episode_s": a.episode_s,
                          "stub_s": a.stub_s, "backend": a.backend,
                          "vs": vs, "wall_s": time.perf_counter() - t0, "usd": stats.get("usd"),
                          "container_h": (stats.get("container_s") or 0.0) / 3600},
               "entries": entries, "records": recs}
        res["analysis"] = analyse(recs, entries, vs, a.seeds, a.n_boot)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.json").write_text(json.dumps(res, indent=1, default=ct._json_default) + "\n")
    md = markdown(res)
    (out_dir / f"{name}.md").write_text(md)
    print(md)
    print(f"wrote {out_dir / name}.json and .md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
