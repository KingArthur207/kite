"""Episode items for kite.fly.behind: one picklable dict per episode, run the same way in a local process pool and in a
Modal container (kite.fly.behind.modal_app).

    item = make_item("V3-JOINT", 7_100_032, "train", x=x)   # a CMA candidate (x in params.space(arm))
    item = make_item("FLOWN", 7_100_032, "train")           # the app as it flies: the normalizer
    item = make_item("v3-joint", 8000, "demo", model="kite/fly/data/behind/v3-joint.json")   # a saved model file
    run_item(item)                                          # -> plain dict (EpisodeResult + ok, error, ...); never
                                                            #    raises
    evaluate_chunk_local(items, workers=8)                  # {"results": [...] in item order, "wall_s": ...}

Item keys: arm, seed, profile, x (None: the arm's init; FLOWN has none), model (a model file, DESIGN section 6: its
follow, steering and brain replace arm / x; arm is then only the label), episode_s, overrides (BehindParams fields),
record (keep the truth traces), stub_s (tests, the Modal smoke: a synthetic episode burning that much CPU instead of
the simulator), tag (echoed back so the trainer can route results).

A result is EpisodeResult as a dict (seed, profile, arm, ret, terms, metrics, n_ticks, wall_s, brain_s, trace) plus
ok, error, traceback, tag, item_wall_s, pid. A failure never raises: it comes back ok False with metrics
{"error": 1.0}, so a broken candidate costs its own score (the trainer charges it as a collision), not the run.
"""

from __future__ import annotations

import os

# One BLAS thread per worker process: episodes are single-threaded and the pools run one worker per core.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import math
import multiprocessing as mp
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np

from kite.fly.interfaces import EpisodeResult
from kite.fly.train.evaluate import MAX_TRACEBACK, default_start_method
from kite.fly.train.evaluate import _plain as plain

ECHO = ("arm", "seed", "profile", "tag")
STUB_OPTIMUM = 0.3  # every stub coordinate scores best here
STUB_FLOWN_D = 0.2  # FLOWN's stub distance (x at 0.7 everywhere with the mean weight 1.25)
STUB_START_FAIL_P = 0.03  # a failed start on this share of seeds, the same for every arm (exercises the exclusion)


def make_item(arm: str, seed: int, profile: str = "train", x=None, episode_s: float = 60.0,
              overrides: dict | None = None, record: bool = False, stub_s: float | None = None, *,
              model: str | os.PathLike | None = None, tag: Any = None) -> dict:
    return {
        "arm": arm,
        "seed": int(seed),
        "profile": profile,
        "x": None if x is None else [float(v) for v in np.asarray(x, dtype=float).ravel()],
        "model": None if model is None else str(model),
        "episode_s": float(episode_s),
        "overrides": dict(overrides) if overrides else None,
        "record": bool(record),
        "stub_s": None if stub_s is None else float(stub_s),
        "tag": tag,
    }


def item_cfg(item: dict):
    """(cfg, brain_x) an item flies: a model file's TunedCfg (its brain read from the file itself), else the arm at
    x."""
    if item.get("model"):
        from kite.follow_tuned import load_model

        return load_model(item["model"]), None
    from kite.fly.behind.params import make_cfg

    x = item.get("x")
    return make_cfg(item["arm"], None if x is None or item["arm"] == "FLOWN" else x)


def _episode(item: dict) -> EpisodeResult:
    if item.get("stub_s") is not None:
        return stub_episode(item)
    from kite.fly.behind.world import run_behind_episode

    cfg, brain_x = item_cfg(item)
    return run_behind_episode(int(item["seed"]), item.get("profile", "train"), cfg, arm=item["arm"], brain_x=brain_x,
                              episode_s=float(item.get("episode_s") or 60.0), overrides=item.get("overrides"),
                              record=bool(item.get("record")))


# Tests swap this for a fake (same signature, in-process only: pool workers import the module fresh).
EPISODE_RUNNER: Callable[[dict], Any] = _episode


def _echo(item: dict) -> dict:
    return {k: item.get(k) for k in ECHO}


def error_result(item: dict, message: str, tb: str | None = None) -> dict:
    return {**_echo(item), "ret": None, "terms": {}, "metrics": {"error": 1.0}, "n_ticks": 0, "wall_s": 0.0,
            "brain_s": 0.0, "trace": None, "ok": False, "error": message, "traceback": tb, "item_wall_s": 0.0,
            "pid": os.getpid()}


def is_error(r: dict) -> bool:
    """A failed episode (raised, a crashed worker or chunk): its metrics mean nothing."""
    return (not r.get("ok", False)) or bool((r.get("metrics") or {}).get("error"))


def run_item(item: dict) -> dict:
    """Run one episode. Never raises: a failure comes back as an error result (ok False, metrics["error"] = 1)."""
    t0 = time.perf_counter()
    try:
        res = EPISODE_RUNNER(item)
        d = asdict(res) if is_dataclass(res) else dict(res)
        if not item.get("record"):
            d["trace"] = None
        d = plain(d)
        ret = d.get("ret")
        if ret is not None and not math.isfinite(float(ret)):
            raise ValueError(f"non-finite episode return {ret!r}")
        if not isinstance(d.get("metrics"), dict) or not d["metrics"]:
            raise ValueError("the episode returned no metrics")
        d.update(_echo(item), ok=True, error=None, traceback=None)
    except Exception as e:  # noqa: BLE001
        d = error_result(item, f"{type(e).__name__}: {e}", traceback.format_exc()[-MAX_TRACEBACK:])
    d["item_wall_s"] = time.perf_counter() - t0
    d["pid"] = os.getpid()
    return d


# ---------------------------------------------------------------------------------------------------------------
# The stub episode (tests, the Modal smoke): no simulator, a smooth objective of x and the seed
# ---------------------------------------------------------------------------------------------------------------


def metric_keys() -> tuple[str, ...]:
    """DESIGN section 4's metric keys (kite.fly.behind.metrics.METRICS_BEHIND): what a stub episode reports."""
    from kite.fly.behind.metrics import METRICS_BEHIND

    return tuple(METRICS_BEHIND)


def stub_episode(item: dict) -> EpisodeResult:
    """Burns item["stub_s"] CPU seconds and returns DESIGN section 4's metrics as smooth functions of the distance
    d = mean(w (x - 0.3)^2) of the candidate from the stub optimum, with per-seed weights and scale (common random
    numbers: every arm sees the same seed noise). FLOWN and model items sit at d = 0.2. The fitness ratio J_arm /
    J_FLOWN then falls as CMA walks x toward 0.3."""
    t0 = time.perf_counter()
    burn = float(item.get("stub_s") or 0.0)
    a = np.ones(64)
    while time.perf_counter() - t0 < burn:
        a = np.sqrt(a * a + 1.0) - 0.5
    seed = int(item["seed"])
    rng = np.random.default_rng([seed, 0xB3])
    scale = 0.7 + 0.6 * float(rng.random())
    failed = float(rng.random()) < STUB_START_FAIL_P
    x = item.get("x")
    if x is None or item.get("arm") == "FLOWN" or item.get("model"):
        d = STUB_FLOWN_D
    else:
        x = np.asarray(x, dtype=float)
        w = 1.0 + 0.5 * rng.random(x.size)
        d = float(np.mean(w * (x - STUB_OPTIMUM) ** 2)) if x.size else STUB_FLOWN_D
    g = 0.5 + 4.0 * d  # 1.3 at FLOWN's d
    m = {k: 0.0 for k in metric_keys()}
    m.update(rms_range_err_m=0.3 * scale * g, rms_bearing_deg=12.0 * scale * g, in_band=max(0.0, 1.0 - 2.0 * d),
             in_view=1.0 - min(0.5, d), min_range_m=1.6 - 0.5 * d, stick_fb=2.0 * g, stick_yaw=1.5 * g,
             stick_ud=1.0 * g, stick_lr=0.5 * g, accel_rms=0.3 * scale * g, speed_track_err=0.15 * scale * g,
             excess_speed=0.1 * d, alt_max_m=1.95, rms_alt_err_m=0.05 * g, max_speed_mps=0.35 + 0.2 * d,
             start_failed=1.0 if failed else 0.0)
    return EpisodeResult(seed=seed, profile=item.get("profile", "train"), arm=item["arm"], ret=-g, terms={},
                         metrics=m, n_ticks=0 if failed else 1200, wall_s=time.perf_counter() - t0)


# ---------------------------------------------------------------------------------------------------------------
# Persistent process pool (the workers' caches, e.g. the loaded brain, survive across generations)
# ---------------------------------------------------------------------------------------------------------------

_POOL: ProcessPoolExecutor | None = None
_POOL_KEY: tuple | None = None


def get_pool(processes: int, start_method: str | None = None) -> ProcessPoolExecutor:
    """spawn on macOS, forkserver on Linux (Modal: workers fork from a clean server, not from a process running
    Modal's threads); both import this module fresh, so nothing but the item crosses."""
    global _POOL, _POOL_KEY
    method = start_method or default_start_method()
    if _POOL is None or _POOL_KEY != (processes, method):
        shutdown_pool()
        ctx = mp.get_context(method)
        if method == "forkserver":
            ctx.set_forkserver_preload(["kite.fly.behind.evaluate"])
        _POOL = ProcessPoolExecutor(max_workers=processes, mp_context=ctx)
        _POOL_KEY = (processes, method)
    return _POOL


def shutdown_pool(wait: bool = False) -> None:
    global _POOL, _POOL_KEY
    if _POOL is not None:
        _POOL.shutdown(wait=wait, cancel_futures=True)
    _POOL, _POOL_KEY = None, None


def evaluate_items(items: list[dict], processes: int = 1, start_method: str | None = None) -> list[dict]:
    """Results in item order. processes > 1 uses the persistent pool; a crashed worker turns its items into error
    results and the pool is rebuilt on the next call. Never raises."""
    if processes <= 1 or len(items) <= 1:
        return [run_item(it) for it in items]
    pool = get_pool(processes, start_method)
    futures = []
    broken = False
    for it in items:
        try:
            futures.append(pool.submit(run_item, it))
        except BrokenProcessPool:
            broken = True
            futures.append(None)
    out = []
    for it, fut in zip(items, futures):
        if fut is None:
            out.append(error_result(it, "BrokenProcessPool: not submitted"))
            continue
        try:
            out.append(fut.result())
        except BrokenProcessPool as e:
            broken = True
            out.append(error_result(it, f"BrokenProcessPool: {e}"))
        except Exception as e:  # noqa: BLE001
            out.append(error_result(it, f"{type(e).__name__}: {e}"))
    if broken:
        shutdown_pool()
    return out


def evaluate_chunk_local(items: list[dict], workers: int = 1, start_method: str | None = None) -> dict:
    """One batch, timed: what a Modal evaluate_chunk returns and the local backend uses."""
    t_start = time.time()
    t0 = time.perf_counter()
    results = evaluate_items(items, processes=int(workers), start_method=start_method)
    return {"results": results, "wall_s": time.perf_counter() - t0, "t_start": t_start, "workers": int(workers),
            "n_items": len(items)}
