"""Episode items: one picklable dict per episode, run identically in a local process pool and in a Modal container.

    item = make_item("FLY-YAW", 10_024, "train", x=x, norm=norm, size_blind=True)
    evaluate_item(item)            # -> plain dict: ok, ret, terms, metrics, wall times (never raises)
    evaluate_items(items, 8)       # in item order, on a persistent pool of 8 worker processes

Item keys: arm, seed, profile, episode_s, size_blind, behind_m (the FollowCfg override of the standoff report; None:
the shared 1.0 m), overrides (sim parameters, e.g. {"fx_px": 1077}), and the controller (x, or params (+ norm), or
params_file; none of them means the arm's reference: KITE-LAW's or PRED-PD's defaults, FLY-YAW-PREV, FLY-YAW-HAND).
stub_s (tests, Modal smoke) swaps the simulator for a synthetic episode burning that much CPU. tag is echoed back.
"""

from __future__ import annotations

import os

# One BLAS thread per worker process: episodes are single-threaded and the pools run one worker per core.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import math
import multiprocessing as mp
import sys
import time
import traceback
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np

MAX_TRACEBACK = 3000


def make_item(arm: str, seed: int, profile: str = "train", *, x=None, params: dict | None = None,
              norm: dict | None = None, params_file: str | None = None, size_blind: bool | None = None,
              episode_s: float = 60.0, behind_m: float | None = None, overrides: dict | None = None,
              stub_s: float | None = None, tag: Any = None) -> dict:
    return {
        "arm": arm,
        "seed": int(seed),
        "profile": profile,
        "x": None if x is None else [float(v) for v in np.asarray(x, dtype=float).ravel()],
        "params": params,
        "norm": norm,
        "params_file": None if params_file is None else str(params_file),
        "size_blind": None if size_blind is None else bool(size_blind),
        "episode_s": float(episode_s),
        "behind_m": None if behind_m is None else float(behind_m),
        "overrides": overrides,
        "stub_s": None if stub_s is None else float(stub_s),
        "tag": tag,
    }


def _run(item: dict):
    from kite.fly.train import arms

    return arms.run_item(item)


# Tests swap this for a fake (same signature): item -> EpisodeResult or dict.
EPISODE_RUNNER: Callable[[dict], Any] = _run


def _plain(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    if isinstance(v, dict):
        return {str(k): _plain(u) for k, u in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(u) for u in v]
    return v


def _echo(item: dict) -> dict:
    return {k: item.get(k) for k in ("arm", "seed", "profile", "size_blind", "tag")}


def error_result(item: dict, message: str, tb: str | None = None) -> dict:
    return {**_echo(item), "ok": False, "ret": None, "terms": {}, "metrics": {}, "n_ticks": 0, "wall_s": 0.0,
            "brain_s": 0.0, "error": message, "traceback": tb, "item_wall_s": 0.0, "pid": os.getpid()}


def evaluate_item(item: dict) -> dict:
    """Run one episode. Never raises: a failure comes back as ok=False with the error text."""
    t0 = time.perf_counter()
    try:
        res = EPISODE_RUNNER(item)
        d = asdict(res) if is_dataclass(res) else dict(res)
        if not item.get("record"):
            d.pop("trace", None)
        d = _plain(d)
        if d.get("ret") is None or not math.isfinite(float(d["ret"])):
            raise ValueError(f"non-finite episode return {d.get('ret')!r}")
        d.update(_echo(item), ok=True, error=None)
    except Exception as e:  # noqa: BLE001
        d = error_result(item, f"{type(e).__name__}: {e}", traceback.format_exc()[-MAX_TRACEBACK:])
    d["item_wall_s"] = time.perf_counter() - t0
    d["pid"] = os.getpid()
    return d


# ---------------------------------------------------------------------------------------------------------------
# Persistent process pool (worker caches, e.g. the brain weights, survive across generations)
# ---------------------------------------------------------------------------------------------------------------

_POOL: ProcessPoolExecutor | None = None
_POOL_KEY: tuple | None = None


def default_start_method() -> str:
    # forkserver on Linux (Modal): workers fork from a clean server, not from a process running Modal's threads
    return "forkserver" if sys.platform.startswith("linux") else "spawn"


def get_pool(processes: int, start_method: str | None = None) -> ProcessPoolExecutor:
    global _POOL, _POOL_KEY
    method = start_method or default_start_method()
    if _POOL is None or _POOL_KEY != (processes, method):
        shutdown_pool()
        ctx = mp.get_context(method)
        if method == "forkserver":
            ctx.set_forkserver_preload(["kite.fly.train.evaluate"])
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
        return [evaluate_item(it) for it in items]
    pool = get_pool(processes, start_method)
    futures = []
    broken = False
    for it in items:
        try:
            futures.append(pool.submit(evaluate_item, it))
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
    """One batch, timed: what Modal's evaluate_chunk returns (and the local backend uses)."""
    t_start = time.time()
    t0 = time.perf_counter()
    results = evaluate_items(items, processes=workers, start_method=start_method)
    return {"results": results, "wall_s": time.perf_counter() - t0, "t_start": t_start, "workers": workers,
            "n_items": len(items)}
