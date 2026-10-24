"""Many episodes at once: seed lists, yaw arms, an optional process pool, mean metrics.

    results = run_seeds(range(20), "demo")                                   # KITE-LAW
    results = run_seeds(BASELINE_SEEDS, "train", arm="FLY-YAW-PREV")          # factory from ARM_SOURCES
    print(format_summary(summarize(results)))

An arm is FollowController's own law (yaw_spec None) or a factory "module:callable" that returns a FRESH YawSource
per episode (it is called in the worker, so it must be importable there). ARM_SOURCES maps arm names to their
default factories (the fly arms point at kite.fly.steer's factories, imported only when used). KITE-LAW-CMA is the
same law with other FollowCfg gains: pass a follow_cfg with the tuned k_yaw / db_yaw_deg (kite.fly.train.arms does
all of this for the trainer).
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from ...follow import FollowCfg, YawSource
from ..interfaces import BASELINE_SEEDS, METRICS, SELECTION_SEEDS, TEST_SEEDS, EpisodeResult
from .world import run_episode

ARM_SOURCES: dict[str, str | None] = {
    "KITE-LAW": None,
    "FLY-YAW-HAND": "kite.fly.steer:make_hand",
    "FLY-YAW-PREV": "kite.fly.steer:make_prev",
    "FLY-YAW": "kite.fly.steer:make_trained",
}


def register_arm(name: str, yaw_spec: str | None) -> None:
    ARM_SOURCES[name] = yaw_spec


def load_callable(spec: str) -> Callable:
    mod, sep, attr = spec.partition(":")
    if not sep or not mod or not attr:
        raise ValueError(f"expected module:callable, got {spec!r}")
    obj = importlib.import_module(mod)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def make_yaw_source(arm: str, yaw_spec: str | None = None) -> YawSource | None:
    """A fresh yaw source for one episode (None: FollowController's own law)."""
    if yaw_spec is None:
        if arm not in ARM_SOURCES:
            raise KeyError(f"arm {arm!r} has no yaw source yet: register_arm() it or pass yaw_spec='module:callable'")
        yaw_spec = ARM_SOURCES[arm]
    return None if yaw_spec is None else load_callable(yaw_spec)()


@dataclass(frozen=True)
class Job:
    seed: int
    profile: str
    arm: str = "KITE-LAW"
    yaw_spec: str | None = None
    follow_cfg: FollowCfg | None = None
    episode_s: float = 60.0
    overrides: tuple[tuple[str, float], ...] = ()  # world parameters (profiles.SimParams fields), e.g. fx_px


def run_job(job: Job) -> EpisodeResult:
    return run_episode(job.seed, job.profile, follow_cfg=job.follow_cfg,
                       yaw_source=make_yaw_source(job.arm, job.yaw_spec), arm=job.arm, episode_s=job.episode_s,
                       overrides=dict(job.overrides) or None)


def run_jobs(jobs: Sequence[Job], workers: int = 1) -> list[EpisodeResult]:
    if workers <= 1 or len(jobs) <= 1:
        return [run_job(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(run_job, jobs, chunksize=max(1, len(jobs) // (4 * workers))))


def run_seeds(seeds: Iterable[int], profile: str, arm: str = "KITE-LAW", yaw_spec: str | None = None,
              follow_cfg: FollowCfg | None = None, episode_s: float = 60.0, workers: int = 1,
              overrides: dict[str, float] | None = None) -> list[EpisodeResult]:
    ov = tuple(sorted((overrides or {}).items()))
    return run_jobs([Job(int(s), profile, arm, yaw_spec, follow_cfg, episode_s, ov) for s in seeds], workers)


def parse_seeds(text: str) -> list[int]:
    """'0-19', '0-9,20,30-35', or a named set: 'baseline' (G1/G2 reports), 'selection', 'test' (G3 only)."""
    named = {"baseline": BASELINE_SEEDS, "test": TEST_SEEDS, "selection": SELECTION_SEEDS}
    if text in named:
        return list(named[text])
    out: list[int] = []
    for part in text.split(","):
        a, _, b = part.strip().partition("-")
        if a:
            out.extend(range(int(a), int(b or a) + 1))
    return out


def summarize(results: Sequence[EpisodeResult]) -> dict[str, float]:
    """Mean of every metric and of the return, plus the per-tick cost."""
    n = max(1, len(results))
    out = {k: sum(r.metrics[k] for r in results) / n for k in METRICS}
    out["ret"] = sum(r.ret for r in results) / n
    out["episodes"] = float(len(results))
    out["collisions"] = float(sum(r.metrics["collision"] for r in results))
    out["start_failed"] = float(sum(r.metrics.get("start_failed", 0.0) for r in results))
    d3 = [r.metrics["min_dist3d_m"] for r in results if "min_dist3d_m" in r.metrics]
    if d3:
        out["min_dist3d_m"] = sum(d3) / len(d3)
    ticks = sum(r.n_ticks for r in results)  # interfaces.DT (0.05 s) ticks
    out["us_per_tick"] = 1e6 * sum(r.wall_s - r.brain_s for r in results) / max(1, ticks)
    out["brain_us_per_tick"] = 1e6 * sum(r.brain_s for r in results) / max(1, ticks)
    terms = sorted({k for r in results for k in r.terms})
    for k in terms:
        out[f"term_{k}"] = sum(r.terms.get(k, 0.0) for r in results) / n
    return out


def format_summary(s: dict[str, float], keys: Sequence[str] = METRICS) -> str:
    lines = [f"  {k:18s} {s[k]:9.3f}" for k in keys if k in s and math.isfinite(s[k])]
    lines.append(f"  {'return':18s} {s['ret']:9.1f}")
    lines.append(f"  {'us per 0.05 s tick':18s} {s['us_per_tick']:9.1f}  sim, "
                 f"+ {s['brain_us_per_tick']:.1f} yaw source")
    return "\n".join(lines)
