"""CMA-ES fine-tuning of a yaw controller in kite's follow simulator (DESIGN section 5, revised by R1 and R7).

    python -m kite.fly.train.cma_train --arm FLY-YAW --cma-seed 1 --gens 60 --pop 24 --k 24 --sigma0 0.03 \\
        --init prev --size-blind on --backend local --tag kite1 --select-every 5 --budget-usd 25
    python -m kite.fly.train.cma_train --arm KITE-LAW-CMA --cma-seed 1 --backend modal --tag kite1
    python -m kite.fly.train.cma_train --arm PRED-PD --cma-seed 1 --backend modal --tag kite1
    python -m kite.fly.train.cma_train --arm STUB --gens 3 --pop 6 --k 4 --stub-s 0 --tag smoke   # no sim needed
    python -m kite.fly.train.cma_train --arm FLY-YAW --cma-seed 1 --dry-run     # projection only

One generation:
1. ask pycma for `pop` candidates on the normalized box [0, 1]^n (pycma's own bound handling);
2. run every candidate on the SAME K fresh training seeds TRAIN_SEED_BASE + 100000 * cma seed + gen * K + k,
   profile `train` (common random numbers; DESIGN R9.4), plus KITE-LAW on those seeds (cached: the normalizer);
3. fitness (DESIGN R1, the lead's revision of R9.3; see fitness()) = the mean over the K seeds of
   clip(r_yaw / max(|r_yaw of KITE-LAW on the seed|, 5), -4, 0) (-4 for a collision or an episode that raised, 0 for
   a seed whose start failed, R10.4), minus ONE stick penalty min(4, max(0, S_arm / max(S_law, 0.1) - 1.8)) on the
   following-step-weighted mean stick_change of the arm and of KITE-LAW over the same seeds (the G3 stick gate
   compares means too). KITE-LAW scores about -1; higher is better;
4. tell pycma -fitness;
5. every `select-every` generations (and at gen 0 = the init, and at the last one) score the distribution mean on
   SELECTION_SEEDS the same way and keep the best ever as best.json;
6. append log.jsonl, rewrite state.pkl atomically. A restarted run resumes from state.pkl exactly: pycma samples
   from a generator reseeded from (cma seed, gen), so generation g draws the same candidates before and after.

Run directory: $KITE_FLY_RUNS (default results/fly/runs) / <arm>_s<cma seed>_<tag>/ with config.json, log.jsonl,
selection.jsonl, best.json, errors.jsonl, state.pkl. Backends: a local process pool, or Modal
(kite.fly.train.modal_app, Evaluator.evaluate_chunk.map). Cost is metered as Modal container time at Modal's list
price (an estimate for the local backend) and the run stops itself before the next generation would pass --budget-usd.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import copy
import datetime as _dt
import json
import math
import pickle
import shutil
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from kite.fly.interfaces import BASELINE_SEEDS, METRICS, SELECTION_SEEDS, TRAIN_SEED_BASE
from kite.fly.train import arms
from kite.fly.train import evaluate as ev

REPO = Path(__file__).resolve().parents[3]
STATE_VERSION = 4  # 2: R1 fitness, 3: R9 seed offsets, 4: one mean-based stick penalty per candidate
DENOM_FLOOR = 5.0
CLIP = (-4.0, 0.0)
ERROR_SCORE = CLIP[0]  # a collision or a failed episode
# DESIGN R1: the reward terms the yaw arm is accountable for. The sim stores every term as its SIGNED, WEIGHTED sum
# (kite.fly.sim.reward: terms["e_x"] = -dt * w_x * sum(e_x^2), "jerk" = -dt * w_j * sum(|d yaw| / 100), "lost" =
# -dt * w_l * sum(lost), "ev_loss" = -10 per loss), so r_yaw is their plain sum.
YAW_TERMS = ("e_x", "jerk", "lost", "ev_loss")
# Stick penalty (revised R9.3): mean stick_change up to 1.8 x KITE-LAW's is free, beyond it costs 1 per 1 x, at most 4
STICK_FLOOR, STICK_FREE, STICK_CAP = 0.1, 1.8, 4.0
SEED_STRIDE = 100_000  # R9.4: training seeds of CMA seed c start at TRAIN_SEED_BASE + SEED_STRIDE * c
SIZE_PROBE_PROFILES = ("train", "demo")  # --size-blind auto: G2's decision, on BASELINE_SEEDS

# DESIGN section 5 plan per trainable arm (CLI flags override)
ARM_DEFAULTS = {
    "FLY-YAW": {"gens": 60, "pop": 24, "k": 24, "sigma0": 0.03, "init": "prev", "init_by_seed": {3: "hand"},
                # readout weights / bias live on +-15 / +-5 but act on O(1) normalized rate differences: the same
                # normalized step that is small for the encoder flips the readout (pycma CMA_stds multipliers)
                "stds": {"dec_w_": 0.08, "dec_b_": 0.2}},
    "KITE-LAW-CMA": {"gens": 30, "pop": 12, "k": 24, "sigma0": 0.1, "init": "default", "stds": {}},
    "PRED-PD": {"gens": 30, "pop": 12, "k": 24, "sigma0": 0.1, "init": "default", "stds": {}},
    "STUB": {"gens": 3, "pop": 8, "k": 4, "sigma0": 0.2, "init": "default", "stds": {}},
}
# A run resumes only if these match its state.pkl (gens, budget, select_every, backend may change)
RESUME_KEYS = ("arm", "cma_seed", "pop", "k", "sigma0", "init", "size_blind", "episode_s", "profile", "n_sel",
               "n_probe", "stds", "maxstd", "stub_s")

# ---------------------------------------------------------------------------------------------------------------
# Modal resources and prices
# ---------------------------------------------------------------------------------------------------------------

# One eval container: 8 physical cores (Modal's CPU unit; 16 hyperthreads), one episode process per hyperthread.
EVAL_SPEC = {"cpu": 8.0, "workers": 16, "memory_mib": 8192, "max_containers": 40, "scaledown_s": 30,
             "timeout_s": 1800, "retries": 2}
DRIVER_SPEC = {"cpu": 0.5, "memory_mib": 1024, "timeout_s": 86400, "retries": 3}
OVERHEAD = 1.25  # cold starts, autoscaler overshoot and idle tails that in-container timers do not see
PACKING = 0.85  # busy fraction of a chunk's workers


@dataclass
class Prices:
    """Modal list prices. Billed per container on max(request, usage), per second."""

    cpu_core_h: float = 0.0473  # `modal billing rates`, 2026-10-24: $0.0473 / physical core / hour
    mem_gib_h: float = 0.008  # $0.008 / GiB / hour
    source: str = "constants from `modal billing rates` (2026-10-24)"

    def usd_per_s(self, cpu: float, memory_mib: float) -> float:
        return (float(cpu) * self.cpu_core_h + float(memory_mib) / 1024.0 * self.mem_gib_h) / 3600.0


def load_prices(live: bool = True) -> Prices:
    """The workspace's current rates from Modal (needs the CLI's credentials), else the constants."""
    if live:
        try:
            import modal

            r = getattr(modal.Workspace.from_context().billing.rates(), "_rates", {})
            return Prices(float(r["cpu_hour_cost"]), float(r["mem_gib_hour_cost"]), "modal billing rates (live)")
        except Exception:  # noqa: BLE001
            pass
    return Prices()


def eval_usd_per_s(prices: Prices, spec: dict = EVAL_SPEC) -> float:
    return prices.usd_per_s(spec["cpu"], spec["memory_mib"])


def project(arm: str, gens: int, pop: int, k: int, select_every: int, s_ep: float, s_law: float,
            prices: Prices | None = None, n_sel: int = 64, chunk: int | None = None, spec: dict = EVAL_SPEC,
            gen_overhead_s: float = 4.0, n_extra: int = 0, s_max: float | None = None) -> dict:
    """Cost and wall time of one run on Modal, from seconds per episode on one loaded Modal worker: s_ep the mean,
    s_max the slowest (default TAIL_X * s_ep; a chunk lasts as long as its slowest worker).

    episodes   N_arm = gens * pop * K + n_selections * n_sel + n_extra (the size probe),
               N_law = gens * K + n_sel (KITE-LAW, cached)
    chunk wall T = ceil(chunk / workers) * s_max;  waves = ceil(chunks per gen / max_containers)
    container  C = max(busy, alive): busy = (N_arm * s_ep + N_law * s_law) / (workers * packing),
               alive = (gens + n_selections) * min(chunks, max_containers) * waves * T   (what ModalBackend meters:
               every container of a map call is billed until its slowest chunk ends)          [container-seconds]
    cost       C * (cpu * $/core-h + GiB * $/GiB-h) / 3600 * overhead + wall * driver $/s
    wall       gens * (waves * T + gen_overhead) + n_selections * (T + gen_overhead)
    """
    prices = prices or Prices()
    chunk = chunk or default_chunk(arm)
    s_max = s_max or TAIL_X * s_ep
    n_selections = 1 + (gens // select_every + (1 if gens % select_every else 0) if select_every else 1)
    n_arm = gens * pop * k + n_selections * n_sel + n_extra
    n_law = gens * k + n_sel
    busy = (n_arm * s_ep + n_law * s_law) / (spec["workers"] * PACKING)
    chunks_gen = math.ceil((pop * k + k) / chunk)
    containers = min(chunks_gen, spec["max_containers"])
    waves = math.ceil(chunks_gen / spec["max_containers"])
    t_chunk = math.ceil(chunk / spec["workers"]) * s_max
    t_sel = math.ceil(min(chunk, n_sel) / spec["workers"]) * s_max
    sel_containers = math.ceil(n_sel / chunk)
    alive = gens * containers * waves * t_chunk + n_selections * sel_containers * t_sel
    alive += math.ceil(n_extra / chunk) * t_chunk
    container_s = max(busy, alive)
    eval_usd = container_s * eval_usd_per_s(prices, spec) * OVERHEAD
    wall = gens * (waves * t_chunk + gen_overhead_s) + n_selections * (t_sel + gen_overhead_s)
    wall += math.ceil(n_extra / (chunk * spec["max_containers"])) * t_chunk
    driver_usd = wall * prices.usd_per_s(DRIVER_SPEC["cpu"], DRIVER_SPEC["memory_mib"])
    return {"arm": arm, "gens": gens, "episodes_arm": n_arm, "episodes_law": n_law, "container_h": container_s / 3600,
            "eval_usd": eval_usd, "driver_usd": driver_usd, "total_usd": eval_usd + driver_usd, "wall_h": wall / 3600,
            "s_per_episode": s_ep, "s_max": s_max, "s_per_law_episode": s_law, "chunks_per_gen": chunks_gen,
            "containers": containers, "usd_per_container_h": eval_usd_per_s(prices, spec) * 3600,
            "prices": prices.source}


def probe_episodes(cfg: dict) -> int:
    """Brain episodes of the --size-blind auto probe (2 modes x SIZE_PROBE_PROFILES x n_probe)."""
    auto = cfg["arm"] == "FLY-YAW" and cfg["size_blind"] == "auto"
    return 2 * len(SIZE_PROBE_PROFILES) * cfg["n_probe"] if auto else 0


def project_cfg(cfg: dict, prices: Prices | None = None, s_ep: float | None = None, s_law: float | None = None,
                chunk: int | None = None, s_max: float | None = None) -> dict:
    """project() for a run config; seconds per episode default to S_EPISODE_GUESS scaled to episode_s."""
    scale = cfg["episode_s"] / 60.0
    s_ep = s_ep or S_EPISODE_GUESS[cfg["arm"]] * scale
    s_law = s_law or S_EPISODE_GUESS["KITE-LAW-CMA"] * scale
    return project(cfg["arm"], cfg["gens"], cfg["pop"], cfg["k"], cfg["select_every"], s_ep, s_law, prices,
                   cfg["n_sel"], chunk, n_extra=probe_episodes(cfg), s_max=s_max)


def format_projection(p: dict) -> str:
    return (f"{p['arm']}: {p['gens']} gens, {p['episodes_arm']} + {p['episodes_law']} law episodes at "
            f"{p['s_per_episode']} / {p['s_per_law_episode']} s, {p['container_h']:.2f} container-h on "
            f"{p['containers']} containers (${p['usd_per_container_h']:.3f}/h, {p['prices']}): ${p['eval_usd']:.2f} "
            f"eval + ${p['driver_usd']:.2f} driver = ${p['total_usd']:.2f}, wall {p['wall_h'] * 60:.0f} min")


# Seconds per 60 s episode on one loaded Modal worker (16 workers on 8 cores). Measured 2026-10-24 (modal_app::probe,
# 32 FLY-YAW-PREV train episodes): mean 1.50 s, p90 2.87 s, max 3.15 s. The defaults keep a margin (a tuned fly
# that follows more often ticks its brain more); the classical arms are a guess from local timings x 2.
S_EPISODE_GUESS = {"FLY-YAW": 2.0, "KITE-LAW-CMA": 0.1, "PRED-PD": 0.2, "STUB": 0.05}
TAIL_X = 2.2  # slowest / mean episode time on a loaded worker (Modal probe 2026-10-24: 3.15 s / 1.50 s, FLY-YAW-PREV)


def default_chunk(arm: str, spec: dict = EVAL_SPEC) -> int:
    """Items per evaluate_chunk call: two episodes per worker for brain arms, more for the cheap law arms."""
    return spec["workers"] * (2 if arm in arms.FLY_ARMS else 8)


# ---------------------------------------------------------------------------------------------------------------
# Seeds and fitness
# ---------------------------------------------------------------------------------------------------------------


def train_seeds(gen: int, k: int, cma_seed: int) -> list[int]:
    """The K common training seeds of generation gen (1-based) of a CMA seed:
    TRAIN_SEED_BASE + 100000 * cma_seed + gen * K + k (each CMA seed sees its own episodes)."""
    if gen < 1 or k < 1 or cma_seed < 0:
        raise ValueError("gen and k start at 1, cma_seed at 0")
    if (gen + 1) * k > SEED_STRIDE:
        raise ValueError(f"generation {gen} x K {k} runs into the next CMA seed's training seeds")
    base = TRAIN_SEED_BASE + SEED_STRIDE * int(cma_seed) + gen * k
    return [base + j for j in range(k)]


def yaw_return(r: dict) -> tuple[float | None, bool]:
    """(r_yaw, collided) of an episode result; (None, False) if it failed. r_yaw = sum of terms[YAW_TERMS], i.e.
    -(w_x sum(e_x^2) + w_j sum(j) + w_l sum(l)) * dt + ev_loss. Every one of those terms is a penalty (<= 0): a
    positive one means the sim changed its convention, which must not pass silently."""
    if not r.get("ok"):
        return None, False
    terms = r.get("terms") or {}
    missing = [k for k in YAW_TERMS if k not in terms]
    if missing:
        raise ValueError(f"episode terms lack {missing} (have {sorted(terms)})")
    if any(float(terms[k]) > 1e-9 for k in YAW_TERMS):
        raise ValueError(f"yaw terms must be signed penalties (<= 0), got {({k: terms[k] for k in YAW_TERMS})}")
    collided = float((r.get("metrics") or {}).get("collision", 0.0)) > 0 or float(terms.get("ev_collision", 0.0)) < 0
    return float(sum(float(terms[k]) for k in YAW_TERMS)), collided


def start_failed(r: dict) -> bool:
    """DESIGN R10.4: the sim could not start following (flagged identically for every arm on that seed)."""
    flag = r.get("start_failed", (r.get("metrics") or {}).get("start_failed", 0.0))
    return bool(flag) and float(flag) > 0


def stick_of(r: dict) -> tuple[float, float] | None:
    """(stick_change, weight) of an episode, the weight being proportional to its number of following control steps;
    None if it failed or its start failed. The sim reports `following` = following steps / the window's control steps,
    and the window is the same for every episode of a run (same episode_s, the control-period jitter averages out
    over ~550 steps), so `following` is that weight up to a common factor. An exact count (metrics n_following) wins
    if the sim provides one. Episodes with no following step weigh 0: they contribute nothing."""
    if not r.get("ok") or start_failed(r):
        return None
    m = r.get("metrics") or {}
    if m.get("stick_change") is None:
        return None
    w = m.get("n_following", m.get("following", 0.0))
    return float(m["stick_change"]), max(0.0, float(w or 0.0))


def weighted_stick(sticks: list[tuple[float, float] | None]) -> float | None:
    """Following-step-weighted mean stick_change; None if no episode has a following step."""
    pairs = [p for p in sticks if p is not None and p[1] > 0]
    w = sum(p[1] for p in pairs)
    return sum(s * wi for s, wi in pairs) / w if w > 0 else None


def stick_penalty(s_arm: float | None, s_law: float | None) -> float:
    """min(4, max(0, S_arm / max(S_law, 0.1) - 1.8)); 0 if the arm never followed, S_law floored at 0.1 (also when
    KITE-LAW never followed)."""
    if s_arm is None:
        return 0.0
    return min(STICK_CAP, max(0.0, s_arm / max(s_law if s_law is not None else 0.0, STICK_FLOOR) - STICK_FREE))


def episode_score(r_yaw: float | None, law_r_yaw: float | None, collided: bool = False) -> float:
    """DESIGN R1: clip(r_yaw / max(|r_yaw(KITE-LAW on the same seed)|, 5), -4, 0); -4 flat for a collision or a
    failed episode."""
    if collided or r_yaw is None or not math.isfinite(r_yaw):
        return ERROR_SCORE
    denom = DENOM_FLOOR if law_r_yaw is None or not math.isfinite(law_r_yaw) else max(abs(law_r_yaw), DENOM_FLOOR)
    return min(max(float(r_yaw) / denom, CLIP[0]), CLIP[1])


def law_record(r: dict) -> dict:
    """What the trainer caches of a KITE-LAW episode: its r_yaw, (stick_change, weight) (None if it failed) and
    whether its start failed."""
    return {"r_yaw": yaw_return(r)[0], "stick": stick_of(r), "start_failed": start_failed(r)}


def fitness_parts(results: list[dict], laws: list[dict | None]) -> dict:
    """The candidate fitness and its parts against the KITE-LAW records (law_record) of the same seeds:
    score = mean episode score - stick penalty, where a seed whose start failed (the arm's or KITE-LAW's) scores 0
    and is left out of both stick means."""
    if not results:
        return {"fitness": ERROR_SCORE, "score": ERROR_SCORE, "penalty": 0.0, "s_arm": None, "s_law": None}
    scores, s_arm, s_law = [], [], []
    for r, law in zip(results, laws):
        law = law or {}
        if start_failed(r) or law.get("start_failed"):
            scores.append(0.0)
            continue
        r_yaw, collided = yaw_return(r)
        scores.append(episode_score(r_yaw, law.get("r_yaw"), collided))
        s_arm.append(stick_of(r))
        s_law.append(law.get("stick"))
    sa, sl = weighted_stick(s_arm), weighted_stick(s_law)
    pen = stick_penalty(sa, sl)
    score = float(np.mean(scores))
    return {"fitness": score - pen, "score": score, "penalty": pen, "s_arm": sa, "s_law": sl}


def fitness(results: list[dict], laws: list[dict | None]) -> float:
    return fitness_parts(results, laws)["fitness"]


# ---------------------------------------------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------------------------------------------


def make_chunks(items: list, size: int) -> list[list]:
    size = max(1, int(size))
    return [items[i:i + size] for i in range(0, len(items), size)]


class LocalBackend:
    """Process pool on this machine. Its cost is what the same episodes would cost in Modal eval containers if they
    ran as fast as here (an estimate: Modal workers are typically slower)."""

    name = "local"

    def __init__(self, processes: int | None = None, prices: Prices | None = None):
        self.processes = int(processes or os.cpu_count() or 1)
        self.prices = prices or Prices()

    def evaluate(self, items: list[dict]) -> tuple[list[dict], dict]:
        out = ev.evaluate_chunk_local(items, workers=self.processes)
        busy = sum(float(r.get("item_wall_s") or 0.0) for r in out["results"])
        container_s = busy / (EVAL_SPEC["workers"] * PACKING)
        return out["results"], {"wall_s": out["wall_s"], "container_s": container_s, "busy_s": busy,
                                "usd": container_s * eval_usd_per_s(self.prices) * OVERHEAD, "n_chunks": 1,
                                "basis": "local estimate"}

    def finish_usd(self) -> float:
        return 0.0

    def close(self) -> None:
        ev.shutdown_pool()


class ModalBackend:
    """Fans chunks out with evaluate_chunk.map. Bills every container seen in a map call for the call's whole wall
    time (they idle while the slowest chunk finishes), and the scaledown tail once at the end."""

    name = "modal"

    def __init__(self, fn: Any, chunk: int, prices: Prices | None = None, spec: dict = EVAL_SPEC):
        self.fn = fn
        self.chunk = int(chunk)
        self.prices = prices or Prices()
        self.spec = spec
        self.peak_containers = 0

    def evaluate(self, items: list[dict]) -> tuple[list[dict], dict]:
        chunks = make_chunks(items, self.chunk)
        t_submit = time.time()
        t0 = time.perf_counter()
        outs = list(self.fn.map(chunks, order_outputs=True, return_exceptions=True))
        wall = time.perf_counter() - t0
        results, walls, containers, cold, queue, errors = [], [], set(), 0, [], 0
        for chunk, out in zip(chunks, outs):
            if not isinstance(out, dict):
                errors += 1
                results.extend(ev.error_result(it, f"chunk failed: {type(out).__name__}: {out}") for it in chunk)
                continue
            results.extend(out["results"])
            walls.append(float(out["wall_s"]))
            containers.add(out.get("container_id"))
            cold += bool(out.get("cold"))
            queue.append(float(out.get("t_start", t_submit)) - t_submit)
        self.peak_containers = max(self.peak_containers, len(containers))
        container_s = max(sum(walls), len(containers) * wall)
        return results, {"wall_s": wall, "container_s": container_s, "busy_s": sum(walls), "n_chunks": len(chunks),
                         "chunk_errors": errors, "containers": len(containers), "cold": cold,
                         "chunk_wall_max_s": max(walls) if walls else None,
                         "queue_max_s": max(queue) if queue else None,
                         "usd": container_s * eval_usd_per_s(self.prices, self.spec) * OVERHEAD, "basis": "modal"}

    def finish_usd(self) -> float:
        return self.peak_containers * self.spec["scaledown_s"] * eval_usd_per_s(self.prices, self.spec)

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------------------------------------------
# CMA-ES helpers and files
# ---------------------------------------------------------------------------------------------------------------


class SeededRandn:
    """pycma's normal sampler (option randn), reseeded every generation. Pickled with the ES (shared by es.opts and
    its sampler), so a resumed run draws exactly what it would have drawn without the restart."""

    def __init__(self, seed: Any = 0):
        self.reseed(seed)

    def reseed(self, seed: Any) -> None:
        self.rng = np.random.default_rng(seed)

    def __call__(self, *shape: int) -> np.ndarray:
        return self.rng.standard_normal(shape)


def cma_stds(names: tuple[str, ...], by_prefix: dict[str, float]) -> list[float] | None:
    if not by_prefix:
        return None
    return [next((float(m) for p, m in by_prefix.items() if n.startswith(p)), 1.0) for n in names]


def runs_root() -> Path:
    return Path(os.environ.get("KITE_FLY_RUNS") or REPO / "results" / "fly" / "runs")


def run_name(arm: str, cma_seed: int, tag: str) -> str:
    return f"{arm}_s{cma_seed}_{tag}"


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")


def _json_default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return str(o)


def atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_json(path: Path, obj: Any) -> None:
    atomic_write(path, (json.dumps(obj, indent=1, default=_json_default) + "\n").encode())


def append_jsonl(path: Path, rec: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=_json_default) + "\n")


def truncate_jsonl(path: Path, max_gen: int) -> None:
    """Drop lines of generations logged after the last checkpoint (a crash between log and save)."""
    if not path.exists():
        return
    keep = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            if int(json.loads(line).get("gen", 0)) <= max_gen:
                keep.append(line)
        except (json.JSONDecodeError, ValueError):
            continue
    atomic_write(path, ("\n".join(keep) + ("\n" if keep else "")).encode())


EXTRA_METRICS = tuple(m for m in ("min_dist3d_m",) if m not in METRICS)  # the sim's R10 additions


def metric_means(results: list[dict]) -> dict[str, float]:
    ok = [r for r in results if r.get("ok") and not start_failed(r)]
    out = {}
    for m in METRICS + EXTRA_METRICS:
        v = [float(r["metrics"][m]) for r in ok if isinstance(r.get("metrics", {}).get(m), (int, float))]
        if v:
            out[m] = float(np.mean(v))
    return out


# ---------------------------------------------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------------------------------------------


def make_cfg(arm: str, cma_seed: int = 1, tag: str = "kite1", **over) -> dict:
    """A run config: DESIGN 5 / R7 / R9's plan for the arm (FLY-YAW: size-blind, init prev, hand on CMA seed 3), then
    the given overrides (None = keep the default)."""
    if arm not in ARM_DEFAULTS:
        raise ValueError(f"{arm} is not trainable; one of {tuple(ARM_DEFAULTS)}")
    d = ARM_DEFAULTS[arm]
    cfg = {"arm": arm, "cma_seed": int(cma_seed), "tag": tag, "gens": d["gens"], "pop": d["pop"], "k": d["k"],
           "sigma0": d["sigma0"], "init": d.get("init_by_seed", {}).get(int(cma_seed), d["init"]),
           "size_blind": "on" if arm == "FLY-YAW" else "off",  # R9.1: size-blind
           "select_every": 5, "budget_usd": 25.0, "episode_s": 60.0, "profile": "train",
           "n_sel": len(SELECTION_SEEDS), "n_probe": len(BASELINE_SEEDS), "stds": dict(d["stds"]), "maxstd": 0.3,
           "stub_s": None}
    for k, v in over.items():
        if k not in cfg:
            raise KeyError(f"unknown config key {k}")
        if v is not None:
            cfg[k] = v
    if arm != "FLY-YAW":
        cfg["size_blind"] = "off"
    if arm == "STUB" and cfg["stub_s"] is None:
        cfg["stub_s"] = 0.0
    return cfg


class Trainer:
    """One CMA-ES run (arm x CMA seed x tag), checkpointed every generation."""

    def __init__(self, cfg: dict, backend: Any, run_dir: str | Path | None = None, *, space: Any = None,
                 log: Callable[[str], None] = print, on_checkpoint: Callable[[], None] | None = None,
                 driver_usd_per_s: float = 0.0):
        self.cfg = copy.deepcopy(cfg)
        self.arm = cfg["arm"]
        self.backend = backend
        self.run_dir = Path(run_dir) if run_dir else runs_root() / run_name(self.arm, cfg["cma_seed"], cfg["tag"])
        self.space = space if space is not None else arms.param_space(self.arm)
        self.log = log
        self.on_checkpoint = on_checkpoint
        self.driver_usd_per_s = float(driver_usd_per_s)
        self.state: dict = {}

    name = property(lambda self: self.run_dir.name)
    state_path = property(lambda self: self.run_dir / "state.pkl")
    log_path = property(lambda self: self.run_dir / "log.jsonl")
    sel_path = property(lambda self: self.run_dir / "selection.jsonl")
    best_path = property(lambda self: self.run_dir / "best.json")
    err_path = property(lambda self: self.run_dir / "errors.jsonl")

    # ---- init / resume
    def load_or_init(self) -> bool:
        """Resume from state.pkl (True) or start a new run (False)."""
        import cma

        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.state_path.exists():
            st = pickle.loads(self.state_path.read_bytes())  # nosec B301: our own resume file in run_dir
            old = {k: st["cfg"].get(k) for k in RESUME_KEYS}
            new = {k: self.cfg.get(k) for k in RESUME_KEYS}
            if st.get("version") != STATE_VERSION or old != new:
                diff = {k: (old[k], new[k]) for k in RESUME_KEYS if old[k] != new[k]}
                raise RuntimeError(f"cannot resume {self.run_dir}: config differs {diff}; use --fresh or another --tag")
            st["cfg"].update({k: self.cfg[k] for k in ("gens", "budget_usd", "select_every")})
            self.state = st
            truncate_jsonl(self.log_path, st["gen"])
            truncate_jsonl(self.sel_path, st["gen"])
            self.log(f"[{self.name}] resumed at generation {st['gen']}")
            return True
        init = arms.init_point(self.arm, self.cfg["init"])
        if self.cfg["stub_s"] is not None and len(init["x0"]) != self.space.dim:  # a test space stands in
            init = {**init, "x0": self.space.default_x().tolist(), "source": "test space default"}
        x0 = np.clip(np.asarray(init["x0"], dtype=float), 0.0, 1.0)
        if x0.size != self.space.dim:
            raise ValueError(f"init has {x0.size} values; {self.arm} has {self.space.dim}")
        self.state = {"version": STATE_VERSION, "cfg": copy.deepcopy(self.cfg), "x0": x0.tolist(),
                      "norm": init["norm"], "init_source": init["source"], "size_blind": None, "size_probe": None,
                      "gen": 0, "law": {}, "best": None, "sel_done": [], "usd": 0.0, "container_s": 0.0,
                      "wall_s": 0.0, "episodes": 0, "errors": 0, "stopped": None, "last_gen_usd": 0.0,
                      "created": _now()}
        self.state["size_blind"] = self._resolve_size_blind()
        opts = {"popsize": int(self.cfg["pop"]), "bounds": [0.0, 1.0], "seed": np.nan, "randn": SeededRandn(0),
                "verbose": -9, "tolfun": 0, "tolfunhist": 0, "tolflatfitness": 10**9, "tolx": 0,
                "tolstagnation": 10**9, "maxiter": 10**9}
        stds = cma_stds(tuple(self.space.names), self.cfg["stds"])
        if stds:
            opts["CMA_stds"] = stds
        if self.cfg.get("maxstd"):
            opts["maxstd"] = float(self.cfg["maxstd"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.state["es"] = cma.CMAEvolutionStrategy(x0.tolist(), float(self.cfg["sigma0"]), opts)
        write_json(self.run_dir / "config.json", {**{k: v for k, v in self.state.items() if k not in ("es", "law")},
                                                   "names": list(self.space.names)})
        self.save()
        self.log(f"[{self.name}] new run: dim {x0.size}, pop {self.cfg['pop']}, K {self.cfg['k']}, init "
                 f"{init['source']}, size_blind {self.state['size_blind']}, dir {self.run_dir}")
        return False

    def _resolve_size_blind(self) -> bool | None:
        """DESIGN 6 / R6: on / off, or auto = G2's call: FLY-YAW-PREV in both modes on BASELINE_SEEDS[:n_probe] x
        SIZE_PROBE_PROFILES; the mode with the lower mean rms_bearing_deg (mean of the profile means) wins."""
        mode = self.cfg["size_blind"]
        if self.arm != "FLY-YAW":
            return None
        if mode in ("on", "off"):
            return mode == "on"
        seeds = BASELINE_SEEDS[: self.cfg["n_probe"]]
        items = [self._item("FLY-YAW-PREV", s, None, {"role": "size", "blind": b}, size_blind=b, norm=None,
                            profile=p) for b in (True, False) for p in SIZE_PROBE_PROFILES for s in seeds]
        results, stats = self.backend.evaluate(items)
        self._account(stats, stats["wall_s"], len(results), results)
        rms = {b: float(np.mean([metric_means([r for r in results if r["tag"]["blind"] == b and r["profile"] == p])
                                 .get("rms_bearing_deg", math.inf) for p in SIZE_PROBE_PROFILES]))
               for b in (True, False)}
        self.state["size_probe"] = {"blind_rms_bearing_deg": rms[True], "size_rms_bearing_deg": rms[False],
                                    "n": len(seeds), "profiles": list(SIZE_PROBE_PROFILES), "seeds": "baseline"}
        self.log(f"[{self.name}] size mode: FLY-YAW-PREV rms bearing {rms[True]:.2f} deg size-blind vs "
                 f"{rms[False]:.2f} deg with size -> {'size-blind' if rms[True] <= rms[False] else 'size'}")
        return bool(rms[True] <= rms[False])

    def save(self) -> None:
        atomic_write(self.state_path, pickle.dumps(self.state, protocol=pickle.HIGHEST_PROTOCOL))
        if self.on_checkpoint:
            self.on_checkpoint()

    # ---- items and scoring
    def _item(self, arm: str, seed: int, x, tag: dict, *, size_blind: bool | None = None, norm: Any = "run",
              profile: str | None = None) -> dict:
        c = self.cfg
        return ev.make_item(arm, seed, profile or c["profile"], x=x, norm=self.state["norm"] if norm == "run" else norm,
                            size_blind=size_blind, episode_s=c["episode_s"], stub_s=c["stub_s"], tag=tag)

    def _law_key(self, seed: int) -> str:
        c = self.cfg
        return f"{seed}|{c['profile']}|{c['episode_s']}|{c['stub_s']}"

    def _law_items(self, seeds: list[int]) -> list[dict]:
        return [self._item("KITE-LAW", s, None, {"role": "law"}, norm=None) for s in seeds
                if self._law_key(s) not in self.state["law"]]

    def _store_law(self, results: list[dict]) -> None:
        for r in results:
            if r["tag"]["role"] == "law":  # r_yaw and stick; a failed normalizer: None (the floors apply)
                self.state["law"][self._law_key(r["seed"])] = law_record(r)

    def _fitness_parts(self, results: list[dict]) -> dict:
        return fitness_parts(results, [self.state["law"].get(self._law_key(r["seed"])) for r in results])

    def _fitness(self, results: list[dict]) -> float:
        return self._fitness_parts(results)["fitness"]

    def _account(self, stats: dict, wall_s: float, n: int, results: list[dict]) -> float:
        usd = float(stats.get("usd") or 0.0) + wall_s * self.driver_usd_per_s
        st = self.state
        st["usd"] += usd
        st["container_s"] += float(stats.get("container_s") or 0.0)
        st["wall_s"] += wall_s
        st["episodes"] += n
        errs = [r for r in results if not r.get("ok")]
        st["errors"] += len(errs)
        for r in errs[:10]:
            append_jsonl(self.err_path, {"gen": st["gen"], "time": _now(),
                                         **{k: r.get(k) for k in ("arm", "seed", "tag", "error", "traceback")}})
        return usd

    def mean_x(self) -> np.ndarray:
        return np.clip(np.asarray(self.state["es"].result.xfavorite, dtype=float), 0.0, 1.0)

    # ---- selection: the distribution mean on SELECTION_SEEDS, best ever = best.json
    def select(self, gen: int, x=None) -> dict:
        t0 = time.perf_counter()
        x = self.mean_x() if x is None else np.asarray(x, dtype=float)
        seeds = list(SELECTION_SEEDS[: self.cfg["n_sel"]])
        sb = self.state["size_blind"]
        items = self._law_items(seeds) + [self._item(self.arm, s, x, {"role": "sel"}, size_blind=sb) for s in seeds]
        results, stats = self.backend.evaluate(items)
        self._store_law(results)
        sel = [r for r in results if r["tag"]["role"] == "sel"]
        parts = self._fitness_parts(sel)
        fit = parts["fitness"]
        usd = self._account(stats, time.perf_counter() - t0, len(results), results)
        best = self.state["best"]
        improved = best is None or fit > best["sel_fitness"]
        rec = {"gen": gen, "time": _now(), "sel_fitness": fit, "metrics": metric_means(sel),
               "n_errors": sum(not r.get("ok") for r in sel), "n_start_failed": sum(map(start_failed, sel)),
               "score": parts["score"], "stick_penalty": parts["penalty"], "stick_arm": parts["s_arm"],
               "stick_law": parts["s_law"], "improved": improved, "usd": usd, "x": x.tolist()}
        if improved:
            self.state["best"] = {"gen": gen, "sel_fitness": fit, "x": x.tolist()}
            self._write_best(rec)
        rec["best_gen"], rec["best_sel_fitness"] = self.state["best"]["gen"], self.state["best"]["sel_fitness"]
        append_jsonl(self.sel_path, rec)
        self.state["sel_done"].append(gen)
        self.log(f"[{self.name}] selection gen {gen}: {fit:.4f} (best {rec['best_sel_fitness']:.4f} at gen "
                 f"{rec['best_gen']})")
        return rec

    def _write_best(self, rec: dict) -> None:
        c, st = self.cfg, self.state
        write_json(self.best_path, {
            "arm": self.arm, "x": rec["x"], "params": self.space.decode(rec["x"]), "norm": st["norm"],
            "size_blind": st["size_blind"], "gen": rec["gen"], "sel_fitness": rec["sel_fitness"],
            "metric_means": rec["metrics"], "n_sel": c["n_sel"], "profile": c["profile"], "episode_s": c["episode_s"],
            "cma_seed": c["cma_seed"], "tag": c["tag"], "init": c["init"], "init_source": st["init_source"],
            "pop": c["pop"], "k": c["k"], "sigma0": c["sigma0"], "stds": c["stds"], "stub_s": c["stub_s"],
            "time": rec["time"]})

    # ---- one generation
    def step(self) -> dict:
        st, c = self.state, self.cfg
        es = st["es"]
        gen = st["gen"] + 1
        t0 = time.perf_counter()
        seeds = train_seeds(gen, c["k"], c["cma_seed"])
        es.opts["randn"].reseed([int(c["cma_seed"]), gen, 0x6B17E])
        X_raw = es.ask()
        X = [np.clip(np.asarray(x, dtype=float), 0.0, 1.0) for x in X_raw]
        sb = st["size_blind"]
        items = self._law_items(seeds)
        n_law = len(items)
        for i, x in enumerate(X):
            items += [self._item(self.arm, s, x, {"role": "cand", "i": i}, size_blind=sb) for s in seeds]
        results, stats = self.backend.evaluate(items)
        self._store_law(results)
        cand = [r for r in results if r["tag"]["role"] == "cand"]
        fits = [self._fitness([r for r in cand if r["tag"]["i"] == i]) for i in range(len(X))]
        es.tell(X_raw, [-f for f in fits])
        st["gen"] = gen
        wall = time.perf_counter() - t0
        usd = self._account(stats, wall, len(results), results)
        f = np.asarray(fits)
        arm_walls = [float(r.get("wall_s") or 0.0) for r in cand if r.get("ok")]
        law_walls = [float(r.get("wall_s") or 0.0) for r in results if r["tag"]["role"] == "law" and r.get("ok")]
        return {"gen": gen, "time": _now(), "fit_best": float(f.max()), "fit_median": float(np.median(f)),
                "fit_mean": float(f.mean()), "fit_worst": float(f.min()), "sigma": float(es.sigma),
                "mean_x": self.mean_x().tolist(), "seeds": [seeds[0], seeds[-1]], "n_items": len(items),
                "n_law": n_law, "n_errors": sum(not r.get("ok") for r in results),
                "n_start_failed": sum(map(start_failed, cand)) // max(1, len(X)),
                "errors": sorted({str(r.get("error"))[:200] for r in results if not r.get("ok")})[:3],
                "metrics": metric_means(cand), "wall_s": wall,
                "s_per_episode": float(np.mean(arm_walls)) if arm_walls else None,
                "s_per_law_episode": float(np.mean(law_walls)) if law_walls else None,
                "backend": {k: v for k, v in stats.items() if k != "usd"}, "usd_gen": usd}

    def _selection_due(self, gen: int) -> bool:
        every = int(self.cfg["select_every"] or 0)
        return gen not in self.state["sel_done"] and ((every > 0 and gen % every == 0) or gen == self.cfg["gens"])

    def run(self, gens: int | None = None, stop_after: int | None = None) -> dict:
        """Generations up to gens (resuming if a checkpoint exists). stop_after: return after that many new
        generations (tests use it to simulate a crash)."""
        if gens is not None:
            self.cfg["gens"] = int(gens)
        self.load_or_init()
        st = self.state
        gens = int(self.cfg["gens"])
        budget = self.cfg.get("budget_usd")
        tail = self.backend.finish_usd()
        if st["stopped"] == "budget" and budget and st["usd"] + st["last_gen_usd"] + tail <= budget:
            st["stopped"] = None  # the budget was raised
        if st["stopped"] == "budget":
            self.log(f"[{self.name}] stopped earlier on budget (${st['usd']:.2f}); raise --budget-usd to continue")
            return self.summary()
        if 0 not in st["sel_done"]:
            self.select(0, st["x0"])
            self.save()
        n_new = 0
        while st["gen"] < gens and (stop_after is None or n_new < stop_after):
            if budget and st["usd"] + st["last_gen_usd"] + self.backend.finish_usd() > budget:
                st["stopped"] = "budget"
                append_jsonl(self.log_path, {"gen": st["gen"], "time": _now(), "event": "budget stop",
                                             "usd": st["usd"], "next_gen_usd": st["last_gen_usd"],
                                             "budget_usd": budget})
                self.save()
                self.log(f"[{self.name}] budget stop at gen {st['gen']}: ${st['usd']:.2f} spent, next generation "
                         f"~${st['last_gen_usd']:.2f}, budget ${budget:.2f}")
                break
            usd0 = st["usd"]
            rec = self.step()
            if self._selection_due(rec["gen"]):
                rec["sel_fitness"] = self.select(rec["gen"])["sel_fitness"]
            rec["best_gen"], rec["best_sel_fitness"] = st["best"]["gen"], st["best"]["sel_fitness"]
            st["last_gen_usd"] = st["usd"] - usd0
            rec["usd_total"] = st["usd"]
            rec["usd_projected"] = st["usd"] + (gens - st["gen"]) * st["last_gen_usd"]
            append_jsonl(self.log_path, rec)
            self.save()
            n_new += 1
            self.log(f"[{self.name}] gen {rec['gen']}/{gens} fit best {rec['fit_best']:.4f} median "
                     f"{rec['fit_median']:.4f} sigma {rec['sigma']:.4f} err {rec['n_errors']} {rec['wall_s']:.1f} s "
                     f"${st['usd']:.3f} (projected ${rec['usd_projected']:.2f})")
        if st["gen"] >= gens and st["stopped"] is None:
            st["usd"] += self.backend.finish_usd()
            st["stopped"] = "done"
            self.save()
        return self.summary()

    def summary(self) -> dict:
        st = self.state
        return {"run": self.name, "run_dir": str(self.run_dir), "gen": st.get("gen"), "stopped": st.get("stopped"),
                "best": {k: v for k, v in (st.get("best") or {}).items() if k != "x"}, "usd": st.get("usd"),
                "container_h": (st.get("container_s") or 0.0) / 3600, "wall_s": st.get("wall_s"),
                "episodes": st.get("episodes"), "errors": st.get("errors"), "size_blind": st.get("size_blind"),
                "size_probe": st.get("size_probe")}


# ---------------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------------


def parse_stds(s: str | None) -> dict | None:
    """'dec_w_=0.08,dec_b_=0.2' -> dict; 'none' -> {} (plain sigma0 everywhere); None -> the arm's default."""
    if s is None:
        return None
    if s.strip().lower() in ("none", ""):
        return {}
    return {k.strip(): float(v) for k, v in (p.split("=") for p in s.split(",") if p.strip())}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m kite.fly.train.cma_train", description=__doc__.split("\n\n")[0])
    p.add_argument("--arm", required=True, choices=list(ARM_DEFAULTS))
    p.add_argument("--cma-seed", type=int, default=1)
    p.add_argument("--gens", type=int)
    p.add_argument("--pop", type=int)
    p.add_argument("--k", type=int, help="training episodes per candidate per generation")
    p.add_argument("--sigma0", type=float)
    p.add_argument("--init", choices=list(arms.INITS), help="FLY-YAW (default R9: prev, hand on CMA seed 3)")
    p.add_argument("--size-blind", choices=["auto", "on", "off"], help="FLY-YAW (default R9: on; auto = G2 probe)")
    p.add_argument("--backend", choices=["local", "modal"], default="local")
    p.add_argument("--tag", default="kite1")
    p.add_argument("--select-every", type=int)
    p.add_argument("--budget-usd", type=float)
    p.add_argument("--episode-s", type=float, help="episode length (smoke tests only; default 60)")
    p.add_argument("--n-sel", type=int, help=f"selection episodes (default all {len(SELECTION_SEEDS)}; smoke tests)")
    p.add_argument("--n-probe", type=int, help=f"size-probe seeds (default all {len(BASELINE_SEEDS)} baseline seeds)")
    p.add_argument("--stds", help="per-name-prefix sigma multipliers, e.g. dec_w_=0.08,dec_b_=0.2, or none")
    p.add_argument("--stub-s", type=float, help="synthetic episodes burning this much CPU (no sim; smoke tests)")
    p.add_argument("--processes", type=int, help="local pool size (default: all cores)")
    p.add_argument("--chunk", type=int, help="items per Modal evaluate_chunk call")
    p.add_argument("--s-per-episode", type=float, help="for the projection (default: measured, else a guess)")
    p.add_argument("--runs-dir", help=f"default $KITE_FLY_RUNS or {REPO / 'results/fly/runs'}")
    p.add_argument("--fresh", action="store_true", help="move an existing run dir aside and start over")
    p.add_argument("--dry-run", action="store_true", help="print the plan and the projection, run nothing")
    return p


def cfg_from_args(a: argparse.Namespace) -> dict:
    return make_cfg(a.arm, a.cma_seed, a.tag, gens=a.gens, pop=a.pop, k=a.k, sigma0=a.sigma0, init=a.init,
                    size_blind=a.size_blind, select_every=a.select_every, budget_usd=a.budget_usd,
                    episode_s=a.episode_s, n_sel=a.n_sel, n_probe=a.n_probe, stds=parse_stds(a.stds), stub_s=a.stub_s)


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    cfg = cfg_from_args(a)
    rdir = Path(a.runs_dir) / run_name(a.arm, a.cma_seed, a.tag) if a.runs_dir else None
    rdir = rdir or runs_root() / run_name(a.arm, a.cma_seed, a.tag)
    prices = load_prices(live=a.backend == "modal" or a.dry_run)
    chunk = a.chunk or default_chunk(a.arm)
    proj = project_cfg(cfg, prices, a.s_per_episode, None, chunk)
    print(json.dumps(cfg, default=_json_default))
    print("projection on Modal: " + format_projection(proj))
    if cfg["budget_usd"] and proj["total_usd"] > cfg["budget_usd"]:
        print(f"WARNING: projected ${proj['total_usd']:.2f} exceeds the budget ${cfg['budget_usd']:.2f}; the run will "
              "stop itself when its metered cost reaches it")
    if a.dry_run:
        return 0
    if a.fresh and rdir.exists():
        bak = rdir.with_name(rdir.name + ".bak-" + time.strftime("%Y%m%d-%H%M%S"))
        shutil.move(str(rdir), str(bak))
        print(f"moved {rdir} to {bak}")
    t0 = time.perf_counter()
    if a.backend == "local":
        backend = LocalBackend(a.processes, prices)
        try:
            summary = Trainer(cfg, backend, rdir).run()
        finally:
            backend.close()
    else:
        import modal

        from kite.fly.train import modal_app

        with modal.enable_output(), modal_app.app.run():
            backend = ModalBackend(modal_app.evaluate_chunk(), chunk, prices)
            summary = Trainer(cfg, backend, rdir).run()
    summary["driver_wall_s"] = time.perf_counter() - t0
    print(json.dumps(summary, indent=1, default=_json_default))
    return 0 if summary.get("stopped") in ("done", "budget", None) else 1


if __name__ == "__main__":
    # Run the importable module, not __main__, so state.pkl refers to kite.fly.train.cma_train.SeededRandn.
    from kite.fly.train import cma_train as _module  # noqa: PLW0406

    sys.exit(_module.main())
