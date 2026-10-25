"""CMA-ES for the behind follow (DESIGN section 8): tune the follow the app flies, and its fly brain, together.

    python -m kite.fly.behind.train --arm V3-JOINT --cma-seed 1 --gens 100 --pop 40 --k 32 --sigma0 0.15 \\
        --backend modal --tag v3 --select-every 5 --budget-usd 40
    python -m kite.fly.behind.train --arm V3-FOLLOW --cma-seed 2 --backend local --tag v3
    python -m kite.fly.behind.train --arm V3-LAW --cma-seed 1 --dry-run            # the projection only
    python -m kite.fly.behind.train --arm V3-LAW --gens 3 --pop 6 --k 4 --n-sel 8 --stub-s 0 --tag smoke   # no sim

One generation:
1. ask pycma for `pop` candidates on [0, 1]^n (kite.fly.behind.params.space(arm); pycma's own bound handling,
   CMA_stds: the knobs search widely, the brain only fine-tunes);
2. run every candidate AND FLOWN (the app as it flies today) on the SAME K fresh training seeds
   BEHIND_TRAIN_BASE + SEED_STRIDE * cma seed + gen * K + k (common random numbers: the person, the video, the
   dynamics are the same for every arm; only the drone's flight differs), seed k on profile k mod the profile list
   (default "train,slow": half of them slow walkers, 0.1-0.35 m/s, where speeding up is never needed). FLOWN never
   changes, so its result per (seed, profile) is cached in state.pkl, keyed by the episode code digest too;
3. fitness = kite.fly.behind.fitness.fitness(): the mean over the seeds of the score against FLOWN on the same seed
   (FLOWN = 1.0, lower is better: the distance relative to FLOWN's, everything else at a fixed price; a collision
   FLOWN did not have makes the candidate worse than FLOWN whatever else it did; an episode that raised or failed
   its own start = 6, a seed where FLOWN failed excluded for every candidate alike). CMA minimizes it directly;
4. every --select-every generations (and at gen 0 = the init, and at the last one) score the distribution mean on
   SELECTION_SEEDS the same way and keep the best ever as best.json: a MODEL FILE (kite.follow_tuned.save_model,
   DESIGN section 6) the app can load as is;
5. append log.jsonl, rewrite state.pkl atomically. A restarted run resumes exactly: pycma samples from a generator
   reseeded from (cma seed, gen), so generation g draws the same candidates before and after a kill (log.jsonl
   records a hash of them: cand_sha). It refuses to resume when the code changed (kite.fly.behind.digest.run_digest:
   the sim, the follow, the governor, the brain, the parameter bounds, the fitness): the cached FLOWN normalizer and
   the CMA state would belong to another objective. --allow-code-change resumes anyway (FLOWN is then re-run).

Run directory: $KITE_BEHIND_RUNS (default results/behind/runs; /runs on Modal) / <arm>_s<cma seed>_<tag>/ with
config.json, log.jsonl, selection.jsonl, best.json, errors.jsonl, state.pkl. The backends, prices, metering, cost
projection and file helpers are kite.fly.train.cma_train's (imported, unchanged); cost is metered as Modal container
time at list price (an estimate for the local backend) and the run stops itself before the next generation would
pass --budget-usd.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import copy
import hashlib
import json
import math
import pickle
import shutil
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from kite.fly.behind import BEHIND_TRAIN_BASE, SEED_STRIDE, SELECTION_SEEDS
from kite.fly.behind import evaluate as bev
from kite.fly.behind import params as bp
from kite.fly.behind.digest import episode_digest, run_digest
from kite.fly.behind.fitness import FitnessCfg, fitness
from kite.fly.behind.profiles import PROFILES
from kite.fly.train import cma_train as ct
from kite.fly.train.cma_train import (TAIL_X, Prices, SeededRandn, _json_default, _now, append_jsonl, atomic_write,
                                      load_prices, run_name, truncate_jsonl, write_json)

REPO = Path(__file__).resolve().parents[3]
MAX_CONTAINERS_ENV = "KITE_BEHIND_MAX_CONTAINERS"
DEFAULT_MAX_CONTAINERS = 160


def max_containers() -> int:
    """kite-behind's eval containers at once ($KITE_BEHIND_MAX_CONTAINERS, default 160; modal_app::probe
    --containers measures what the workspace really runs)."""
    try:
        return max(1, int(os.environ.get(MAX_CONTAINERS_ENV) or DEFAULT_MAX_CONTAINERS))
    except ValueError:
        return DEFAULT_MAX_CONTAINERS


# kite-behind's eval container (modal_app's Evaluator): kite.fly's shape (8 physical cores, 16 worker processes,
# 8 GiB) under this app's own container cap, not kite.fly's 40. The one spec the projections, the chunk choice, the
# Modal metering and the app's decorator all use, so a dry run projects what a launch then does.
EVAL_SPEC = {**ct.EVAL_SPEC, "cpu": 8.0, "workers": 16, "memory_mib": 8192, "max_containers": max_containers(),
             "scaledown_s": 30, "timeout_s": 1800, "retries": 2}
STATE_VERSION = 2  # 2: FLOWN cached per (seed, profile, code digest), the code digest checked on resume
RANDN_SALT = 0xBE41D  # mixed into the per-generation sampler seed (kite.fly's runs use their own)
TRAINABLE = ("V3-FOLLOW", "V3-JOINT", "V3-LAW")  # the round-1 plan; the V3B arms run only when asked
# DESIGN section 8's plan per arm (CLI flags override); the fly arms on 4 CMA seeds, the law on 2
ARM_PLAN = {
    "V3-FOLLOW": {"gens": 60, "pop": 24, "k": 32, "sigma0": 0.15, "cma_seeds": (1, 2, 3, 4)},
    "V3-JOINT": {"gens": 100, "pop": 40, "k": 32, "sigma0": 0.15, "cma_seeds": (1, 2, 3, 4)},
    "V3-LAW": {"gens": 60, "pop": 20, "k": 32, "sigma0": 0.15, "cma_seeds": (1, 2)},
    # round 2: warm-started from a round-1 best (--init), so a smaller step and fewer generations
    "V3B-FOLLOW": {"gens": 50, "pop": 24, "k": 32, "sigma0": 0.08, "cma_seeds": (1, 2, 3, 4)},
    "V3B-JOINT": {"gens": 50, "pop": 32, "k": 32, "sigma0": 0.08, "cma_seeds": (1, 2)},
}
DEFAULT_BUDGET_USD = 40.0
DEFAULT_TAG = "v3"
# A run resumes only if these match its state.pkl (gens, budget_usd, select_every may change)
RESUME_KEYS = ("arm", "cma_seed", "pop", "k", "sigma0", "episode_s", "profile", "n_sel", "maxstd", "stub_s", "fitness",
               "init")
DEFAULT_PROFILE = "train,slow"  # training / selection seed k flies profile k mod 2: speed only when needed is tested
# Seconds per 60 s episode on one loaded Modal worker (16 workers on 8 cores): guesses until modal_app::probe has
# measured them. The fly arms tick the 1,446-neuron brain at 20 Hz plus the pose model; the law arm only the follow.
S_EPISODE_GUESS = {"FLOWN": 3.5, "V3-FOLLOW": 3.5, "V3-JOINT": 3.5, "V3-LAW": 0.4, "V3B-FOLLOW": 3.5, "V3B-JOINT": 3.5}


# ---------------------------------------------------------------------------------------------------------------
# Seeds, chunks, configs, projection
# ---------------------------------------------------------------------------------------------------------------


def runs_root() -> Path:
    return Path(os.environ.get("KITE_BEHIND_RUNS") or REPO / "results" / "behind" / "runs")


def train_seeds(gen: int, k: int, cma_seed: int) -> list[int]:
    """The K common training seeds of generation gen (1-based) of a CMA seed:
    BEHIND_TRAIN_BASE + SEED_STRIDE * cma_seed + gen * K + j (each CMA seed sees its own episodes)."""
    if gen < 1 or k < 1 or cma_seed < 0:
        raise ValueError("gen and k start at 1, cma_seed at 0")
    if (gen + 1) * k > SEED_STRIDE:
        raise ValueError(f"generation {gen} x K {k} runs into the next CMA seed's training seeds")
    base = BEHIND_TRAIN_BASE + SEED_STRIDE * int(cma_seed) + gen * k
    return [base + j for j in range(k)]


def default_chunk(arm: str, spec: dict = EVAL_SPEC) -> int:
    """Items per Modal evaluate_chunk call: two episodes per worker for the fly arms, more for the cheap law arm."""
    return spec["workers"] * (2 if arm in bp.FLY_ARMS else 8)


def make_run_cfg(arm: str, cma_seed: int = 1, tag: str = DEFAULT_TAG, **over) -> dict:
    """A run config: DESIGN section 8's plan for the arm, then the given overrides (None = keep the default).
    fitness: FitnessCfg overrides ({} = its defaults; recorded in full in config.json)."""
    if arm not in ARM_PLAN:
        raise ValueError(f"{arm} is not trainable; one of {tuple(ARM_PLAN)}")
    d = ARM_PLAN[arm]
    cfg = {"arm": arm, "cma_seed": int(cma_seed), "tag": tag, "gens": d["gens"], "pop": d["pop"], "k": d["k"],
           "sigma0": d["sigma0"], "select_every": 5, "budget_usd": DEFAULT_BUDGET_USD, "episode_s": 60.0,
           "profile": DEFAULT_PROFILE, "n_sel": len(SELECTION_SEEDS), "maxstd": 0.3, "stub_s": None, "fitness": {},
           "allow_code_change": False, "init": None}
    for k, v in over.items():
        if k not in cfg:
            raise KeyError(f"unknown config key {k}")
        if v is not None:
            cfg[k] = v
    if isinstance(cfg["init"], (str, Path)):  # a model file (a round-1 best.json): embedded, so a Modal driver has it
        rec = json.loads(Path(cfg["init"]).read_text(encoding="utf-8"))
        cfg["init"] = {"source": str(cfg["init"]), "follow": rec.get("follow") or {}, "x": rec.get("x")}
    FitnessCfg(**cfg["fitness"])  # unknown fields fail here, not after an hour
    profiles(cfg)
    if cfg["pop"] < 2 or cfg["k"] < 1 or cfg["n_sel"] < 1 or cfg["n_sel"] > len(SELECTION_SEEDS):
        raise ValueError(f"pop >= 2, k >= 1 and 1 <= n_sel <= {len(SELECTION_SEEDS)}")
    return cfg


def profiles(cfg: dict) -> tuple[str, ...]:
    """The run's profile list (cfg["profile"], comma separated): seed k of a generation or of the selection flies
    profiles[k % len]."""
    out = tuple(p.strip() for p in str(cfg["profile"]).split(",") if p.strip())
    bad = [p for p in out if p not in PROFILES]
    if not out or bad:
        raise ValueError(f"profile must be a comma list of {PROFILES}, got {cfg['profile']!r}")
    return out


def plan_runs(tag: str = DEFAULT_TAG, arms: tuple[str, ...] = TRAINABLE, **over) -> list[dict]:
    """Every run of DESIGN section 8's plan (each arm on its CMA seeds)."""
    return [make_run_cfg(a, s, tag, **over) for a in arms for s in ARM_PLAN[a]["cma_seeds"]]


def s_episode_guess(cfg: dict, arm: str | None = None) -> float:
    """Seconds per episode on a loaded Modal worker for the run's arm (or FLOWN), scaled to its episode length."""
    if cfg.get("stub_s") is not None:
        return max(float(cfg["stub_s"]), 0.01)
    return S_EPISODE_GUESS[arm or cfg["arm"]] * float(cfg["episode_s"]) / 60.0


def project_run(cfg: dict, prices: Prices | None = None, s_ep: float | None = None, s_flown: float | None = None,
                chunk: int | None = None, s_max: float | None = None, spec: dict = EVAL_SPEC) -> dict:
    """kite.fly.train.cma_train.project() for a run: its 'law' episodes are FLOWN's (K per generation + the
    selection seeds once, cached like KITE-LAW there). A chunk lasts as long as its slowest worker, and train
    interleaves FLOWN's K episodes through every chunk, so the slowest worker runs one FLOWN episode and r - 1 of the
    arm's (r = ceil(chunk / workers) rounds): s_max = TAIL_X x max(s_ep, (s_flown + (r - 1) s_ep) / r) per round
    (TAIL_X x s_ep for the fly arms, whose episodes cost what FLOWN's do)."""
    s_ep = s_ep or s_episode_guess(cfg)
    s_flown = s_flown or s_episode_guess(cfg, "FLOWN")
    chunk = chunk or default_chunk(cfg["arm"], spec)
    pop, k = int(cfg["pop"]), int(cfg["k"])
    r = math.ceil(chunk / spec["workers"])
    s_max = s_max or TAIL_X * max(s_ep, (s_flown + (r - 1) * s_ep) / r)
    p = ct.project(cfg["arm"], int(cfg["gens"]), pop, k, int(cfg["select_every"]), s_ep, s_flown, prices,
                   int(cfg["n_sel"]), chunk, spec=spec, s_max=s_max)
    p["episodes_flown"] = p.pop("episodes_law")
    p["s_per_flown_episode"] = p.pop("s_per_law_episode")
    p["chunk"] = chunk
    return p


CHUNK_ROUNDS = (1, 2, 3, 4, 6, 8, 12, 16)  # episodes per worker per chunk that auto_chunk tries


def auto_chunk(cfg: dict, prices: Prices | None = None, s_ep: float | None = None, s_flown: float | None = None,
               spec: dict = EVAL_SPEC) -> int:
    """The chunk size with the lowest projected cost (then wall). Every container of a map call is billed until its
    slowest chunk ends, so one chunk too many for max_containers costs a whole second wave: V3-JOINT's (40 + 1) x 32
    items in chunks of 32 make 41 chunks on 40 containers and double the bill; chunks of 48 make 28."""
    best = None
    for m in CHUNK_ROUNDS:
        chunk = spec["workers"] * m
        p = project_run(cfg, prices, s_ep, s_flown, chunk, spec=spec)
        key = (round(p["total_usd"], 6), p["wall_h"], chunk)
        best = key if best is None or key < best else best
    return int(best[2])


def format_projection(p: dict) -> str:
    return (f"{p['arm']}: {p['gens']} gens, {p['episodes_arm']} + {p['episodes_flown']} FLOWN episodes at "
            f"{p['s_per_episode']:.2f} / {p['s_per_flown_episode']:.2f} s, {p['container_h']:.2f} container-h on "
            f"{p['containers']} containers (${p['usd_per_container_h']:.3f}/h, {p['prices']}): ${p['eval_usd']:.2f} "
            f"eval + ${p['driver_usd']:.2f} driver = ${p['total_usd']:.2f}, wall {p['wall_h'] * 60:.0f} min")


def interleave(heavy: list, light: list) -> list:
    """heavy spread evenly through light (deterministic). A Modal chunk is a consecutive slice of the items, so the
    3.5 s FLOWN episodes must not all land in one chunk of 0.4 s law episodes."""
    if not heavy or not light:
        return list(heavy) + list(light)
    out, m, n = [], len(heavy), len(light)
    for i, h in enumerate(heavy):
        out.append(h)
        out.extend(light[i * n // m:(i + 1) * n // m])
    return out


def is_start_failed(r: dict) -> bool:
    v = (r.get("metrics") or {}).get("start_failed", 0.0)
    return v is not None and float(v) > 0.0


def metric_means(results: list[dict]) -> dict[str, float]:
    """Mean of every finite numeric metric over the episodes that ran and started."""
    ok = [r["metrics"] for r in results if not bev.is_error(r) and not is_start_failed(r)]
    keys = sorted({k for m in ok for k, v in m.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
    out = {}
    for k in keys:
        v = [float(m[k]) for m in ok if isinstance(m.get(k), (int, float)) and math.isfinite(float(m[k]))]
        if v:
            out[k] = float(np.mean(v))
    return out


def finite(obj: Any) -> Any:
    """obj as plain JSON data with every non-finite float replaced by None (model files are strict JSON)."""
    if isinstance(obj, dict):
        return {str(k): finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [finite(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _sha(X: list[np.ndarray]) -> str:
    return hashlib.sha1(np.ascontiguousarray(np.asarray(X, dtype=np.float64)).tobytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------------------------------------------
# Backends: kite.fly's metering, kite.fly.behind's items
# ---------------------------------------------------------------------------------------------------------------


class BehindLocalBackend(ct.LocalBackend):
    """kite.fly's local process-pool backend on kite.fly.behind.evaluate's items (same cost estimate)."""

    def evaluate(self, items: list[dict]) -> tuple[list[dict], dict]:
        out = bev.evaluate_chunk_local(items, workers=self.processes)
        busy = sum(float(r.get("item_wall_s") or 0.0) for r in out["results"])
        container_s = busy / (EVAL_SPEC["workers"] * ct.PACKING)
        return out["results"], {"wall_s": out["wall_s"], "container_s": container_s, "busy_s": busy,
                                "usd": container_s * ct.eval_usd_per_s(self.prices, EVAL_SPEC) * ct.OVERHEAD,
                                "n_chunks": 1,
                                "basis": "local estimate"}

    def close(self) -> None:
        bev.shutdown_pool()


class BehindModalBackend(ct.ModalBackend):
    """kite.fly's Modal backend (evaluate_chunk.map, the results, the scaledown tail), metered by THIS run's chunks.
    kite.fly bills every container a map call touched for the call's whole wall time. That is right for one run
    alone on its containers, but a launch runs 10 drivers on one Evaluator capped at max_containers: a call's wall
    then includes waiting for containers busy with other runs' chunks, the same container-seconds are billed to several
    runs, and every run stops on budget at ~60 % of its generations having spent ~60 % of the budget (review). Here
    a call is billed max(the busy time of its chunks, its containers x its slowest chunk): what it used, plus the
    idle its own stragglers cause, never another run's queue (x ct.OVERHEAD for cold starts and tails, as kite.fly).
    A chunk that failed as a whole comes back as kite.fly error results; they are marked metrics["error"] so the
    fitness charges them."""

    def __init__(self, fn: Any, chunk: int, prices: Prices | None = None, spec: dict = EVAL_SPEC):
        super().__init__(fn, chunk, prices, spec)

    def evaluate(self, items: list[dict]) -> tuple[list[dict], dict]:
        results, stats = super().evaluate(items)
        for r in results:
            if not r.get("ok"):
                r["metrics"] = {**(r.get("metrics") or {}), "error": 1.0}
        busy = float(stats.get("busy_s") or 0.0)
        slowest = float(stats.get("chunk_wall_max_s") or 0.0)
        container_s = max(busy, int(stats.get("containers") or 0) * slowest)
        stats.update(container_s_shared=stats["container_s"], container_s=container_s, basis="modal own chunks",
                     usd=container_s * ct.eval_usd_per_s(self.prices, self.spec) * ct.OVERHEAD)
        return results, stats


def modal_evaluate_fn(modal_app: Any) -> Any:
    """The handle the Modal backend maps over: modal_app.evaluate_chunk (a Modal function) or evaluate_chunk() (a
    helper returning one, as kite.fly's app has)."""
    f = getattr(modal_app, "evaluate_chunk", None)
    if f is None:
        raise RuntimeError("kite.fly.behind.modal_app has no evaluate_chunk")
    return f if hasattr(f, "map") else f()


# ---------------------------------------------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------------------------------------------


class Trainer:
    """One CMA-ES run (arm x CMA seed x tag), checkpointed every generation."""

    def __init__(self, cfg: dict, backend: Any, run_dir: str | Path | None = None, *,
                 log: Callable[[str], None] = print, on_checkpoint: Callable[[], None] | None = None,
                 driver_usd_per_s: float = 0.0):
        self.cfg = copy.deepcopy(cfg)
        self.arm = cfg["arm"]
        self.backend = backend
        self.run_dir = Path(run_dir) if run_dir else runs_root() / run_name(self.arm, cfg["cma_seed"], cfg["tag"])
        self.space = bp.space(self.arm)
        self.fc = FitnessCfg(**(cfg.get("fitness") or {}))
        self.log = log
        self.on_checkpoint = on_checkpoint
        self.driver_usd_per_s = float(driver_usd_per_s)
        self.state: dict = {}
        self.profiles = profiles(cfg)
        self.code = run_digest()  # what the scores depend on: a resume with other code is refused
        self.episode_code = episode_digest()  # what FLOWN's cached metrics depend on: part of their key

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
            if st.get("version") != STATE_VERSION:
                raise RuntimeError(f"cannot resume {self.run_dir}: state version {st.get('version')}, this trainer "
                                   f"writes {STATE_VERSION}; use --fresh or another --tag")
            old = {k: st["cfg"].get(k) for k in RESUME_KEYS}
            new = {k: self.cfg.get(k) for k in RESUME_KEYS}
            if old != new:
                diff = {k: (old[k], new[k]) for k in RESUME_KEYS if old[k] != new[k]}
                raise RuntimeError(f"cannot resume {self.run_dir}: config differs {diff}; use --fresh or another --tag")
            if st.get("code") != self.code:
                if not self.cfg.get("allow_code_change"):
                    raise RuntimeError(
                        f"cannot resume {self.run_dir}: the code changed since generation 0 (run digest "
                        f"{st.get('code')} -> {self.code}: the sim, the follow, the bounds or the fitness), so its "
                        "FLOWN normalizer and CMA state belong to another objective; use --fresh, another --tag, or "
                        "--allow-code-change to go on anyway (FLOWN is re-run on the new code)")
                st.setdefault("code_changes", []).append({"gen": st["gen"], "time": _now(), "from": st.get("code"),
                                                          "to": self.code})
                append_jsonl(self.log_path, {"gen": st["gen"], "time": _now(), "event": "code change",
                                             "from": st.get("code"), "to": self.code})
                self.log(f"[{self.name}] WARNING: resuming across a code change ({st.get('code')} -> {self.code})")
                st["code"] = self.code
            st["cfg"].update({k: self.cfg[k] for k in ("gens", "budget_usd", "select_every")})
            self.state = st
            truncate_jsonl(self.log_path, st["gen"])
            truncate_jsonl(self.sel_path, st["gen"])
            self.log(f"[{self.name}] resumed at generation {st['gen']}")
            return True
        # the flown follow, or a warm start from a model file (make_run_cfg embedded it; knobs outside this arm's
        # bounds, e.g. a round-1 fwd_cap_fast 41 under V3B's 50-63, are clipped onto them)
        c = self.cfg
        x0 = np.clip(bp.encode_model(self.arm, c["init"]) if c.get("init") else bp.init_x(self.arm), 0.0, 1.0)
        self.state = {"version": STATE_VERSION, "cfg": copy.deepcopy(c), "x0": x0.tolist(),
                      "names": list(self.space.names), "init_source": "flown follow" + (
                          " + smooth2's brain x" if self.space.brain_names else ""),
                      "gen": 0, "flown": {}, "best": None, "sel_done": [], "usd": 0.0, "container_s": 0.0,
                      "wall_s": 0.0, "episodes": 0, "errors": 0, "stopped": None, "last_gen_usd": 0.0,
                      "created": _now(), "code": self.code, "episode_code": self.episode_code}
        opts = {"popsize": int(c["pop"]), "bounds": [0.0, 1.0], "seed": np.nan, "randn": SeededRandn(0),
                "verbose": -9, "tolfun": 0, "tolfunhist": 0, "tolflatfitness": 10**9, "tolx": 0,
                "tolstagnation": 10**9, "maxiter": 10**9, "CMA_stds": bp.cma_stds(self.arm, float(c["sigma0"]))}
        if c.get("maxstd"):
            opts["maxstd"] = float(c["maxstd"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.state["es"] = cma.CMAEvolutionStrategy(x0.tolist(), float(c["sigma0"]), opts)
        write_json(self.run_dir / "config.json", {
            **{k: v for k, v in self.state.items() if k not in ("es", "flown")},
            "fitness_cfg": asdict(self.fc), "cma_stds": opts["CMA_stds"], "space": bp.describe(self.arm),
            "seeds": {"train_base": BEHIND_TRAIN_BASE, "stride": SEED_STRIDE,
                      "selection": [SELECTION_SEEDS[0], SELECTION_SEEDS[c["n_sel"] - 1]]}})
        self.save()
        self.log(f"[{self.name}] new run: dim {x0.size}, pop {c['pop']}, K {c['k']}, sigma0 {c['sigma0']}, init "
                 f"{self.state['init_source']}, dir {self.run_dir}")
        return False

    def save(self) -> None:
        atomic_write(self.state_path, pickle.dumps(self.state, protocol=pickle.HIGHEST_PROTOCOL))
        if self.on_checkpoint:
            self.on_checkpoint()

    # ---- items and scoring. A slot is (seed, profile): seed k of a generation / the selection flies profile k mod n
    def slots(self, seeds: list[int]) -> list[tuple[int, str]]:
        return [(s, self.profiles[j % len(self.profiles)]) for j, s in enumerate(seeds)]

    def _item(self, arm: str, slot: tuple[int, str], x, tag: dict) -> dict:
        c = self.cfg
        return bev.make_item(arm, slot[0], slot[1], x=x, episode_s=c["episode_s"], stub_s=c["stub_s"], tag=tag)

    def _flown_key(self, slot: tuple[int, str]) -> str:
        c = self.cfg
        return f"{slot[0]}|{slot[1]}|{c['episode_s']}|{c['stub_s']}|{self.episode_code}"

    def _flown_items(self, slots: list[tuple[int, str]]) -> list[dict]:
        return [self._item("FLOWN", sl, None, {"role": "flown"}) for sl in slots
                if self._flown_key(sl) not in self.state["flown"]]

    def _store_flown(self, results: list[dict]) -> dict[tuple[int, str], dict]:
        """Cache FLOWN's metrics per slot (they never change for this code). A failed FLOWN episode is not cached (it
        is re-run when the slot comes back); for this call its slot is excluded for every candidate alike."""
        failed = {}
        for r in results:
            if r["tag"]["role"] != "flown":
                continue
            sl = (int(r["seed"]), r["profile"])
            if bev.is_error(r):
                failed[sl] = {"error": 1.0}
            else:
                self.state["flown"][self._flown_key(sl)] = r["metrics"]
        return failed

    def _flown_metrics(self, slots: list[tuple[int, str]], failed: dict) -> dict[tuple[int, str], dict]:
        return {sl: self.state["flown"].get(self._flown_key(sl), failed.get(sl, {"error": 1.0})) for sl in slots}

    def _score(self, results: list[dict], flown: dict[tuple[int, str], dict]) -> dict:
        """fitness() of one candidate's episodes against FLOWN's on the same slots."""
        return fitness([r["metrics"] for r in results], [flown[(int(r["seed"]), r["profile"])] for r in results],
                       self.fc)

    def _account(self, stats: dict, wall_s: float, n: int, results: list[dict]) -> float:
        usd = float(stats.get("usd") or 0.0) + wall_s * self.driver_usd_per_s
        st = self.state
        st["usd"] += usd
        st["container_s"] += float(stats.get("container_s") or 0.0)
        st["wall_s"] += wall_s
        st["episodes"] += n
        errs = [r for r in results if bev.is_error(r)]
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
        slots = self.slots(list(SELECTION_SEEDS[: self.cfg["n_sel"]]))
        items = interleave(self._flown_items(slots),
                           [self._item(self.arm, sl, x, {"role": "sel"}) for sl in slots])
        results, stats = self.backend.evaluate(items)
        flown = self._flown_metrics(slots, self._store_flown(results))
        sel = [r for r in results if r["tag"]["role"] == "sel"]
        f = self._score(sel, flown)
        fit = float(f["fitness"])
        usd = self._account(stats, time.perf_counter() - t0, len(results), results)
        best = self.state["best"]
        improved = best is None or fit < best["sel_fitness"]
        by_prof = {p: [r for r in sel if r["profile"] == p] for p in self.profiles}
        rec = {"gen": gen, "time": _now(), "sel_fitness": fit, "mean_score": f.get("mean_score"), "n": f["n"],
               "excluded": f["excluded"], "collisions": f["collisions"],
               "excess_collisions": f.get("excess_collisions"), "errors": f["errors"], "cost": f.get("cost"),
               "flown_cost": f.get("flown_cost"), "parts": f.get("parts"), "flown_parts": f.get("flown_parts"),
               "by_profile": {p: self._score(rs, flown)["fitness"] for p, rs in by_prof.items() if rs},
               "metrics": metric_means(sel),
               "flown_metrics": metric_means([{"ok": True, "metrics": flown[sl]} for sl in slots]),
               "improved": improved, "usd": usd, "x": x.tolist()}
        if improved:
            self.state["best"] = {"gen": gen, "sel_fitness": fit, "x": x.tolist()}
            self._write_best(rec)
        rec["best_gen"], rec["best_sel_fitness"] = self.state["best"]["gen"], self.state["best"]["sel_fitness"]
        append_jsonl(self.sel_path, rec)
        self.state["sel_done"].append(gen)
        note = ""
        if gen == 0 and self.arm != "V3-LAW" and abs(fit - 1.0) > 1e-9 and self.cfg["stub_s"] is None \
                and not self.cfg.get("init"):
            note = " (the init is the flown follow: expected 1.0, so the tuned controller or the world differs)"
        per = ", ".join(f"{p} {v:.4f}" for p, v in rec["by_profile"].items())
        self.log(f"[{self.name}] selection gen {gen}: {fit:.4f} on {f['n']} seeds ({per}), {f['collisions']} "
                 f"collisions (best {rec['best_sel_fitness']:.4f} at gen {rec['best_gen']}){note}")
        return rec

    def model_name(self) -> str:
        return f"{self.arm.lower()}-s{self.cfg['cma_seed']}-{self.cfg['tag']}"

    def _write_best(self, rec: dict) -> None:
        """best.json as a model file (DESIGN section 6; save_model writes it atomically and refuses NaN, so the
        trained record carries None where a mean was not finite)."""
        from kite.fly.behind.world import MAX_RC
        from kite.follow_tuned import save_model

        c, st = self.cfg, self.state
        x = rec["x"]
        trained = {"behind_arm": self.arm, "cma_seed": c["cma_seed"], "tag": c["tag"], "gen": rec["gen"],
                   "fitness": rec["sel_fitness"], "time": rec["time"], "run": self.name, "x": list(x),
                   "names": list(self.space.names), "n_sel": c["n_sel"], "n": rec["n"], "excluded": rec["excluded"],
                   "collisions": rec["collisions"], "parts": rec["parts"], "flown_parts": rec["flown_parts"],
                   "metric_means": rec["metrics"], "flown_metric_means": rec["flown_metrics"],
                   "profile": c["profile"], "episode_s": c["episode_s"], "pop": c["pop"], "k": c["k"],
                   "sigma0": c["sigma0"], "stub_s": c["stub_s"], "init_source": st["init_source"],
                   "fitness_cfg": asdict(self.fc), "by_profile": rec.get("by_profile"),
                   "excess_collisions": rec.get("excess_collisions"), "code": st.get("code")}
        save_model(self.best_path, name=self.model_name(), follow=bp.model_follow(self.arm, x),
                   brain_x=self.space.brain_x(x), steering=bp.steering(self.arm), trained=finite(trained),
                   governor={"trained_max_rc": MAX_RC})

    # ---- one generation
    def step(self) -> dict:
        st, c = self.state, self.cfg
        es = st["es"]
        gen = st["gen"] + 1
        t0 = time.perf_counter()
        seeds = train_seeds(gen, c["k"], c["cma_seed"])
        slots = self.slots(seeds)
        es.opts["randn"].reseed([int(c["cma_seed"]), gen, RANDN_SALT])
        X_raw = es.ask()
        X = [np.clip(np.asarray(x, dtype=float), 0.0, 1.0) for x in X_raw]
        flown_items = self._flown_items(slots)
        cand = [self._item(self.arm, sl, x, {"role": "cand", "i": i}) for i, x in enumerate(X) for sl in slots]
        items = interleave(flown_items, cand)
        results, stats = self.backend.evaluate(items)
        flown = self._flown_metrics(slots, self._store_flown(results))
        by_i: list[list[dict]] = [[] for _ in X]
        for r in results:
            if r["tag"]["role"] == "cand":
                by_i[r["tag"]["i"]].append(r)
        scored = [self._score(rs, flown) for rs in by_i]
        fits = [float(f["fitness"]) for f in scored]
        es.tell(X_raw, fits)
        st["gen"] = gen
        wall = time.perf_counter() - t0
        usd = self._account(stats, wall, len(results), results)
        f = np.asarray(fits)
        ib = int(np.argmin(f))
        cand_res = [r for rs in by_i for r in rs]
        arm_walls = [float(r.get("wall_s") or 0.0) for r in cand_res if not bev.is_error(r)]
        fl_walls = [float(r.get("wall_s") or 0.0) for r in results
                    if r["tag"]["role"] == "flown" and not bev.is_error(r)]
        return {"gen": gen, "time": _now(), "fit_best": float(f.min()), "fit_median": float(np.median(f)),
                "fit_mean": float(f.mean()), "fit_worst": float(f.max()), "sigma": float(es.sigma),
                "mean_x": self.mean_x().tolist(), "cand_sha": _sha(X), "seeds": [seeds[0], seeds[-1]],
                "n_items": len(items), "n_flown": len(flown_items),
                "n_errors": sum(bev.is_error(r) for r in results),
                "n_excluded": int(np.median([s["excluded"] for s in scored])),
                "collisions": int(sum(s["collisions"] for s in scored)),
                "excess_collisions": int(sum(s.get("excess_collisions") or 0 for s in scored)),
                "best_parts": scored[ib].get("parts"),
                "errors": sorted({str(r.get("error"))[:200] for r in results if bev.is_error(r)})[:3],
                "metrics": metric_means(cand_res), "wall_s": wall,
                "s_per_episode": float(np.mean(arm_walls)) if arm_walls else None,
                "s_per_flown_episode": float(np.mean(fl_walls)) if fl_walls else None,
                "backend": {k: v for k, v in stats.items() if k != "usd"}, "usd_gen": usd}

    def _selection_due(self, gen: int) -> bool:
        every = int(self.cfg["select_every"] or 0)
        return gen not in self.state["sel_done"] and ((every > 0 and gen % every == 0) or gen == self.cfg["gens"])

    def run(self, gens: int | None = None, stop_after: int | None = None) -> dict:
        """Generations up to gens (resuming if a checkpoint exists). stop_after: return after that many new
        generations (tests use it to simulate a kill)."""
        if gens is not None:
            self.cfg["gens"] = int(gens)
        self.load_or_init()
        st = self.state
        gens = int(self.cfg["gens"])
        budget = self.cfg.get("budget_usd")
        tail = self.backend.finish_usd()
        if st["stopped"] == "budget" and budget and st["usd"] + st["last_gen_usd"] + tail <= budget:
            st["stopped"] = None  # the budget was raised
        if st["stopped"] == "done" and st["gen"] < gens:
            st["stopped"] = None  # more generations were asked for
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
                "episodes": st.get("episodes"), "errors": st.get("errors"),
                "best_json": str(self.best_path) if self.best_path.exists() else None}


# ---------------------------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m kite.fly.behind.train", description=__doc__.split("\n\n")[0])
    p.add_argument("--arm", required=True, choices=list(ARM_PLAN))
    p.add_argument("--cma-seed", type=int, default=1)
    p.add_argument("--gens", type=int)
    p.add_argument("--pop", type=int)
    p.add_argument("--k", type=int, help="training episodes per candidate per generation")
    p.add_argument("--sigma0", type=float, help="initial step on the normalized box (default 0.15)")
    p.add_argument("--backend", choices=["local", "modal"], default="local")
    p.add_argument("--tag", default=DEFAULT_TAG)
    p.add_argument("--select-every", type=int)
    p.add_argument("--budget-usd", type=float, help=f"metered stop (default ${DEFAULT_BUDGET_USD:.0f})")
    p.add_argument("--episode-s", type=float, help="episode length (smoke tests only; default 60)")
    p.add_argument("--n-sel", type=int, help=f"selection episodes (default all {len(SELECTION_SEEDS)}; smoke tests)")
    p.add_argument("--maxstd", type=float, help="pycma maxstd on the normalized box (default 0.3)")
    p.add_argument("--fitness-json", help="FitnessCfg overrides, a JSON object or a path to one (e.g. calibrated "
                                          "stick scales)")
    p.add_argument("--stub-s", type=float, help="synthetic episodes burning this much CPU (no sim; smoke tests)")
    p.add_argument("--processes", type=int, help="local pool size (default: all cores)")
    p.add_argument("--chunk", help="items per Modal evaluate_chunk call (default: default_chunk, what modal_app's "
                                   "train uses; 'auto': the cheapest projected)")
    p.add_argument("--profile", help=f"profiles of the training and selection seeds, round robin (default "
                                     f"{DEFAULT_PROFILE})")
    p.add_argument("--allow-code-change", action="store_true",
                   help="resume although the code changed since the run started (FLOWN is re-run on the new code)")
    p.add_argument("--s-episode", "--s-per-episode", dest="s_episode", type=float,
                   help="s per arm episode on a loaded Modal worker, for the projection (default a guess: "
                        "3.5 fly arms, 0.4 law)")
    p.add_argument("--s-flown", type=float, help="s per FLOWN episode for the projection (default 3.5)")
    p.add_argument("--prices", choices=["live", "constants"], default="live",
                   help="Modal's live rates (modal backend and dry runs) or the built-in constants")
    p.add_argument("--runs-dir", help=f"default $KITE_BEHIND_RUNS or {REPO / 'results/behind/runs'}")
    p.add_argument("--fresh", action="store_true", help="move an existing run dir aside and start over")
    p.add_argument("--dry-run", action="store_true", help="print the plan and the projection, run nothing")
    return p


def parse_fitness(s: str | None) -> dict | None:
    if not s:
        return None
    text = Path(s).read_text(encoding="utf-8") if not s.lstrip().startswith("{") else s
    d = json.loads(text)
    if not isinstance(d, dict):
        raise ValueError("--fitness-json must be a JSON object of FitnessCfg fields")
    return d


def cfg_from_args(a: argparse.Namespace) -> dict:
    return make_run_cfg(a.arm, a.cma_seed, a.tag, gens=a.gens, pop=a.pop, k=a.k, sigma0=a.sigma0,
                        select_every=a.select_every, budget_usd=a.budget_usd, episode_s=a.episode_s, n_sel=a.n_sel,
                        maxstd=a.maxstd, stub_s=a.stub_s, fitness=parse_fitness(a.fitness_json), profile=a.profile,
                        allow_code_change=a.allow_code_change or None)


def pick_chunk(arg: str | None, cfg: dict, prices: Prices | None, s_ep: float | None, s_flown: float | None) -> int:
    """--chunk: a number, 'auto' (auto_chunk) or nothing (default_chunk, the size modal_app's train() uses, so a run
    driven from this laptop and a launched one project and bill the same way)."""
    if arg is None or str(arg).strip() == "":
        return default_chunk(cfg["arm"])
    if str(arg).strip().lower() == "auto":
        return auto_chunk(cfg, prices, s_ep, s_flown)
    return max(1, int(arg))


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    cfg = cfg_from_args(a)
    rdir = Path(a.runs_dir) / run_name(a.arm, a.cma_seed, a.tag) if a.runs_dir else None
    rdir = rdir or runs_root() / run_name(a.arm, a.cma_seed, a.tag)
    prices = load_prices(live=a.prices == "live" and (a.backend == "modal" or a.dry_run))
    chunk = pick_chunk(a.chunk, cfg, prices, a.s_episode, a.s_flown)
    proj = project_run(cfg, prices, a.s_episode, a.s_flown, chunk)
    print(json.dumps(cfg, default=_json_default))
    print("projection on Modal: " + format_projection(proj))
    if cfg["budget_usd"] and proj["total_usd"] > cfg["budget_usd"]:
        print(f"WARNING: projected ${proj['total_usd']:.2f} exceeds the budget ${cfg['budget_usd']:.2f}; the run will "
              "stop itself when its metered cost reaches it")
    if a.dry_run:
        runs = plan_runs(a.tag, episode_s=cfg["episode_s"], stub_s=cfg["stub_s"], fitness=cfg["fitness"],
                         profile=cfg["profile"])
        projs = []
        for r in runs:
            s_ep = a.s_episode if r["arm"] == a.arm else None
            projs.append(project_run(r, prices, s_ep, a.s_flown, pick_chunk(a.chunk, r, prices, s_ep, a.s_flown)))
        for r, p in zip(runs, projs):
            print(f"  plan {run_name(r['arm'], r['cma_seed'], r['tag']):22s} ${p['total_usd']:7.2f}  "
                  f"wall {p['wall_h'] * 60:5.0f} min  {p['containers']} containers x chunk {p['chunk']}")
        print(f"  plan total ${sum(p['total_usd'] for p in projs):.2f} for {len(runs)} runs; wall "
              f"{max(p['wall_h'] for p in projs) * 60:.0f} min if they get their containers in parallel")
        return 0
    if a.fresh and rdir.exists():
        bak = rdir.with_name(rdir.name + ".bak-" + time.strftime("%Y%m%d-%H%M%S"))
        shutil.move(str(rdir), str(bak))
        print(f"moved {rdir} to {bak}")
    t0 = time.perf_counter()
    if a.backend == "local":
        backend = BehindLocalBackend(a.processes, prices)
        try:
            summary = Trainer(cfg, backend, rdir).run()
        finally:
            backend.close()
    else:
        import modal

        from kite.fly.behind import modal_app

        with modal.enable_output(), modal_app.app.run():
            backend = BehindModalBackend(modal_evaluate_fn(modal_app), chunk, prices, spec=EVAL_SPEC)
            summary = Trainer(cfg, backend, rdir).run()
    summary["driver_wall_s"] = time.perf_counter() - t0
    print(json.dumps(summary, indent=1, default=_json_default))
    return 0 if summary.get("stopped") in ("done", "budget", None) else 1


if __name__ == "__main__":
    # Run the importable module, not __main__, so state.pkl refers to importable classes.
    from kite.fly.behind import train as _module  # noqa: PLW0406

    sys.exit(_module.main())
