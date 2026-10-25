"""Modal app `kite-behind`: the CMA-ES runs of kite.fly.behind (DESIGN section 8) and their evaluations in Modal.

    .venv/bin/modal run -m kite.fly.behind.modal_app::smoke                        # stub episodes, image self-check
    .venv/bin/modal run -m kite.fly.behind.modal_app::probe --containers 160       # s per real episode + concurrency
    .venv/bin/modal run -m kite.fly.behind.modal_app::launch                       # the plan and its projection only
    .venv/bin/modal run --detach -m kite.fly.behind.modal_app::launch --confirm --budget-usd 40
    .venv/bin/modal run -m kite.fly.behind.modal_app::status
    .venv/bin/modal run -m kite.fly.behind.modal_app::fetch --run V3-JOINT_s1_v3   # or --tag v3: every run of a tag
    python -m kite.fly.behind.train --arm V3-JOINT --backend modal ...            # or drive one run from this laptop
    python -m kite.fly.behind.signoff ... --backend modal                         # the sign-off fan-out

Pieces (a new app next to kite.fly's `kite-fly`, which it never imports: that module builds its own image on import):
  image          debian_slim (Python 3.12) + numpy / scipy / cma / pyyaml pinned to the local venv (pyyaml because
                 kite.flybrain reads configs/controllers.yaml), and only the kite files the episode path needs: every
                 module an import probe sees (in a fresh interpreter that refuses cv2 / djitellopy / av / ultralytics /
                 PIL / torch, so a stray import fails on this laptop, not in 160 containers), all of kite/fly and
                 kite/flybrain (FlySteer imports its brain lazily, so the probe alone would miss it), the brain data
                 (kite/flybrain/data/brains/**, configs/*.yaml), kite/follow_behind.py and kite/follow_tuned.py.
                 Staged as a snapshot first: files being edited while the image builds cannot tear it.
  Evaluator      an eval container (EVAL_SPEC: 8 cores, 16 worker processes, max_containers from
                 $KITE_BEHIND_MAX_CONTAINERS, default 160). evaluate_chunk runs a batch of episode items with
                 kite.fly.behind.evaluate.evaluate_chunk_local and returns the results with its timing and container
                 id. On start it checks that the brain data is there and that smooth2 loads: FlyYaw silently falls
                 back to the yaw law when the brain cannot load, which would turn every fly arm into V3-LAW unseen.
  train          one CMA-ES driver (kite.fly.behind.train.Trainer) in a small container; state, logs and best.json
                 live on the Volume `kite-behind-runs` at /runs ($KITE_BEHIND_RUNS), committed every generation, so a
                 preempted or retried call resumes exactly; a lease file keeps a second driver off the same run
  train_all      the launch: spawns every planned run, records the call ids on the Volume, re-spawns a run that dies
                 (handing it the dead call's id, so the new driver may take over the lease the old one left)
Budget: launch splits --budget-usd over the runs in proportion to their projected cost; every driver meters its own
container time (kite.fly.behind.train.BehindModalBackend: its own chunks, not the queue behind the other runs on the
shared containers) and stops before a generation would pass its share. When the plan wants more containers at once
than the cap, the runs queue: launch prints the wall stretched by wanted / cap (the cost does not change).
Determinism: probe --determinism N runs one fixed episode on N containers at once and compares the metric hashes
(numpy picks SIMD kernels per CPU, so exp / sin / arctan2 can differ in the last bit between hosts); if they differ,
set KITE_BEHIND_NPY_DISABLE (e.g. "AVX512F AVX512_SKX") when building the image: it becomes NPY_DISABLE_CPU_FEATURES.
"""

from __future__ import annotations

import atexit
import inspect
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import modal

from kite.fly.behind import train as bt
from kite.fly.train import cma_train as ct

REPO = Path(__file__).resolve().parents[3]  # the repo here, /root in a container (the image mirrors kite/ there)
APP_NAME = "kite-behind"
VOLUME_NAME = "kite-behind-runs"
RUNS_MOUNT = "/runs"
RUNS_ENV = "KITE_BEHIND_RUNS"
REMOTE_ROOT = "/root"
PROBE_ENV = "KITE_BEHIND_IMPORT_PROBE"  # set inside the probe: a module importing this one must not stage again
LEASE_TTL_S = 15 * 60
MIN_SHARE_USD = 1.0  # budget floor of one run in a launch
DEFAULT_MAX_CONTAINERS = bt.DEFAULT_MAX_CONTAINERS
PROBE_SEED0 = 9_000  # probe seeds: outside training (7M+), selection (6000-6095) and test (8000-8199)
# = the local venv (tests check it), so the trainer here and the containers compute the same numbers
PACKAGES = ("numpy==2.5.4", "scipy==1.18.1", "cma==4.5.0", "pyyaml==6.0.3")
BLOCKED = ("cv2", "djitellopy", "av", "ultralytics", "PIL", "torch")  # must never be needed in a container
BLOCK_MARK = "is not available in the Modal image"
PROBE_MODULES = ("kite.fly.behind", "kite.fly.behind.world", "kite.fly.behind.evaluate", "kite.fly.behind.train",
                 "kite.fly.behind.digest",
                 "kite.fly.behind.signoff", "kite.fly.behind.params", "kite.fly.behind.fitness",
                 "kite.fly.behind.metrics", "kite.fly.behind.profiles", "kite.fly.behind.walker",
                 "kite.fly.behind.pose", "kite.follow_tuned", "kite.follow_behind", "kite.flybrain.flyfollow.steer",
                 "kite.drone.safety", "kite.fly.sim.drone")
PY_TREES = ("kite/fly", "kite/flybrain")  # every .py under these (lazy imports: FlySteer's brain, the sim pieces)
DATA_GLOBS = ("kite/flybrain/data/brains/**/*.npz", "kite/flybrain/data/brains/**/*.json",
              "kite/flybrain/configs/*.yaml", "kite/flybrain/flydrones/LICENSE",
              "kite/fly/data/*.npz", "kite/fly/data/*.json", "kite/fly/data/init/*.json",
              "kite/fly/data/reference/*.json", "kite/fly/data/behind/*.json")
EXTRA_FILES = ("kite/__init__.py", "kite/follow_behind.py", "kite/follow_tuned.py", "kite/types.py", "kite/camera.py")
# What an eval container refuses to run without (Evaluator.start)
REQUIRED = ("kite/follow_behind.py", "kite/follow_tuned.py", "kite/flybrain/flyfollow/steer.py",
            "kite/flybrain/data/brains/pursuit_core1.npz",
            "kite/flybrain/data/brains/trained/FLY-YAW_smooth2_best.json",
            "kite/flybrain/configs/controllers.yaml", "kite/fly/behind/__init__.py", "kite/fly/behind/world.py",
            "kite/fly/behind/evaluate.py", "kite/fly/behind/params.py", "kite/fly/behind/metrics.py")
THREADS = {v: "1" for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
NPY_DISABLE_ENV = "KITE_BEHIND_NPY_DISABLE"  # -> NPY_DISABLE_CPU_FEATURES in the image (only if probe saw drift)
# iCloud's conflict copies ("world 2.py", "data 2/"): never shipped
_ICLOUD_DUP = re.compile(r" \d+(\.[^./]*)?$")


# One eval container: 8 physical cores (Modal's CPU unit; 16 hyperthreads), one episode process per hyperthread, up
# to $KITE_BEHIND_MAX_CONTAINERS (default 160) at once. kite.fly.behind.train's EVAL_SPEC: its projections, chunk
# choice and metering use the same one, so a dry run there projects what this app then runs.
EVAL_SPEC = bt.EVAL_SPEC
DRIVER_SPEC = {**ct.DRIVER_SPEC, "cpu": 0.5, "memory_mib": 1024, "timeout_s": 86400, "retries": 3}

_PROBE = r"""
import importlib, importlib.abc, json, sys
BLOCKED, MODS, MARK = set(sys.argv[1].split(',')), [m for m in sys.argv[2].split(',') if m], sys.argv[3]
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in BLOCKED:
            raise ImportError(f'{name} {MARK} (imported while loading the episode path)')
        return None
sys.meta_path.insert(0, Block())
skipped = {}
for m in MODS:
    try:
        importlib.import_module(m)
    except ImportError as e:
        if MARK in str(e) or not (e.name == 'kite' or (e.name or '').startswith('kite.')):
            raise
        skipped[m] = f'{type(e).__name__}: {e}'  # a kite module that is not written (or not finished) yet
files = sorted({getattr(mod, '__file__', None) or '' for n, mod in list(sys.modules.items())
                if n == 'kite' or n.startswith('kite.')} - {''})
print(json.dumps({'files': files, 'skipped': skipped}))
"""


def _excluded(rel: Path) -> bool:
    return any(p == "__pycache__" or p == "tests" or _ICLOUD_DUP.search(p) for p in rel.parts)


def run_probe(modules: tuple[str, ...] | list[str] = PROBE_MODULES, repo: Path = REPO,
              extra_paths: tuple | list = ()) -> dict:
    """Import `modules` in a fresh interpreter that refuses BLOCKED packages: {"files": the kite files it loaded,
    "skipped": {module: why} for kite modules that do not import yet}. A blocked import, a missing third-party
    package or any other error raises (a container would fail the same way)."""
    path = os.pathsep.join([str(repo), *(str(p) for p in extra_paths)])
    out = subprocess.run([sys.executable, "-c", _PROBE, ",".join(BLOCKED), ",".join(modules), BLOCK_MARK],
                         cwd=repo, capture_output=True, text=True, timeout=600,
                         env={**os.environ, "PYTHONPATH": path, PROBE_ENV: "1", "KITE_NO_DEVICE_TESTS": "1"})
    if out.returncode != 0:
        raise RuntimeError(f"import probe failed (a container would fail the same way):\n{out.stderr[-3000:]}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def source_files(repo: Path = REPO) -> tuple[list[Path], dict[str, str]]:
    """The files the image ships (absolute, sorted) and the probe modules that do not import yet."""
    probe = run_probe(PROBE_MODULES, repo)
    files = {Path(f).resolve() for f in probe["files"]}
    for tree in PY_TREES:
        files |= {p.resolve() for p in (repo / tree).rglob("*.py")}
    for g in DATA_GLOBS + EXTRA_FILES:
        files |= {p.resolve() for p in repo.glob(g)}
    root = repo.resolve()
    keep = sorted(f for f in files if f.is_relative_to(root) and f.is_file() and not _excluded(f.relative_to(root)))
    return keep, dict(probe["skipped"])


def stage_sources(dest: Path, repo: Path = REPO) -> tuple[list[str], dict[str, str]]:
    """Copy source_files() into dest with the same relative layout: the upload then sees a snapshot."""
    files, skipped = source_files(repo)
    root = repo.resolve()
    rel = []
    for f in files:
        r = f.relative_to(root)
        (dest / r).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest / r)
        rel.append(r.as_posix())
    return rel, skipped


STAGED: dict = {"files": [], "skipped": {}}  # what this process put in the image (local side only)


def _image() -> modal.Image:
    env = {"PYTHONPATH": REMOTE_ROOT, RUNS_ENV: RUNS_MOUNT, "KITE_NO_DEVICE_TESTS": "1", **THREADS}
    if os.environ.get(NPY_DISABLE_ENV):
        env["NPY_DISABLE_CPU_FEATURES"] = os.environ[NPY_DISABLE_ENV]
    img = modal.Image.debian_slim(python_version="3.12").uv_pip_install(*PACKAGES).env(env)
    if modal.is_local() and not os.environ.get(PROBE_ENV):
        stage = Path(tempfile.mkdtemp(prefix="kite-behind-src-"))
        atexit.register(shutil.rmtree, stage, ignore_errors=True)
        STAGED["files"], STAGED["skipped"] = stage_sources(stage)
        img = img.add_local_dir(stage / "kite", f"{REMOTE_ROOT}/kite")
    return img


image = _image()
app = modal.App(APP_NAME, include_source=False)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
E, D = EVAL_SPEC, DRIVER_SPEC


def _retries(n: int) -> modal.Retries:
    return modal.Retries(max_retries=int(n), backoff_coefficient=2.0, initial_delay=2.0, max_delay=60.0)


def _versions() -> dict[str, str | None]:
    from importlib import metadata

    out = {}
    for pin in PACKAGES:
        name = pin.split("==")[0]
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def container_selfcheck(root: Path = REPO, load_fly: bool = True) -> dict:
    """What an eval container needs to run the episode path faithfully: the REQUIRED files under root, the pinned
    package versions, and smooth2 loading as FollowMission loads it (FlyYaw.load(BehindCfg())). ok False = refuse."""
    missing = [r for r in REQUIRED if not (Path(root) / r).is_file()]
    vers = _versions()
    wrong = {p.split("==")[0]: vers.get(p.split("==")[0]) for p in PACKAGES
             if vers.get(p.split("==")[0]) != p.split("==")[1]}
    out = {"root": str(root), "missing": missing, "python": sys.version.split()[0], "packages": vers,
           "wrong_versions": wrong, "fly": None, "fly_error": None}
    if load_fly and not missing:
        try:
            from kite.follow_behind import BehindCfg, FlyYaw

            steer = FlyYaw.load(BehindCfg())
            st = steer.stats
            out["fly"] = {"arm": st.get("arm"), "params": st.get("params")}
        except Exception as e:  # noqa: BLE001
            out["fly_error"] = f"{type(e).__name__}: {e}"
    out["ok"] = not missing and out["fly_error"] is None
    return out


@app.cls(image=image, cpu=E["cpu"], memory=E["memory_mib"], max_containers=E["max_containers"],
         scaledown_window=E["scaledown_s"], timeout=E["timeout_s"], retries=_retries(E["retries"]))
class Evaluator:
    """An eval container. A class (not a function) so its process pool is shut down on exit: a live pool thread
    otherwise holds every container 30 s past its scaledown."""

    @modal.enter()
    def start(self) -> None:
        self.t0, self.served = time.time(), 0
        self.check = container_selfcheck()
        if not self.check["ok"]:
            raise RuntimeError(f"this image cannot run the behind episodes faithfully: {self.check}")

    @modal.method()
    def evaluate_chunk(self, items: list[dict]) -> dict:
        """A batch of episode items on this container's process pool (one worker per hyperthread). t_start / t_end
        are this container's wall clock around the batch (the probe's concurrency count reads them)."""
        from kite.fly.behind import evaluate as ev

        cold, age = self.served == 0, time.time() - self.t0
        t_start = time.time()
        out = ev.evaluate_chunk_local(items, workers=E["workers"])
        self.served += 1
        out.update(t_start=t_start, t_end=time.time(), cold=cold, container_age_s=age, served=self.served,
                   container_id=os.environ.get("MODAL_TASK_ID"), n_cpu=os.cpu_count(),
                   n_affinity=len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                   cpu=cpu_isa())
        return out

    @modal.method()
    def selfcheck(self) -> dict:
        return {**self.check, "container_id": os.environ.get("MODAL_TASK_ID"), "n_cpu": os.cpu_count()}

    @modal.exit()
    def stop(self) -> None:
        from kite.fly.behind import evaluate as ev

        ev.shutdown_pool(wait=True)


def cpu_isa() -> dict:
    """This host's CPU model and the SIMD families numpy may dispatch on (Linux /proc/cpuinfo; {} elsewhere)."""
    try:
        text = Path("/proc/cpuinfo").read_text()
    except OSError:
        return {}
    model = next((ln.split(":", 1)[1].strip() for ln in text.splitlines() if ln.startswith("model name")), None)
    flags = next((ln.split(":", 1)[1].split() for ln in text.splitlines() if ln.startswith("flags")), [])
    return {"model": model, "avx2": "avx2" in flags, "avx512f": "avx512f" in flags}


def metrics_hash(metrics: dict) -> str:
    """A hash of an episode's metrics, bit for bit (the determinism probe compares hosts with it)."""
    import hashlib

    return hashlib.sha256(json.dumps(metrics, sort_keys=True).encode()).hexdigest()[:16]


def evaluate_chunk():
    """What the backends fan out over: evaluate_chunk().map(chunks) (Evaluator.evaluate_chunk). kite.fly.behind.train's
    modal_evaluate_fn calls this."""
    return Evaluator().evaluate_chunk


def default_chunk(arm: str) -> int:
    return bt.default_chunk(arm, EVAL_SPEC)


def modal_backend(chunk: int, prices: ct.Prices | None = None):
    """kite.fly.behind.train's Modal backend on this app's Evaluator, metered with EVAL_SPEC. Use inside app.run()."""
    return bt.BehindModalBackend(evaluate_chunk(), int(chunk), prices or ct.Prices(), spec=EVAL_SPEC)


def _lease(run_dir: Path, call_id: str | None, check: bool, takeover: str | None = None) -> None:
    """Refuse to drive a run another live call is driving (a double launch); else take or refresh the lease.
    takeover: the call id of a predecessor known to be over (train_all re-spawning a driver whose retries ran out):
    its lease is fresh, but nobody holds it any more."""
    path = run_dir / "lease.json"
    if check and path.exists():
        try:
            d = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            d = {}
        holder = d.get("call_id")
        if holder not in (None, call_id, takeover) and time.time() - float(d.get("heartbeat", 0)) < LEASE_TTL_S:
            raise RuntimeError(f"{run_dir.name} is driven by call {d['call_id']}")
    run_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"call_id": call_id, "heartbeat": time.time()}))


def run_name(cfg: dict) -> str:
    return ct.run_name(cfg["arm"], cfg["cma_seed"], cfg["tag"])


def new_trainer(cfg: dict, backend, run_dir: Path, on_checkpoint=None, driver_usd_per_s: float = 0.0):
    """kite.fly.behind.train.Trainer with the keywords it takes (on_checkpoint, driver_usd_per_s)."""
    from kite.fly.behind import train as bt

    sig = inspect.signature(bt.Trainer).parameters
    kw = {"on_checkpoint": on_checkpoint, "driver_usd_per_s": driver_usd_per_s}
    if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.values()):
        kw = {k: v for k, v in kw.items() if k in sig}
    return bt.Trainer(cfg, backend, run_dir, **kw)


@app.function(image=image, volumes={RUNS_MOUNT: volume}, cpu=D["cpu"], memory=D["memory_mib"], timeout=D["timeout_s"],
              retries=_retries(D["retries"]))
def train(cfg: dict, prices: dict | None = None, chunk: int | None = None, takeover: str | None = None) -> dict:
    """One CMA-ES run (a kite.fly.behind.train.make_run_cfg dict). Idempotent: resumes from the Volume. takeover: the
    call id of a dead predecessor whose lease this driver may take (train_all's re-spawn)."""
    volume.reload()
    p = ct.Prices(**prices) if prices else ct.Prices()
    run_dir = Path(RUNS_MOUNT) / run_name(cfg)
    call_id = modal.current_function_call_id()
    _lease(run_dir, call_id, check=True, takeover=takeover)
    volume.commit()

    def checkpoint() -> None:
        _lease(run_dir, call_id, check=False)
        volume.commit()

    trainer = new_trainer(cfg, modal_backend(chunk or default_chunk(cfg["arm"]), p), run_dir, checkpoint,
                          p.usd_per_s(D["cpu"], D["memory_mib"]))
    summary = trainer.run()
    (run_dir / "lease.json").unlink(missing_ok=True)
    volume.commit()
    return summary


@app.function(image=image, volumes={RUNS_MOUNT: volume}, cpu=0.125, memory=512, timeout=D["timeout_s"],
              retries=_retries(2))
def train_all(launch_id: str, runs: list[dict], prices: dict | None = None, max_restarts: int = 2) -> list[dict]:
    """Spawn train for every run, wait, re-spawn one that died (it resumes from its pickle). Call ids are kept in
    /runs/_launch/<launch_id>.json, so a restarted train_all re-attaches instead of spawning twins."""
    volume.reload()
    rec_path = Path(RUNS_MOUNT) / "_launch" / f"{launch_id}.json"
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    rec = json.loads(rec_path.read_text()) if rec_path.exists() else {"launch_id": launch_id, "runs": {}}

    def save() -> None:
        rec_path.write_text(json.dumps(rec, indent=1, default=str))
        volume.commit()

    calls = {}
    for cfg in runs:
        name = run_name(cfg)
        e = rec["runs"].get(name)
        if e and e["status"] in ("done", "failed"):
            continue
        if e and e["status"] == "running":
            calls[name] = modal.FunctionCall.from_id(e["call_id"])
            continue
        calls[name] = train.spawn(cfg, prices)
        rec["runs"][name] = {"call_id": calls[name].object_id, "status": "running", "restarts": 0, "cfg": cfg}
        save()
    while calls:
        time.sleep(20)
        for name in list(calls):
            e = rec["runs"][name]
            try:
                e["summary"] = calls[name].get(timeout=0)
                e["status"] = "done"
                del calls[name]
            except TimeoutError:
                continue
            except Exception as err:  # noqa: BLE001
                e["last_error"] = f"{type(err).__name__}: {err}"[:500]
                if e["restarts"] >= max_restarts:
                    e["status"] = "failed"
                    del calls[name]
                else:  # .get() raised: that call is over (Modal's retries spent), so its lease is free to take
                    e["restarts"] += 1
                    dead = e["call_id"]
                    calls[name] = train.spawn(e["cfg"], prices, None, dead)
                    e["call_id"] = calls[name].object_id
            save()
    return list(rec["runs"].values())


# ---------------------------------------------------------------------------------------------------------------
# Plan, projection, budget (pure: tests run them without Modal)
# ---------------------------------------------------------------------------------------------------------------


def _seeds(csv: str | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if csv is None:
        return tuple(default)
    return tuple(int(s) for s in str(csv).split(",") if s.strip())


def plan(tag: str = "v3", follow_seeds: str | None = None, joint_seeds: str | None = None,
         law_seeds: str | None = None, gens: int = 0, episode_s: float = 0.0,
         stub_s: float | None = None, profile: str = "", allow_code_change: bool = False) -> list[dict]:
    """DESIGN section 8's runs (kite.fly.behind.train.ARM_PLAN: V3-FOLLOW and V3-JOINT on CMA seeds 1-4, V3-LAW on
    1-2), each a make_run_cfg dict; a seeds csv of "" drops that arm. profile "" = train's default (train,slow)."""
    from kite.fly.behind import train as bt

    seeds = {"V3-FOLLOW": follow_seeds, "V3-JOINT": joint_seeds, "V3-LAW": law_seeds}
    over = {"gens": gens or None, "episode_s": episode_s or None, "stub_s": stub_s, "profile": profile or None,
            "allow_code_change": allow_code_change or None}
    return [bt.make_run_cfg(arm, s, tag, **over) for arm in ("V3-FOLLOW", "V3-JOINT", "V3-LAW")
            for s in _seeds(seeds[arm], bt.ARM_PLAN[arm]["cma_seeds"])]


def project(cfg: dict, prices: ct.Prices | None = None, s_ep: float | None = None, s_flown: float | None = None,
            s_max: float | None = None, chunk: int | None = None) -> dict:
    """kite.fly.behind.train.project_run on THIS app's containers (EVAL_SPEC: up to max_containers per map call, not
    kite.fly's 40, so a 41-chunk V3-JOINT generation is one wave, not two) with the chunk train() uses. s_ep / s_flown:
    seconds per arm / FLOWN episode on a loaded worker (default train's guesses); s_max: the slowest one."""
    return bt.project_run(cfg, prices, s_ep, s_flown, chunk or default_chunk(cfg["arm"]), s_max, spec=EVAL_SPEC)


def split_budget(runs: list[dict], prices: ct.Prices, budget_usd: float, s_by_arm: dict | None = None,
                 s_flown: float | None = None, s_max_by_arm: dict | None = None) -> tuple[list[dict], float]:
    """Each run's budget_usd: a share of the total in proportion to its projected cost, but at least
    min(MIN_SHARE_USD, total / runs) (the cheap law runs would otherwise get cents, below Modal's fixed per-container
    overheads). The shares add up to budget_usd and every driver hard-stops at its share, so the whole launch cannot
    pass budget_usd. Returns (projections, projected total)."""
    s_by_arm, s_max_by_arm = s_by_arm or {}, s_max_by_arm or {}
    projs = [project(c, prices, s_by_arm.get(c["arm"]), s_flown, s_max_by_arm.get(c["arm"])) for c in runs]
    if not runs:
        return projs, 0.0
    total = sum(p["total_usd"] for p in projs)
    floor = min(MIN_SHARE_USD, budget_usd / len(runs))
    low = [total <= 0 or budget_usd * p["total_usd"] / total < floor for p in projs]
    rest, big = budget_usd - floor * sum(low), sum(p["total_usd"] for p, lo in zip(projs, low) if not lo)
    for c, p, lo in zip(runs, projs, low):
        c["budget_usd"] = floor if lo else rest * p["total_usd"] / big
    return projs, total


def peak_concurrency(intervals: list[tuple[float, float]]) -> int:
    """Most intervals open at one instant (an end at t closes before a start at t opens)."""
    ev = sorted([(float(a), 1) for a, _ in intervals] + [(float(b), -1) for _, b in intervals],
                key=lambda e: (e[0], e[1]))
    peak = cur = 0
    for _, d in ev:
        cur += d
        peak = max(peak, cur)
    return peak


def timing_stats(values: list[float]) -> dict:
    import numpy as np

    v = np.asarray([float(a) for a in values if a is not None and math.isfinite(float(a))])
    if v.size == 0:
        return {"n": 0, "mean": None, "median": None, "p90": None, "max": None}
    return {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)),
            "p90": float(np.percentile(v, 90)), "max": float(v.max())}


def _require_complete(what: str, allow_missing: bool = False) -> bool:
    """False (with the reason printed) if the staged image lacks probe modules that do not import yet."""
    skipped = STAGED["skipped"]
    if skipped and not allow_missing:
        print(f"REFUSED {what}: the image would lack modules that do not import on this laptop yet:")
        for m, why in skipped.items():
            print(f"  {m}: {why}")
        return False
    return True


def _f(v, d: int = 2) -> str:
    return "-" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{d}f}"


# ---------------------------------------------------------------------------------------------------------------
# Local entrypoints
# ---------------------------------------------------------------------------------------------------------------


def _prices_dict(p: ct.Prices) -> dict:
    return {"cpu_core_h": p.cpu_core_h, "mem_gib_h": p.mem_gib_h, "source": p.source}


@app.local_entrypoint()
def smoke(n: int = 32, stub_s: float = 0.5, chunk: int = 16, rounds: int = 2, gen: bool = True) -> None:
    """Prove the image and evaluate_chunk.map work: the container self-check (brain data, smooth2 loads, pinned
    versions), `rounds` waves of n stub episodes (stub_s CPU seconds each, no sim), then one tiny stub CMA
    generation of V3-LAW through the trainer's Modal path. Prints timings, per-item overhead and the metered cost."""
    import numpy as np

    from kite.fly.behind import evaluate as ev

    for m, why in STAGED["skipped"].items():
        print(f"WARNING: {m} does not import yet ({why}); real episodes would fail")
    prices = ct.load_prices()
    chk = Evaluator().selfcheck.remote()
    print(f"container self-check: ok {chk['ok']}, missing {chk['missing']}, fly {chk['fly']} {chk['fly_error'] or ''}, "
          f"python {chk['python']}, packages {chk['packages']} (wrong {chk['wrong_versions']}), {chk['n_cpu']} cpus")
    be = modal_backend(chunk, prices)
    usd = 0.0
    for r in range(rounds):
        items = [ev.make_item("FLOWN", PROBE_SEED0 + i, "train", stub_s=stub_s, tag={"i": i}) for i in range(n)]
        t0 = time.perf_counter()
        res, st = be.evaluate(items)
        wall = time.perf_counter() - t0
        usd += st["usd"]
        ok = [x for x in res if not ev.is_error(x)]
        item_s = [x["item_wall_s"] for x in ok] or [math.nan]
        print(f"round {r + 1}: {len(ok)}/{len(res)} ok, map wall {wall:.2f} s, {st['n_chunks']} chunks on "
              f"{st['containers']} containers ({st['cold']} cold), slowest chunk {_f(st['chunk_wall_max_s'])} s, "
              f"queue up to {_f(st['queue_max_s'])} s; item wall mean {np.mean(item_s):.3f} s for {stub_s} s of work "
              f"(overhead {np.mean(item_s) - stub_s:+.3f} s per item); metered ${st['usd']:.4f}")
        for e in [x.get("error") for x in res if ev.is_error(x)][:3]:
            print("  error:", e)
    if gen:
        from kite.fly.behind import train as bt

        cfg = bt.make_run_cfg("V3-LAW", 1, "smoke", gens=1, pop=4, k=2, select_every=0, n_sel=4, stub_s=0.05,
                              budget_usd=0.5)
        with tempfile.TemporaryDirectory() as d:
            s = new_trainer(cfg, modal_backend(chunk, prices), Path(d) / run_name(cfg)).run()
        usd += float(s.get("usd") or 0.0)
        print(f"stub V3-LAW CMA generation via Modal: gen {s['gen']}, stopped {s['stopped']}, best {s['best']}, "
              f"errors {s['errors']}, metered ${float(s.get('usd') or 0.0):.4f}")
    usd += be.finish_usd()
    print(f"smoke metered total incl. one scaledown tail: ${usd:.4f} ({prices.source}; eval container "
          f"${ct.eval_usd_per_s(prices, EVAL_SPEC) * 3600:.3f}/h)")


def _concurrency(fn, m: int, hold_s: float, prices: ct.Prices) -> dict:
    """Ask for m chunks at once, each one stub item per worker burning hold_s: how many ran concurrently?"""
    from kite.fly.behind import evaluate as ev

    w = EVAL_SPEC["workers"]
    est = m * (hold_s + 20.0 + EVAL_SPEC["scaledown_s"]) * ct.eval_usd_per_s(prices, EVAL_SPEC)
    print(f"concurrency: asking for {m} chunks of {w} x {hold_s:g} s stub items at once (function cap "
          f"{EVAL_SPEC['max_containers']}); estimated ${est:.2f}")
    chunks = [[ev.make_item("FLOWN", PROBE_SEED0 + j, "train", stub_s=hold_s, tag={"c": c, "j": j}) for j in range(w)]
              for c in range(m)]
    t_submit = time.time()
    t0 = time.perf_counter()
    outs = list(fn.map(chunks, order_outputs=False, return_exceptions=True))
    wall = time.perf_counter() - t0
    ok = [o for o in outs if isinstance(o, dict)]
    intervals = [(o["t_start"], o["t_end"]) for o in ok]
    starts = sorted(o["t_start"] - t_submit for o in ok)
    ids = {o.get("container_id") for o in ok}
    container_s = len(ids) * wall
    usd = container_s * ct.eval_usd_per_s(prices, EVAL_SPEC) * ct.OVERHEAD
    res = {"asked": m, "ok": len(ok), "failed": len(outs) - len(ok), "peak": peak_concurrency(intervals),
           "containers": len(ids), "first_start_s": starts[0] if starts else None,
           "last_start_s": starts[-1] if starts else None, "wall_s": wall, "usd": usd}
    print(f"  peak {res['peak']} chunks ran at once on {res['containers']} distinct containers ({res['ok']} of {m} "
          f"chunks ok, {res['failed']} failed); starts {_f(res['first_start_s'], 1)} to {_f(res['last_start_s'], 1)} "
          f"s after submit; map wall {wall:.1f} s; metered ~${usd:.2f}")
    for o in [o for o in outs if not isinstance(o, dict)][:3]:
        print(f"  chunk error: {type(o).__name__}: {o}")
    return res


def _determinism(fn, m: int, episode_s: float, profile: str, seed0: int) -> dict:
    """The same FLOWN episode on m containers at once (each chunk fills a container's workers with copies of it, so
    the chunks spread out): one metric hash if the hosts compute bit-identical episodes."""
    from kite.fly.behind import evaluate as ev

    w = EVAL_SPEC["workers"]
    item = ev.make_item("FLOWN", seed0, profile, episode_s=episode_s)
    outs = list(fn.map([[dict(item, tag={"c": c, "j": j}) for j in range(w)] for c in range(m)],
                       order_outputs=False, return_exceptions=True))
    rows = []
    for o in outs:
        if not isinstance(o, dict):
            print(f"  determinism chunk failed: {type(o).__name__}: {o}")
            continue
        hs = {metrics_hash(r["metrics"]) for r in o["results"] if not ev.is_error(r)}
        rows.append((o.get("container_id"), json.dumps(o.get("cpu") or {}, sort_keys=True), hs))
    hashes = set().union(*(h for _, _, h in rows)) if rows else set()
    by_cpu: dict[str, set] = {}
    for _, cpu, hs in rows:
        by_cpu.setdefault(cpu, set()).update(hs)
    res = {"chunks": len(rows), "containers": len({c for c, _, _ in rows}), "hashes": sorted(hashes),
           "by_cpu": {k: sorted(v) for k, v in by_cpu.items()}}
    print(f"determinism: {res['chunks']} chunks on {res['containers']} containers, {len(hashes)} distinct metric "
          f"hash(es) of one {episode_s:g} s FLOWN episode")
    for cpu, hs in res["by_cpu"].items():
        print(f"  {cpu}: {hs}")
    if len(hashes) > 1:
        print(f"  DIFFERENT across hosts: rebuild the image with {NPY_DISABLE_ENV}='AVX512F AVX512_SKX' (or the "
              "families the differing hosts add) so every container runs the same kernels")
    return res


@app.local_entrypoint()
def probe(n: int = 32, episode_s: float = 60.0, profile: str = "train", seed0: int = PROBE_SEED0,
          arms: str = "FLOWN,V3-LAW", containers: int = 160, hold_s: float = 30.0, determinism: int = 8,
          allow_missing: bool = False) -> None:
    """Seconds per REAL episode on a loaded Modal worker: n episodes of each arm (FLOWN = the app with smooth2,
    V3-LAW = the follow without the brain, at its init) in one chunk each, so one eval container runs them on its 16
    workers at once. Then, with --determinism N > 0, one fixed episode on N containers (bit-identical across hosts?),
    and with --containers M > 0, how many eval containers the workspace really runs at once when asked for M chunks
    together. Prints the distributions and the launch projection from the measured numbers."""
    from kite.fly.behind import evaluate as ev

    if not _require_complete("probe", allow_missing):
        return
    prices = ct.load_prices()
    fn = evaluate_chunk()
    arm_list = [a.strip() for a in arms.split(",") if a.strip()]
    chunks = [[ev.make_item(a, seed0 + i, profile, episode_s=episode_s, tag={"i": i}) for i in range(n)]
              for a in arm_list]
    t0 = time.perf_counter()
    outs = list(fn.map(chunks, order_outputs=True, return_exceptions=True))
    wall = time.perf_counter() - t0
    timing: dict[str, dict] = {}
    for a, out in zip(arm_list, outs):
        if not isinstance(out, dict):
            print(f"{a}: chunk failed: {type(out).__name__}: {out}")
            continue
        ok = [r for r in out["results"] if not ev.is_error(r)]
        w, b = timing_stats([r["wall_s"] for r in ok]), timing_stats([r.get("brain_s") or 0.0 for r in ok])
        timing[a] = w
        print(f"{a} {profile} {episode_s:g} s episodes: {len(ok)}/{len(out['results'])} ok on container "
              f"{out.get('container_id')} ({'cold' if out.get('cold') else 'warm'}), chunk wall {out['wall_s']:.1f} s")
        print(f"  s per episode on a loaded worker: mean {_f(w['mean'])}, median {_f(w['median'])}, p90 "
              f"{_f(w['p90'])}, max {_f(w['max'])}; brain mean {_f(b['mean'])} s")
        for r in [r for r in out["results"] if ev.is_error(r)][:3]:
            print("  error:", r.get("error"))
    usd = len(outs) * wall * ct.eval_usd_per_s(prices, EVAL_SPEC) * ct.OVERHEAD
    print(f"timing map wall {wall:.1f} s, metered ~${usd:.3f}")
    if determinism > 0:
        _determinism(fn, int(determinism), min(episode_s, 20.0), profile, seed0)
    conc = _concurrency(fn, int(containers), float(hold_s), prices) if containers > 0 else None
    scale = 60.0 / episode_s
    fl = timing.get("FLOWN") or {}
    law = timing.get("V3-LAW") or {}
    if fl.get("mean"):
        s_fly, s_law = fl["mean"] * scale, (law.get("mean") or 0.0) * scale or None
        runs = plan()
        s_by = {"V3-FOLLOW": s_fly, "V3-JOINT": s_fly, "V3-LAW": s_law}
        s_max = {"V3-FOLLOW": fl["max"] * scale, "V3-JOINT": fl["max"] * scale}
        projs, total = split_budget(runs, prices, 40.0, s_by, s_fly, s_max)
        peak = sum(p["containers"] for p in projs)
        print(f"launch projection at {s_fly:.2f} s per fly / FLOWN episode (slowest {fl['max'] * scale:.2f} s), "
              f"{_f(s_law)} s per law episode: ${total:.2f} for {len(runs)} runs, wall "
              f"{max(p['wall_h'] for p in projs) * 60:.0f} min if parallel; {peak} eval containers wanted at once"
              + (f", the workspace ran {conc['peak']}" if conc else ""))
        print(f"  pass to launch: --s-per-episode {s_fly:.2f} --s-per-law-episode {_f(s_law)} --s-max "
              f"{fl['max'] * scale:.2f}")


@app.local_entrypoint()
def launch(confirm: bool = False, tag: str = "v3", budget_usd: float = 40.0, follow_seeds: str = "1,2,3,4",
           joint_seeds: str = "1,2,3,4", law_seeds: str = "1,2", gens: int = 0, episode_s: float = 0.0,
           s_per_episode: float = 0.0, s_per_law_episode: float = 0.0, s_max: float = 0.0, profile: str = "",
           allow_code_change: bool = False, allow_missing: bool = False) -> None:
    """Print the planned runs (V3-FOLLOW and V3-JOINT on CMA seeds 1-4, V3-LAW on 1-2), their budget shares and
    projected cost and wall time; with --confirm, spawn them (train_all). Refuses when the projection exceeds
    --budget-usd or the image would miss modules. Use `modal run --detach` so the runs outlive this laptop.
    --s-per-episode / --s-per-law-episode / --s-max: probe's measured seconds per 60 s episode (fly arms and FLOWN /
    the law arm / the slowest fly episode); default train's conservative guesses. A run whose state on the Volume
    was made by other code refuses to resume: launch a new --tag, or --allow-code-change to go on anyway."""
    prices = ct.load_prices()
    runs = plan(tag, follow_seeds, joint_seeds, law_seeds, gens, episode_s, profile=profile,
                allow_code_change=allow_code_change)
    s_fly = s_per_episode or None
    s_by = {"V3-FOLLOW": s_fly, "V3-JOINT": s_fly, "V3-LAW": s_per_law_episode or None}
    s_mx = {"V3-FOLLOW": s_max or None, "V3-JOINT": s_max or None}
    projs, total = split_budget(runs, prices, budget_usd, s_by, s_fly, s_mx)
    from kite.fly.behind import train as bt

    for cfg, p in zip(runs, projs):
        print(f"{run_name(cfg):22s} budget ${cfg['budget_usd']:6.2f}  " + bt.format_projection(p))
    peak = sum(p["containers"] for p in projs)
    wall = max(p["wall_h"] for p in projs) * 60
    stretch = max(1.0, peak / EVAL_SPEC["max_containers"])
    print(f"projected total ${total:.2f} of ${budget_usd:.2f}; wall {wall:.0f} min if the runs got their containers "
          f"in parallel, ~{wall * stretch:.0f} min on the cap ({peak} eval containers wanted at once, function cap "
          f"{EVAL_SPEC['max_containers']}: the runs queue, the cost is the same) + {len(runs) + 1} small drivers")
    if total > budget_usd:
        print("REFUSED: the projection exceeds the budget; shrink the plan (--*-seeds, --gens) or raise the budget")
        return
    if not _require_complete("launch", allow_missing):
        return
    if not confirm:
        print("not launching: add --confirm (with `modal run --detach`) to spawn the runs")
        return
    launch_id = time.strftime("launch-%Y%m%d-%H%M%S")
    call = train_all.spawn(launch_id, runs, _prices_dict(prices))
    print(f"launched {launch_id}: train_all call {call.object_id}; watch with ::status")
    for r in call.get():
        print(json.dumps({k: r.get(k) for k in ("status", "summary", "last_error")}, default=str)[:600])


def _read(path: str) -> str | None:
    try:
        return b"".join(volume.read_file(path)).decode()
    except Exception:  # noqa: BLE001
        return None


def _run_names(tag: str = "") -> list[str]:
    try:
        entries = sorted(volume.listdir("/"), key=lambda e: e.path)
    except Exception as e:  # noqa: BLE001
        print(f"no runs yet ({e})")
        return []
    names = [e.path.strip("/").split("/")[-1] for e in entries]
    return [n for n in names if n and not n.startswith("_") and (not tag or n.endswith("_" + tag))]


def status_row(name: str, log_text: str | None, best_text: str | None, lease_text: str | None,
               now: float | None = None) -> list[str]:
    """One status line from a run's log.jsonl, best.json and lease.json texts (pure; tests feed it strings)."""
    recs = []
    for line in (log_text or "").splitlines():
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    g = [r for r in recs if "fit_best" in r]
    last = g[-1] if g else {}
    ev = next((r["event"] for r in reversed(recs) if r.get("event")), "")
    best = {}
    try:
        best = (json.loads(best_text) if best_text else {}).get("trained") or {}
    except json.JSONDecodeError:
        pass
    driving = "-"
    try:
        hb = float(json.loads(lease_text).get("heartbeat", 0)) if lease_text else None
        if hb is not None:
            age = (now or time.time()) - hb
            driving = f"{age / 60:.0f}m ago" if age < LEASE_TTL_S else "stale"
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    sel = last.get("best_sel_fitness", best.get("fitness"))
    sel_gen = last.get("best_gen", best.get("gen", "-"))
    return [name, str(last.get("gen", 0)), _f(last.get("fit_best"), 3), _f(last.get("fit_median"), 3),
            _f(last.get("sigma"), 4), f"{_f(sel, 3)}@{sel_gen}", str(sum(int(r.get("n_errors", 0)) for r in g)),
            str(last.get("collisions", "-")), _f(last.get("usd_total")), _f(last.get("usd_projected")),
            _f(last.get("s_per_episode")), driving, (last.get("time") or "-")[11:19] + (f" {ev}" if ev else "")]


STATUS_HEAD = ["run", "gen", "fit_best", "fit_med", "sigma", "best_sel@gen", "errors", "coll", "usd", "proj_usd",
               "s/ep", "lease", "last"]


@app.local_entrypoint()
def status(tag: str = "") -> None:
    """Every run on the Volume: last generation, fitness (lower is better, FLOWN = 1), best selection, errors,
    metered spend and its projection, the driver's lease heartbeat, the stop reason."""
    rows = [status_row(n, _read(f"/{n}/log.jsonl"), _read(f"/{n}/best.json"), _read(f"/{n}/lease.json"))
            for n in _run_names(tag)]
    w = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(STATUS_HEAD)] if rows else \
        [len(h) for h in STATUS_HEAD]
    for r in [STATUS_HEAD] + rows:
        print("  ".join(c.ljust(n) for c, n in zip(r, w)))
    if rows:
        print(f"total metered ${sum(float(r[8]) for r in rows if r[8] != '-'):.2f}")


FETCH_FILES = ("best.json", "log.jsonl", "selection.jsonl", "config.json", "errors.jsonl")


@app.local_entrypoint()
def fetch(run: str = "", tag: str = "", state: bool = False) -> None:
    """Copy best.json, the logs and config (and state.pkl with --state) of one run (--run) or of every run of a tag
    (--tag; neither: every run) from the Volume to results/behind/runs/<run>/."""
    names = [run] if run else _run_names(tag)
    for name in names:
        dest = REPO / "results" / "behind" / "runs" / name
        dest.mkdir(parents=True, exist_ok=True)
        print(f"{name}:")
        for fn in FETCH_FILES + (("state.pkl",) if state else ()):
            try:
                data = b"".join(volume.read_file(f"/{name}/{fn}"))
            except Exception:  # noqa: BLE001
                print(f"  {fn}: not on the Volume")
                continue
            (dest / fn).write_bytes(data)
            print(f"  {fn}: {len(data)} bytes")
        print(f"  -> {dest}")
