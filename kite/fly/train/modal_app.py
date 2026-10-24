"""Modal app `kite-fly`: the CMA-ES fine-tune (DESIGN section 5) and evaluations in Modal containers.

    .venv/bin/modal run -m kite.fly.train.modal_app::smoke                        # stub episodes: deploy + map check
    .venv/bin/modal run -m kite.fly.train.modal_app::launch                       # the plan and projection only
    .venv/bin/modal run --detach -m kite.fly.train.modal_app::launch --confirm    # FLY-YAW x 3, KITE-LAW-CMA, PRED-PD
    .venv/bin/modal run -m kite.fly.train.modal_app::status
    .venv/bin/modal run -m kite.fly.train.modal_app::fetch --run FLY-YAW_s1_kite1  # best.json, logs -> results/fly/runs
    python -m kite.fly.train.cma_train --arm FLY-YAW --backend modal ...          # or drive one run from this laptop
    python -m kite.fly.train.report ... --backend modal                           # evaluation fan-out

Pieces:
  image          debian_slim (Python 3.12) + numpy / scipy / cma pinned to the local venv, plus only the kite modules
                 the episode path imports (found by an import probe that refuses cv2 / djitellopy / av / ultralytics)
                 and kite/fly/data (brain .npz, reference and init parameters)
  evaluate_chunk an eval container (EVAL_SPEC: 8 cores, 16 worker processes) runs a batch of episode items
  train          one CMA-ES driver (cma_train.Trainer) in a small container; state, logs and best.json live on the
                 Volume `kite-fly-runs` (committed every generation), so a preempted or retried call resumes exactly;
                 a lease file keeps a second driver off the same run
  train_all      the launch: spawns every planned run, records the call ids on the Volume, re-spawns a run that dies
Budget: launch splits --budget-usd over the runs in proportion to their projected cost; every driver meters its own
container time (cma_train.ModalBackend) and stops before a generation would pass its share.
"""

from __future__ import annotations

import atexit
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import modal

from kite.fly.train import cma_train as ct

APP_NAME = "kite-fly"
VOLUME_NAME = "kite-fly-runs"
RUNS_MOUNT = "/runs"
REMOTE_ROOT = "/root"
LEASE_TTL_S = 15 * 60
MIN_SHARE_USD = 1.0  # budget floor of one run in a launch
PACKAGES = ("numpy==2.5.4", "scipy==1.18.1", "cma==4.5.0")  # = the local venv, so both compute the same numbers
BLOCKED = ("cv2", "djitellopy", "av", "ultralytics", "PIL", "torch")  # must never be needed in a container
PROBE_MODULES = ("kite.fly.train.cma_train", "kite.fly.train.evaluate", "kite.fly.train.arms", "kite.fly.train.report",
                 "kite.fly.sim", "kite.fly.steer", "kite.fly.params", "kite.fly.brain", "kite.fly.encoder",
                 "kite.fly.decoder", "kite.fly.calibrate")
DATA_GLOBS = ("kite/fly/data/*.npz", "kite/fly/data/*.json", "kite/fly/data/init/*.json",
              "kite/fly/data/reference/*.json")
THREADS = {v: "1" for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}

_PROBE = r"""
import importlib, importlib.abc, json, sys
BLOCKED, MODS = set(sys.argv[1].split(',')), sys.argv[2].split(',')
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in BLOCKED:
            raise ImportError(f'{name} is not available in the Modal image')
sys.meta_path.insert(0, Block())
skipped = []
for m in MODS:
    try:
        importlib.import_module(m)
    except ModuleNotFoundError as e:
        if e.name is None or not m.startswith(e.name):
            raise
        skipped.append(m)  # not written yet
files = sorted({getattr(mod, '__file__', None) or '' for n, mod in list(sys.modules.items())
                if n == 'kite' or n.startswith('kite.')} - {''})
print(json.dumps({'files': files, 'skipped': skipped}))
"""


def source_files(repo: Path = ct.REPO) -> tuple[list[Path], list[str]]:
    """The kite files the container needs: every module the episode path imports (in a fresh interpreter that refuses
    BLOCKED packages, so a stray cv2 import fails here, not in 40 containers), all of kite/fly, and its data."""
    out = subprocess.run([sys.executable, "-c", _PROBE, ",".join(BLOCKED), ",".join(PROBE_MODULES)], cwd=repo,
                         capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(repo)})
    if out.returncode != 0:
        raise RuntimeError(f"import probe failed (a container would fail the same way):\n{out.stderr[-3000:]}")
    probe = json.loads(out.stdout.strip().splitlines()[-1])
    files = {Path(f).resolve() for f in probe["files"]}
    files |= {p.resolve() for p in (repo / "kite" / "fly").rglob("*.py") if "__pycache__" not in p.parts}
    for g in DATA_GLOBS:
        files |= {p.resolve() for p in repo.glob(g)}
    files = sorted(f for f in files if f.is_relative_to(repo.resolve()))
    return files, probe["skipped"]


def stage_sources(dest: Path) -> list[str]:
    """Copy source_files() into dest (same relative layout): the upload then sees a snapshot, not files that are
    being edited while the image builds."""
    files, _ = source_files()
    root = ct.REPO.resolve()
    rel = []
    for f in files:
        r = f.relative_to(root)
        (dest / r).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest / r)
        rel.append(r.as_posix())
    return rel


def _image() -> modal.Image:
    img = (modal.Image.debian_slim(python_version="3.12").uv_pip_install(*PACKAGES)
           .env({"PYTHONPATH": REMOTE_ROOT, "KITE_FLY_RUNS": RUNS_MOUNT, **THREADS}))
    if modal.is_local():
        stage = Path(tempfile.mkdtemp(prefix="kite-fly-src-"))
        atexit.register(shutil.rmtree, stage, ignore_errors=True)
        stage_sources(stage)
        img = img.add_local_dir(stage / "kite", f"{REMOTE_ROOT}/kite")
    return img


image = _image()
app = modal.App(APP_NAME, include_source=False)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
E, D = ct.EVAL_SPEC, ct.DRIVER_SPEC


def _retries(n: int) -> modal.Retries:
    return modal.Retries(max_retries=int(n), backoff_coefficient=2.0, initial_delay=2.0, max_delay=60.0)


@app.cls(image=image, cpu=E["cpu"], memory=E["memory_mib"], max_containers=E["max_containers"],
         scaledown_window=E["scaledown_s"], timeout=E["timeout_s"], retries=_retries(E["retries"]))
class Evaluator:
    """An eval container. A class (not a function) so its process pool is shut down on exit: a live pool thread
    otherwise holds every container 30 s past its scaledown."""

    @modal.enter()
    def start(self) -> None:
        self.t0, self.served = time.time(), 0

    @modal.method()
    def evaluate_chunk(self, items: list[dict]) -> dict:
        """A batch of episode items on this container's process pool (one worker per hyperthread)."""
        from kite.fly.train import evaluate as ev

        cold, age = self.served == 0, time.time() - self.t0
        out = ev.evaluate_chunk_local(items, workers=E["workers"])
        self.served += 1
        out.update(cold=cold, container_age_s=age, container_id=os.environ.get("MODAL_TASK_ID"), n_cpu=os.cpu_count(),
                   n_affinity=len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None)
        return out

    @modal.exit()
    def stop(self) -> None:
        from kite.fly.train import evaluate as ev

        ev.shutdown_pool(wait=True)


def evaluate_chunk():
    """What the backends fan out over: evaluate_chunk().map(chunks) (Evaluator.evaluate_chunk)."""
    return Evaluator().evaluate_chunk


def _lease(run_dir: Path, call_id: str | None, check: bool) -> None:
    """Refuse to drive a run another live call is driving (a double launch); else take or refresh the lease."""
    path = run_dir / "lease.json"
    if check and path.exists():
        try:
            d = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            d = {}
        if d.get("call_id") not in (None, call_id) and time.time() - float(d.get("heartbeat", 0)) < LEASE_TTL_S:
            raise RuntimeError(f"{run_dir.name} is driven by call {d['call_id']}")
    run_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"call_id": call_id, "heartbeat": time.time()}))


@app.function(image=image, volumes={RUNS_MOUNT: volume}, cpu=D["cpu"], memory=D["memory_mib"], timeout=D["timeout_s"],
              retries=_retries(D["retries"]))
def train(cfg: dict, prices: dict | None = None, chunk: int | None = None) -> dict:
    """One CMA-ES run. Idempotent: resumes from the Volume after a preemption or retry."""
    volume.reload()
    p = ct.Prices(**prices) if prices else ct.Prices()
    run_dir = Path(RUNS_MOUNT) / ct.run_name(cfg["arm"], cfg["cma_seed"], cfg["tag"])
    call_id = modal.current_function_call_id()
    _lease(run_dir, call_id, check=True)
    volume.commit()

    def checkpoint() -> None:
        _lease(run_dir, call_id, check=False)
        volume.commit()

    backend = ct.ModalBackend(evaluate_chunk(), chunk or ct.default_chunk(cfg["arm"]), p)
    trainer = ct.Trainer(cfg, backend, run_dir, on_checkpoint=checkpoint,
                         driver_usd_per_s=p.usd_per_s(D["cpu"], D["memory_mib"]))
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
        name = ct.run_name(cfg["arm"], cfg["cma_seed"], cfg["tag"])
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
                else:
                    e["restarts"] += 1
                    calls[name] = train.spawn(e["cfg"], prices)
                    e["call_id"] = calls[name].object_id
            save()
    return list(rec["runs"].values())


# ---------------------------------------------------------------------------------------------------------------
# Local entrypoints
# ---------------------------------------------------------------------------------------------------------------


def _prices_dict(p: ct.Prices) -> dict:
    return {"cpu_core_h": p.cpu_core_h, "mem_gib_h": p.mem_gib_h, "source": p.source}


@app.local_entrypoint()
def smoke(n: int = 16, stub_s: float = 0.5, chunk: int = 8, rounds: int = 2, gen: bool = True) -> None:
    """Prove the image and evaluate_chunk.map work: `rounds` waves of n stub episodes (stub_s CPU seconds each, no
    sim, no brain), then one tiny STUB CMA generation through the trainer's Modal path. Prints per-chunk timings,
    the per-item overhead and the metered cost."""
    import numpy as np

    from kite.fly.train import evaluate as ev

    prices = ct.load_prices()
    be = ct.ModalBackend(evaluate_chunk(), chunk, prices)
    usd = 0.0
    for r in range(rounds):
        items = [ev.make_item("STUB", 10_000 + i, "train", stub_s=stub_s, tag={"i": i}) for i in range(n)]
        t0 = time.perf_counter()
        res, st = be.evaluate(items)
        wall = time.perf_counter() - t0
        usd += st["usd"]
        ok = [x for x in res if x.get("ok")]
        item_s = [x["item_wall_s"] for x in ok]
        print(f"round {r + 1}: {len(ok)}/{len(res)} ok, map wall {wall:.2f} s, {st['n_chunks']} chunks on "
              f"{st['containers']} containers ({st['cold']} cold), slowest chunk {st['chunk_wall_max_s']:.2f} s, "
              f"queue up to {st['queue_max_s']:.2f} s; item wall mean {np.mean(item_s):.3f} s for {stub_s} s of "
              f"work (overhead {np.mean(item_s) - stub_s:+.3f} s per item); metered ${st['usd']:.4f}")
        if r == rounds - 1:
            per_chunk = (wall - st["chunk_wall_max_s"]) if st["chunk_wall_max_s"] else None
            print(f"  map overhead beyond the slowest chunk: {per_chunk:.2f} s per call")
        errs = [x["error"] for x in res if not x.get("ok")]
        if errs:
            print("  errors:", errs[:3])
    if gen:
        import tempfile

        cfg = ct.make_cfg("STUB", 1, "smoke", gens=1, pop=4, k=2, select_every=0, stub_s=0.05, n_sel=4, budget_usd=0.5)
        with tempfile.TemporaryDirectory() as d:
            s = ct.Trainer(cfg, ct.ModalBackend(evaluate_chunk(), chunk, prices), Path(d) / "STUB_s1_smoke").run()
        usd += s["usd"]
        print(f"STUB CMA generation via Modal: gen {s['gen']}, stopped {s['stopped']}, best {s['best']}, "
              f"errors {s['errors']}, metered ${s['usd']:.4f}")
    usd += be.finish_usd()
    print(f"smoke metered total incl. one scaledown tail: ${usd:.4f} ({prices.source}; eval container "
          f"${ct.eval_usd_per_s(prices) * 3600:.3f}/h)")


def plan(tag: str = "kite1", fly_seeds: str = "1,2,3", init: str | None = None, size_blind: str | None = None,
         classical: bool = True, gens: int = 0, episode_s: float = 0.0) -> list[dict]:
    """DESIGN 5 / R7 / R9's runs: FLY-YAW on each CMA seed (size-blind; init prev, hand on seed 3, unless given),
    then KITE-LAW-CMA and PRED-PD on seed 1."""
    over = {"gens": gens or None, "episode_s": episode_s or None}
    runs = [ct.make_cfg("FLY-YAW", int(s), tag, init=init or None, size_blind=size_blind or None, **over)
            for s in fly_seeds.split(",") if s.strip()]
    if classical:
        runs += [ct.make_cfg(arm, 1, tag, **over) for arm in ("KITE-LAW-CMA", "PRED-PD")]
    return runs


def split_budget(runs: list[dict], prices: ct.Prices, budget_usd: float, s_per_episode: float = 0.0,
                 s_per_law_episode: float = 0.0, s_max: float = 0.0) -> tuple[list[dict], float]:
    """Each run's budget_usd: a share of the total in proportion to its projected cost, but at least
    min(MIN_SHARE_USD, total / runs) (the cheap classical runs would otherwise get cents, below Modal's fixed
    per-container overheads). The shares add up to budget_usd and every driver hard-stops at its share, so the whole
    launch cannot pass budget_usd. Returns (projections, projected total)."""
    projs = []
    for cfg in runs:
        s_ep = s_per_episode if cfg["arm"] == "FLY-YAW" else s_per_law_episode
        projs.append(ct.project_cfg(cfg, prices, s_ep or None, s_per_law_episode or None,
                                    s_max=(s_max or None) if cfg["arm"] == "FLY-YAW" else None))
    total = sum(p["total_usd"] for p in projs)
    floor = min(MIN_SHARE_USD, budget_usd / len(runs))
    low = [total <= 0 or budget_usd * p["total_usd"] / total < floor for p in projs]
    rest, big = budget_usd - floor * sum(low), sum(p["total_usd"] for p, lo in zip(projs, low) if not lo)
    for cfg, p, lo in zip(runs, projs, low):
        cfg["budget_usd"] = floor if lo else rest * p["total_usd"] / big
    return projs, total


@app.local_entrypoint()
def probe(arm: str = "FLY-YAW-PREV", n: int = 32, episode_s: float = 60.0, profile: str = "train",
          seed0: int = 3000) -> None:
    """Seconds per REAL episode on a loaded Modal worker: n episodes of an arm (reference arms only) in one chunk, so
    one eval container runs them on its 16 workers at once. Prints the distribution and the launch projection."""
    import numpy as np

    from kite.fly.train import evaluate as ev

    prices = ct.load_prices()
    be = ct.ModalBackend(evaluate_chunk(), n, prices)
    items = [ev.make_item(arm, seed0 + i, profile, size_blind=True, episode_s=episode_s, tag={"i": i})
             for i in range(n)]
    res, st = be.evaluate(items)
    ok = [r for r in res if r.get("ok")]
    w = np.array([r["wall_s"] for r in ok]) if ok else np.zeros(1)
    b = np.array([r.get("brain_s") or 0.0 for r in ok]) if ok else np.zeros(1)
    usd = st["usd"] + be.finish_usd()
    print(f"{arm} {profile} {episode_s:g} s episodes: {len(ok)}/{len(res)} ok on {st['containers']} container(s), "
          f"chunk wall {st['chunk_wall_max_s']:.1f} s, map wall {st['wall_s']:.1f} s")
    print(f"  s per episode on a loaded worker: mean {w.mean():.2f}, median {np.median(w):.2f}, p90 "
          f"{np.percentile(w, 90):.2f}, max {w.max():.2f}; brain {b.mean():.2f} s; metered ${usd:.4f}")
    for e in [r["error"] for r in res if not r.get("ok")][:3]:
        print("  error:", e)
    runs = plan()
    scale = 60.0 / episode_s
    projs, total = split_budget(runs, prices, 25.0, float(np.mean(w)) * scale, s_max=float(w.max()) * scale)
    print(f"launch projection at {np.mean(w) * scale:.2f} s (slowest {w.max() * scale:.2f} s) per 60 s FLY-YAW "
          f"episode: ${total:.2f}, wall {max(p['wall_h'] for p in projs) * 60:.0f} min if parallel")


@app.local_entrypoint()
def launch(confirm: bool = False, tag: str = "kite1", budget_usd: float = 25.0, fly_seeds: str = "1,2,3",
           init: str = "", size_blind: str = "", classical: bool = True, gens: int = 0,
           episode_s: float = 0.0, s_per_episode: float = 0.0, s_per_law_episode: float = 0.0,
           s_max: float = 0.0) -> None:
    """Print the planned runs and their projected cost and wall time; with --confirm, spawn them (train_all).
    Refuses to launch when the projection exceeds --budget-usd. Use `modal run --detach` so the runs outlive this
    laptop. --s-per-episode: measured seconds per FLY-YAW episode on a loaded Modal worker (a run's log.jsonl
    `s_per_episode`); default a conservative guess."""
    prices = ct.load_prices()
    runs = plan(tag, fly_seeds, init, size_blind, classical, gens, episode_s)
    projs, total = split_budget(runs, prices, budget_usd, s_per_episode, s_per_law_episode, s_max)
    for cfg, p in zip(runs, projs):
        print(f"{ct.run_name(cfg['arm'], cfg['cma_seed'], tag):24s} budget ${cfg['budget_usd']:.2f}  "
              + ct.format_projection(p))
    peak = sum(p["containers"] for p in projs)
    print(f"projected total ${total:.2f} of ${budget_usd:.2f}; wall {max(p['wall_h'] for p in projs) * 60:.0f} min "
          f"if the runs get their containers in parallel (peak {peak} eval containers, function cap "
          f"{ct.EVAL_SPEC['max_containers']}: above it they queue) + {len(runs) + 1} small driver containers")
    if total > budget_usd:
        print("REFUSED: the projection exceeds the budget; shrink the plan (--fly-seeds, --gens) or raise the budget")
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


@app.local_entrypoint()
def status(tag: str = "") -> None:
    """Every run on the Volume: last generation, fitness, best selection, errors, metered cost, stop reason."""
    try:
        entries = sorted(volume.listdir("/"), key=lambda e: e.path)
    except Exception as e:  # noqa: BLE001
        print(f"no runs yet ({e})")
        return
    head = ["run", "gen", "fit_best", "fit_med", "sigma", "best_sel@gen", "errors", "usd", "proj_usd", "s/ep", "last"]
    rows = []
    for e in entries:
        name = e.path.strip("/").split("/")[-1]
        if name.startswith("_") or (tag and not name.endswith("_" + tag)):
            continue
        recs = [json.loads(x) for x in (_read(f"/{name}/log.jsonl") or "").splitlines() if x.strip()]
        g = [r for r in recs if "fit_best" in r]
        last = g[-1] if g else {}
        ev = next((r["event"] for r in reversed(recs) if r.get("event")), "")
        f = lambda v, d=3: "-" if v is None else f"{v:.{d}f}"  # noqa: E731
        rows.append([name, str(last.get("gen", 0)), f(last.get("fit_best")), f(last.get("fit_median")),
                     f(last.get("sigma"), 4), f"{f(last.get('best_sel_fitness'))}@{last.get('best_gen', '-')}",
                     str(sum(r.get("n_errors", 0) for r in g)), f(last.get("usd_total"), 2),
                     f(last.get("usd_projected"), 2), f(last.get("s_per_episode"), 2),
                     (last.get("time") or "-")[11:19] + (f" {ev}" if ev else "")])
    w = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(head)] if rows else [len(h) for h in head]
    for r in [head] + rows:
        print("  ".join(c.ljust(n) for c, n in zip(r, w)))


@app.local_entrypoint()
def fetch(run: str, state: bool = False) -> None:
    """Copy one run's best.json, logs and config (and state.pkl with --state) from the Volume to results/fly/runs."""
    dest = ct.runs_root() / run
    dest.mkdir(parents=True, exist_ok=True)
    names = ["best.json", "log.jsonl", "selection.jsonl", "config.json", "errors.jsonl"]
    names += ["state.pkl"] if state else []
    for n in names:
        try:
            data = b"".join(volume.read_file(f"/{run}/{n}"))
        except Exception:  # noqa: BLE001
            print(f"  {n}: not on the Volume")
            continue
        (dest / n).write_bytes(data)
        print(f"  {n}: {len(data)} bytes")
    print(f"fetched into {dest}")
