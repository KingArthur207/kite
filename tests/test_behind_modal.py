"""kite.fly.behind's Modal app and sign-off without Modal or the simulator: the image ships the episode path and the
fly brain's data (and the import probe refuses cv2), the pins match this venv, the plan / projection / budget split,
the concurrency count, the status line; and the sign-off's statistics and DESIGN section 9 gates on synthetic per-seed
metric tables (a clearly better model passes; more collisions, or more excess_speed on slow, fail), its model
reconstruction against load_model, its episode cache, and an end-to-end run on stub episodes."""

import json
import math
from importlib import metadata
from types import SimpleNamespace

import numpy as np
import pytest

from kite.fly.behind import signoff as so

# ---------------------------------------------------------------------------------------------------------------
# Modal app (skipped without the modal package)
# ---------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ma():
    pytest.importorskip("modal")
    from kite.fly.behind import modal_app

    return modal_app


def _train_module():
    from kite.fly.behind import train

    return train


def test_image_ships_the_episode_path_and_the_brain_data(ma):
    files, skipped = ma.source_files()
    root = ma.REPO.resolve()
    rel = {f.relative_to(root).as_posix() for f in files}
    must = {"kite/__init__.py", "kite/types.py", "kite/follow_behind.py", "kite/follow_tuned.py",
            "kite/drone/safety.py", "kite/fly/sim/drone.py", "kite/fly/interfaces.py",
            "kite/fly/behind/modal_app.py", "kite/fly/behind/signoff.py", "kite/fly/behind/evaluate.py",
            "kite/fly/behind/params.py", "kite/fly/train/cma_train.py",
            "kite/flybrain/flyfollow/steer.py", "kite/flybrain/flydrones/brain/lif.py",
            "kite/flybrain/flyfollow/rl/params.py",
            "kite/flybrain/data/brains/pursuit_core1.npz",
            "kite/flybrain/data/brains/trained/FLY-YAW_smooth2_best.json",
            "kite/flybrain/data/brains/init/FLY-CMA__pursuit_core1.json",
            "kite/flybrain/configs/controllers.yaml"}
    assert must <= rel, sorted(must - rel)
    assert not any(" 2" in r or "__pycache__" in r or r.startswith("tests/") for r in rel)
    assert "kite/record.py" not in rel and not any(r.startswith(("earpiece/", "run/", "phone/")) for r in rel)
    assert set(skipped) <= set(ma.PROBE_MODULES)


def test_import_probe_refuses_cv2_and_skips_unwritten_kite_modules(ma, tmp_path):
    (tmp_path / "behind_probe_cv2_case.py").write_text("import numpy\nimport cv2\n")
    with pytest.raises(RuntimeError, match="cv2"):
        ma.run_probe(["behind_probe_cv2_case"], extra_paths=[tmp_path])
    (tmp_path / "behind_probe_torch_case.py").write_text("from torch import nn\n")
    with pytest.raises(RuntimeError, match="torch"):
        ma.run_probe(["behind_probe_torch_case"], extra_paths=[tmp_path])
    optional = "try:\n    import cv2\nexcept ImportError:\n    cv2 = None\n"
    (tmp_path / "behind_probe_optional_case.py").write_text(optional)
    assert ma.run_probe(["behind_probe_optional_case"], extra_paths=[tmp_path])["skipped"] == {}
    out = ma.run_probe(["kite.fly.behind.no_such_module_xyz", "kite.types"])
    assert "kite.fly.behind.no_such_module_xyz" in out["skipped"]
    assert any(f.endswith("kite/types.py") for f in out["files"])
    with pytest.raises(RuntimeError, match="no_such_package_xyz"):
        ma.run_probe(["no_such_package_xyz"])


def test_pins_match_this_venv(ma):
    for pin in ma.PACKAGES:
        name, version = pin.split("==")
        assert metadata.version(name) == version, pin
    assert any(p.startswith("pyyaml==") for p in ma.PACKAGES)


def test_container_selfcheck(ma, tmp_path):
    empty = ma.container_selfcheck(tmp_path)
    assert not empty["ok"] and empty["missing"] == list(ma.REQUIRED) and empty["fly"] is None
    chk = ma.container_selfcheck(ma.REPO)
    assert not chk["missing"], chk["missing"]
    assert chk["ok"] and not chk["wrong_versions"], chk
    assert chk["fly"]["arm"] == "FLY-YAW" and chk["fly"]["params"].endswith("FLY-YAW_smooth2_best.json")


def test_eval_spec_and_peak_concurrency(ma):
    assert (ma.EVAL_SPEC["cpu"], ma.EVAL_SPEC["workers"], ma.EVAL_SPEC["memory_mib"]) == (8.0, 16, 8192)
    assert ma.EVAL_SPEC["max_containers"] == int(__import__("os").environ.get("KITE_BEHIND_MAX_CONTAINERS") or 160)
    assert ma.peak_concurrency([]) == 0
    assert ma.peak_concurrency([(0, 1), (1, 2), (2, 3)]) == 1  # touching, never overlapping
    assert ma.peak_concurrency([(0, 10), (1, 5), (2, 3), (5, 6), (9, 12)]) == 3
    assert ma.peak_concurrency([(0, 5)] * 7) == 7
    t = ma.timing_stats([1.0, 2.0, 3.0, 4.0, float("nan")])
    assert t["n"] == 4 and t["mean"] == pytest.approx(2.5) and t["max"] == 4.0


def test_plan_projection_and_budget_split(ma):
    bt = _train_module()
    from kite.fly.train import cma_train as ct

    runs = ma.plan()
    assert [(r["arm"], r["cma_seed"]) for r in runs] == ([("V3-FOLLOW", s) for s in (1, 2, 3, 4)]
                                                         + [("V3-JOINT", s) for s in (1, 2, 3, 4)]
                                                         + [("V3-LAW", 1), ("V3-LAW", 2)])
    shape = {r["arm"]: (r["gens"], r["pop"], r["k"]) for r in runs}
    assert shape == {"V3-FOLLOW": (60, 24, 32), "V3-JOINT": (100, 40, 32), "V3-LAW": (60, 20, 32)}
    assert {ma.run_name(r) for r in runs} >= {"V3-JOINT_s1_v3", "V3-LAW_s2_v3"}
    assert [r["cma_seed"] for r in ma.plan(joint_seeds="2", follow_seeds="", law_seeds="")] == [2]
    joint = next(r for r in runs if r["arm"] == "V3-JOINT")
    p = ma.project(joint, ct.Prices(), 2.0, 2.0, 4.4)
    assert p["chunks_per_gen"] == math.ceil((40 * 32 + 32) / 32) == 41
    assert p["containers"] == 41  # one wave under this app's cap (kite.fly's 40 would need two)
    assert p == bt.project_run(joint, ct.Prices(), 2.0, 2.0, 32, 4.4)  # the trainer projects what the app runs
    assert p["wall_h"] < bt.project_run(joint, ct.Prices(), 2.0, 2.0, 32, 4.4, spec=ct.EVAL_SPEC)["wall_h"]
    projs, total = ma.split_budget(runs, ct.Prices(), 40.0)
    assert sum(r["budget_usd"] for r in runs) == pytest.approx(40.0)
    assert all(r["budget_usd"] >= min(ma.MIN_SHARE_USD, 40.0 / len(runs)) - 1e-9 for r in runs)
    assert total == pytest.approx(sum(q["total_usd"] for q in projs))


def test_status_row(ma):
    log = "\n".join(json.dumps(r) for r in [
        {"gen": 1, "fit_best": 0.95, "fit_median": 1.01, "sigma": 0.15, "best_gen": 0, "best_sel_fitness": 1.0,
         "usd_total": 0.4, "usd_projected": 40.0, "s_per_episode": 2.1, "n_errors": 1, "collisions": 3,
         "time": "2026-10-24T22:00:00+00:00"},
        {"gen": 2, "fit_best": 0.9, "fit_median": 0.97, "sigma": 0.14, "best_gen": 0, "best_sel_fitness": 1.0,
         "usd_total": 0.8, "usd_projected": 40.0, "s_per_episode": 2.0, "n_errors": 0, "collisions": 1,
         "time": "2026-10-24T22:01:00+00:00"},
        {"gen": 2, "event": "budget stop"}]) + "\nnot json\n"
    row = ma.status_row("V3-JOINT_s1_v3", log, None, json.dumps({"heartbeat": 1000.0}), now=1060.0)
    d = dict(zip(ma.STATUS_HEAD, row))
    assert d["gen"] == "2" and d["fit_best"] == "0.900" and d["best_sel@gen"] == "1.000@0"
    assert d["errors"] == "1" and d["usd"] == "0.80" and d["lease"] == "1m ago" and d["last"].endswith("budget stop")
    empty = dict(zip(ma.STATUS_HEAD, ma.status_row("x", None, json.dumps({"trained": {"fitness": 0.8, "gen": 5}}),
                                                   None)))
    assert empty["gen"] == "0" and empty["best_sel@gen"] == "0.800@5" and empty["lease"] == "-"


# ---------------------------------------------------------------------------------------------------------------
# Sign-off statistics and gates on synthetic per-seed tables
# ---------------------------------------------------------------------------------------------------------------

FC = SimpleNamespace(s_fb=2.0, s_yaw=1.5, s_ud=1.0, s_lr=0.5)
BASE = dict(collided=0.0, rms_range_err_m=0.30, in_band=0.6, too_close_s=0.5, very_close_s=0.05, min_range_m=1.3,
            overshoot_m=0.10, far_s=2.0, rms_bearing_deg=10.0, in_view=0.95, losses=0.3, time_lost_s=1.0,
            wrong_lock_s=0.1, stick_fb=2.0, stick_yaw=1.5, stick_ud=1.0, stick_lr=0.5, accel_rms=0.3, alt_max_m=2.0,
            ceiling_s=0.1, rms_alt_err_m=0.05, speed_track_err=0.15, excess_speed=0.02, fast_unneeded_s=0.3,
            max_speed_mps=0.35, governor_events=0.0, start_failed=0.0)
FLAGS = ("collided", "start_failed", "governor_events")


def flown_table(n: int = 40, seed: int = 0) -> list[dict]:
    """FLOWN's episodes: BASE with a per-seed factor (the seed's difficulty, common to every arm)."""
    rng = np.random.default_rng(seed)
    out = []
    for s in range(n):
        f = float(np.exp(0.2 * rng.standard_normal()))
        out.append({"seed": 8000 + s, "ok": True,
                    "metrics": {k: (v if k in FLAGS else v * f) for k, v in BASE.items()}})
    return out


def derive(table: list[dict], fn) -> list[dict]:
    """Another arm on the same seeds: fn(index, metrics copy) edits it."""
    out = []
    for i, r in enumerate(table):
        m = dict(r["metrics"])
        fn(i, m)
        out.append({"seed": r["seed"], "ok": True, "metrics": m})
    return out


def better(i: int, m: dict) -> None:
    m["rms_range_err_m"] *= 0.6
    m["far_s"] *= 0.5
    for k in so.STICKS:
        m[k] *= 0.8


def J(m: dict) -> float:
    return m["rms_range_err_m"] / 0.3 + (m["stick_fb"] / 2.0 + m["stick_yaw"] / 1.5) / 2 + 10 * m["excess_speed"]


def score(ma_: dict, mf: dict) -> float | None:
    """A stand-in for fitness.episode_score with the same conventions (FLOWN = 1, collision 6, None excluded)."""
    if ma_.get("start_failed") or mf.get("start_failed") or mf.get("error"):
        return None
    if ma_.get("error") or ma_.get("collided"):
        return 6.0
    return min(6.0, J(ma_) / max(J(mf), 0.5))


def recs_for(model_by_profile: dict, profiles=so.PROFILES, n: int = 40) -> dict:
    recs = {so.FLOWN: {}, "m": {}}
    for k, p in enumerate(profiles):
        f = flown_table(n, seed=k)
        recs[so.FLOWN][p] = f
        recs["m"][p] = derive(f, model_by_profile.get(p, better))
    return recs


ENTRIES = [{"label": so.FLOWN, "arm": "FLOWN", "gated": False, "kind": "baseline"},
           {"label": "m", "arm": "V3-JOINT", "gated": True, "kind": "model"}]


def run_gates(recs: dict, profiles=None) -> dict:
    return so.analyse(recs, ENTRIES, profiles, fc=FC, n_boot=2000, score_fn=score)


def test_bootstrap_ci_known_cases():
    m, lo, hi = so.bootstrap_ci([-0.1] * 50, 1000)
    assert (m, lo, hi) == pytest.approx((-0.1, -0.1, -0.1))
    m, lo, hi = so.bootstrap_ci([0.0, 1.0], 20_000)  # resampled means 0, 0.5, 1 with p 1/4, 1/2, 1/4
    assert (m, lo, hi) == (0.5, 0.0, 1.0)
    v = np.random.default_rng(3).normal(0.4, 2.0, 400)
    m, lo, hi = so.bootstrap_ci(v.tolist(), 10_000)
    half = 1.96 * v.std(ddof=1) / math.sqrt(v.size)
    assert m == pytest.approx(v.mean()) and (hi - lo) / 2 == pytest.approx(half, rel=0.1)
    assert lo < m < hi


def test_paired_differences_exclude_collisions_and_failed_starts():
    f = flown_table(20)
    a = derive(f, lambda i, m: m.update(rms_range_err_m=m["rms_range_err_m"] - 0.1))
    a[3]["metrics"]["collided"] = 1.0
    a[7]["metrics"]["start_failed"] = 1.0
    q = so.paired(a, f, "rms_range_err_m", 1000)
    assert q["n"] == 18 and q["n_excluded"] == 2
    assert (q["mean"], q["lo"], q["hi"]) == pytest.approx((-0.1, -0.1, -0.1))
    c = so.collision_pairs(a, f)
    assert (c["n"], c["a_only"], c["b_only"], c["both"]) == (19, 1, 0, 0)
    s = so.score_pairs(a, f, score, 1000)
    assert s["n"] == 19 and s["mean_flown"] == pytest.approx(1.0)  # the failed start is excluded, the collision is 6
    assert 6.0 in [score(r["metrics"], g["metrics"]) for r, g in zip(a, f)]


def test_a_clearly_better_model_is_recommended():
    res = run_gates(recs_for({}))
    g = res["gates"]["m"]
    assert g["pass"], [c for c in g["checks"] if not c["pass"]]
    assert {c["criterion"] for c in g["checks"]} == {1, 2, 3, 4, 5}
    assert res["verdict"].startswith("RECOMMENDED: m")
    sc = res["compare"]["m"]["train"]["score"]
    assert sc["mean_flown"] == pytest.approx(1.0) and sc["mean_model"] < 0.9 and sc["hi"] < 0


def test_more_collisions_fail_criterion_2_seed_by_seed():
    def crash(k):
        return lambda i, m: (better(i, m), m.update(collided=1.0 if i < k else 0.0))

    g = run_gates(recs_for({"stress": crash(3)}))["gates"]["m"]
    assert g["failed_criteria"] == [2]
    bad = [c for c in g["checks"] if not c["pass"]]
    assert len(bad) == 1 and bad[0]["check"].startswith("stress") and "alone on 3 seeds" in bad[0]["detail"]
    assert run_gates(recs_for({"stress": crash(1)}))["gates"]["m"]["pass"]  # the +1 margin
    both = recs_for({"stress": crash(3)})
    for r in both[so.FLOWN]["stress"][:2]:
        r["metrics"]["collided"] = 1.0  # FLOWN crashes on 2 of those seeds too: 1 alone vs 0 alone, within the margin
    assert run_gates(both)["gates"]["m"]["pass"]


def test_more_excess_speed_on_slow_fails_criterion_5():
    def rush(i, m):
        better(i, m)
        m["excess_speed"] += 0.1

    g = run_gates(recs_for({"slow": rush}))["gates"]["m"]
    assert g["failed_criteria"] == [5]
    assert [c["check"] for c in g["checks"] if not c["pass"]] == ["slow: excess_speed not worse than FLOWN"]

    def fast_when_slow(i, m):
        better(i, m)
        m["fast_unneeded_s"] += 2.0

    assert run_gates(recs_for({"slow": fast_when_slow}))["gates"]["m"]["failed_criteria"] == [5]

    def falls_behind(i, m):
        better(i, m)
        m["far_s"] *= 2.5  # better() halved it: now 1.25 x FLOWN's

    g = run_gates(recs_for({"fast": falls_behind}))["gates"]["m"]
    assert g["failed_criteria"] == [5] and "far_s" in next(c["check"] for c in g["checks"] if not c["pass"])


def test_other_gates_fail_on_their_own_criterion():
    g = run_gates(recs_for({"demo": lambda i, m: (better(i, m), m.update(overshoot_m=m["overshoot_m"] + 0.3))}))
    assert g["gates"]["m"]["failed_criteria"] == [3]
    g = run_gates(recs_for({"train": lambda i, m: (better(i, m), m.update(stick_lr=m["stick_lr"] * 4))}))
    assert 4 in g["gates"]["m"]["failed_criteria"]
    g = run_gates(recs_for({"train": lambda i, m: None, "demo": lambda i, m: None}))  # = FLOWN on the main profiles
    assert g["gates"]["m"]["failed_criteria"] == [1] and g["verdict"].startswith("NO MODEL RECOMMENDED")
    recs = recs_for({})
    recs["m"]["fast"][0] = {"seed": recs["m"]["fast"][0]["seed"], "ok": False, "metrics": {"error": 1.0}}
    assert run_gates(recs)["gates"]["m"]["failed_criteria"] == [2]  # the model's own error is never excused
    recs[so.FLOWN]["fast"][0] = {"seed": recs[so.FLOWN]["fast"][0]["seed"], "ok": False, "metrics": {"error": 1.0}}
    g = run_gates(recs)["gates"]["m"]
    assert g["pass"]  # FLOWN failed on that seed too: the simulator's error, excluded for both
    assert "1 more on seeds where FLOWN failed too" in next(c["detail"] for c in g["checks"]
                                                            if c["check"].startswith("fast: collisions"))


def test_missing_profiles_fail_their_criteria():
    g = run_gates(recs_for({}, profiles=("train", "demo")), profiles=["train", "demo"])["gates"]["m"]
    assert g["failed_criteria"] == [2, 5]
    assert sum("not evaluated" in c["detail"] for c in g["checks"]) == 3 + 4  # fast/slow/stress + 2 slow, 2 fast


def test_identity_check_of_an_init_arm():
    recs = recs_for({})
    recs["V3-FOLLOW@init"] = {p: [dict(r) for r in rs] for p, rs in recs[so.FLOWN].items()}
    entries = ENTRIES + [{"label": "V3-FOLLOW@init", "arm": "V3-FOLLOW", "gated": False, "kind": "init"}]
    res = so.analyse(recs, entries, fc=FC, n_boot=500, score_fn=score)
    assert res["identity"]["V3-FOLLOW@init"]["differ"] == 0 and "V3-FOLLOW@init" not in res["gates"]
    recs["V3-FOLLOW@init"]["demo"][4] = {**recs["V3-FOLLOW@init"]["demo"][4],
                                         "metrics": {**recs[so.FLOWN]["demo"][4]["metrics"], "far_s": 9.0}}
    q = so.analyse(recs, entries, fc=FC, n_boot=500, score_fn=score)["identity"]["V3-FOLLOW@init"]
    assert q["differ"] == 1 and q["worst_metric"] == "far_s"


def test_the_real_fitness_scores_flown_at_one():
    from kite.fly.behind.fitness import FitnessCfg

    res = so.analyse(recs_for({}), ENTRIES, fc=FitnessCfg(), n_boot=1000)
    for p in so.PROFILES:
        sc = res["compare"]["m"][p]["score"]
        assert sc["mean_flown"] == pytest.approx(1.0) and sc["mean_model"] < 1.0
    assert res["gates"]["m"]["pass"], [c for c in res["gates"]["m"]["checks"] if not c["pass"]]


# ---------------------------------------------------------------------------------------------------------------
# Model files, the cache, the CLI
# ---------------------------------------------------------------------------------------------------------------


def _save(tmp_path, arm: str, x, name: str):
    from kite.fly.behind import params as bp
    from kite.follow_tuned import save_model

    sp = bp.space(arm)
    return save_model(tmp_path / f"{name}.json", name=name, follow=bp.model_follow(arm, x), brain_x=sp.brain_x(x),
                      steering=bp.steering(arm), trained={"behind_arm": arm, "cma_seed": 1, "gen": 5})


@pytest.mark.parametrize("arm", ["V3-FOLLOW", "V3-JOINT", "V3-LAW"])
def test_a_saved_model_is_reconstructed_exactly(tmp_path, arm):
    from kite.fly.behind import params as bp

    x = np.clip(bp.init_x(arm) + np.random.default_rng(1).normal(0, 0.1, bp.space(arm).dim), 0, 1)
    path = _save(tmp_path, arm, x, f"t-{arm.lower()}")
    e = so.load_entry(path)
    assert e["arm"] == arm and e["label"] == f"t-{arm.lower()}" and e["kind"] == "model"
    cfg_e, brain_e = bp.make_cfg(arm, e["x"])
    cfg_x, brain_x = bp.make_cfg(arm, x)
    assert cfg_e == cfg_x and brain_e == brain_x


def test_a_model_that_would_fly_something_else_is_refused(tmp_path):
    from kite.fly.behind import params as bp

    path = _save(tmp_path, "V3-FOLLOW", bp.init_x("V3-FOLLOW"), "follow")
    rec = json.loads(path.read_text())
    rec["x"] = [min(1.0, v + 0.05) for v in rec["x"]]  # a different brain: V3-FOLLOW would fly smooth2's
    path.write_text(json.dumps(rec))
    with pytest.raises(SystemExit, match="brain"):
        so.load_entry(path)
    rec = json.loads(_save(tmp_path, "V3-LAW", bp.init_x("V3-LAW"), "law").read_text())
    rec["follow"]["fly_deadband"] = 6.0  # a knob outside the law arm's space
    (tmp_path / "law.json").write_text(json.dumps(rec))
    with pytest.raises(SystemExit, match="fly_deadband"):
        so.load_entry(tmp_path / "law.json")
    rec["trained"]["behind_arm"] = "FLOWN"
    (tmp_path / "law.json").write_text(json.dumps(rec))
    with pytest.raises(SystemExit, match="behind_arm"):
        so.load_entry(tmp_path / "law.json")


class FakeBackend:
    def __init__(self, fail_seed=None):
        self.calls, self.fail_seed = [], fail_seed

    def evaluate(self, items):
        self.calls.append(len(items))
        out = []
        for it in items:
            ok = it["seed"] != self.fail_seed
            out.append({"arm": it["arm"], "seed": it["seed"], "profile": it["profile"], "ok": ok,
                        "metrics": {**BASE, "far_s": float(it["seed"] % 7)} if ok else {"error": 1.0},
                        "wall_s": 0.01, "error": None if ok else "boom"})
        return out, {"usd": 0.001}


def test_the_episode_cache_reruns_nothing_twice(tmp_path):
    entries = [{"label": so.FLOWN, "arm": "FLOWN", "x": None}, {"label": "a", "arm": "V3-LAW", "x": [0.5] * 19},
               {"label": "a-copy", "arm": "V3-LAW", "x": [0.5] * 19}]
    seeds, profiles, path = [8000, 8001, 8002], ["train", "slow"], tmp_path / "cache" / "episodes.jsonl"
    be = FakeBackend(fail_seed=8002)
    recs, st = so.run_eval(entries, seeds, profiles, be, so.EpisodeCache(path), "d1", batch=5, log=lambda s: None)
    assert sum(be.calls) == 2 * 3 * 2 and st["errors"] == 2 * 2  # a-copy shares a's episodes; 8002 fails
    assert recs["a"]["slow"] == recs["a-copy"]["slow"] and [r["seed"] for r in recs["a"]["train"]] == seeds
    assert not recs["a"]["train"][2]["ok"] and recs["a"]["train"][2]["metrics"]["error"] == 1.0
    be2 = FakeBackend()
    recs2, st2 = so.run_eval(entries, seeds, profiles, be2, so.EpisodeCache(path), "d1", log=lambda s: None)
    # only the 4 failed episodes re-run; the other 18 - 6 lookups (seed 8002 of FLOWN, a and a-copy) hit the cache
    assert sum(be2.calls) == 4 and st2["cache_hits"] == 3 * 2 * 3 - 3 * 2
    assert all(r["ok"] for per in recs2.values() for rs in per.values() for r in rs)
    be3 = FakeBackend()
    so.run_eval(entries, seeds, profiles, be3, so.EpisodeCache(path), "d2", log=lambda s: None)
    assert sum(be3.calls) == 12  # another sim digest: nothing is reused
    be4 = FakeBackend()
    so.run_eval(entries, seeds, profiles, be4, so.EpisodeCache(path), "d0", reuse_stale=True, log=lambda s: None)
    assert be4.calls == []
    with pytest.raises(SystemExit, match="cache-only"):
        so.run_eval(entries, seeds + [8003], profiles, FakeBackend(), so.EpisodeCache(path), "d2", cache_only=True,
                    log=lambda s: None)
    assert so.episode_key("V3-LAW", [0.5], 1, "train", 60.0) != so.episode_key("V3-LAW", [0.5], 1, "train", 60.0,
                                                                               stub_s=0.0)


def test_sim_digest_tracks_the_episode_path(tmp_path):
    (tmp_path / "kite" / "fly" / "behind").mkdir(parents=True)
    (tmp_path / "kite" / "fly" / "behind" / "world.py").write_text("a = 1\n")
    (tmp_path / "kite" / "fly" / "behind" / "signoff.py").write_text("x\n")
    d0 = so.sim_digest(tmp_path)
    (tmp_path / "kite" / "fly" / "behind" / "signoff.py").write_text("y\n")  # not on the episode path
    (tmp_path / "kite" / "fly" / "behind" / "world 2.py").write_text("an iCloud copy\n")
    assert so.sim_digest(tmp_path) == d0
    (tmp_path / "kite" / "fly" / "behind" / "world.py").write_text("a = 2\n")
    assert so.sim_digest(tmp_path) != d0


def test_signoff_end_to_end_on_stub_episodes(tmp_path):
    _train_module()
    from kite.fly.behind import params as bp

    x = np.clip(bp.init_x("V3-LAW") - 0.2, 0, 1)
    model = _save(tmp_path, "V3-LAW", x, "stub-law")
    out = tmp_path / "report"
    argv = ["--models", str(model), "--seeds", "selection", "--n", "6", "--stub-s", "0", "--processes", "1",
            "--n-boot", "200", "--out", str(out), "--init-arms", "V3-FOLLOW"]
    assert so.main(argv) == 0
    res = json.loads((out / "signoff.json").read_text())
    assert set(res["records"]) == {so.FLOWN, "V3-FOLLOW@init", "stub-law"}
    assert all(len(res["records"]["stub-law"][p]) == 6 for p in so.PROFILES)
    # computed on every seed (the stub places an arm by its x, so only the real sim makes it 0: see the next test)
    assert res["analysis"]["identity"]["V3-FOLLOW@init"]["n"] == 6 * len(so.PROFILES)
    g = res["analysis"]["gates"]["stub-law"]
    assert {c["criterion"] for c in g["checks"]} == {1, 2, 3, 4, 5} and "Verdict" in (out / "signoff.md").read_text()
    n_cached = len((out / "cache" / "episodes.jsonl").read_text().splitlines())
    assert so.main(argv + ["--cache-only"]) == 0  # everything comes from the cache
    assert len((out / "cache" / "episodes.jsonl").read_text().splitlines()) == n_cached
    assert so.main(["--from", str(out / "signoff.json"), "--out", str(out), "--n-boot", "200"]) == 0
    again = json.loads((out / "signoff.json").read_text())
    assert again["analysis"]["gates"]["stub-law"]["failed_criteria"] == g["failed_criteria"]
    rec = so.signoff_record(again, "stub-law")
    assert rec["recommended"] == g["pass"] and set(rec["score"]) == set(so.PROFILES)


def test_tuned_fly_arms_at_their_init_fly_exactly_flown_in_the_real_sim(tmp_path):
    """The sign-off's path (arm + x through kite.fly.behind.evaluate) on the real simulator: V3-FOLLOW and V3-JOINT at
    their init are the flown follow with smooth2, so every metric of every seed must equal FLOWN's."""
    bt = _train_module()
    from kite.fly.train import cma_train as ct

    entries = so.build_entries([], ["V3-FOLLOW", "V3-JOINT"])
    backend = bt.BehindLocalBackend(1, ct.Prices())
    try:
        recs, st = so.run_eval(entries, [8000, 8001], ["train"], backend, so.EpisodeCache(None), "real",
                               episode_s=10.0, log=lambda s: None)
    finally:
        backend.close()
    assert st["errors"] == 0
    for lab in ("V3-FOLLOW@init", "V3-JOINT@init"):
        q = so.identity_check(recs, lab)
        assert q["n"] == 2 and q["differ"] == 0, (lab, q)


def test_a_record_without_fitness_fields_is_still_summarized():
    r = {"seed": 1, "ok": True, "metrics": {"far_s": 1.0}}
    s = so.summarize([r, {**r, "seed": 2, "ok": False, "metrics": {"error": 1.0}}])
    assert (s["n"], s["n_errors"], s["metrics"]) == (2, 1, {"far_s": 1.0})
    assert so.stick_norm({"stick_fb": 2.0, "stick_yaw": 1.5, "stick_ud": 1.0, "stick_lr": 0.5}, FC) == 1.0
    assert so.stick_norm({"stick_fb": 2.0}, FC) is None


# ---------------------------------------------------------------------------------------------------------------
# Review fixes: the code digest, the lease takeover, determinism helpers, the governor of a model
# ---------------------------------------------------------------------------------------------------------------


def test_the_code_digest_covers_the_episode_path_and_ships_in_the_image(ma):
    """Every kite file an episode imports (the import probe, kite/config.py's SafetyCfg and kite/camera.py included)
    is in kite.fly.behind.digest, so a change there invalidates cached episodes and refuses a resume; and every file
    the digest hashes is in the Modal image, so the laptop and a container agree on it."""
    from kite.fly.behind import digest as dg

    root = ma.REPO.resolve()
    probe = {str(__import__("pathlib").Path(f).resolve().relative_to(root).as_posix())
             for f in ma.run_probe()["files"]}
    covered = {p.as_posix() for p in dg.source_files(root, scoring=True)}
    assert {"kite/config.py", "kite/camera.py", "kite/sources/pose.py", "kite/fly/behind/fitness.py"} <= covered
    assert probe - covered <= dg.DRIVER_ONLY, sorted(probe - covered - dg.DRIVER_ONLY)
    shipped = {f.relative_to(root).as_posix() for f in ma.source_files()[0]}
    assert covered <= shipped, sorted(covered - shipped)
    assert "kite/fly/behind/fitness.py" not in {p.as_posix() for p in dg.source_files(root)}  # episode digest


def test_the_digest_tracks_config_and_fitness(tmp_path):
    from kite.fly.behind import digest as dg

    (tmp_path / "kite" / "fly" / "behind").mkdir(parents=True)
    (tmp_path / "kite" / "config.py").write_text("max_rc = 50\n")
    (tmp_path / "kite" / "fly" / "behind" / "fitness.py").write_text("w = 1\n")
    (tmp_path / "kite" / "fly" / "behind" / "train.py").write_text("driver\n")
    e0, r0 = dg.episode_digest(tmp_path), dg.run_digest(tmp_path)
    (tmp_path / "kite" / "fly" / "behind" / "train.py").write_text("another driver\n")
    assert (dg.episode_digest(tmp_path), dg.run_digest(tmp_path)) == (e0, r0)
    (tmp_path / "kite" / "fly" / "behind" / "fitness.py").write_text("w = 2\n")
    assert dg.episode_digest(tmp_path) == e0 and dg.run_digest(tmp_path) != r0
    (tmp_path / "kite" / "config.py").write_text("max_rc = 65\n")
    assert dg.episode_digest(tmp_path) != e0


def test_a_respawned_driver_takes_over_the_dead_drivers_lease(ma, tmp_path):
    d = tmp_path / "V3-JOINT_s1_v3"
    ma._lease(d, "fc-OLD", True)
    ma._lease(d, "fc-OLD", False)  # its last heartbeat: fresh, though the call has died
    with pytest.raises(RuntimeError, match="fc-OLD"):
        ma._lease(d, "fc-NEW", True)  # a second launch is still refused
    ma._lease(d, "fc-NEW", True, takeover="fc-OLD")  # train_all's re-spawn names its dead predecessor
    assert json.loads((d / "lease.json").read_text())["call_id"] == "fc-NEW"
    import inspect

    raw = getattr(ma.train, "_raw_f_", None) or getattr(getattr(ma.train, "info", None), "raw_f", None)
    assert raw is not None and "takeover" in inspect.signature(raw).parameters


def test_determinism_helpers(ma):
    m = {"a": 1.0, "b": 0.1 + 0.2}
    assert ma.metrics_hash(m) == ma.metrics_hash(dict(reversed(list(m.items()))))
    assert ma.metrics_hash(m) != ma.metrics_hash({"a": 1.0, "b": 0.3})  # the last bit counts
    assert isinstance(ma.cpu_isa(), dict)


def test_a_model_trained_under_another_governor_is_refused(tmp_path):
    from kite.follow_tuned import save_model

    ok = save_model(tmp_path / "ok.json", name="ok", follow={"fwd_cap_fast": 63, "fwd_cap_per_mps": 100.0},
                    brain_x=None, steering="pid", trained={"behind_arm": "V3-LAW"}, governor={"trained_max_rc": 65})
    g = so.load_entry(ok)["governor"]
    assert g == {"needs_max_rc": 63, "trained_max_rc": 65, "sim_max_rc": 65}
    bad = save_model(tmp_path / "bad.json", name="bad", follow={}, brain_x=None, steering="pid",
                     trained={"behind_arm": "V3-LAW"}, governor={"trained_max_rc": 50})
    with pytest.raises(SystemExit, match="governor"):
        so.load_entry(bad)
