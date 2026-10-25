"""kite.fly.behind's trainer (DESIGN sections 7 and 8): the parameter spaces decode their init to EXACTLY the flown
follow, the item runner never raises, and the CMA-ES driver, on stub episodes (no simulator), improves, resumes
exactly after a kill, stops on its budget and writes best.json as a model file the app can load."""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, fields

import numpy as np
import pytest

pytest.importorskip("cma")

from kite.fly.behind import BEHIND_TRAIN_BASE, SEED_STRIDE, SELECTION_SEEDS, TEST_SEEDS  # noqa: E402
from kite.fly.behind import evaluate as bev  # noqa: E402
from kite.fly.behind import params as bp  # noqa: E402
from kite.fly.behind import train as tr  # noqa: E402
from kite.fly.behind.fitness import fitness  # noqa: E402
from kite.fly.interfaces import EpisodeResult  # noqa: E402
from kite.fly.train import cma_train as ct  # noqa: E402
from kite.flybrain.flyfollow.rl.params import param_space  # noqa: E402
from kite.flybrain.flyfollow.steer import load_params  # noqa: E402
from kite.follow_behind import BehindCfg  # noqa: E402
from kite.follow_tuned import SMOOTH2_PATH, TunedCfg, load_model, save_model  # noqa: E402

SMOOTH2_X = [float(v) for v in load_params(SMOOTH2_PATH)[0]]
DIMS = {"FLOWN": 0, "V3-FOLLOW": 25, "V3-JOINT": 59, "V3-LAW": 23}


# ---------------------------------------------------------------------------------------------------------------
# Parameter spaces
# ---------------------------------------------------------------------------------------------------------------


def test_arm_spaces_have_the_design_dimensions_and_names():
    assert bp.ARMS == ("FLOWN", "V3-FOLLOW", "V3-JOINT", "V3-LAW", "V3B-FOLLOW", "V3B-JOINT")
    for arm, dim in DIMS.items():
        assert bp.space(arm).dim == dim == len(bp.init_x(arm)) == len(bp.cma_stds(arm))
    follow = [r[0] for r in bp.FOLLOW_KNOBS]
    assert len(follow) == 20 and len(bp.FLY_KNOBS) == 5 and len(bp.LAW_KNOBS) == 3
    assert bp.space("V3-FOLLOW").names == tuple(follow + [r[0] for r in bp.FLY_KNOBS])
    assert bp.space("V3-LAW").names == tuple(follow + [r[0] for r in bp.LAW_KNOBS])
    joint = bp.space("V3-JOINT")
    assert joint.knob_names == bp.space("V3-FOLLOW").names
    assert joint.brain_names == tuple("brain." + n for n in param_space("FLY-YAW").names)


def test_the_design_tables_flown_column_is_the_cfg_default():
    d = TunedCfg()
    for name, lo, hi, scale, flown in bp.FOLLOW_KNOBS + bp.FLY_KNOBS + bp.LAW_KNOBS:
        assert lo <= flown <= hi and scale in ("lin", "log"), name
        assert (scale == "log") <= (lo > 0), name
        want = 0.0 if name == "fwd_slew" else flown  # the top of fwd_slew is TunedCfg's 0 = off
        assert getattr(d, name) == want, name
        assert (name in bp.INT_KNOBS) == (type(getattr(d, name)) is int), name  # integer knobs are the int fields
    assert {f.name for f in fields(TunedCfg)} >= set(bp.space("V3-FOLLOW").names) | set(bp.space("V3-LAW").names)


def test_decode_encode_round_trip():
    rng = np.random.default_rng(3)
    for arm in bp.TUNED_ARMS:
        sp = bp.space(arm)
        for _ in range(20):
            x = rng.random(sp.dim)
            d = sp.decode(x)
            x2 = sp.encode(d)
            assert sp.decode(x2) == d  # idempotent
            for i, n in enumerate(sp.names):
                if n in bp.INT_KNOBS:
                    assert isinstance(d[n], int)
                elif n == "fwd_slew" and d[n] == 0.0:
                    assert x2[i] == 1.0
                else:
                    assert x2[i] == pytest.approx(x[i], abs=1e-9), n
            if sp.brain_names:  # brain dims pass through bit-exact
                assert sp.brain_x(x) == [float(v) for v in x[sp.n_knobs:]] == [d[n] for n in sp.brain_names]
        assert np.all(sp.encode({}) == sp.default_x())  # a missing name takes the flown value
        with pytest.raises(ValueError):
            sp.decode(np.zeros(sp.dim + 1))


def test_the_init_is_exactly_the_flown_follow():
    flown = asdict(TunedCfg())
    for arm in tr.TRAINABLE:  # round 1 starts at the flown follow; V3B warm-starts from a round-1 model
        cfg, brain_x = bp.make_cfg(arm, bp.init_x(arm))
        assert type(cfg) is TunedCfg
        got = asdict(cfg)
        assert got.pop("steering") == ("pid" if arm == "V3-LAW" else "fly")
        assert got == {k: v for k, v in flown.items() if k != "steering"}, arm  # every field, bit for bit
        assert cfg.fly_params_path == str(SMOOTH2_PATH) and cfg.max_rc_yaw == 40
        for n in bp.space(arm).knob_names:
            assert type(getattr(cfg, n)) is type(getattr(TunedCfg(), n)), n
        assert brain_x == (SMOOTH2_X if arm == "V3-JOINT" else None)
    assert bp.make_cfg("V3-JOINT")[1] == SMOOTH2_X  # x None = the init
    cfg, brain_x = bp.make_cfg("FLOWN")
    assert type(cfg) is BehindCfg and cfg == BehindCfg() and brain_x is None
    assert bp.model_follow("FLOWN") == {} and bp.init_x("FLOWN").shape == (0,)
    with pytest.raises(ValueError):
        bp.make_cfg("FLOWN", [0.5])
    with pytest.raises(ValueError):
        bp.make_cfg("V3-NOPE")


def test_fly_arms_set_max_rc_yaw_with_the_fly_clamp():
    sp = bp.space("V3-FOLLOW")
    x = sp.encode({"fly_max_rc_yaw": 55, "dist_gain": 70.0, "fwd_cap_fast": 60, "fwd_slew": 300.0})
    cfg, _ = bp.make_cfg("V3-FOLLOW", x)
    assert cfg.fly_max_rc_yaw == cfg.max_rc_yaw == 55 and cfg.fwd_cap_fast == 60
    assert cfg.dist_gain == pytest.approx(70.0) and cfg.fwd_slew == pytest.approx(300.0)
    law, _ = bp.make_cfg("V3-LAW", bp.space("V3-LAW").encode({"max_rc_yaw": 25, "yaw_gain": 2.0}))
    assert law.steering == "pid" and law.max_rc_yaw == 25 and law.fly_max_rc_yaw == 40


def test_cma_stds_search_the_knobs_widely_and_fine_tune_the_brain():
    assert bp.cma_stds("V3-FOLLOW") == [1.0] * 25 and bp.cma_stds("V3-LAW") == [1.0] * 23
    s = dict(zip(bp.space("V3-JOINT").names, bp.cma_stds("V3-JOINT", sigma0=0.15)))
    assert all(s[n] == 1.0 for n in bp.space("V3-FOLLOW").names)
    assert s["brain.enc_r_max_hz"] == pytest.approx(0.2)  # 0.15 x 0.2 = kite.fly FLY-YAW's step 0.03
    assert s["brain.dec_w_yaw_DNa02"] == pytest.approx(0.2 * 0.08) and s["brain.dec_b_yaw"] == pytest.approx(0.04)
    s3 = dict(zip(bp.space("V3-JOINT").names, bp.cma_stds("V3-JOINT", sigma0=0.3)))
    assert 0.3 * s3["brain.enc_width_deg"] == pytest.approx(0.03) and s3["dist_gain"] == 1.0


def test_a_saved_model_loads_back_as_the_same_cfg(tmp_path):
    rng = np.random.default_rng(5)
    for arm in bp.TUNED_ARMS:
        sp = bp.space(arm)
        x = rng.random(sp.dim)
        cfg, brain_x = bp.make_cfg(arm, x)
        path = save_model(tmp_path / f"{arm}.json", name=arm.lower(), follow=bp.model_follow(arm, x),
                          brain_x=brain_x, steering=bp.steering(arm), trained={"behind_arm": arm})
        back = load_model(path)
        a, b = asdict(cfg), asdict(back)
        for k in ("fly_params_path", "model_name"):
            a.pop(k), b.pop(k)
        assert a == b, arm
        rec = json.loads(path.read_text())
        assert sp.decode(bp.encode_model(arm, rec)) == sp.decode(x)
        if arm == "V3-LAW":
            assert "x" not in rec and back.steering == "pid"
        else:
            assert rec["x"] == (brain_x if arm in bp.JOINT_ARMS else SMOOTH2_X)  # a FlySteer params file
            assert back.fly_params_path == str(path.resolve())
            assert [float(v) for v in load_params(path)[0]] == rec["x"]


# ---------------------------------------------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------------------------------------------


def test_items_are_plain_picklable_dicts():
    it = bev.make_item("V3-JOINT", 7, "demo", x=np.full(57, 0.5), stub_s=0.0, tag={"i": 3})
    assert json.loads(json.dumps(it)) == it and pickle.loads(pickle.dumps(it)) == it
    assert it["x"] == [0.5] * 57 and it["model"] is None and it["record"] is False


def test_run_item_never_raises(monkeypatch):
    def boom(item):
        raise RuntimeError("kaput")

    it = bev.make_item("V3-LAW", 3, "train", tag={"role": "cand", "i": 0})
    monkeypatch.setattr(bev, "EPISODE_RUNNER", boom)
    r = bev.run_item(it)
    assert r["ok"] is False and r["metrics"] == {"error": 1.0} and "kaput" in r["error"] and r["traceback"]
    assert r["tag"] == {"role": "cand", "i": 0} and bev.is_error(r) and r["item_wall_s"] >= 0
    monkeypatch.setattr(bev, "EPISODE_RUNNER", lambda it: EpisodeResult(3, "train", "V3-LAW", float("nan"),
                                                                        metrics={"a": 1.0}))
    assert bev.run_item(it)["ok"] is False
    monkeypatch.setattr(bev, "EPISODE_RUNNER", lambda it: EpisodeResult(3, "train", "V3-LAW", 0.0))
    assert "no metrics" in bev.run_item(it)["error"]
    monkeypatch.setattr(bev, "EPISODE_RUNNER", lambda it: EpisodeResult(
        3, "train", "V3-LAW", -1.0, metrics={"a": np.float64(2.0)}, trace={"t": [1]}))
    r = bev.run_item(it)
    assert r["ok"] and r["metrics"] == {"a": 2.0} and type(r["metrics"]["a"]) is float and r["trace"] is None
    pickle.dumps(r)


def test_run_item_flies_the_arms_cfg(monkeypatch, tmp_path):
    calls = []

    def fake(seed, profile, cfg, *, arm="FLOWN", brain_x=None, episode_s=60.0, overrides=None, record=False):
        calls.append((seed, profile, cfg, arm, brain_x, episode_s, overrides, record))
        return EpisodeResult(seed, profile, arm, 0.0, metrics={"collided": 0.0})

    from kite.fly.behind import world

    monkeypatch.setattr(world, "run_behind_episode", fake)
    x = bp.init_x("V3-JOINT")
    assert bev.run_item(bev.make_item("V3-JOINT", 11, "fast", x=x, episode_s=30.0, overrides={"a": 1}))["ok"]
    assert bev.run_item(bev.make_item("FLOWN", 12, "demo"))["ok"]
    model = save_model(tmp_path / "m.json", name="m", follow={"dist_gain": 50.0}, brain_x=None, steering="fly",
                       trained={})
    assert bev.run_item(bev.make_item("m", 13, "slow", model=model, record=True))["ok"]
    (s1, p1, c1, a1, b1, e1, o1, r1), (s2, p2, c2, a2, b2, *_), (s3, p3, c3, a3, b3, _, _, r3) = calls
    assert (s1, p1, a1, e1, o1, r1) == (11, "fast", "V3-JOINT", 30.0, {"a": 1}, False)
    assert type(c1) is TunedCfg and b1 == SMOOTH2_X
    assert type(c2) is BehindCfg and b2 is None and a2 == "FLOWN"
    assert type(c3) is TunedCfg and c3.dist_gain == 50.0 and c3.fly_params_path == str(model.resolve())
    assert b3 is None and r3 is True


def test_stub_episodes_carry_every_metric_and_flown_scores_one():
    keys = bev.metric_keys()
    fl = [bev.run_item(bev.make_item("FLOWN", s, stub_s=0.0)) for s in range(40)]
    assert all(set(r["metrics"]) >= set(keys) for r in fl)
    f = fitness([r["metrics"] for r in fl], [r["metrics"] for r in fl])
    assert f["fitness"] == pytest.approx(1.0) and f["n"] + f["excluded"] == 40 and f["excluded"] >= 1
    good = [bev.run_item(bev.make_item("V3-LAW", s, x=np.full(21, 0.3), stub_s=0.0)) for s in range(40)]
    assert fitness([r["metrics"] for r in good], [r["metrics"] for r in fl])["fitness"] < 0.8


def test_the_pool_keeps_item_order():
    items = [bev.make_item("V3-LAW", 100 + i, x=np.full(21, 0.4), stub_s=0.0, tag={"i": i}) for i in range(6)]
    out = bev.evaluate_chunk_local(items, workers=2)
    bev.shutdown_pool()
    assert [r["tag"]["i"] for r in out["results"]] == list(range(6)) and all(r["ok"] for r in out["results"])
    assert out["wall_s"] > 0 and out["n_items"] == 6


# ---------------------------------------------------------------------------------------------------------------
# CMA-ES on the stub
# ---------------------------------------------------------------------------------------------------------------


def stub_cfg(arm="V3-LAW", **over):
    base = {"gens": 3, "pop": 8, "k": 4, "select_every": 1, "n_sel": 8, "stub_s": 0.0}
    base.update(over)
    return tr.make_run_cfg(arm, 1, "test", **base)


def trainer(tmp_path, cfg=None, backend=None, name="run"):
    return tr.Trainer(cfg or stub_cfg(), backend or tr.BehindLocalBackend(processes=1), tmp_path / name,
                      log=lambda s: None)


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_seeds_are_common_fresh_and_disjoint():
    s = tr.train_seeds(1, 32, 1)
    assert s[0] == BEHIND_TRAIN_BASE + SEED_STRIDE + 32 and len(s) == 32
    assert not set(tr.train_seeds(2, 32, 1)) & set(s) and not set(tr.train_seeds(1, 32, 2)) & set(s)
    last = tr.train_seeds(SEED_STRIDE // 32 - 1, 32, 1)
    assert max(last) < BEHIND_TRAIN_BASE + 2 * SEED_STRIDE
    assert min(s) > max(TEST_SEEDS) > max(SELECTION_SEEDS)
    with pytest.raises(ValueError):
        tr.train_seeds(SEED_STRIDE // 32, 32, 1)


def test_a_stub_run_improves_and_logs_everything(tmp_path):
    t = trainer(tmp_path, stub_cfg(gens=4, pop=10))
    s = t.run()
    assert s["gen"] == 4 and s["stopped"] == "done" and s["errors"] == 0
    sel = lines(t.sel_path)
    assert [r["gen"] for r in sel] == [0, 1, 2, 3, 4] and sel[0]["x"] == bp.init_x("V3-LAW").tolist()
    assert s["best"]["sel_fitness"] < sel[0]["sel_fitness"] - 0.1  # lower is better
    log = lines(t.log_path)
    assert [r["gen"] for r in log] == [1, 2, 3, 4]
    assert log[0]["n_items"] == 10 * 4 + 4 and all(r["n_flown"] == 4 for r in log)  # fresh seeds every generation
    assert log[0]["seeds"] == [BEHIND_TRAIN_BASE + SEED_STRIDE + 4, BEHIND_TRAIN_BASE + SEED_STRIDE + 7]
    assert log[0]["fit_best"] <= log[0]["fit_median"] <= log[0]["fit_worst"] and log[0]["best_parts"]
    # FLOWN runs once per seed: 8 selection seeds once, then K per generation
    assert len(t.state["flown"]) == 8 + 4 * 4 and s["episodes"] == 5 * 8 + 8 + 4 * (10 * 4 + 4)
    conf = json.loads((t.run_dir / "config.json").read_text())
    assert conf["fitness_cfg"]["w_collided"] == 4.0 and len(conf["space"]) == 23 and conf["names"][0] == "dist_gain"


def test_resume_reproduces_the_same_candidates_after_a_kill(tmp_path):
    cfg = stub_cfg("V3-JOINT", gens=3)
    straight = trainer(tmp_path, cfg, name="a")
    straight.run()
    killed = trainer(tmp_path, cfg, name="b")
    killed.run(stop_after=1)
    assert killed.state["gen"] == 1
    np.random.seed(12345)
    np.random.rand(7)  # the global RNG must not matter
    resumed = trainer(tmp_path, cfg, name="b")
    assert resumed.load_or_init() is True
    resumed.run()
    a, b = lines(straight.log_path), lines(resumed.log_path)
    assert [r["gen"] for r in b] == [1, 2, 3]
    for ra, rb in zip(a, b):
        for k in ("cand_sha", "fit_best", "fit_median", "fit_mean", "sigma", "mean_x", "seeds"):
            assert ra[k] == rb[k], k
    ba, bb = (json.loads(t.best_path.read_text()) for t in (straight, resumed))
    assert ba["x"] == bb["x"] and ba["follow"] == bb["follow"] and ba["trained"]["fitness"] == bb["trained"]["fitness"]


def test_resume_drops_a_generation_logged_but_not_saved_and_refuses_a_changed_config(tmp_path):
    t = trainer(tmp_path, stub_cfg(gens=1))
    t.run()
    ct.append_jsonl(t.log_path, {"gen": 2, "fit_best": 0.0})  # a kill between the log and the checkpoint
    again = trainer(tmp_path, stub_cfg(gens=1))
    again.load_or_init()
    assert [r["gen"] for r in lines(t.log_path)] == [1]
    with pytest.raises(RuntimeError, match="pop"):
        trainer(tmp_path, stub_cfg(gens=1, pop=6)).load_or_init()
    with pytest.raises(RuntimeError, match="fitness"):
        trainer(tmp_path, stub_cfg(gens=1, fitness={"w_collided": 8.0})).load_or_init()
    s = trainer(tmp_path, stub_cfg(gens=2)).run()  # more generations on the same run resume it
    assert s["gen"] == 2 and s["stopped"] == "done"


class PricedBackend(tr.BehindLocalBackend):
    def evaluate(self, items):
        res, st = super().evaluate(items)
        st["usd"] = 0.01 * len(items)
        return res, st


def test_budget_stops_before_overspending(tmp_path):
    cfg = stub_cfg(gens=5, budget_usd=1.5, select_every=0)
    t = trainer(tmp_path, cfg, PricedBackend(processes=1))
    s = t.run()  # selection at 0: 8 FLOWN + 8 arm items = $0.16; each generation 8 x 4 + 4 items = $0.36
    assert s["stopped"] == "budget" and s["gen"] == 3 and s["usd"] <= 1.5
    assert lines(t.log_path)[-1]["event"] == "budget stop"
    s = trainer(tmp_path, dict(cfg, budget_usd=10.0), PricedBackend(processes=1)).run()  # a raised budget resumes
    assert s["gen"] == 5 and s["stopped"] == "done"


def test_errors_cost_the_candidate_and_a_failed_flown_seed_is_rerun(tmp_path, monkeypatch):
    real = bev.EPISODE_RUNNER
    failed_flown = []

    def flaky(item):
        if item["arm"] == "FLOWN" and item["seed"] == SELECTION_SEEDS[1] and not failed_flown:
            failed_flown.append(item["seed"])
            raise RuntimeError("flown broke once")
        if item["arm"] != "FLOWN" and item["seed"] % 4 == 0 and item["tag"]["role"] == "cand":
            raise RuntimeError("candidate broke")
        return real(item)

    monkeypatch.setattr(bev, "EPISODE_RUNNER", flaky)
    t = trainer(tmp_path, stub_cfg(gens=1, k=4))
    s = t.run()
    assert s["gen"] == 1 and s["stopped"] == "done" and s["errors"] == 1 + 8  # 1 FLOWN + 1 seed x 8 candidates
    sel = lines(t.sel_path)
    assert sel[0]["excluded"] >= 1 and sel[1]["n"] + sel[1]["excluded"] == 8  # the seed came back and ran
    assert len(t.state["flown"]) == 8 + 4
    log = lines(t.log_path)[0]
    assert log["n_errors"] == 8 and "candidate broke" in log["errors"][0] and log["fit_best"] > 1.0
    assert len(lines(t.err_path)) == 1 + 8


@pytest.mark.parametrize("arm", ["V3-FOLLOW", "V3-JOINT", "V3-LAW"])
def test_best_json_is_a_model_file(tmp_path, arm):
    t = trainer(tmp_path, stub_cfg(arm, gens=1, select_every=1), name=arm)
    s = t.run()
    rec = json.loads(t.best_path.read_text())
    assert s["best_json"] == str(t.best_path) and rec["name"] == f"{arm.lower()}-s1-test"
    tr_ = rec["trained"]
    assert tr_["behind_arm"] == arm and tr_["gen"] == s["best"]["gen"] and tr_["fitness"] == s["best"]["sel_fitness"]
    assert tr_["names"] == list(bp.space(arm).names) and len(tr_["x"]) == DIMS[arm]
    cfg = load_model(t.best_path)
    want, brain_x = bp.make_cfg(arm, tr_["x"])
    a, b = asdict(want), asdict(cfg)
    for k in ("fly_params_path", "model_name"):
        a.pop(k), b.pop(k)
    assert a == b and cfg.model_name == rec["name"]
    if arm == "V3-LAW":
        assert rec["steering"] == "pid" and "x" not in rec
    else:
        assert rec["steering"] == "fly" and rec["arm"] == "FLY-YAW" and rec["brain"] == "pursuit_core1.npz"
        assert rec["x"] == (brain_x if arm == "V3-JOINT" else SMOOTH2_X)
        x, brain_arm, _ = load_params(t.best_path)  # the app's FlySteer loader reads the brain from it
        assert brain_arm == "FLY-YAW" and [float(v) for v in x] == rec["x"]


# ---------------------------------------------------------------------------------------------------------------
# Projection and CLI
# ---------------------------------------------------------------------------------------------------------------


def test_the_projection_counts_the_plan():
    p = tr.project_run(tr.make_run_cfg("V3-JOINT", 1), ct.Prices(), chunk=48)
    assert p["episodes_arm"] == 100 * 40 * 32 + 21 * 96 and p["episodes_flown"] == 100 * 32 + 96
    assert p["s_per_episode"] == p["s_per_flown_episode"] == 3.5 and p["chunks_per_gen"] == 28
    assert 0 < p["eval_usd"] < 40 and 0 < p["wall_h"] < 24
    law = tr.project_run(tr.make_run_cfg("V3-LAW", 1), ct.Prices())
    assert law["s_per_episode"] == 0.4 and law["total_usd"] < p["total_usd"] / 5
    assert law["s_max"] > ct.TAIL_X * 0.4  # a chunk's slowest worker runs one 3.5 s FLOWN episode
    half = tr.project_run(tr.make_run_cfg("V3-JOINT", 1, episode_s=30.0), ct.Prices(), chunk=48)
    assert half["s_per_episode"] == 1.75 and half["total_usd"] < p["total_usd"]


def test_auto_chunk_avoids_a_second_wave():
    cfg = tr.make_run_cfg("V3-JOINT", 1)
    for spec in (ct.EVAL_SPEC, tr.EVAL_SPEC):  # kite.fly's 40-container cap, and kite-behind's own
        chunk = tr.auto_chunk(cfg, ct.Prices(), spec=spec)
        best = tr.project_run(cfg, ct.Prices(), chunk=chunk, spec=spec)
        assert best["chunks_per_gen"] <= spec["max_containers"]
        for m in tr.CHUNK_ROUNDS:
            assert best["total_usd"] <= tr.project_run(cfg, ct.Prices(), chunk=16 * m, spec=spec)["total_usd"] + 1e-9
    # 41 chunks of 32 on kite.fly's 40 containers take a second wave
    assert tr.auto_chunk(cfg, ct.Prices(), spec=ct.EVAL_SPEC) != 32
    assert tr.EVAL_SPEC["max_containers"] == tr.max_containers() and tr.EVAL_SPEC["workers"] == 16


def test_dry_run_prints_the_projection_and_the_plan(capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("KITE_BEHIND_RUNS", str(tmp_path))
    assert tr.main(["--arm", "V3-JOINT", "--cma-seed", "2", "--gens", "10", "--dry-run", "--prices", "constants"]) == 0
    out = capsys.readouterr().out
    cfg = json.loads(out.splitlines()[0])
    assert cfg["arm"] == "V3-JOINT" and cfg["gens"] == 10 and cfg["pop"] == 40 and cfg["k"] == 32
    assert "projection on Modal: V3-JOINT: 10 gens" in out and "plan total $" in out
    assert out.count("  plan V3-") == 10 and not list(tmp_path.iterdir())  # 4 + 4 + 2 runs, nothing written
    assert tr.runs_root() == tmp_path


def test_cli_runs_a_stub_locally(tmp_path, capsys):
    rc = tr.main(["--arm", "V3-LAW", "--gens", "1", "--pop", "4", "--k", "2", "--n-sel", "4", "--stub-s", "0",
                  "--processes", "1", "--tag", "smoke", "--prices", "constants", "--runs-dir", str(tmp_path),
                  "--fitness-json", '{"w_collided": 5.0}'])
    assert rc == 0
    run = tmp_path / "V3-LAW_s1_smoke"
    assert (run / "best.json").exists() and (run / "state.pkl").exists()
    assert json.loads((run / "config.json").read_text())["fitness_cfg"]["w_collided"] == 5.0
    assert '"stopped": "done"' in capsys.readouterr().out


def test_extra_speed_comes_only_through_the_person_speed_gate():
    """The search space cannot buy speed whatever the person does: the base cap tops out at the flown 35, so any
    forward stick above it needs the gated rise (fwd_cap_per_mps, fwd_cap_fast). Nor can it switch off the back-away
    from a person walking at the drone (approach_rate_mps <= 0.3), or let the drone sag toward head height."""
    rows = {r[0]: r for r in bp.FOLLOW_KNOBS}
    assert rows["max_rc_forward"][2] == 35.0 == TunedCfg().max_rc_forward
    assert rows["approach_rate_mps"][2] <= 0.3 and rows["alt_deadband_m"][2] <= 0.08
    assert rows["fwd_cap_v0_mps"][1] >= 0.3  # "faster" starts above a slow walk
    for arm in bp.TUNED_ARMS:
        hi, _ = bp.make_cfg(arm, np.ones(bp.space(arm).dim))
        assert hi.max_rc_forward == 35 and hi.forward_cap(None) == 35 and hi.forward_cap(0.3) == 35


def test_seeds_alternate_train_and_slow_and_flown_is_cached_per_profile(tmp_path):
    t = trainer(tmp_path, stub_cfg(gens=1))
    assert t.cfg["profile"] == "train,slow" and t.profiles == ("train", "slow")
    assert t.slots([5, 6, 7]) == [(5, "train"), (6, "slow"), (7, "train")]
    t.run()
    keys = list(t.state["flown"])
    assert {k.split("|")[1] for k in keys} == {"train", "slow"} and all(k.endswith(t.episode_code) for k in keys)
    log = lines(t.sel_path)
    assert set(log[0]["by_profile"]) == {"train", "slow"}
    with pytest.raises(ValueError, match="profile"):
        stub_cfg(profile="train,street")


def test_resume_refuses_a_code_change_unless_allowed(tmp_path, monkeypatch):
    """A run started on other code (the sim, the follow, the bounds, the fitness) must not continue silently: its
    cached FLOWN metrics and its CMA state were scored by another objective."""
    t = trainer(tmp_path, stub_cfg(gens=1))
    t.run()
    n_flown = len(t.state["flown"])
    monkeypatch.setattr(tr, "run_digest", lambda: "other-code")
    monkeypatch.setattr(tr, "episode_digest", lambda: "other-episode-code")
    with pytest.raises(RuntimeError, match="code changed"):
        trainer(tmp_path, stub_cfg(gens=2)).load_or_init()
    s = trainer(tmp_path, stub_cfg(gens=2, allow_code_change=True))
    out = s.run()
    assert out["gen"] == 2 and s.state["code"] == "other-code" and s.state["code_changes"][0]["gen"] == 1
    assert any(r.get("event") == "code change" for r in lines(s.log_path))
    # FLOWN was re-run on the new code: its old entries are keyed by the old episode digest and never read
    assert sum(k.endswith("other-episode-code") for k in s.state["flown"]) == 4 + 8  # gen 2's K + the last selection
    assert len(s.state["flown"]) == n_flown + 12


class FakeMap:
    """A Modal function's .map over chunks: every chunk took wall_s on its container, but the call's wall (the
    caller's clock) also holds the time the chunks queued behind another run's (the launch's shared containers)."""

    def __init__(self, wall_s: float, n_containers: int, queue_s: float):
        self.wall_s, self.n, self.queue_s = wall_s, n_containers, queue_s

    def map(self, chunks, order_outputs=True, return_exceptions=True):
        import time as _t

        _t.sleep(self.queue_s)
        for i, ch in enumerate(chunks):
            res = [dict(bev.run_item(it)) for it in ch]
            yield {"results": res, "wall_s": self.wall_s * (1.0 + 0.1 * (i % 3)), "t_start": _t.time(),
                   "container_id": f"c{i % self.n}", "cold": False}


def test_modal_metering_bills_a_runs_own_chunks_not_the_shared_queue():
    items = [bev.make_item("V3-LAW", 100 + i, stub_s=0.0, tag={"i": i}) for i in range(12)]
    be = tr.BehindModalBackend(FakeMap(0.1, 4, 0.5), chunk=2, prices=ct.Prices(), spec=tr.EVAL_SPEC)
    res, st = be.evaluate(items)
    assert len(res) == 12 and all(r["ok"] for r in res)
    busy, slowest = st["busy_s"], st["chunk_wall_max_s"]
    assert st["container_s"] == pytest.approx(max(busy, 4 * slowest))
    assert st["container_s_shared"] >= 4 * 0.5  # kite.fly's meter: every container x the call's wall, queue included
    assert st["container_s"] < 0.5 * st["container_s_shared"]
    assert st["usd"] == pytest.approx(st["container_s"] * ct.eval_usd_per_s(ct.Prices(), tr.EVAL_SPEC) * ct.OVERHEAD)


def test_cli_chunk_is_the_launchs_unless_asked():
    cfg = tr.make_run_cfg("V3-JOINT", 1)
    assert tr.pick_chunk(None, cfg, ct.Prices(), None, None) == tr.default_chunk("V3-JOINT") == 32
    assert tr.pick_chunk("48", cfg, ct.Prices(), None, None) == 48
    assert tr.pick_chunk("auto", cfg, ct.Prices(), None, None) == tr.auto_chunk(cfg, ct.Prices())


def test_round_two_arms_keep_the_speed_up_and_warm_start_from_a_model(tmp_path):
    """V3B: the same knobs as V3 except a cap of 50-63 when the person-speed gate opens (rising >= 40 per m/s), and
    a warm start from a round-1 model file whose out-of-range knobs are clipped onto those bounds."""
    from kite.follow_tuned import save_model

    assert bp.space("V3B-FOLLOW").dim == DIMS["V3-FOLLOW"] and bp.space("V3B-JOINT").dim == DIMS["V3-JOINT"]
    rows = {r[0]: r for r in bp.ARM_KNOBS["V3B-FOLLOW"]}
    assert rows["fwd_cap_fast"][1:3] == (50.0, 63.0) and rows["fwd_cap_per_mps"][1:3] == (40.0, 150.0)
    assert rows["max_rc_forward"][2] == 35.0  # below the gate it is never faster than the flown follow
    follow = bp.model_follow("V3-JOINT", bp.init_x("V3-JOINT"))
    follow.update(fwd_cap_fast=41, fwd_cap_per_mps=2.0, dist_gain=52.0)
    path = save_model(tmp_path / "r1.json", name="r1", follow=follow, brain_x=bp.smooth2_x(), steering="fly",
                      trained={"behind_arm": "V3-JOINT"})
    cfg = tr.make_run_cfg("V3B-JOINT", 1, "t", init=str(path))
    assert cfg["init"]["source"] == str(path) and len(cfg["init"]["x"]) == 34
    x0 = np.clip(bp.encode_model("V3B-JOINT", cfg["init"]), 0.0, 1.0)
    d = bp.space("V3B-JOINT").decode(x0)
    assert d["fwd_cap_fast"] == pytest.approx(50.0) and d["fwd_cap_per_mps"] == pytest.approx(40.0)
    assert d["dist_gain"] == pytest.approx(52.0)
    assert bp.space("V3B-JOINT").brain_x(x0) == pytest.approx(bp.smooth2_x())

