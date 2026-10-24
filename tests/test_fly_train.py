"""kite.fly.train without the simulator or the brain: parameter spaces, the arm adapters (against stubs), the R1
yaw fitness, CMA-ES runs on a synthetic objective (improvement, exact resume, budget stop, size-mode probe), the
paired bootstrap and the G2 / G3 gate logic on synthetic episodes, the cost projection and the Modal image probe."""

import json
import math
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from kite.fly.interfaces import (BASELINE_SEEDS, METRICS, PREV_CALIBRATION, PREV_PARAMS, SELECTION_SEEDS, TEST_SEEDS,
                                 TRAIN_SEED_BASE)
from kite.fly.interfaces import EpisodeResult
from kite.fly.train import arms
from kite.fly.train import cma_train as ct
from kite.fly.train import evaluate as ev
from kite.fly.train import report as rp

# ---------------------------------------------------------------------------------------------------------------
# Parameter spaces and arms
# ---------------------------------------------------------------------------------------------------------------


def test_law_space_round_trip_and_defaults():
    sp = arms.LAW_SPACE
    assert sp.names == ("k_yaw", "db_yaw_deg") and sp.dim == 2
    d = sp.decode(sp.default_x())
    assert d["k_yaw"] == pytest.approx(1.5) and d["db_yaw_deg"] == pytest.approx(2.0)
    lo, hi = sp.decode(np.zeros(2)), sp.decode(np.ones(2))
    assert (lo["k_yaw"], hi["k_yaw"], lo["db_yaw_deg"], hi["db_yaw_deg"]) == pytest.approx((0.5, 4.0, 0.0, 5.0))
    assert sp.decode(np.full(2, 0.5))["k_yaw"] == pytest.approx(math.sqrt(0.5 * 4.0))  # log: geometric midpoint
    for x in np.random.default_rng(0).random((20, 2)):
        assert sp.encode(sp.decode(x)) == pytest.approx(x)
    assert sp.decode(np.array([2.0, -1.0])) == sp.decode(np.array([1.0, 0.0]))  # decode clips to the box
    with pytest.raises(ValueError):
        sp.decode(np.array([0.5]))


def test_spaces_per_arm():
    from kite.fly.params import PD_SPACE, SPACE

    assert arms.param_space("FLY-YAW") is SPACE and SPACE.dim == 34
    assert arms.param_space("PRED-PD") is PD_SPACE and PD_SPACE.names == ("kp", "kd", "deadband_deg")
    for arm in ("FLY-YAW", "PRED-PD", "KITE-LAW-CMA"):
        sp = arms.param_space(arm)
        for x in np.random.default_rng(1).random((5, sp.dim)):
            assert sp.encode(sp.decode(x)) == pytest.approx(x)
    with pytest.raises(ValueError):
        arms.param_space("KITE-LAW")


def test_init_points():
    from kite.fly.params import HAND_FILE, load_norm

    prev = json.loads(PREV_PARAMS.read_text())
    p = arms.init_point("FLY-YAW", "prev")
    assert p["x0"] == pytest.approx(prev["x"], abs=1e-9) and p["source"] == str(PREV_PARAMS)
    assert list(p["norm"].values()) == pytest.approx(load_norm(PREV_CALIBRATION).tolist())
    h = arms.init_point("FLY-YAW", "hand")
    assert h["source"] == str(HAND_FILE) and h["norm"] == pytest.approx(json.loads(HAND_FILE.read_text())["norm"])
    assert arms.LAW_SPACE.decode(np.asarray(arms.init_point("KITE-LAW-CMA", "prev")["x0"])) == pytest.approx(
        {"k_yaw": 1.5, "db_yaw_deg": 2.0})
    assert arms.param_space("PRED-PD").decode(np.asarray(arms.init_point("PRED-PD", "default")["x0"])) == \
        pytest.approx({"kp": 1.5, "kd": 0.1, "deadband_deg": 2.0})
    with pytest.raises(ValueError):
        arms.init_point("FLY-YAW-PREV", "prev")


def test_follow_cfg_per_arm():
    from kite.fly.sim.world import default_follow_cfg

    for arm in ("KITE-LAW", "PRED-PD", "FLY-YAW-PREV", "FLY-YAW"):
        c = arms.make_follow_cfg(arm)
        assert c == default_follow_cfg()
        assert (c.behind_m, c.above_m, c.max_yaw, c.k_yaw, c.db_yaw_deg) == (1.0, 0.30, 70, 1.5, 2.0)
    assert arms.make_follow_cfg("KITE-LAW") is not arms.make_follow_cfg("KITE-LAW")  # one instance per episode
    c = arms.make_follow_cfg("KITE-LAW-CMA", np.array([1.0, 0.0]), behind_m=1.5)
    assert (c.k_yaw, c.db_yaw_deg, c.behind_m, c.max_yaw) == pytest.approx((4.0, 0.0, 1.5, 70))
    c = arms.make_follow_cfg("KITE-LAW-CMA", params={"k_yaw": 2.2, "db_yaw_deg": 1.0})
    assert (c.k_yaw, c.db_yaw_deg) == (2.2, 1.0)
    with pytest.raises(ValueError):
        arms.make_follow_cfg("KITE-LAW-CMA")


@pytest.fixture
def sim_calls(monkeypatch):
    """arms with the simulator replaced by a recorder (the yaw sources are the real FlySteer / PredPD)."""
    calls = []

    def sim(**kw):
        calls.append(kw)
        return EpisodeResult(seed=kw["seed"], profile=kw["profile"], arm=kw["arm"], ret=-20.0)

    monkeypatch.setattr(arms, "_sim_run_episode", sim)
    return calls


def test_run_item_wires_each_arm_to_the_real_yaw_sources(sim_calls):
    from kite.fly.steer import FlySteer, PredPD

    arms.run_item(ev.make_item("KITE-LAW", 5, "demo", episode_s=30, overrides={"fx_px": 1000.0}))
    kw = sim_calls[-1]
    assert kw["yaw_source"] is None and kw["follow_cfg"].k_yaw == 1.5 and kw["follow_cfg"].max_yaw == 70
    assert (kw["seed"], kw["profile"], kw["episode_s"], kw["arm"], kw["overrides"]) == (5, "demo", 30.0, "KITE-LAW",
                                                                                         {"fx_px": 1000.0})
    arms.run_item(ev.make_item("KITE-LAW-CMA", 5, x=[1.0, 0.0]))
    assert sim_calls[-1]["yaw_source"] is None and sim_calls[-1]["follow_cfg"].k_yaw == pytest.approx(4.0)
    arms.run_item(ev.make_item("FLY-YAW-PREV", 7, size_blind=True, behind_m=1.5))
    kw = sim_calls[-1]
    ys = kw["yaw_source"]
    assert isinstance(ys, FlySteer) and ys.cfg is kw["follow_cfg"] and ys.behind_m == 1.5 == kw["follow_cfg"].behind_m
    assert ys.size_blind is True and ys.seed == 7
    prev = json.loads(PREV_PARAMS.read_text())["params"]
    assert all(ys.params[k] == pytest.approx(v) for k, v in prev.items())
    norm = arms.init_point("FLY-YAW", "prev")["norm"]
    x = np.asarray(arms.init_point("FLY-YAW", "prev")["x0"])
    arms.run_item(ev.make_item("FLY-YAW", 8, x=x, norm=norm, size_blind=False))
    ys = sim_calls[-1]["yaw_source"]
    assert isinstance(ys, FlySteer) and ys.size_blind is False and ys.cfg is sim_calls[-1]["follow_cfg"]
    with pytest.raises(ValueError):  # a candidate without its readout norm is refused
        arms.run_item(ev.make_item("FLY-YAW", 8, x=x))
    arms.run_item(ev.make_item("PRED-PD", 9, x=[1.0, 0.5, 0.25]))
    ys = sim_calls[-1]["yaw_source"]
    assert isinstance(ys, PredPD) and ys.cfg is sim_calls[-1]["follow_cfg"]
    assert ys.params == pytest.approx({"kp": 4.0, "kd": 0.5, "deadband_deg": 1.0})
    arms.run_item(ev.make_item("PRED-PD", 9))
    assert sim_calls[-1]["yaw_source"].params == pytest.approx({"kp": 1.5, "kd": 0.1, "deadband_deg": 2.0})


class FakeSteer:
    def __init__(self, cfg, behind_m=None, own_cfg=None):
        self.cfg = own_cfg or cfg
        self.behind_m = cfg.behind_m if behind_m is None else behind_m
        self.video_lag_s = cfg.video_lag_s


def test_a_yaw_source_must_share_the_controller_cfg():
    cfg = arms.make_follow_cfg("FLY-YAW")
    assert arms.check_shared_cfg(FakeSteer(cfg), cfg)
    with pytest.raises(AssertionError, match="behind_m"):
        arms.check_shared_cfg(FakeSteer(cfg, behind_m=1.2), cfg)
    with pytest.raises(AssertionError, match="instance"):
        arms.check_shared_cfg(FakeSteer(cfg, own_cfg=arms.make_follow_cfg("FLY-YAW")), cfg)


# ---------------------------------------------------------------------------------------------------------------
# Fitness, seeds, items
# ---------------------------------------------------------------------------------------------------------------


def ep(e_x=0.0, jerk=0.0, lost=0.0, ev_loss=0.0, collision=0.0, ok=True, stick=1.0, following=0.8, **other):
    """An episode result with signed, weighted terms as the sim stores them."""
    terms = {"e_x": e_x, "jerk": jerk, "lost": lost, "ev_loss": ev_loss, "e_z": -50.0, "too_close": 0.0,
             "ev_too_close": 0.0, "ev_collision": -200.0 if collision else 0.0, **other}
    return {"ok": ok, "ret": sum(terms.values()), "terms": terms,
            "metrics": {"collision": collision, "stick_change": stick, "following": following}}


def test_yaw_return_is_the_sum_of_the_yaw_terms():
    r_yaw, col = ct.yaw_return(ep(e_x=-6.0, jerk=-1.0, lost=-2.0, ev_loss=-10.0))
    assert r_yaw == pytest.approx(-19.0) and col is False  # e_z, too_close and their events do not count
    assert ct.yaw_return(ep(e_x=-1.0, collision=1.0))[1] is True
    assert ct.yaw_return(ep(ok=False)) == (None, False)
    with pytest.raises(ValueError, match="signed"):
        ct.yaw_return(ep(e_x=+6.0))  # raw (unsigned) sums would mean the sim changed its convention
    with pytest.raises(ValueError, match="lack"):
        ct.yaw_return({"ok": True, "ret": -1.0, "terms": {"e_x": -1.0}, "metrics": {}})
    assert ct.law_record(ep(e_x=-7.0, stick=0.4)) == {"r_yaw": -7.0, "stick": (0.4, 0.8), "start_failed": False}


def test_the_sims_terms_give_the_r1_formula():
    from kite.fly.sim import reward

    cfg = reward.RewardCfg()
    rw = reward.Reward(cfg, z_ref=1.0, min_range_m=0.7)
    dt, steps = 0.1, [(1.0, 0.10, 3.0, False), (1.2, -0.20, -5.0, False), (1.4, 0.05, 0.0, True), (1.0, 0.0, 8.0, True),
                      (1.0, 0.3, 1.0, False), (1.0, 0.0, 0.0, True)]
    for z, b, dyaw, lost in steps:
        rw.step(dt, z, b, dyaw, lost)
    explicit = (-(cfg.w_x * sum(b * b for _, b, _, _ in steps) + cfg.w_j * sum(abs(d) / 100 for _, _, d, _ in steps)
                  + cfg.w_l * sum(lo for *_, lo in steps)) * dt + 2 * cfg.ev_loss)  # two separate losses
    r = {"ok": True, "ret": rw.total, "terms": dict(rw.sums), "metrics": {}}
    assert ct.yaw_return(r)[0] == pytest.approx(explicit)


def test_episode_score_r1():
    assert ct.episode_score(-50.0, -50.0) == pytest.approx(-1.0)  # KITE-LAW itself scores -1
    assert ct.episode_score(-25.0, -50.0) == pytest.approx(-0.5)
    assert ct.episode_score(-2.0, -1.0) == pytest.approx(-0.4)  # |law| < 5: the floor of 5
    assert ct.episode_score(-1000.0, -50.0) == -4.0  # clipped below
    assert ct.episode_score(0.0, -50.0) == 0.0  # upper clip 0
    assert ct.episode_score(-10.0, -50.0, collided=True) == -4.0  # any collision scores -4, flat
    assert ct.episode_score(None, -50.0) == -4.0 and ct.episode_score(float("nan"), -50.0) == -4.0
    assert ct.episode_score(-20.0, None) == pytest.approx(-4.0)  # failed normalizer: the floor (-20 / 5, clipped)
    assert ct.fitness([], []) == ct.ERROR_SCORE


def law(r_yaw=-50.0, stick=0.5, following=0.8, start_failed=False):
    return {"r_yaw": r_yaw, "stick": (stick, following), "start_failed": start_failed}


def test_one_stick_penalty_on_the_following_weighted_means():
    # no penalty while the arm's mean stick stays within 1.8 x KITE-LAW's
    res = [ep(e_x=-50.0, stick=0.9), ep(e_x=-25.0, stick=0.9)]
    p = ct.fitness_parts(res, [law(), law()])
    assert (p["score"], p["penalty"], p["fitness"]) == pytest.approx((-0.75, 0.0, -0.75))
    # beyond it: once per candidate, S_arm / S_law - 1.8
    p = ct.fitness_parts([ep(e_x=-50.0, stick=1.5), ep(e_x=-25.0, stick=1.5)], [law(), law()])
    assert p["penalty"] == pytest.approx(1.5 / 0.5 - 1.8) and p["fitness"] == pytest.approx(-0.75 - 1.2)
    # the means are weighted by following steps: one wild episode that barely followed cannot dominate
    wild = [ep(e_x=-50.0, stick=0.6, following=0.9) for _ in range(23)] + [ep(e_x=-50.0, stick=11.0, following=0.02)]
    p = ct.fitness_parts(wild, [law(stick=0.5, following=0.9)] * 24)
    s_arm = (23 * 0.6 * 0.9 + 11.0 * 0.02) / (23 * 0.9 + 0.02)
    assert p["s_arm"] == pytest.approx(s_arm) and p["penalty"] == 0.0 and p["fitness"] == pytest.approx(-1.0)
    assert ct.fitness([ep(e_x=-50.0, stick=0.5, following=0.0)] * 2 + [ep(e_x=-50.0, stick=0.5)],
                      [law()] * 3) == pytest.approx(-1.0)  # 0-step episodes contribute nothing
    # the cap: never more than 4, even against a KITE-LAW that hardly moved its stick (floored at 0.1)
    p = ct.fitness_parts([ep(e_x=-50.0, stick=50.0)], [law(stick=0.0)])
    assert p["penalty"] == 4.0 and p["fitness"] == pytest.approx(-5.0)
    assert ct.stick_penalty(1.0, None) == 4.0 and ct.stick_penalty(None, 0.5) == 0.0  # law / arm never followed
    # failed starts leave both stick means alone
    bad = ep(e_x=-5.0, stick=40.0)
    bad["metrics"]["start_failed"] = 1.0
    assert ct.fitness_parts([ep(e_x=-50.0, stick=0.5), bad], [law(), law()])["penalty"] == 0.0


def test_failed_starts_are_excluded_from_the_fitness_not_punished():
    bad = ep(e_x=-5.0, collision=1.0)
    bad["metrics"]["start_failed"] = 1.0
    law_ok, law_bad = law(stick=1.0), law(stick=1.0, start_failed=True)
    assert ct.start_failed(bad) and not ct.start_failed(ep())
    assert ct.start_failed({"ok": True, "start_failed": True, "metrics": {}})  # a top-level flag works too
    assert ct.fitness([ep(e_x=-25.0), bad], [law_ok, law_ok]) == pytest.approx((-0.5 + 0.0) / 2)  # 0, not -4
    assert ct.fitness([ep(e_x=-25.0), ep(e_x=-1.0)], [law_ok, law_bad]) == pytest.approx(-0.25)  # the law's start
    assert ct.law_record(bad)["start_failed"] is True and "collision" not in ct.metric_means([bad])


def test_train_seeds_are_common_fresh_and_disjoint_per_cma_seed():
    assert ct.train_seeds(1, 24, 1) == list(range(TRAIN_SEED_BASE + 100_000 + 24, TRAIN_SEED_BASE + 100_000 + 48))
    assert ct.train_seeds(3, 4, 2) == [TRAIN_SEED_BASE + 200_000 + 12 + k for k in range(4)]
    seen = set()
    for c in (1, 2, 3):
        for g in range(1, 61):
            s = set(ct.train_seeds(g, 24, c))
            assert not s & seen
            seen |= s
    assert not seen & (set(SELECTION_SEEDS) | set(TEST_SEEDS) | set(BASELINE_SEEDS))
    with pytest.raises(ValueError):
        ct.train_seeds(0, 24, 1)
    with pytest.raises(ValueError):
        ct.train_seeds(5000, 24, 1)  # would run into CMA seed 2's range


def test_r9_run_defaults():
    cfgs = [ct.make_cfg("FLY-YAW", c) for c in (1, 2, 3)]
    assert [c["init"] for c in cfgs] == ["prev", "prev", "hand"] and all(c["size_blind"] == "on" for c in cfgs)
    assert ct.make_cfg("FLY-YAW", 3, init="prev")["init"] == "prev"
    assert ct.make_cfg("PRED-PD", 1)["size_blind"] == "off" and ct.make_cfg("PRED-PD", 1)["gens"] == 30


def test_evaluate_item_never_raises(monkeypatch):
    item = ev.make_item("KITE-LAW", 3, tag={"role": "law"})

    def boom(it):
        raise RuntimeError("sim exploded")

    monkeypatch.setattr(ev, "EPISODE_RUNNER", boom)
    r = ev.evaluate_item(item)
    assert r["ok"] is False and "sim exploded" in r["error"] and r["tag"] == {"role": "law"} and r["seed"] == 3
    monkeypatch.setattr(ev, "EPISODE_RUNNER", lambda it: EpisodeResult(3, "train", "KITE-LAW", float("inf")))
    assert ev.evaluate_item(item)["ok"] is False
    monkeypatch.setattr(ev, "EPISODE_RUNNER",
                        lambda it: EpisodeResult(3, "train", "KITE-LAW", -12.5, metrics={"losses": np.float64(1)},
                                                 trace={"t": [0.0]}))
    r = ev.evaluate_item(item)
    assert r["ok"] and r["ret"] == -12.5 and type(r["metrics"]["losses"]) is float and "trace" not in r
    pickle.dumps(r)


def test_process_pool_keeps_item_order():
    items = [ev.make_item("STUB", 100 + i, x=[0.3] * 4, stub_s=0.0, tag={"i": i}) for i in range(6)]
    out = ev.evaluate_items(items, processes=2)
    ev.shutdown_pool()
    assert [r["tag"]["i"] for r in out] == list(range(6)) and all(r["ok"] for r in out)
    assert len({r["pid"] for r in out}) >= 1 and out == sorted(out, key=lambda r: r["seed"])


# ---------------------------------------------------------------------------------------------------------------
# CMA-ES runs on the stub objective
# ---------------------------------------------------------------------------------------------------------------


def stub_cfg(**over):
    base = {"gens": 2, "pop": 8, "k": 4, "sigma0": 0.2, "select_every": 1, "n_sel": 8, "stub_s": 0.0}
    base.update(over)
    return ct.make_cfg("STUB", 1, "test", **base)


def trainer(tmp_path, cfg=None, backend=None, name="run"):
    return ct.Trainer(cfg or stub_cfg(), backend or ct.LocalBackend(processes=1), tmp_path / name, log=lambda s: None)


def logs(path):
    return [json.loads(x) for x in (path / "log.jsonl").read_text().splitlines()]


def test_two_generations_improve_on_the_stub(tmp_path):
    t = trainer(tmp_path)
    s = t.run()
    assert s["gen"] == 2 and s["stopped"] == "done" and s["errors"] == 0
    sel = [json.loads(x) for x in (t.run_dir / "selection.jsonl").read_text().splitlines()]
    assert [r["gen"] for r in sel] == [0, 1, 2]
    assert sel[0]["sel_fitness"] == pytest.approx(-1.0)  # the init is the stub's law: -1 by construction
    assert s["best"]["sel_fitness"] > sel[0]["sel_fitness"] + 0.2
    best = json.loads((t.run_dir / "best.json").read_text())
    assert best["arm"] == "STUB" and best["gen"] == s["best"]["gen"] and len(best["x"]) == 4
    assert set(best["params"]) == set(arms.STUB_SPACE.names) and "rms_bearing_deg" in best["metric_means"]
    log = logs(t.run_dir)
    assert [r["gen"] for r in log] == [1, 2] and log[0]["n_items"] == 8 * 4 + 4 and log[1]["n_law"] == 4
    assert log[0]["seeds"] == [TRAIN_SEED_BASE + 100_000 + 4, TRAIN_SEED_BASE + 100_000 + 7]


def test_resume_reproduces_the_next_generations(tmp_path):
    cfg = stub_cfg(gens=3)
    straight = trainer(tmp_path, cfg, name="a")
    straight.run()
    crashed = trainer(tmp_path, cfg, name="b")
    crashed.run(stop_after=1)
    assert crashed.state["gen"] == 1
    np.random.seed(12345)
    np.random.rand(7)  # the global RNG must not matter
    resumed = trainer(tmp_path, cfg, name="b")
    assert resumed.load_or_init() is True
    resumed.run()
    a, b = logs(straight.run_dir), logs(resumed.run_dir)
    assert [r["gen"] for r in b] == [1, 2, 3]
    for ra, rb in zip(a, b):
        for k in ("fit_best", "fit_median", "fit_mean", "sigma", "mean_x", "seeds"):
            assert ra[k] == rb[k], k
    ba, bb = (json.loads((t.run_dir / "best.json").read_text()) for t in (straight, resumed))
    assert ba["x"] == bb["x"] and ba["sel_fitness"] == bb["sel_fitness"]


def test_resume_drops_a_generation_logged_but_not_saved_and_refuses_a_changed_config(tmp_path):
    t = trainer(tmp_path, stub_cfg(gens=1))
    t.run()
    ct.append_jsonl(t.log_path, {"gen": 2, "fit_best": 0.0})  # crash between log and checkpoint
    again = trainer(tmp_path, stub_cfg(gens=1))
    again.load_or_init()
    assert [r["gen"] for r in logs(t.run_dir)] == [1]
    with pytest.raises(RuntimeError, match="pop"):
        trainer(tmp_path, stub_cfg(gens=1, pop=6)).load_or_init()


def test_best_json_loads_with_the_steer_classes(tmp_path):
    from kite.fly.params import load_params
    from kite.fly.steer import FlySteer, PredPD

    fly = ct.Trainer(ct.make_cfg("FLY-YAW", 3, "t", gens=1, pop=4, k=2, n_sel=4, stub_s=0.0),
                     ct.LocalBackend(processes=1), tmp_path / "fly", log=lambda s: None)
    fly.run()
    best = fly.best_path
    pf = load_params(best)
    rec = json.loads(best.read_text())
    assert pf.arm == "FLY-YAW" and pf.size_blind is True and pf.norm_from == best  # norm and size mode in the file
    assert rec["init"] == "hand" and rec["norm"] == pytest.approx(arms.init_point("FLY-YAW", "hand")["norm"])
    cfg = arms.make_follow_cfg("FLY-YAW")
    steer = FlySteer.from_file(best, cfg=cfg)
    assert steer.size_blind is True and steer.cfg is cfg
    assert all(steer.params[k] == pytest.approx(v) for k, v in rec["params"].items())
    pd = ct.Trainer(ct.make_cfg("PRED-PD", 1, "t", gens=1, pop=4, k=2, n_sel=4, stub_s=0.0),
                    ct.LocalBackend(processes=1), tmp_path / "pd", log=lambda s: None)
    pd.run()
    rec = json.loads(pd.best_path.read_text())
    steer = PredPD.from_file(pd.best_path, cfg=cfg)
    assert steer.params == pytest.approx(rec["params"]) and set(rec["params"]) == {"kp", "kd", "deadband_deg"}


class PricedBackend(ct.LocalBackend):
    def evaluate(self, items):
        res, st = super().evaluate(items)
        st["usd"] = 0.01 * len(items)
        return res, st


def test_budget_stops_before_overspending(tmp_path):
    cfg = stub_cfg(gens=5, budget_usd=1.5, select_every=0)
    t = trainer(tmp_path, cfg, PricedBackend(processes=1))
    s = t.run()  # selection at 0: 8 + 4 items = $0.12; each generation 32 + 4 items = $0.36
    assert s["stopped"] == "budget" and s["gen"] == 3 and s["usd"] <= 1.5
    assert logs(t.run_dir)[-1]["event"] == "budget stop"
    s = trainer(tmp_path, dict(cfg, budget_usd=10.0), PricedBackend(processes=1)).run()  # a raised budget resumes
    assert s["gen"] == 5 and s["stopped"] == "done"


def test_size_mode_auto_picks_the_lower_rms_bearing(tmp_path, monkeypatch):
    def runner(item):
        if item["arm"] == "FLY-YAW-PREV":
            rms = 4.0 if item["size_blind"] else 6.0
            return EpisodeResult(item["seed"], item["profile"], item["arm"], -20.0, metrics={"rms_bearing_deg": rms})
        return arms.stub_episode(item)

    cfg = ct.make_cfg("FLY-YAW", 1, "t", gens=1, pop=4, k=2, n_sel=4, n_probe=4, stub_s=0.0, size_blind="auto",
                      stds={})
    seen = []
    monkeypatch.setattr(ev, "EPISODE_RUNNER", lambda it: seen.append((it["seed"], it["profile"])) or runner(it))
    t = ct.Trainer(cfg, ct.LocalBackend(processes=1), tmp_path / "fly", space=arms.STUB_SPACE, log=lambda s: None)
    t.load_or_init()
    assert t.state["size_blind"] is True
    assert t.state["size_probe"] == {"blind_rms_bearing_deg": 4.0, "size_rms_bearing_deg": 6.0, "n": 4,
                                     "profiles": ["train", "demo"], "seeds": "baseline"}
    probe = seen[:16]  # G2's decision: baseline seeds, train and demo, both modes
    assert {s for s, _ in probe} == set(BASELINE_SEEDS[:4]) and {p for _, p in probe} == {"train", "demo"}
    cfg_off = dict(cfg, size_blind="off")
    t2 = ct.Trainer(cfg_off, ct.LocalBackend(processes=1), tmp_path / "fly2", space=arms.STUB_SPACE, log=lambda s: 0)
    t2.load_or_init()
    assert t2.state["size_blind"] is False and t2.state["size_probe"] is None


def test_cma_stds_by_prefix():
    names = ("enc_r_max_hz", "dec_w_yaw_DNa02", "dec_b_yaw", "dec_g_yaw")
    assert ct.cma_stds(names, {"dec_w_": 0.08, "dec_b_": 0.2}) == [1.0, 0.08, 0.2, 1.0]
    assert ct.cma_stds(names, {}) is None
    assert ct.parse_stds("dec_w_=0.08, dec_b_=0.2") == {"dec_w_": 0.08, "dec_b_": 0.2}
    assert ct.parse_stds("none") == {} and ct.parse_stds(None) is None


def test_projection_formula():
    p = ct.Prices(0.0473, 0.008)
    usd_s = (8 * 0.0473 + 8 * 0.008) / 3600
    proj = ct.project("FLY-YAW", 60, 24, 24, 5, 3.0, 0.3, p, n_sel=64, chunk=32, n_extra=400, s_max=6.0)
    assert proj["episodes_arm"] == 60 * 24 * 24 + 13 * 64 + 400 and proj["episodes_law"] == 60 * 24 + 64
    assert proj["chunks_per_gen"] == 19 and proj["containers"] == 19
    busy = (proj["episodes_arm"] * 3.0 + proj["episodes_law"] * 0.3) / (16 * ct.PACKING)
    alive = 60 * 19 * (2 * 6.0) + 13 * 2 * (2 * 6.0) + 13 * (2 * 6.0)  # every container waits for the slowest
    assert alive > busy and proj["container_h"] == pytest.approx(alive / 3600)
    assert proj["eval_usd"] == pytest.approx(alive * usd_s * ct.OVERHEAD)
    assert proj["wall_h"] * 3600 == pytest.approx(60 * (12.0 + 4.0) + 13 * (12.0 + 4.0) + 12.0)
    even = ct.project("FLY-YAW", 60, 24, 24, 5, 3.0, 0.3, p, n_sel=64, chunk=32, s_max=1.0)  # busy dominates
    assert even["container_h"] * 3600 == pytest.approx((even["episodes_arm"] * 3.0 + even["episodes_law"] * 0.3)
                                                       / (16 * ct.PACKING))
    assert ct.project("FLY-YAW", 60, 24, 24, 5, 3.0, 0.3, p)["s_max"] == pytest.approx(ct.TAIL_X * 3.0)
    cfg = ct.make_cfg("FLY-YAW", 1, "t", size_blind="auto")
    assert ct.probe_episodes(cfg) == 400 and ct.probe_episodes(ct.make_cfg("FLY-YAW", 1, "t")) == 0
    assert ct.project_cfg(cfg, p, 3.0, 0.3, 32, s_max=6.0)["episodes_arm"] == proj["episodes_arm"]


def test_the_whole_fine_tune_fits_the_budget_with_shares():
    pytest.importorskip("modal")
    from kite.fly.train import modal_app

    runs = modal_app.plan()
    assert [(r["arm"], r["cma_seed"]) for r in runs] == [("FLY-YAW", 1), ("FLY-YAW", 2), ("FLY-YAW", 3),
                                                         ("KITE-LAW-CMA", 1), ("PRED-PD", 1)]
    assert all((r["gens"], r["pop"], r["k"], r["sigma0"], r["size_blind"]) == (60, 24, 24, 0.03, "on")
               for r in runs[:3])
    assert [r["init"] for r in runs[:3]] == ["prev", "prev", "hand"]
    assert all(r["gens"] == 30 for r in runs[3:])
    projs, total = modal_app.split_budget(runs, ct.Prices(), 25.0)  # the conservative episode-time guesses
    assert total <= 25.0 and sum(r["budget_usd"] for r in runs) == pytest.approx(25.0)


# ---------------------------------------------------------------------------------------------------------------
# Statistics and gates on synthetic episodes
# ---------------------------------------------------------------------------------------------------------------


def test_side_sets_parse_and_reach_the_items():
    assert rp.parse_side_set("demo@1.2") == ("demo", 1.2, None)
    assert rp.parse_side_set("demo@p_head0.3") == ("demo", None, {"p_head": 0.3})
    with pytest.raises(SystemExit):
        rp.parse_side_set("demo")
    e = {"label": "KITE-LAW", "arm": "KITE-LAW", "size_blind": None}
    it = rp.entry_items(e, [3000], "demo", 60.0, None, "demo@p_head0.3")[0]
    assert (it["profile"], it["behind_m"], it["overrides"], it["tag"]["set"]) == ("demo", None, {"p_head": 0.3},
                                                                                  "demo@p_head0.3")
    it = rp.entry_items(e, [3000], "demo", 60.0, None, "demo@1.5")[0]
    assert (it["behind_m"], it["overrides"]) == (1.5, None)


def test_bootstrap_ci():
    v = np.random.default_rng(1).normal(1.0, 1.0, 400)
    m, lo, hi = rp.bootstrap_ci(v, n_boot=4000)
    assert m == pytest.approx(v.mean()) and lo < 1.0 < hi
    assert (hi - lo) == pytest.approx(2 * 1.96 / 20, rel=0.15)
    assert rp.bootstrap_ci(v, n_boot=4000) == (m, lo, hi)  # deterministic
    assert math.isnan(rp.bootstrap_ci([])[0]) and math.isnan(rp.bootstrap_ci([2.0])[1])


def episodes(seeds, rng, **means):
    """Synthetic episode records: each metric = mean + a per-seed effect shared across arms (rms bearing) + noise."""
    out = []
    for s in seeds:
        shared = np.random.default_rng([s, 99]).normal(0, 2.0)
        m = {k: 0.0 for k in METRICS}
        for k, v in means.items():
            m[k] = float(v + (shared + rng.normal(0, 0.1) if k == "rms_bearing_deg" else rng.normal(0, 0.01)))
        r_yaw = -20.0 - m["rms_bearing_deg"]
        out.append({"seed": s, "ok": True, "ret": r_yaw - 30.0, "metrics": m,
                    "terms": {"e_x": r_yaw, "jerk": 0.0, "lost": 0.0, "ev_loss": 0.0, "e_z": -30.0}})
    return out


def test_paired_differences_use_the_shared_seed_effect():
    seeds, rng = list(range(50)), np.random.default_rng(0)
    a = episodes(seeds, rng, rms_bearing_deg=5.0)
    b = episodes(seeds, rng, rms_bearing_deg=5.3)
    p = rp.paired(a, b, "rms_bearing_deg", n_boot=2000)
    assert p["n"] == 50 and p["hi"] < 0 and p["mean"] == pytest.approx(-0.3, abs=0.05)
    assert rp.mean_metric(a, "rms_bearing_deg") - rp.mean_metric(b, "rms_bearing_deg") == pytest.approx(p["mean"])
    b[0]["ok"] = False
    assert rp.paired(a, b, "rms_bearing_deg", n_boot=200)["n"] == 49


BASE = {"rms_bearing_deg": 6.0, "losses": 1.0, "stick_change": 2.0, "in_view": 0.9, "following": 0.9,
        "time_lost_s": 3.0, "rms_range_err_m": 0.4, "too_close_s": 0.5, "pitch_exits": 0.3}  # collision stays 0


def world(fly=None, law_cma=None, pred=None, profiles=("train", "demo", "stress", "fast"), n=60):
    rng = np.random.default_rng(3)
    seeds = list(range(2000, 2000 + n))
    arms_means = {"KITE-LAW": BASE, "KITE-LAW-CMA": {**BASE, **(law_cma or {})}, "PRED-PD": {**BASE, **(pred or {})},
                  "FLY-YAW-PREV": {**BASE, "rms_bearing_deg": 5.5}, "FLY-YAW": {**BASE, "losses": 0.9, **(fly or {})}}
    recs = {lab: {p: episodes(seeds, rng, **m) for p in profiles} for lab, m in arms_means.items()}
    entries = [{"label": lab, "arm": lab, "size_blind": None, "params": {"k_yaw": 2.0, "db_yaw_deg": 1.0}}
               for lab in arms_means]
    return recs, entries


def failed(out, lab="FLY-YAW"):
    return [(c["criterion"], c["check"]) for c in out["candidates"][lab]["checks"] if not c["pass"]]


def test_g3_passes_the_fly_when_every_criterion_holds():
    recs, entries = world(fly={"rms_bearing_deg": 4.5, "stick_change": 3.5})
    out = rp.g3(recs, entries, n_boot=1000)
    assert out["candidates"]["FLY-YAW"]["pass"], failed(out)
    assert out["verdict"].startswith("FLY-YAW PASSES G3") and "opt-in" in out["verdict"]


@pytest.mark.parametrize("fly, criterion, check", [
    ({"rms_bearing_deg": 5.6}, 1, "below FLY-YAW-PREV"),  # better than the law, not than the previous fly
    ({"rms_bearing_deg": 5.2}, 1, "below FLY-YAW-PREV"),  # CI below 0 but only 5.5 % lower: < 10 %
    ({"rms_bearing_deg": 4.5, "losses": 1.3}, 2, "losses non-inferior"),
    ({"rms_bearing_deg": 4.5, "time_lost_s": 3.8}, 2, "time_lost_s non-inferior"),  # margin max(10 % = 0.3, 0.5)
    ({"rms_bearing_deg": 4.5, "in_view": 0.87}, 3, "in_view lower CI"),
    ({"rms_bearing_deg": 4.5, "stick_change": 4.5}, 5, "stick_change <= 2 x"),
])
def test_g3_names_the_failed_criterion(fly, criterion, check):
    recs, entries = world(fly=fly)
    out = rp.g3(recs, entries, n_boot=1000)
    assert not out["candidates"]["FLY-YAW"]["pass"]
    assert any(c == criterion and check in name for c, name in failed(out)), failed(out)
    assert criterion in out["candidates"]["FLY-YAW"]["failed_criteria"]
    assert "fails G3 on criteria" in out["verdict"] and "nothing beats KITE-LAW" in out["verdict"]


def test_g3_noninferiority_margin_tolerates_a_small_increase():
    recs, entries = world(fly={"rms_bearing_deg": 4.5, "time_lost_s": 3.2, "losses": 1.05})
    out = rp.g3(recs, entries, n_boot=1000)
    assert out["candidates"]["FLY-YAW"]["pass"], failed(out)


def collide(recs, lab, prof, idx):
    for i in idx:
        recs[lab][prof][i]["metrics"]["collision"] = 1.0


def test_paired_skips_collisions_and_failed_starts_and_says_how_many():
    seeds, rng = list(range(40)), np.random.default_rng(0)
    a, b = episodes(seeds, rng, rms_bearing_deg=5.0), episodes(seeds, rng, rms_bearing_deg=6.0)
    a[0]["metrics"]["collision"] = 1.0
    a[1]["metrics"]["rms_bearing_deg"] = b[1]["metrics"]["rms_bearing_deg"] = 1e6  # skipped: its start failed
    a[1]["start_failed"] = b[1]["start_failed"] = True
    b[2]["metrics"]["collision"] = 1.0
    b[3]["ok"] = False
    p = rp.paired(a, b, "rms_bearing_deg", n_boot=500)
    assert (p["n"], p["n_excluded"]) == (36, 4) and p["mean"] == pytest.approx(-1.0, abs=0.1)
    assert p["mean_a"] - p["mean_b"] == pytest.approx(p["mean"])
    assert rp.collision_pairs(a, b) == {"n": 38, "a_only": 1, "b_only": 1, "both": 0}  # failed start / episode out
    assert "4 seeds excluded" in rp._ci(p)


def test_g3_collisions_seed_by_seed_edges_and_the_honest_outcomes():
    recs, entries = world(fly={"rms_bearing_deg": 4.5}, law_cma={"rms_bearing_deg": 5.0})
    collide(recs, "FLY-YAW", "demo", [7, 8])  # collides alone on 2 seeds: allowed (KITE-LAW's 0 + 2)
    assert rp.g3(recs, entries, n_boot=500)["candidates"]["FLY-YAW"]["pass"]
    collide(recs, "FLY-YAW", "demo", [9])  # 3 > 0 + 2
    out = rp.g3(recs, entries, n_boot=1000)
    assert failed(out) == [(4, "demo: collisions seed by seed vs KITE-LAW")]
    assert "collides alone on 3 seeds" in next(c["detail"] for c in out["candidates"]["FLY-YAW"]["checks"]
                                               if c["criterion"] == 4 and not c["pass"])
    assert out["candidates"]["KITE-LAW-CMA"]["pass"] and "KITE-LAW-CMA beats KITE-LAW (k_yaw 2.000" in out["verdict"]
    collide(recs, "KITE-LAW", "demo", [9, 20])  # shared seed 9 no longer counts; KITE-LAW alone on 20: 2 <= 1 + 2
    assert rp.g3(recs, entries, n_boot=500)["candidates"]["FLY-YAW"]["pass"]
    collide(recs, "FLY-YAW", "stress", [0, 1, 2])  # ... and the same rule on stress and fast
    assert failed(rp.g3(recs, entries, n_boot=500)) == [(6, "stress: rms_bearing <= 1.25 x KITE-LAW, collisions seed "
                                                            "by seed")]
    recs, entries = world(fly={"rms_bearing_deg": 4.5}, profiles=("train", "demo", "stress"))
    out = rp.g3(recs, entries, n_boot=500)
    assert [c for c, _ in failed(out)] == [6] and "fast" in failed(out)[0][1]
    recs, entries = world(fly={"rms_bearing_deg": 4.5})
    for r in recs["FLY-YAW"]["fast"]:
        r["metrics"]["rms_bearing_deg"] *= 2.0  # 9 vs 1.25 x 6
    out = rp.g3(recs, entries, n_boot=500)
    assert [c for c, _ in failed(out)] == [6]
    recs, entries = world(fly={"rms_bearing_deg": 4.6}, pred={"rms_bearing_deg": 5.0})
    out = rp.g3(recs, entries, n_boot=1000)  # PRED-PD beats the law (-17 %), the fly is only 8 % below PRED-PD
    assert out["candidates"]["PRED-PD"]["pass"] and not out["candidates"]["FLY-YAW"]["pass"]
    assert out["verdict"].startswith("prediction helps, the fly does not")
    recs, entries = world(fly={"rms_bearing_deg": 4.5})
    out = rp.g3(recs, [e for e in entries if e["arm"] != "PRED-PD"], n_boot=500)
    assert (1, "comparator PRED-PD") in failed(out)  # every competitor must have been evaluated
    assert rp.g3({}, [])["verdict"].startswith("NO VERDICT")


def test_g2_size_mode_init_and_g1():
    rng = np.random.default_rng(5)
    seeds = list(range(2000, 2080))
    good = {"rms_bearing_deg": 4.0, "p95_bearing_deg": 7.0, "in_view": 0.95, "following": 0.97, "time_lost_s": 0.5,
            "losses": 0.2, "rms_range_err_m": 0.2, "in_band": 0.8, "too_close_s": 0.1, "rms_alt_err_m": 0.05,
            "stick_change": 1.0}
    law = {"rms_bearing_deg": 6.0, "p95_bearing_deg": 10.0, "in_view": 0.90, "following": 0.92, "time_lost_s": 1.0,
           "losses": 0.5, "rms_range_err_m": 0.3, "in_band": 0.7, "too_close_s": 0.3, "rms_alt_err_m": 0.08,
           "stick_change": 1.5}
    recs = {"KITE-LAW": {p: episodes(seeds, rng, **law) for p in ("train", "demo")},
            "FLY-YAW-PREV@blind": {p: episodes(seeds, rng, **good) for p in ("train", "demo")},
            "FLY-YAW-PREV@size": {p: episodes(seeds, rng, **{**good, "rms_bearing_deg": 4.6})
                                  for p in ("train", "demo")}}
    entries = [{"label": "KITE-LAW", "arm": "KITE-LAW", "size_blind": None},
               {"label": "FLY-YAW-PREV@blind", "arm": "FLY-YAW-PREV", "size_blind": True},
               {"label": "FLY-YAW-PREV@size", "arm": "FLY-YAW-PREV", "size_blind": False}]
    out = rp.g2(recs, entries)
    assert out["size_mode"]["choice"] == "FLY-YAW-PREV@blind" and out["size_mode"]["size_blind"] is True
    assert "already_good" not in out  # R6: no shortcut
    assert out["init"]["choice"] == "prev" and out["g1_law"]["demo"]["collisions"] == 0


def test_report_end_to_end_on_stub_episodes(tmp_path):
    best = tmp_path / "law_best.json"
    best.write_text(json.dumps({"arm": "KITE-LAW-CMA", "x": [0.6, 0.3], "params": {"k_yaw": 1.8, "db_yaw_deg": 1.5},
                                "cma_seed": 1, "gen": 5, "sel_fitness": -0.9}))
    common = ["--best", str(best), "--stub-s", "0", "--n", "6", "--profiles", "train,demo", "--processes", "1",
              "--n-boot", "200", "--out", str(tmp_path)]
    assert rp.main(["--arms", "KITE-LAW,FLY-YAW-PREV@blind,FLY-YAW-PREV@size", "--name", "g2", *common]) == 0
    res = json.loads((tmp_path / "g2.json").read_text())
    assert [r["seed"] for r in res["records"]["KITE-LAW"]["demo"]] == list(BASELINE_SEEDS[:6])  # G2: baseline only
    assert res["analysis"]["summary"]["KITE-LAW"]["train"]["fitness"] == pytest.approx(-1.0)
    assert "g2" in res["analysis"] and "g3" not in res["analysis"]
    assert set(res["analysis"]["g2"]["g1_law"]) == {"demo", "demo@p_head0.3"}  # R10.3: G1 at p_head 0.3 too
    assert rp.main(["--arms", "KITE-LAW,KITE-LAW-CMA", "--seeds", "test", "--name", "g3", *common]) == 0
    res = json.loads((tmp_path / "g3.json").read_text())
    assert [r["seed"] for r in res["records"]["KITE-LAW"]["demo"]] == list(TEST_SEEDS[:6])
    assert "g3" in res["analysis"] and "g2" not in res["analysis"]
    assert set(res["records"]["KITE-LAW-CMA"]) == {"train", "demo", "demo@1.2", "demo@1.5", "demo@p_head0.3"}
    assert res["config"]["side_sets"] == ["demo@1.2", "demo@1.5", "demo@p_head0.3"]
    assert "demo@1.5" in res["analysis"]["summary"]["KITE-LAW"]
    assert "## G3" in (tmp_path / "g3.md").read_text()
    assert rp.main(["--from", str(tmp_path / "g3.json"), "--out", str(tmp_path), "--n-boot", "100"]) == 0
    with pytest.raises(SystemExit):
        rp.main(["--arms", "KITE-LAW,PRED-PD", "--stub-s", "0", "--out", str(tmp_path)])  # PRED-PD needs --best


# ---------------------------------------------------------------------------------------------------------------
# Modal image
# ---------------------------------------------------------------------------------------------------------------


def test_modal_image_holds_only_the_episode_path():
    pytest.importorskip("modal")
    from kite.fly.train import modal_app

    files, skipped = modal_app.source_files()
    rel = {f.relative_to(ct.REPO.resolve()).as_posix() for f in files}
    assert {"kite/__init__.py", "kite/follow.py", "kite/camera.py", "kite/types.py", "kite/fly/interfaces.py",
            "kite/fly/train/evaluate.py", "kite/fly/train/modal_app.py", "kite/fly/data/pursuit_core1.npz",
            "kite/fly/data/reference/fly_calibration_prev.json"} <= rel
    assert "kite/record.py" not in rel and not any(r.startswith("tests/") for r in rel)
    assert set(skipped) <= set(modal_app.PROBE_MODULES)
