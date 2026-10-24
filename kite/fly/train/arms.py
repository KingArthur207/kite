"""The one adapter between kite.fly.train and the simulator / fly brain: arm (+ x or a params file) -> FollowCfg and
yaw source, and the parameter spaces CMA-ES searches.

    make_follow_cfg("KITE-LAW-CMA", x)                                      # shared config + tuned k_yaw / deadband
    make_follow_cfg("KITE-LAW", follow={"predict": True})                   # + other FollowCfg fields (a variant)
    make_yaw_source("FLY-YAW", x, norm=norm, size_blind=True, seed=2000, follow_cfg=cfg)   # a kite.follow.YawSource
    run_item({"arm": "FLY-YAW-PREV", "seed": 2000, "profile": "demo"})      # one EpisodeResult

Arms (interfaces.ARMS). Every arm flies the sim's shared FollowCfg(behind_m=1.0, above_m=0.30, max_yaw=70) (DESIGN
R3), or that config with another behind_m (the G3 standoff side report) and other FollowCfg fields (an item's
`follow`: e.g. predict, main's opt-in prediction planner, on the law arms only; pitch_comp); a yaw source is always
built from the very instance the controller uses (asserted):
  KITE-LAW      yaw_source None, the law's defaults (k_yaw 1.5 stick/deg, 2 deg deadband)
  KITE-LAW-CMA  yaw_source None, k_yaw and db_yaw_deg from x in LAW_SPACE
  PRED-PD       kite.fly.steer.PredPD: FlySteer's latency filter and stick conditioning around a PD (x in PD_SPACE)
  FLY-YAW-HAND  kite.fly.steer.make_fly_steer: data/init/fly_yaw_hand.json with its own readout norm
  FLY-YAW-PREV  make_fly_steer: interfaces.PREV_PARAMS with the PREV_CALIBRATION norm
  FLY-YAW       FlySteer from x in kite.fly.params.SPACE (34) and the norm of its init
STUB (not an interface arm) is a cheap synthetic episode for tests and the Modal smoke test: no sim, no brain.
"""

from __future__ import annotations

import time
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import numpy as np

from kite.fly.interfaces import ARMS, METRICS, OUTPUT_GROUPS, PREV_PARAMS, EpisodeResult
from kite.fly.params import HAND_FILE, PD_SPACE, SPACE, ParamSpace, load_params
from kite.follow import FollowCfg

FLY_ARMS = ("FLY-YAW-HAND", "FLY-YAW-PREV", "FLY-YAW")
LAW_ARMS = ("KITE-LAW", "KITE-LAW-CMA")
TRAINABLE = ("FLY-YAW", "KITE-LAW-CMA", "PRED-PD", "STUB")
INITS = ("prev", "hand", "default")

_LAW = FollowCfg()
LAW_SPACE = ParamSpace.from_rows([("k_yaw", 0.5, 4.0, "log", _LAW.k_yaw),
                                  ("db_yaw_deg", 0.0, 5.0, "lin", _LAW.db_yaw_deg)])
STUB_SPACE = ParamSpace.from_rows([(f"stub_{i}", 0.0, 1.0, "lin", 0.7) for i in range(4)])
STUB_OPTIMUM = 0.3  # every stub coordinate scores best here


def param_space(arm: str) -> ParamSpace:
    spaces = {"FLY-YAW": SPACE, "KITE-LAW-CMA": LAW_SPACE, "PRED-PD": PD_SPACE, "STUB": STUB_SPACE}
    if arm not in spaces:
        raise ValueError(f"{arm} has no parameter space")
    return spaces[arm]


def norm_dict(norm) -> dict[str, float]:
    """A readout norm (10 rates in OUTPUT_GROUPS order, or a dict) as a JSON-ready {DN: Hz}."""
    if isinstance(norm, dict):
        return {k: float(norm[k]) for k in OUTPUT_GROUPS}
    return dict(zip(OUTPUT_GROUPS, np.asarray(norm, dtype=float).tolist()))


def init_point(arm: str, init: str) -> dict:
    """CMA starting point of a trainable arm: {"x0", "norm", "source"}. FLY-YAW: prev (FLY-YAW-PREV and its norm),
    hand (FLY-YAW-HAND and its norm) or default (the brain's defaults with the hand norm). The classical arms start
    from their defaults (the law's gains)."""
    if arm in ("KITE-LAW-CMA", "PRED-PD", "STUB"):
        return {"x0": param_space(arm).default_x().tolist(), "norm": None, "source": f"{arm} defaults"}
    if arm != "FLY-YAW":
        raise ValueError(f"{arm} is not trainable; one of {TRAINABLE}")
    if init not in INITS:
        raise ValueError(f"init must be one of {INITS}, not {init!r}")
    pf = load_params(PREV_PARAMS if init == "prev" else HAND_FILE)
    x0 = pf.x if init != "default" else SPACE.default_x()
    return {"x0": np.asarray(x0, dtype=float).tolist(), "norm": norm_dict(pf.norm),
            "source": str(pf.path) if init != "default" else f"brain defaults, norm {pf.norm_from}"}


# ---------------------------------------------------------------------------------------------------------------
# Arm -> FollowCfg, yaw source
# ---------------------------------------------------------------------------------------------------------------


def check_follow(arm: str, follow: dict | None) -> dict:
    """An item's FollowCfg overrides, typed like the fields ({"predict": 1.0} -> {"predict": True}). behind_m is the
    side sets' (the item's behind_m); predict replaces the yaw law, so only the law arms take it (a yaw source and
    FollowCfg.predict are exclusive in FollowController)."""
    kinds = {f.name: type(getattr(_LAW, f.name)) for f in fields(FollowCfg)}
    out: dict = {}
    for k, v in (follow or {}).items():
        if k not in kinds:
            raise ValueError(f"{k!r} is not a FollowCfg field")
        if k == "behind_m":
            raise ValueError("behind_m is set by a side set (<profile>@<metres>), not a FollowCfg override")
        kind = kinds[k]
        out[k] = bool(v) if kind is bool else int(round(float(v))) if kind is int else float(v)
    if out.get("predict") and arm not in LAW_ARMS:
        raise ValueError(f"FollowCfg.predict steers instead of a yaw source: {arm} has one; use a law arm")
    return out


def make_follow_cfg(arm: str, x=None, params: dict | None = None, behind_m: float | None = None,
                    follow: dict | None = None) -> FollowCfg:
    """The sim's shared config (behind_m and the `follow` fields overridden if given, see check_follow); KITE-LAW-CMA
    takes k_yaw / db_yaw_deg from x or params. One new instance per call: build the yaw source from the instance the
    controller gets."""
    from kite.fly.sim.world import default_follow_cfg

    cfg = default_follow_cfg()
    if behind_m is not None:
        cfg = replace(cfg, behind_m=float(behind_m))
    if follow:
        cfg = replace(cfg, **check_follow(arm, follow))
    if arm == "KITE-LAW-CMA":
        if params is None:
            if x is None:
                raise ValueError("KITE-LAW-CMA needs x or params")
            params = LAW_SPACE.decode(np.asarray(x, dtype=float))
        cfg = replace(cfg, k_yaw=float(params["k_yaw"]), db_yaw_deg=float(params["db_yaw_deg"]))
    return cfg


def check_shared_cfg(ys: Any, cfg: FollowCfg) -> Any:
    """DESIGN R3: a yaw source holds the controller's own FollowCfg instance and believes its behind_m and
    video_lag_s."""
    if getattr(ys, "cfg", cfg) is not cfg:
        raise AssertionError(f"{type(ys).__name__} was built from another FollowCfg instance")
    for k in ("behind_m", "video_lag_s"):
        v = getattr(ys, k, None)
        if v is not None and abs(float(v) - float(getattr(cfg, k))) > 1e-12:
            raise AssertionError(f"{type(ys).__name__}.{k} = {v}, FollowCfg.{k} = {getattr(cfg, k)}")
    return ys


def make_yaw_source(arm: str, x=None, params_file: str | Path | None = None, size_blind: bool = True, seed: int = 0,
                    params: dict | None = None, norm: dict | None = None, follow_cfg: FollowCfg | None = None):
    """None for the law arms (FollowController's own law), else the arm's kite.follow.YawSource, built from
    follow_cfg (default: a fresh shared config). A FLY-YAW candidate (x or params) must come with its readout norm;
    without x / params / params_file the reference arms load their own files."""
    if arm in LAW_ARMS or arm == "STUB":
        return None
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; one of {ARMS}")
    from kite.fly.steer import FlySteer, PredPD, make_fly_steer

    cfg = follow_cfg or make_follow_cfg(arm)
    if arm == "PRED-PD":
        if params_file is not None:
            ys = PredPD.from_file(params_file, cfg=cfg, seed=seed)
        elif params is None and x is not None:
            ys = PredPD.from_x(np.asarray(x, dtype=float), cfg=cfg, seed=seed)
        else:
            ys = PredPD(params, cfg=cfg, seed=seed)
    elif params is not None or x is not None:
        if norm is None:
            raise ValueError(f"{arm}: params / x given without the readout norm they belong to")
        params = params if params is not None else SPACE.decode(np.asarray(x, dtype=float))
        ys = FlySteer(params, norm, cfg=cfg, size_blind=bool(size_blind), seed=seed, arm=arm)
    elif params_file is not None:
        ys = FlySteer.from_file(params_file, cfg=cfg, size_blind=bool(size_blind), seed=seed)
    elif arm == "FLY-YAW":
        raise ValueError("FLY-YAW needs x, params or a params_file")
    else:
        ys = make_fly_steer(arm, cfg=cfg, size_blind=bool(size_blind), seed=seed)
    return check_shared_cfg(ys, cfg)


# ---------------------------------------------------------------------------------------------------------------
# One episode
# ---------------------------------------------------------------------------------------------------------------


def _sim_run_episode(**kw) -> EpisodeResult:
    from kite.fly.sim import run_episode

    return run_episode(**kw)


def run_item(item: dict) -> EpisodeResult:
    """Run the episode an item describes (see kite.fly.train.evaluate.make_item)."""
    if item.get("stub_s") is not None:
        return stub_episode(item)
    arm, seed = item["arm"], int(item["seed"])
    x = item.get("x")
    law_cma = arm == "KITE-LAW-CMA"
    cfg = make_follow_cfg(arm, x if law_cma else None, item.get("params") if law_cma else None,
                          behind_m=item.get("behind_m"), follow=item.get("follow"))
    sb = item.get("size_blind")
    ys = make_yaw_source(arm, x=x, params_file=item.get("params_file"), size_blind=True if sb is None else bool(sb),
                         seed=seed, params=item.get("params"), norm=item.get("norm"), follow_cfg=cfg)
    return _sim_run_episode(seed=seed, profile=item.get("profile", "train"), follow_cfg=cfg, yaw_source=ys, arm=arm,
                            record=bool(item.get("record", False)), episode_s=float(item.get("episode_s") or 60.0),
                            overrides=item.get("overrides"))


def stub_episode(item: dict) -> EpisodeResult:
    """Synthetic episode: burns item["stub_s"] seconds of CPU, return depends on x and the seed (common random
    numbers: every arm sees the same seed noise). The law arms score like x at the stub default."""
    t0 = time.perf_counter()
    burn = float(item.get("stub_s") or 0.0)
    a = np.ones(64)
    while time.perf_counter() - t0 < burn:
        a = np.sqrt(a * a + 1.0) - 0.5
    seed = int(item["seed"])
    rng = np.random.default_rng([seed, 7])
    x = item.get("x")
    x = STUB_SPACE.default_x() if x is None or item["arm"] in LAW_ARMS else np.asarray(x, dtype=float)
    w = 1.0 + 0.5 * rng.random(x.size)  # per-seed weights: a noisy but consistent objective
    cost = float(np.sum(w * (x - STUB_OPTIMUM) ** 2))
    scale = 20.0 * (0.7 + 0.6 * rng.random())
    r_yaw = -scale * (0.2 + 5.0 * cost)
    terms = {"e_x": 0.7 * r_yaw, "jerk": 0.1 * r_yaw, "lost": 0.2 * r_yaw, "ev_loss": 0.0, "e_z": -0.5 * scale,
             "too_close": 0.0, "ev_too_close": 0.0, "ev_collision": 0.0}  # signed, weighted, as the sim's
    rms = 3.0 + 20.0 * cost + rng.random()
    metrics = {k: 0.0 for k in METRICS}
    metrics.update(rms_bearing_deg=rms, p95_bearing_deg=2 * rms, in_view=1.0 - min(0.5, cost), following=0.95,
                   losses=float(rng.random() < cost), stick_change=1.0 + cost, min_range_m=0.9)
    return EpisodeResult(seed=seed, profile=item.get("profile", "train"), arm=item["arm"], ret=sum(terms.values()),
                         terms=terms, metrics=metrics, n_ticks=600, wall_s=time.perf_counter() - t0)
