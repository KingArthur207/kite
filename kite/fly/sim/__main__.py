"""Command line for the follow simulator.

    python -m kite.fly.sim --arm KITE-LAW --profile demo --seeds baseline      # mean metrics of one arm
    python -m kite.fly.sim --profile train --seeds 0-19 --set k_yaw=2.2        # FollowCfg overrides (KITE-LAW-CMA)
    python -m kite.fly.sim --arm FLY-YAW-PREV --seeds baseline                 # a fly arm (batch.ARM_SOURCES)
    python -m kite.fly.sim --yaw-source mypkg.mod:factory --arm MINE           # any YawSource factory
    python -m kite.fly.sim --profile demo --world fx_px=1077                   # change a world parameter
    python -m kite.fly.sim --sweep --profile demo --seeds baseline             # G1: geometry x true fx
    python -m kite.fly.sim --speed-caps                                         # walker speed cap per max_rc

--yaw-source names a callable returning a fresh kite.follow.YawSource per episode (called in each worker process).
Report on BASELINE_SEEDS ('baseline'); TEST_SEEDS ('test') are for G3 only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, replace

from ...follow import FollowCfg
from ..interfaces import METRICS
from .batch import format_summary, parse_seeds, run_seeds, summarize
from .profiles import PARAM_KEYS, PROFILES, TRAIN, walker_speed_cap
from .world import BELIEVED_FX, default_follow_cfg

SWEEP = ((1.0, 0.15), (1.0, 0.30), (1.0, 0.50), (1.2, 0.30), (1.5, 0.30))
SWEEP_FX = (920.0, 1000.0, 1077.0)


def _parse_value(cur, v: str):
    if isinstance(cur, bool):
        return v.lower() in ("1", "true", "yes")
    if v == "None":
        return None
    return type(cur)(float(v)) if cur is not None else float(v)


def _cfg_from_args(args) -> FollowCfg:
    cfg = default_follow_cfg()
    if args.behind is not None:
        cfg = replace(cfg, behind_m=args.behind)
    if args.above is not None:
        cfg = replace(cfg, above_m=args.above)
    for kv in args.set:
        k, sep, v = kv.partition("=")
        if not sep or not hasattr(cfg, k):
            raise SystemExit(f"--set expects a FollowCfg field=value, got {kv!r}")
        cfg = replace(cfg, **{k: _parse_value(getattr(cfg, k), v)})
    return cfg


def _world_from_args(args) -> dict[str, float]:
    out = {}
    for kv in args.world:
        k, sep, v = kv.partition("=")
        if not sep or k not in PARAM_KEYS:
            raise SystemExit(f"--world expects a profiles.SimParams field=value, got {kv!r}")
        out[k] = float(v)
    return out


def geometry(behind: float, above: float, fx: float) -> tuple[float, float, float]:
    """(target row, deg left below it before the bottom edge, nearest range with the head in frame), level drone.
    The controller believes fx 920; with another true fx it settles where the TRUE row is its target row."""
    depression = math.atan(BELIEVED_FX * above / behind / fx)  # true angle of the target row
    edge = math.atan(360.0 / fx)
    return 360.0 + BELIEVED_FX * above / behind, math.degrees(edge - depression), above / math.tan(edge)


def speed_caps() -> str:
    lo, hi = TRAIN["fwd_gain_mps"]
    lines = ["walker speed cap = min(1.5, 0.9 * fwd_gain * max_rc / 100); drone top speed = fwd_gain * max_rc / 100",
             f"  {'max_rc':>6s} {'demo 0.96 m/s':>14s} {f'train {lo}':>11s} {f'train {hi}':>11s} "
             f"{'drone top (demo)':>17s}"]
    for m in (50, 60, 80, 100):
        lines.append(f"  {m:6d} {walker_speed_cap(0.96, m):14.2f} {walker_speed_cap(lo, m):11.2f} "
                     f"{walker_speed_cap(hi, m):11.2f} {0.96 * m / 100:17.2f}")
    return "\n".join(lines)


def sweep(profile: str, seeds: list[int], base: FollowCfg, workers: int, episode_s: float,
          fxs: tuple[float, ...] = SWEEP_FX, geometries=SWEEP) -> str:
    cols = ("in_view", "following", "rms_bearing_deg", "rms_range_err_m", "in_band", "losses", "time_lost_s",
            "pitch_exits", "corner_losses")
    head = (f"  {'fx':>5s} {'behind':>6s} {'above':>5s} {'margin':>6s} {'near':>5s} "
            + " ".join(f"{c[:11]:>11s}" for c in cols) + f" {'collisions':>10s}")
    lines = [f"geometry sweep, KITE-LAW (max_yaw {base.max_yaw}), {profile}, {len(seeds)} seeds. margin: degrees "
             "between the target row and the bottom edge; near: closest range with the head in frame (level)", head]
    for fx in fxs:
        for behind, above in geometries:
            cfg = replace(base, behind_m=behind, above_m=above)
            res = run_seeds(seeds, profile, follow_cfg=cfg, workers=workers, episode_s=episode_s,
                            overrides={"fx_px": fx})
            s = summarize(res)
            _, margin, near = geometry(behind, above, fx)
            lines.append(f"  {fx:5.0f} {behind:6.2f} {above:5.2f} {margin:+5.1f}° {near:5.2f} "
                         + " ".join(f"{s[c]:11.3f}" for c in cols) + f" {int(s['collisions']):>4d} / {len(res)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m kite.fly.sim", description="Box-level follow simulator (kite.fly).")
    p.add_argument("--arm", default="KITE-LAW", help="arm label (and default yaw source from batch.ARM_SOURCES)")
    p.add_argument("--yaw-source", default=None, help="module:callable returning a fresh YawSource per episode")
    p.add_argument("--profile", default="demo", choices=PROFILES)
    p.add_argument("--seeds", default="baseline", help="'0-19', '0-9,20', 'baseline', 'selection' or 'test'")
    p.add_argument("--behind", type=float, default=None, help="FollowCfg.behind_m (default 1.0)")
    p.add_argument("--above", type=float, default=None, help="FollowCfg.above_m (default 0.30)")
    p.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE", help="any FollowCfg field")
    p.add_argument("--world", action="append", default=[], metavar="FIELD=VALUE",
                   help="any profiles.SimParams field, e.g. fx_px=1077")
    p.add_argument("--episode-s", type=float, default=60.0)
    p.add_argument("--workers", type=int, default=1, help="worker processes")
    p.add_argument("--json", default=None, help="write every episode's result here")
    p.add_argument("--sweep", action="store_true", help="KITE-LAW over (behind_m, above_m) x true fx 920/1000/1077")
    p.add_argument("--speed-caps", action="store_true", help="walker speed cap per max_rc")
    args = p.parse_args(argv)

    if args.speed_caps:
        print(speed_caps())
        return 0
    seeds = parse_seeds(args.seeds)
    cfg = _cfg_from_args(args)
    world = _world_from_args(args)
    t0 = time.time()
    if args.sweep:
        print(sweep(args.profile, seeds, cfg, args.workers, args.episode_s))
        print(f"({time.time() - t0:.1f} s)")
        return 0
    res = run_seeds(seeds, args.profile, arm=args.arm, yaw_spec=args.yaw_source, follow_cfg=cfg,
                    episode_s=args.episode_s, workers=args.workers, overrides=world or None)
    s = summarize(res)
    print(f"{args.arm} on {args.profile}{' ' + str(world) if world else ''}, {len(res)} seeds ({args.seeds}), "
          f"behind {cfg.behind_m} m, above {cfg.above_m} m, max_yaw {cfg.max_yaw}: mean over episodes")
    print(format_summary(s, METRICS))
    print(f"  {'collision episodes':18s} {int(s['collisions']):9d}")
    print(f"({time.time() - t0:.1f} s)")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump([{k: v for k, v in asdict(r).items() if k != "trace"} for r in res], fh, indent=1)
        print(f"-> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
