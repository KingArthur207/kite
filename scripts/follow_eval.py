"""Follow mode, reactive controller vs prediction planner, in the closed-loop Moss St simulation (tests/test_follow.py).

    python scripts/follow_eval.py              # markdown table, 3 seeds per row (about a minute)
    python scripts/follow_eval.py --seeds 5

Scenarios: the Moss walk (stand, 12 m straight, stop at the curb, left turn on a 1.5 m radius, 8 m, stop) at
0.5 m/s with SafetyCfg.max_rc 50, 10% dropped frames and a 0.4 s blackout; then one thing changed per row.
Columns: standoff error (m from the point 1.2 m back along the walker's path), max lag (furthest beyond 1.2 m),
closest approach, max bearing of the walker off the camera axis, walker-lost events (hover), stick reversals a
minute (oscillation, |change| > 10 on lr / fb / yaw).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

from kite.follow import FollowCfg  # noqa: E402
from test_follow import moss_route, run_sim, sim_metrics  # noqa: E402

SCENARIOS = {
    "Moss walk, 0.5 m/s": {},
    "video lag 0.4 s (planned for 0.25)": {"video_lag_s": 0.4},
    "30% dropped frames": {"drop": 0.3},
    "blackout 0.8 s (> lost_s)": {"blackout": (20.0, 20.8)},
    "0.65 m/s, max_rc 70": {"speed": 0.65, "max_rc": 70},
}
CONTROLLERS = {
    "reactive": (FollowCfg(), None),
    "predict": (FollowCfg(predict=True), None),
    "predict + route prior": (FollowCfg(predict=True), "route"),
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", type=int, default=3)
    args = p.parse_args(argv)
    print("| scenario | controller | standoff mean / p95 (m) | max lag (m) | closest (m) | max bearing (deg) "
          "| lost | reversals/min |")
    print("|---|---|---|---|---|---|---|---|")
    for name, kw in SCENARIOS.items():
        for label, (cfg, route) in CONTROLLERS.items():
            ms = [sim_metrics(run_sim(cfg=cfg, route=moss_route() if route else None, seed=s, **kw))
                  for s in range(args.seeds)]
            m = {k: float(np.mean([x[k] for x in ms])) for k in ms[0]}
            lost = sum(x["lost"] for x in ms)
            print(f"| {name} | {label} | {m['standoff_mean']:.2f} / {m['standoff_p95']:.2f} | {m['max_lag']:.2f} "
                  f"| {min(x['min_dist'] for x in ms):.2f} | {m['max_bearing']:.1f} | {lost} in {args.seeds} runs "
                  f"| {m['flips_per_min']:.1f} |", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
