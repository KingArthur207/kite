"""Box-level follow simulator used to tune and evaluate the yaw controllers (DESIGN 2, 4).

kite's real WalkerTracker and FollowController run in the loop against a simulated walker, drone and camera:

    from kite.fly.sim import run_episode
    res = run_episode(seed, "demo")                          # KITE-LAW (FollowController's own yaw law)
    res = run_episode(seed, "train", yaw_source=my_source)   # any kite.follow.YawSource decides the yaw

Modules: profiles (randomized worlds, common random numbers), drone (Tello response), walker (street walk),
camera (projection, fake detector), reward (DESIGN 4), world (the episode loop), batch (many seeds, arms, workers).
CLI: python -m kite.fly.sim --help
"""

from .batch import ARM_SOURCES, register_arm, run_seeds, summarize
from .profiles import PROFILES, draw_params, walker_speed_cap
from .world import run_episode

__all__ = ["ARM_SOURCES", "PROFILES", "draw_params", "register_arm", "run_episode", "run_seeds", "summarize",
           "walker_speed_cap"]
