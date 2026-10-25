"""kite.fly.behind: simulate the follow the app flies (kite.follow_behind.FollowMission) and train it (DESIGN.md).

The yaw-only study next door (kite.fly.sim) simulated FollowController 1.0 m behind the head with boxes. This package
runs the app's own loop, 1.8 m behind and 1.9 m above the floor, on a fake YOLO pose detector (box + 17 keypoints),
so the range cues, the lock, the fly brain smooth2 and the safety governor are the real code.

    from kite.follow_behind import BehindCfg
    from kite.fly.behind.world import run_behind_episode
    res = run_behind_episode(7, "demo", BehindCfg())      # FLOWN: the app today

Seeds. Training draws fresh seeds per generation, BEHIND_TRAIN_BASE + SEED_STRIDE * cma_seed + gen * K + k;
selection and test are fixed, disjoint from each other, from training and from kite.fly's own seed sets (1000-3199,
10000+), so a test number was never seen by training or checkpoint selection.
"""

BEHIND_TRAIN_BASE = 7_000_000
SEED_STRIDE = 100_000  # per CMA seed: room for 100k training episodes each
SELECTION_SEEDS = tuple(range(6000, 6096))
TEST_SEEDS = tuple(range(8000, 8200))
