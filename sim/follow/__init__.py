"""Rendered person-following sim: kite's whole follow stack against a simulated Tello and street.

    python -m sim.follow                                   # 60 s walk, summary only
    python -m sim.follow --preview                         # watch it live (q quits)
    python -m sim.follow --out runs/sim_follow             # follow.mp4 + ticks.csv + summary.json
    python -m sim.follow --speed 0.9 --max-rc 70 --seed 3  # a faster walker, a looser stick clamp

What runs is the code that flies the real drone: SafetyGovernor (kite/drone/safety.py) wrapping SimTello, which
implements kite.drone.base.Drone on kite's Tello response model (kite/fly/sim/drone.py), and WalkerTracker +
FollowController (kite/follow.py) at 10 Hz, as kite.record runs them. Camera frames are rendered at 30 fps from the
drone's pose and arrive video_lag_s late, as the Tello's do.

kite/fly/sim is the other follow sim: box level (no images), fast, many seeds, for tuning the yaw controllers. This
one renders, so it is slower, but it shows the walk as the drone's camera would see it and exercises the Drone
interface, the governor and the discrete commands (takeoff, up, land).
"""
