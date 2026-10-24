"""Shared constants, types and conventions for kite's fly-brain yaw steering (see DESIGN.md).

The same code runs on the drone laptop (inside kite.follow.FollowController as its YawSource), inside the
training simulator and on Modal. Never redefine these locally.

Conventions
- Image: the Tello stream, 960 x 720 px, x right, y down.
- Bearing (deg in kite.follow, rad inside the fly): + = walker RIGHT of the image centre.
- Sticks: Tello rc values in -100..100. yaw + = clockwise (turn right), fb + = forward, ud + = climb, lr + = right.
- Body pitch (deg): + = nose UP. The camera is fixed to the body: nose-up moves the head further down the frame.
- The fly's clock: one brain tick = BRAIN_TICK_MS of neuron time, run at DT = 0.05 s of wall/sim time whatever
  the caller's rate (kite.record calls the follow controller at 10 Hz, so two ticks per call).
- Follow geometry (kite spec): kite.follow.FollowCfg(behind_m=1.0, above_m=0.30). The range comes from the head
  top's image row (FollowController.measure), never from the box height (the box is cut by the bottom edge).
- Normalized size s = behind_m / range_m: 1 at the standoff, > 1 = too close (the encoder's size input).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DT = 0.05
BRAIN_TICK_MS = 50.0

DATA_DIR = Path(__file__).resolve().parent / "data"
BRAIN_FILE = DATA_DIR / "pursuit_core1.npz"  # MaleCNS v1.0 pursuit subgraph, 1,446 neurons (CC-BY 4.0 data)
# The tuned yaw sources kite.record --follow-yaw / kite.follow --yaw can use (kite.fly.steer.make_yaw_source). Opt in
# only: neither passed G3 (DESIGN section 11), so kite's own yaw law stays the default.
TRAINED_FILE = DATA_DIR / "fly_yaw.json"  # the kite-retuned fly (FLY-YAW, size-blind, CMA seed 1 gen 50)
PD_FILE = DATA_DIR / "pred_pd.json"  # the CMA-tuned PRED-PD gains (kp 2.33, kd 0.09, deadband 1.52 deg)
REFERENCE_DIR = DATA_DIR / "reference"  # the previously trained yaw parameters and their readout calibration
PREV_PARAMS = REFERENCE_DIR / "fly_yaw_prev_smooth.json"
PREV_CALIBRATION = REFERENCE_DIR / "fly_calibration_prev.json"

# Seed sets. Training draws fresh seeds per generation from TRAIN_SEED_BASE up; selection and test are fixed
# and disjoint, so a reported test number was never seen by training or checkpoint selection.
TRAIN_SEED_BASE = 10_000
SELECTION_SEEDS = tuple(range(1_000, 1_064))  # checkpoint selection during training
BASELINE_SEEDS = tuple(range(3_000, 3_100))  # G1/G2 decisions (init, size mode, go/no-go): never TEST_SEEDS
TEST_SEEDS = tuple(range(2_000, 2_200))  # G3 only, once, after training

# Neuron groups in the brain .npz ("groups": json name -> global neuron indices; "meta" has azimuth_bins).
INPUT_GROUPS = ("LC10a_L", "LC10a_R", "LC9_L", "LC9_R", "LC11_L", "LC11_R", "AROUSAL_L", "AROUSAL_R")
DN_TYPES = ("DNa02", "DNa01", "DNb05", "DNg13", "DNb06")  # steering descending neurons, one cell per side
OUTPUT_GROUPS = tuple(f"{t}_{s}" for t in DN_TYPES for s in "LR")  # type-major, side-minor
N_AZIMUTH_BINS = 8  # LC10a azimuth bins per side, bin 0 frontal
CHANNELS = (tuple(f"LC10a_L_b{k}" for k in range(N_AZIMUTH_BINS)) + tuple(f"LC10a_R_b{k}" for k in range(N_AZIMUTH_BINS))
            + ("LC9_L", "LC9_R", "LC11_L", "LC11_R", "AROUSAL_L", "AROUSAL_R"))
N_CHANNELS = len(CHANNELS)  # 22: what the encoder emits and the brain consumes

# Arms. Every arm decides ONLY the yaw stick while following; forward, lateral, altitude, coasting and the lost
# hover are kite.follow.FollowController's for all of them, so arms differ in steering alone.
ARMS = ("KITE-LAW", "KITE-LAW-CMA", "PRED-PD", "FLY-YAW-HAND", "FLY-YAW-PREV", "FLY-YAW")
#   KITE-LAW      FollowController's own yaw law as on main (k_yaw 1.5 stick/deg, 2 deg deadband): the baseline
#   KITE-LAW-CMA  the same law with CMA-ES-tuned k_yaw / deadband: the fair classical competitor
#   PRED-PD       ablation: FlySteer's latency-compensating filter and stick conditioning around a classical PD
#                 (yaw = kp * theta_hat + kd * dtheta_hat), CMA-tuned; separates "the fly helps" from "prediction helps"
#   FLY-YAW-HAND  fly brain with the hand (ridge) calibration of the readout, untrained
#   FLY-YAW-PREV  fly brain with the previously trained yaw parameters (REFERENCE_DIR), not retuned for kite
#   FLY-YAW       fly brain fine-tuned for kite (the candidate to ship)


@dataclass(frozen=True)
class TargetFeatures:
    """Pure-geometry features the fly's encoder starts from (no trained parameters)."""

    valid: bool
    theta: float  # bearing (rad), + = walker right
    s: float  # normalized size behind_m / range (1.0 when size-blind or no range)
    dtheta: float  # rad/s
    ds: float  # 1/s


NO_TARGET = TargetFeatures(False, 0.0, 1.0, 0.0, 0.0)


@dataclass
class EpisodeResult:
    """One simulated episode, returned to the trainer and to eval. Plain data: it pickles across processes."""

    seed: int
    profile: str
    arm: str
    ret: float  # total return (reward sum incl. events), higher is better, usually negative
    terms: dict[str, float] = field(default_factory=dict)  # summed contribution of each reward term and event
    metrics: dict[str, float] = field(default_factory=dict)  # METRICS
    n_ticks: int = 0
    wall_s: float = 0.0
    brain_s: float = 0.0
    trace: dict[str, list] | None = None  # per-tick arrays when run with record=True


# Episode metrics from simulator TRUTH, after METRIC_START_S (controllers never see truth).
METRIC_START_S = 3.0
METRICS = (
    "rms_bearing_deg",  # RMS true bearing (wrapped) from the drone heading to the walker's head over ALL ticks
                        # after METRIC_START_S, in view or not: the steering error (losing the walker can't hide it)
    "p95_bearing_deg",
    "in_view",  # fraction of ticks with the head top inside the frame
    "following",  # fraction of control steps FollowController was in state "following" (not coast / lost)
    "time_lost_s",  # time with no true detection for > 1 s
    "losses",  # number of separate > 1 s detection gaps
    "rms_range_err_m",  # RMS of (true horizontal range - behind_m)
    "in_band",  # fraction of ticks with |range - behind_m| <= 0.15 behind_m
    "min_range_m",
    "too_close_s",  # time with true range < FollowCfg.min_range_m
    "collision",  # 1 if the true horizontal range fell below 0.3 m (the episode ends)
    "rms_alt_err_m",  # RMS of (true altitude - (true head top + above_m))
    "stick_change",  # mean |change of the sent yaw stick| per control step (lower = calmer)
    "mean_abs_yaw",
    "pitch_exits",  # times the head left the frame while the walker was within 1.5 m (braking pitch, too close)
    "corner_losses",  # > 1 s detection gaps that started within 3 s of a walker corner turn
    "min_dist3d_m",  # closest 3D approach of the drone to the walker's head
    "start_failed",  # 1 if follow could never start (excluded for every arm alike)
)
# Reported, never trained on (kite.fly.sim.metrics): the forward and lateral sticks' stick_change, and the share of
# "following" control steps the prediction planner steered (FollowCfg.predict, kite/predict.py).
SIM_EXTRA_METRICS = ("fb_change", "lr_change", "planned")
# "collision" means overhead: the drone within 0.3 m horizontally of the head while above it (blind over the head).
# After one, the rest of the episode is charged as lost at a 90 deg bearing, so a crash can never look better.
