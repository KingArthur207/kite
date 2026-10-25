"""Behind-follow profiles: one episode's randomized world, drawn with common random numbers (DESIGN section 3).

Every episode is decided by (seed, profile) before its first tick. Each kind of randomness has its own stream
`numpy.random.default_rng([seed, profile_id, stream_id])` and everything is drawn at reset, so the same (seed, profile)
is the same world for every arm: the person's walk, the detector's luck per frame, the telemetry noise, the control
clock and the drone's drift are shared, and only the drone's own path differs between arms.

BehindParams takes one uniform per field, always in field order (fixed values still consume theirs), so narrowing one
range never shifts another field's draw. A spec is (lo, hi) for a uniform, a number for a fixed value, Choice(...) for
one of a few values, or Bern(p) for a yes/no drawn with probability p. `demo`, `fast`, `slow` and `stress` are
`train` with the overrides of DESIGN's profile table; whatever the table does not fix stays randomized as in train
(the person, the camera, the start, the telemetry noise).

Telemetry pitch (review, 2026-10-25). DESIGN drew a fresh N(0, 0.5 deg) per packet. At the follow geometry (the
camera 0.2 m above the head, 1.8 m back) 1 deg of pitch is ~16 % of range, and white noise the camera never sees
read as range rate, fired the flown approach feed-forward and set off a fore-aft limit cycle that left FLOWN 3 m off
even behind a 0.2 m/s walker. Whether a real Tello's integer-degree attitude has that per-packet noise is not
measured (a flight log of telemetry pitch next to the range would settle it); an IMU attitude error is mostly slow.
So the error is now an OU process: train draws its sd U(0, 0.5) deg and its correlation time U(0.5, 3) s (from
none to DESIGN's level: a trained model must not depend on which is true), demo 0.25 deg / 1.5 s, stress keeps
DESIGN's white 0.5 deg (correlation 0.05 s, one control tick).
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np

PROFILES = ("train", "demo", "fast", "slow", "stress")

# stream ids: one generator per kind of randomness (a new kind gets a new id: the others never shift)
S_PARAMS, S_WALK, S_POSE, S_DRONE, S_TELEMETRY, S_TIMING, S_VIDEO, S_SECOND = range(8)


@dataclass(frozen=True)
class Choice:
    values: tuple[float, ...]


@dataclass(frozen=True)
class Bern:
    p: float


@dataclass(frozen=True)
class BehindParams:
    """One episode's world (truth). The follow never reads it: it sees boxes, keypoints and telemetry."""

    # drone response per axis (the team's Tello lag test): gain at stick 100, first-order lag, dead time
    fwd_gain_mps: float  # forward and lateral
    tau_fwd_s: float
    fwd_dead_s: float
    yaw_gain_dps: float
    tau_yaw_s: float
    yaw_dead_s: float
    vz_gain_mps: float
    tau_z_s: float
    vz_dead_s: float
    pitch_scale: float  # pitch = -pitch_scale * atan(a_fwd / g), + nose up (DroneModel) ...
    att_tau_s: float  # ... reached through the attitude loop: a first-order lag of this time constant (world.py)
    drift_sigma_mps: float  # OU hover drift (tau 2 s) per horizontal axis: indoor calm
    wind_sigma_mps: float  # OU wind (tau 3 s)
    hold_tau_s: float  # the position hold at rc 0 pulls the drift displacement back with this time constant
    # video: capture -> frame on the laptop (Frame.t); YOLO time per frame U(yolo_lo_s, yolo_hi_s) on top
    video_latency_s: float
    yolo_lo_s: float
    yolo_hi_s: float
    # detector (pose.FakePose)
    det_base_p: float  # p(detected) = det_base_p * min(1, visible body height / 80 px)^2
    burst_rate_hz: float  # Poisson rate of 0.2-1.0 s detection blackouts on the person
    false_box_p: float  # per frame, one false person box somewhere
    box_noise_px: float  # sigma on each box edge
    kp_noise_px: float  # sigma on each keypoint coordinate
    swap_p: float  # per frame, the left / right shoulder labels are swapped
    # true camera (the follow believes BehindCfg.camera_fx and a level mount)
    fx_px: float  # fx = fy, principal point (480, 360), 960 x 720
    mount_pitch_deg: float  # camera tilt on the body, + up
    # telemetry
    alt_bias_m: float  # 'h' bias
    alt_noise_m: float  # 'h' noise sigma
    pitch_noise_deg: float
    tof_noise_m: float
    # the person
    person_h_m: float  # true height with shoes (head top)
    height_err_m: float  # the operator's --walker-height = round(person_h_m + this, 2)
    shoulder_m: float  # shoulder width (joint to joint)
    speed_lo: float  # walking speeds U(speed_lo, speed_hi)
    speed_hi: float
    stretch_mps: float  # demo: one stretch at this speed (0 = none) ...
    stretch_at: float  # ... starting at this fraction of the episode ...
    stretch_s: float  # ... this long
    p_step_back: float  # one step toward the drone in this episode with this probability (walker stream)
    # a second person crossing the view (exercises the lock)
    second_person: bool
    q_height_m: float
    q_shoulder_m: float
    q_at: float  # crossing starts at this fraction of the episode
    q_dist_m: float  # crossing line this far ahead of the drone (where it looks when the crossing starts)
    q_speed_mps: float
    q_side: float  # +1 enters from the drone's right, -1 from its left
    q_angle_deg: float  # crossing direction off perpendicular
    # start (DESIGN: directly in FOLLOW)
    start_range_off_m: float  # person at distance_m + this, horizontally
    start_bearing_deg: float  # + right of the drone's heading
    start_facing_deg: float  # facing away, + clockwise
    alt_offset_m: float  # drone altitude = altitude_m + this
    stand_s: float  # the person stands this long before walking
    # appended last, so adding it shifted no other field's draw: the telemetry pitch error is an OU process of sd
    # pitch_noise_deg and this correlation time (then rounded to whole degrees, as the Tello reports)
    pitch_noise_tau_s: float

    @property
    def operator_height_m(self) -> float:
        """What the operator types as --walker-height (cm resolution, as the app's argument)."""
        return round(self.person_h_m + self.height_err_m, 2)


KEYS = tuple(f.name for f in fields(BehindParams))

TRAIN: dict[str, object] = {
    "fwd_gain_mps": (0.80, 1.10), "tau_fwd_s": (0.31, 0.59), "fwd_dead_s": (0.33, 0.61),
    "yaw_gain_dps": (45.0, 65.0), "tau_yaw_s": (0.02, 0.10), "yaw_dead_s": (0.13, 0.24),
    "vz_gain_mps": (0.25, 0.40), "tau_z_s": (0.02, 0.10), "vz_dead_s": (0.29, 0.53),
    "pitch_scale": (0.5, 1.0), "att_tau_s": (0.08, 0.20), "drift_sigma_mps": (0.0, 0.04),
    "wind_sigma_mps": (0.0, 0.05), "hold_tau_s": (1.0, 2.0),
    "video_latency_s": (0.15, 0.35), "yolo_lo_s": 0.02, "yolo_hi_s": 0.06,
    "det_base_p": (0.92, 1.0), "burst_rate_hz": (0.0, 0.15), "false_box_p": (0.0, 0.01),
    "box_noise_px": (2.0, 6.0), "kp_noise_px": (2.0, 5.0), "swap_p": 0.01,
    "fx_px": (1040.0, 1110.0), "mount_pitch_deg": (-1.0, 1.0),
    "alt_bias_m": (-0.05, 0.05), "alt_noise_m": (0.01, 0.03), "pitch_noise_deg": (0.0, 0.5), "tof_noise_m": 0.01,
    "person_h_m": (1.55, 1.90), "height_err_m": (-0.02, 0.02), "shoulder_m": (0.36, 0.46),
    "speed_lo": 0.2, "speed_hi": 0.75, "stretch_mps": 0.0, "stretch_at": (0.25, 0.55), "stretch_s": (6.0, 10.0),
    "p_step_back": 0.15,
    "second_person": Bern(0.3), "q_height_m": (1.55, 1.90), "q_shoulder_m": (0.36, 0.46), "q_at": (0.1, 0.8),
    "q_dist_m": (1.0, 4.0), "q_speed_mps": (0.8, 1.4), "q_side": Choice((-1.0, 1.0)), "q_angle_deg": (-20.0, 20.0),
    "start_range_off_m": (-0.3, 0.6), "start_bearing_deg": (-10.0, 10.0), "start_facing_deg": (-20.0, 20.0),
    "alt_offset_m": (-0.12, 0.08), "stand_s": (1.0, 2.5), "pitch_noise_tau_s": (0.5, 3.0),
}

OVERRIDES: dict[str, dict[str, object]] = {
    "train": {},
    "demo": {  # the lag-test medians, the app's video lag, one walking speed with one faster stretch, nobody else
        "fwd_gain_mps": 0.96, "tau_fwd_s": 0.45, "fwd_dead_s": 0.47, "yaw_gain_dps": 55.0, "tau_yaw_s": 0.02,
        "yaw_dead_s": 0.18, "vz_gain_mps": 0.32, "tau_z_s": 0.02, "vz_dead_s": 0.41, "pitch_scale": 0.75,
        "att_tau_s": 0.12, "video_latency_s": 0.25, "det_base_p": 0.97, "burst_rate_hz": 0.05, "drift_sigma_mps": 0.02,
        "wind_sigma_mps": 0.02, "speed_lo": 0.35, "speed_hi": 0.35, "stretch_mps": 0.6, "second_person": Bern(0.0),
        "pitch_noise_deg": 0.25, "pitch_noise_tau_s": 1.5,
    },
    "fast": {"speed_lo": 0.5, "speed_hi": 0.8},
    "slow": {"speed_lo": 0.1, "speed_hi": 0.35},
    "stress": {"fwd_gain_mps": (0.75, 1.2), "video_latency_s": (0.35, 0.5), "det_base_p": (0.85, 0.95),
               "burst_rate_hz": 0.2, "drift_sigma_mps": 0.05, "wind_sigma_mps": 0.1, "second_person": Bern(0.5),
               "pitch_noise_deg": 0.5, "pitch_noise_tau_s": 0.05},  # DESIGN's first model: white 0.5 deg per packet
}


def profile_id(profile: str) -> int:
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; one of {PROFILES}")
    return PROFILES.index(profile)


def profile_specs(profile: str) -> dict[str, object]:
    profile_id(profile)
    return {**TRAIN, **OVERRIDES[profile]}


def stream(seed: int, profile: str, stream_id: int, attempt: int = 0) -> np.random.Generator:
    """The generator of one kind of randomness: key [seed, profile_id, stream_id] (+ [attempt] when > 0, a fresh
    stream for a redrawn start; attempt 0 is the plain three-number key)."""
    key = [int(seed), profile_id(profile), int(stream_id)] + ([int(attempt)] if attempt else [])
    return np.random.default_rng(key)


def _draw(spec: object, u: float) -> float | bool:
    if isinstance(spec, Bern):
        return bool(u < spec.p)
    if isinstance(spec, Choice):
        return float(spec.values[min(int(u * len(spec.values)), len(spec.values) - 1)])
    if isinstance(spec, tuple):
        return float(spec[0]) + u * (float(spec[1]) - float(spec[0]))
    return float(spec)


def draw(seed: int, profile: str, overrides: dict | None = None) -> BehindParams:
    """The episode's world (a pure function of seed, profile and overrides, e.g. {"speed_lo": 0.6, "speed_hi": 0.6}).
    Overrides replace drawn values after the draw, so they never shift any other field."""
    specs = profile_specs(profile)
    u = stream(seed, profile, S_PARAMS).random(len(KEYS)).tolist()
    p = BehindParams(**{k: _draw(specs[k], u[i]) for i, k in enumerate(KEYS)})
    if overrides:
        unknown = set(overrides) - set(KEYS)
        if unknown:
            raise ValueError(f"unknown BehindParams fields {sorted(unknown)}")
        p = replace(p, **{k: (bool(v) if k == "second_person" else float(v)) for k, v in overrides.items()})
    return p
