"""Simulation profiles: the randomized world of one episode, drawn with common random numbers (DESIGN 2.4, 8).

Every episode is decided by (seed, profile) before its first tick. Each random quantity comes from its own stream
`numpy.random.default_rng([seed, profile_id, stream_id])` and everything is drawn at reset, so the same (seed,
profile) is the same world for every yaw arm; only the drone's own path differs between arms.

Episode parameters take one uniform each, always in PARAM_KEYS order (fixed values still consume theirs), so
narrowing one range never shifts another parameter's draw. A spec is (lo, hi) for a uniform, a number for a fixed
value, or Choice((a, b, ...)) for one of a few values. `demo`, `stress` and `fast` are `train` with overrides.

Revision 1 (DESIGN 8): video latency sits on top of the measured dynamics (train 0-0.25 s, demo 0.10 s, stress
0.45-0.65 s); max_rc train 50-100, demo 50 (today's SafetyCfg), fast 100; true fx 880-1100 while the controller
always believes 920; the person box has a head part (0.14-0.17 m wide) and a torso part.
Revision 3 (DESIGN 10): a position-held hover (hold_tau_s), an uncertain head-only detection (p_head 0.3-0.8, demo
0.55), and the detector rate capped at 1 / detector time.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np

PROFILES = ("train", "demo", "stress", "fast")

# stream ids: one generator per kind of randomness
S_PARAMS, S_START, S_WALKER, S_DRONE, S_DETECTOR, S_TELEMETRY, S_TIMING = range(7)


@dataclass(frozen=True)
class Choice:
    values: tuple[float, ...]


@dataclass(frozen=True)
class SimParams:
    """One episode's world (truth). The controller never reads it."""

    # drone response per axis (the team's Tello lag test, stick 30): gain at stick 100, first-order lag, dead time
    fwd_gain_mps: float  # forward and lateral
    yaw_gain_dps: float
    vz_gain_mps: float
    tau_fwd_s: float
    tau_yaw_s: float
    tau_z_s: float
    fwd_dead_s: float
    yaw_dead_s: float
    vz_dead_s: float
    pitch_scale: float  # pitch = pitch_scale * atan(a_fwd / g)
    drift_sigma_mps: float  # OU hover drift, tau 2 s, per horizontal axis
    wind_sigma_mps: float  # OU street wind, tau 3 s, per horizontal axis
    # telemetry
    alt_bias_m: float
    alt_noise_m: float
    pitch_noise_deg: float
    # camera and detector
    fx_px: float  # true focal length (fy = fx, principal point at the centre); the controller believes 920
    video_latency_s: float  # capture -> frame on the laptop (the frame's t), on top of the dynamics' dead times
    det_time_s: float  # frame on the laptop -> boxes reach the tracker
    det_rate_hz: float  # frames the detector processes per second
    box_noise_px: float  # sigma on each box edge
    dropout: float  # iid per processed frame
    burst_rate_hz: float  # Poisson rate of detection blackouts
    false_box_frac: float  # per frame, an extra person-sized box somewhere
    # SafetyCfg.max_rc: FollowController(max_rc) and the governor's clamp
    max_rc: int
    # walker
    head_top_m: float
    head_m: float  # head top to chin
    head_w_m: float
    shoulder_m: float  # torso width
    walk_speed_lo: float  # a straight segment's new speed is U(lo, hi), capped at v_cap
    walk_speed_hi: float
    # Revision 3 (appended last, so the draws above are unchanged)
    hold_tau_s: float  # position hold at rc 0: the time constant pulling drift / wind displacement back
    p_head: float  # head-only detection: p = p_head * min(1, (h_vis / 40)^2)

    @property
    def v_cap_mps(self) -> float:
        """Walker speed cap: every episode can physically be followed at this profile's max_rc."""
        return walker_speed_cap(self.fwd_gain_mps, self.max_rc)


def walker_speed_cap(fwd_gain_mps: float, max_rc: float) -> float:
    return min(1.5, 0.9 * fwd_gain_mps * max_rc / 100.0)


PARAM_KEYS = tuple(f.name for f in fields(SimParams))

TRAIN: dict[str, object] = {
    "fwd_gain_mps": (0.67, 1.5), "yaw_gain_dps": (39.0, 72.0), "vz_gain_mps": (0.23, 0.42),
    "tau_fwd_s": (0.31, 0.59), "tau_yaw_s": (0.02, 0.10), "tau_z_s": (0.02, 0.10),
    "fwd_dead_s": (0.33, 0.61), "yaw_dead_s": (0.13, 0.24), "vz_dead_s": (0.29, 0.53),
    "pitch_scale": (0.5, 1.0), "drift_sigma_mps": (0.0, 0.05), "wind_sigma_mps": (0.0, 0.15),
    "alt_bias_m": (-0.05, 0.05), "alt_noise_m": (0.01, 0.04), "pitch_noise_deg": 0.5,
    "fx_px": (880.0, 1100.0),
    "video_latency_s": (0.0, 0.25), "det_time_s": (0.03, 0.10), "det_rate_hz": (6.0, 25.0),
    "box_noise_px": (1.0, 5.0), "dropout": (0.0, 0.15), "burst_rate_hz": (0.0, 0.2), "false_box_frac": (0.0, 0.02),
    "max_rc": (50.0, 100.0),
    "head_top_m": (1.50, 1.95), "head_m": (0.20, 0.26), "head_w_m": (0.14, 0.17), "shoulder_m": (0.40, 0.50),
    "walk_speed_lo": 0.5, "walk_speed_hi": 1.5,
    "hold_tau_s": (1.0, 2.0), "p_head": (0.3, 0.8),
}

OVERRIDES: dict[str, dict[str, object]] = {
    "train": {},
    "demo": {  # the lag-test medians and today's settings; everything else stays randomized as in train
        "fwd_gain_mps": 0.96, "yaw_gain_dps": 55.0, "vz_gain_mps": 0.32, "tau_fwd_s": 0.45, "tau_yaw_s": 0.02,
        "tau_z_s": 0.02, "fwd_dead_s": 0.47, "yaw_dead_s": 0.18, "vz_dead_s": 0.41, "fx_px": 920.0,
        "video_latency_s": 0.10, "det_rate_hz": 15.0, "wind_sigma_mps": 0.05, "max_rc": 50.0, "p_head": 0.55,
    },
    "stress": {"video_latency_s": (0.45, 0.65), "dropout": 0.30, "fx_px": Choice((880.0, 1100.0)),
               "wind_sigma_mps": 0.15},
    "fast": {"walk_speed_lo": 1.1, "walk_speed_hi": 1.5, "max_rc": 100.0},
}


def profile_specs(profile: str) -> dict[str, object]:
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; one of {PROFILES}")
    return {**TRAIN, **OVERRIDES[profile]}


def stream(seed: int, profile: str, stream_id: int, attempt: int = 0) -> np.random.Generator:
    """attempt > 0: a fresh stream for a redrawn start (attempt 0 is the plain [seed, profile_id, stream_id])."""
    key = [int(seed), PROFILES.index(profile), int(stream_id)] + ([int(attempt)] if attempt else [])
    return np.random.default_rng(key)


def _draw(spec: object, u: float) -> float:
    if isinstance(spec, Choice):
        return float(spec.values[min(int(u * len(spec.values)), len(spec.values) - 1)])
    if isinstance(spec, tuple):
        return float(spec[0]) + u * (float(spec[1]) - float(spec[0]))
    return float(spec)


def draw_params(seed: int, profile: str, overrides: dict[str, float] | None = None) -> SimParams:
    """The episode's world parameters (pure function of seed, profile and overrides, e.g. {"fx_px": 1077}).
    The detector rate is capped at 1 / detector time: one detector thread cannot run faster."""
    specs = profile_specs(profile)
    u = stream(seed, profile, S_PARAMS).random(len(PARAM_KEYS)).tolist()
    v = {k: _draw(specs[k], u[i]) for i, k in enumerate(PARAM_KEYS)}
    v["max_rc"] = int(round(v["max_rc"]))
    p = SimParams(**v)
    if overrides:
        unknown = set(overrides) - set(PARAM_KEYS)
        if unknown:
            raise ValueError(f"unknown SimParams fields {sorted(unknown)}")
        p = replace(p, **{k: (int(round(x)) if k == "max_rc" else float(x)) for k, x in overrides.items()})
    if p.det_rate_hz > 1.0 / p.det_time_s:
        p = replace(p, det_rate_hz=1.0 / p.det_time_s)
    return p
