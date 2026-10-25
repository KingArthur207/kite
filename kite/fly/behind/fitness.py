"""Fitness of the behind follow (DESIGN section 5), minimized: an episode cost J, scored against FLOWN on the same seed.

    J = cost_terms(m) summed         (each term dimensionless, about 1 for a mediocre follow)
    score = episode_score(m_arm, m_flown)  1 = as good as the flown app; lower is better
    fitness(arm_metrics, flown_metrics)    the mean score over seed-paired episodes (+ counts and per-term means)

Terms (FitnessCfg holds every weight and scale; it is recorded in every run, so a change is never silent):

    distance     w_range * huber((rms_range_err_m / s_range)^2, cap_range): the square up to cap_range, then its
                 tangent line (range_tail "linear"; "cap" = DESIGN's literal flat cap)
    bearing      w_bearing * min(cap_bearing, (rms_bearing_deg / s_bearing)^2)
    lock         w_lost * time_lost_s / s_lost + w_losses * losses
    sticks       w_sticks * (stick_fb / s_fb + stick_yaw / s_yaw + stick_ud / s_ud + stick_lr / s_lr) / 4
    accel        w_accel * accel_rms / s_accel
    speed_track  w_track * speed_track_err / s_track
    speed_excess w_excess * excess_speed / s_excess + w_fast * fast_unneeded_s       speed only when needed
    altitude     w_alt * rms_alt_err_m / s_alt                                         stay at the follow height
    collided, very_close, too_close, overshoot, ceiling, wrong_lock: weight * metric (penalties)

The distance tail. DESIGN caps the squared distance term at 9 (0.9 m RMS) so a long loss cannot swamp the rest. But
the flown follow sits far beyond that cap on most training seeds (stick 35 is ~0.34 m/s and the person walks 0.2-0.75
m/s: it trails metres behind), and a flat cap there is a plateau: a candidate 9.6 m behind and one 3 m behind both
score FLOWN's 9. Past the knee the term goes on as the tangent of the square (value and slope continuous: cap + 2
sqrt(cap) (r - sqrt(cap)), r = rms / s_range), linear, so every metre of falling behind costs the same 20 and a long
loss still cannot grow as its square.

The score (review, 2026-10-25). DESIGN divided the whole J by FLOWN's J on the same seed. Where FLOWN trails, its J
is ~100, nearly all distance, so every other term counted ~100 x less than on a seed where it holds the distance: a
second spent under 0.8 m cost 0.011, a metre of overshoot 0.03, and a candidate that made near passes or dropped the
back-off still scored better than FLOWN. Only the distance term spans orders of magnitude between seeds (how far a
walker can outwalk the drone), so only it is scored relative to FLOWN; everything else is an absolute price:

    score = 1 + (D_arm - D_FLOWN) / max(D_FLOWN, floor) + (Q_arm - Q_FLOWN) / q_scale

with D = the rel_terms (distance) and Q = every other term, q_scale = 3 (the J of a follow that holds the distance,
DESIGN's "about 1 per term"): a second under 0.8 m costs (w_very_close + w_too_close) / 3 = 0.43 on every seed, a
second of unneeded fast flight 0.2. FLOWN scores exactly 1.0 on every seed. Clipped to [min_score, max_score].

Collisions are a hard constraint. A collision scores collision_score; on top, fitness() adds collision_penalty per
seed where the candidate collided and FLOWN did not, after lifting the mean to at least 1: a colliding candidate is
never better than the flown follow, whatever it gains elsewhere (a flat 6 among scores of ~0.5 let 1 collision in 13
episodes pay for itself). A failed start: excluded when FLOWN's failed (the seed is unusable), charged max_score when
only the candidate's did. An arm episode that errored scores error_score (a candidate that crashes the controller must
not be excluded, which would reward it); a FLOWN episode that errored is excluded (the baseline's failure is not the
candidate's).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

TERMS = ("distance", "bearing", "lock", "sticks", "accel", "speed_track", "speed_excess", "altitude",
         "collided", "very_close", "too_close", "overshoot", "ceiling", "wrong_lock")


@dataclass(frozen=True)
class FitnessCfg:
    # distance: the RMS range error, normalized by 0.3 m (the healthy band's half width), squared up to 9 (0.9 m
    # RMS), then linear (range_tail "linear", the module docstring) or flat ("cap", DESIGN's literal cap): after a
    # long loss the person walks on and an unbounded square would swamp every other term
    w_range: float = 1.0
    s_range: float = 0.3
    cap_range: float = 9.0
    range_tail: str = "linear"
    # bearing: RMS true bearing to the head, by 15 deg (half the frame's margin before the head nears the edge), capped
    w_bearing: float = 0.5
    s_bearing: float = 15.0
    cap_bearing: float = 9.0
    # lock: seconds without the true person per 10 s, plus a fixed price per separate loss (each one is a re-lock risk)
    w_lost: float = 1.0
    s_lost: float = 10.0
    w_losses: float = 0.5
    # sticks: mean |change| per FOLLOW control tick on each axis, each by FLOWN's own value on the demo profile (so the
    # four axes weigh alike and FLOWN's stick term is about w_sticks). Measured 2026-10-25 (after the review's
    # simulator changes: OU telemetry pitch, the ToF footprint): FLOWN (FollowMission, BehindCfg(), smooth2) on demo
    # seeds 0-39, 60 s each, world.run_behind_episode. fb is the fore-aft swing of the approach feed-forward; yaw is
    # small because smooth2's output conditioning (deadband 4, hysteresis 3) holds the stick; lr only moves in the
    # lost-while-close side step (orbit is off as flown), so it is rare and noisy. Means 4.24 / 0.173 / 0.154 /
    # 0.089 (per-seed sd 2.2 / 0.06 / 0.15 / 0.13; before the review's changes 4.76 / 0.170 / 0.156 / 0.090).
    # Re-measure after a simulator change.
    w_sticks: float = 0.6
    s_fb: float = 4.24
    s_yaw: float = 0.173
    s_ud: float = 0.154
    s_lr: float = 0.089
    # accel: RMS true horizontal acceleration by 0.3 m/s^2 (a calm follow; braking hard for a stop is ~1)
    w_accel: float = 0.3
    s_accel: float = 0.3
    # speed_track: time-weighted RMS (drone speed along the person's heading - their speed) while they walk near, by
    # 0.15 m/s
    w_track: float = 0.8
    s_track: float = 0.15
    # speed only when needed: excess speed toward them while near (m/s, time-averaged) by 0.1 m/s, plus 0.6 per second
    # of flying at them > 0.40 m/s after they walked < 0.30 m/s (any range): the request's "faster IF and ONLY IF the
    # person is faster", so a second of it costs 0.2 score
    w_excess: float = 1.0
    s_excess: float = 0.1
    w_fast: float = 0.6
    # altitude: RMS true altitude error by 0.05 m (the follow's deadband): sagging toward head height is not free
    w_alt: float = 0.3
    s_alt: float = 0.05
    # penalties, per unit of the metric: a collision, seconds < 0.8 m and < 1.2 m, overshoot metres (summed over the
    # person's stops / slow-downs / step backs / turn arounds), seconds above the ceiling margin, seconds locked on
    # the second person
    w_collided: float = 4.0
    w_very_close: float = 1.0
    w_too_close: float = 0.3
    w_overshoot: float = 3.0
    w_ceiling: float = 0.5
    w_wrong_lock: float = 0.5
    # the score: the terms scored relative to FLOWN's on the same seed, their J_FLOWN floor (a near-perfect FLOWN
    # episode must not make the ratio explode), the price scale of every other term, clip, fixed scores
    rel_terms: tuple[str, ...] = ("distance",)
    floor: float = 0.5
    q_scale: float = 3.0
    min_score: float = -6.0
    max_score: float = 6.0
    collision_score: float = 6.0
    collision_penalty: float = 1.0  # fitness(): + this per seed the candidate collided on and FLOWN did not
    error_score: float = 6.0


def _v(m: Mapping, key: str) -> float:
    return float(m[key])


def range_term(r: float, cap: float, tail: str = "linear") -> float:
    """r^2 up to cap; beyond, its tangent line cap + 2 sqrt(cap) (r - sqrt(cap)) ("linear") or cap ("cap")."""
    sq = r * r
    if sq <= cap:
        return sq
    if tail == "cap":
        return cap
    if tail != "linear":
        raise ValueError(f"range_tail must be 'linear' or 'cap', got {tail!r}")
    k = math.sqrt(cap)
    return cap + 2.0 * k * (abs(r) - k)


def cost_terms(m: Mapping, fc: FitnessCfg = FitnessCfg()) -> dict[str, float]:
    """Each term of J (TERMS order) for one episode's metrics (kite.fly.behind.metrics.episode_metrics)."""
    r = _v(m, "rms_range_err_m") / fc.s_range
    b = _v(m, "rms_bearing_deg") / fc.s_bearing
    return {
        "distance": fc.w_range * range_term(r, fc.cap_range, fc.range_tail),
        "bearing": fc.w_bearing * min(fc.cap_bearing, b * b),
        "lock": fc.w_lost * _v(m, "time_lost_s") / fc.s_lost + fc.w_losses * _v(m, "losses"),
        "sticks": fc.w_sticks * (_v(m, "stick_fb") / fc.s_fb + _v(m, "stick_yaw") / fc.s_yaw
                                 + _v(m, "stick_ud") / fc.s_ud + _v(m, "stick_lr") / fc.s_lr) / 4.0,
        "accel": fc.w_accel * _v(m, "accel_rms") / fc.s_accel,
        "speed_track": fc.w_track * _v(m, "speed_track_err") / fc.s_track,
        "speed_excess": fc.w_excess * _v(m, "excess_speed") / fc.s_excess + fc.w_fast * _v(m, "fast_unneeded_s"),
        "altitude": fc.w_alt * _v(m, "rms_alt_err_m") / fc.s_alt,
        "collided": fc.w_collided * _v(m, "collided"),
        "very_close": fc.w_very_close * _v(m, "very_close_s"),
        "too_close": fc.w_too_close * _v(m, "too_close_s"),
        "overshoot": fc.w_overshoot * _v(m, "overshoot_m"),
        "ceiling": fc.w_ceiling * _v(m, "ceiling_s"),
        "wrong_lock": fc.w_wrong_lock * _v(m, "wrong_lock_s"),
    }


def episode_cost(m: Mapping, fc: FitnessCfg = FitnessCfg()) -> float:
    """J: the sum of cost_terms."""
    return float(sum(cost_terms(m, fc).values()))


def _flag(m: Mapping, key: str) -> bool:
    v = m.get(key, 0.0)
    return v is not None and float(v) > 0.0


def split_cost(m: Mapping, fc: FitnessCfg = FitnessCfg()) -> tuple[float, float]:
    """(D, Q): the sum of the terms scored relative to FLOWN (fc.rel_terms) and of the absolutely priced rest."""
    terms = cost_terms(m, fc)
    d = sum(v for k, v in terms.items() if k in fc.rel_terms)
    return float(d), float(sum(v for k, v in terms.items() if k not in fc.rel_terms))


def episode_score(m_arm: Mapping, m_flown: Mapping, fc: FitnessCfg = FitnessCfg()) -> float | None:
    """The arm's score on one seed against FLOWN's on the same seed (module docstring). None = excluded."""
    if _flag(m_flown, "start_failed") or _flag(m_flown, "error"):
        return None
    if _flag(m_arm, "error"):
        return fc.error_score
    if _flag(m_arm, "start_failed"):
        return fc.max_score  # FLOWN started on this seed: the candidate's failure is its own
    if _flag(m_arm, "collided"):
        return fc.collision_score
    try:
        (d_a, q_a), (d_f, q_f) = split_cost(m_arm, fc), split_cost(m_flown, fc)
    except (KeyError, TypeError, ValueError):
        return fc.error_score
    if not (math.isfinite(d_f) and math.isfinite(q_f)):
        return None
    if not (math.isfinite(d_a) and math.isfinite(q_a)):
        return fc.error_score
    s = 1.0 + (d_a - d_f) / max(d_f, fc.floor) + (q_a - q_f) / fc.q_scale
    return min(max(s, fc.min_score), fc.max_score)


def fitness(arm_metrics: list[Mapping], flown_metrics: list[Mapping], fc: FitnessCfg = FitnessCfg()) -> dict:
    """The mean score over seed-paired episodes (arm_metrics[i] and flown_metrics[i] are the same seed), lower is
    better, and what drives it:

        fitness      mean score over the used pairs (fc.max_score when none could be used, so CMA never sees nan);
                     with excess collisions max(mean, 1) + collision_penalty x excess_collisions
        n, excluded  used pairs / pairs left out (FLOWN's start failed or it errored)
        collisions   used arm episodes that collided; excess_collisions: of them, on seeds where FLOWN did not
        errors       used arm episodes that errored
        scores       per pair, None where excluded (for paired statistics)
        mean_score   the plain mean (before the collision constraint)
        parts        mean of each cost term of the arm over the used pairs that have metrics (TERMS)
        flown_parts  the same for FLOWN; cost / flown_cost: mean J of each
    """
    if len(arm_metrics) != len(flown_metrics):
        raise ValueError(f"{len(arm_metrics)} arm episodes but {len(flown_metrics)} FLOWN episodes: pair them by seed")
    scores: list[float | None] = []
    used: list[float] = []
    collisions = excess = errors = 0
    sums = dict.fromkeys(TERMS, 0.0)
    fsums = dict.fromkeys(TERMS, 0.0)
    n_parts = 0
    for ma, mf in zip(arm_metrics, flown_metrics):
        s = episode_score(ma, mf, fc)
        scores.append(s)
        if s is None:
            continue
        used.append(s)
        if _flag(ma, "error"):
            errors += 1
            continue
        if _flag(ma, "collided"):
            collisions += 1
            excess += not _flag(mf, "collided")
        try:
            ta, tf = cost_terms(ma, fc), cost_terms(mf, fc)
        except (KeyError, TypeError, ValueError):
            continue
        n_parts += 1
        for k in TERMS:
            sums[k] += ta[k]
            fsums[k] += tf[k]
    parts = {k: sums[k] / n_parts for k in TERMS} if n_parts else {}
    fparts = {k: fsums[k] / n_parts for k in TERMS} if n_parts else {}
    mean = sum(used) / len(used) if used else fc.max_score
    fit = mean if not excess else max(mean, 1.0) + fc.collision_penalty * excess
    return {
        "fitness": fit,
        "mean_score": mean,
        "n": len(used),
        "excluded": len(scores) - len(used),
        "collisions": collisions,
        "excess_collisions": excess,
        "errors": errors,
        "scores": scores,
        "parts": parts,
        "flown_parts": fparts,
        "cost": sum(parts.values()) if parts else None,
        "flown_cost": sum(fparts.values()) if fparts else None,
    }
