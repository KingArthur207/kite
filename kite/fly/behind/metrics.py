"""Episode metrics of the behind follow (DESIGN section 4), from simulator truth after the first 3 s.

    m = episode_metrics(phys, ctrl, info)   # once per episode: ~6000 physics samples, ~1100 control ticks

The world (world.py) records two truth traces from the follow start and hands them over at the end; this module is a
pure function of them, so it is tested on hand-built traces (tests/test_behind_metrics.py) without the simulator.

    phys  one sample per 0.01 s physics tick: t, x, y, z, heading (rad, clockwise from north), pitch_deg, vx, vy,
          px, py (the person's head top), p_heading, p_speed, p_event (EV_*), in_view, qx, qy (second person)
    ctrl  one sample per control tick: t, state, lr, fb, ud, yaw (the sticks SimTello got, after the governor),
          lock (LOCK_*), seen, gov_event
    info  distance_m, altitude_m, person_height_m, metric_start_s, episode_s, collided, collision_t, start_failed

Window. Only samples with t >= metric_start_s count (the start, the lock's 3 confirming hits and the first settle are
the same for every arm and say nothing about the follow). Fractions (in_view, in_band, following) and the speed
averages are over the FULL window (metric_start_s to episode_s), so an episode cut short never looks better.

Collision. It ends the episode (the traces stop at collision_t). The rest of the window is charged as kite.fly does:
lost (one more loss, unless a loss was already running, and its seconds in time_lost_s), out of view, out of band, at
a true bearing of LOST_BEARING_DEG. Range, altitude and speed metrics cover the time that ran.

Definitions (x east, y north, z up; heading and p_heading clockwise from north):
- range = horizontal distance drone (x, y) -> head top (px, py); bearing = its angle off the drone's heading.
- far_s / in_band are relative to distance_m (2.6 m and 1.5-2.1 m at 1.8 m); too_close_s (< 1.2 m) and
  very_close_s (< 0.8 m) are absolute: they are about the person's safety, not the setpoint.
- overshoot_m: person events are the contiguous runs of p_event in {stop, slow-down, turn around, step back} (a speed
  change counts only when the new speed is lower), each extended by OVERSHOOT_AFTER_S; overlapping windows merge so
  one close approach is charged once. Per window max(0, (distance_m - BAND_M) - smallest range in it) (1.5 m at the
  1.8 m follow); overshoot_m is the SUM over the windows, overshoot_max_m the worst, overshoot_events how many.
- losses / time_lost_s: periods longer than LOSS_MIN_S in which the follow did not see its locked TRUE person on a new
  frame (state != FOLLOW, not seen, or lock != LOCK_TRUE: a lock on the second person or a false box is lost). Each
  period counts whole (its first second too), clipped to the window; a period that began before the window counts
  if its full length is over LOSS_MIN_S.
- stick_*: mean |change of the sent stick| between consecutive control ticks that are both in state FOLLOW.
- accel_rms: RMS of the horizontal acceleration from (v(t + 0.1 s) - v(t)) / 0.1 s, i.e. the 0.01 s finite
  differences box-averaged over 0.1 s, so the step-wise sub-step physics (a new stick every 0.05 s) does not dominate.
- Speeds: ground speed |(vx, vy)|; speed along the person's heading u = (vx, vy) . (sin p_heading, cos p_heading);
  speed TOWARD the person v_to = (vx, vy) . (unit drone -> person) (backing away and side steps are < 0 / ~0: never
  "too fast"); the person's recent speed p_recent = the most they walked over the last FAST_RECENT_S (so braking
  for a person who just stopped is not charged until the drone has had time to brake).
  speed_track_err = sqrt(sum (u - p_speed)^2 / window samples) over the samples where the person walks (p_speed >
  WALKS_MPS) and range < distance_m + 0.6 (further behind the drone SHOULD be faster): a time-weighted RMS, so a
  drone that is rarely near is charged for the few samples it has, not their mean at full weight.
  excess_speed = time-average over the window of max(0, v_to - p_speed - 0.15) counted while range < distance_m +
  0.3, zero elsewhere (closing in on them faster than they walk: the overrun risk).
  fast_unneeded_s = seconds, at ANY range, with v_to > max(0.40, p_recent + 0.15) while p_recent < 0.30: the person
  has been slow and the drone flies at them faster than the flown cap allows (speed only when needed; catching up
  with a slow walker from far behind at more than ~0.4 m/s is not needed either).
- governor_events = control ticks in the window with gov_event.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from kite.fly.interfaces import METRIC_START_S
from kite.fly.sim.metrics import LOST_BEARING_DEG

# Person event codes in phys["p_event"] (the walker's segment kind on that tick; walker.py writes them).
EV_WALK, EV_STOP, EV_SPEED, EV_TURN_AROUND, EV_STEP_BACK, EV_CORNER = 0, 1, 2, 3, 4, 5
# Lock codes in ctrl["lock"].
LOCK_NONE, LOCK_TRUE, LOCK_SECOND, LOCK_FALSE = 0, 1, 2, 3
FOLLOW_STATE = "FOLLOW"

BAND_M = 0.3  # in_band half width around distance_m; distance_m - BAND_M (1.5 m) is the overshoot reference
TOO_CLOSE_M = 1.2  # never closer than this in normal walking (DESIGN goal 1)
VERY_CLOSE_M = 0.8
FAR_MARGIN_M = 0.8  # far_s: range > distance_m + 0.8 (2.6 m): fell behind
OVERSHOOT_AFTER_S = 2.0  # an event's window runs this long past its end (the drone is still braking)
SLOWDOWN_EPS_MPS = 0.02  # a speed change is a slow-down when it ends this much below where it started
CEILING_MARGIN_M = 0.25  # ceiling_s: true altitude > altitude_m + 0.25 (the flown back-off climb limit)
WALKS_MPS = 0.15  # "the person walks"
TRACK_MARGIN_M = 0.6  # speed_track_err only while range < distance_m + 0.6
NEAR_MARGIN_M = 0.3  # excess_speed / fast_unneeded_s only while range < distance_m + 0.3
EXCESS_SLACK_MPS = 0.15  # free speed above the person's before excess_speed counts
FAST_MPS = 0.40  # fast_unneeded_s: the drone toward the person above this (and p_recent + EXCESS_SLACK_MPS) ...
SLOW_PERSON_MPS = 0.30  # ... while the person has walked below this ...
FAST_RECENT_S = 1.5  # ... over the last this long
LOSS_MIN_S = 1.0  # a loss is a period longer than this without the true person
ACCEL_WINDOW_S = 0.1  # finite-difference span of accel_rms
PHYS_DT = 0.01  # the physics tick, when the trace is too short to read it from
_EPS = 1e-9

METRICS_BEHIND = (
    "collided",  # 1 if the episode ended in a collision
    "rms_range_err_m",  # RMS of (horizontal range - distance_m)
    "in_band",  # fraction of the window with |range - distance_m| <= BAND_M
    "too_close_s",  # seconds with range < TOO_CLOSE_M
    "very_close_s",  # seconds with range < VERY_CLOSE_M
    "min_range_m",  # smallest horizontal range
    "overshoot_m",  # sum over person stop / slow-down / step back / turn around of max(0, 1.5 - smallest range)
    "far_s",  # seconds with range > distance_m + FAR_MARGIN_M
    "rms_bearing_deg",  # RMS true bearing to the head top, all window ticks (LOST_BEARING_DEG after a collision)
    "in_view",  # fraction of the window with the head top inside the frame
    "losses",  # periods > LOSS_MIN_S without the locked true person
    "time_lost_s",  # their total seconds in the window
    "wrong_lock_s",  # seconds locked on the second person
    "stick_fb", "stick_yaw", "stick_ud", "stick_lr",  # mean |change of the sent stick| per FOLLOW control tick
    "accel_rms",  # RMS true horizontal acceleration (m/s^2), 0.1 s finite differences
    "alt_max_m",  # highest true altitude
    "ceiling_s",  # seconds above altitude_m + CEILING_MARGIN_M
    "rms_alt_err_m",  # RMS of (true altitude - altitude_m)
    "speed_track_err",  # time-weighted RMS of (speed along the person's heading - person speed), walking, near
    "excess_speed",  # time-average of max(0, speed toward them - person speed - 0.15) while near (m/s)
    "fast_unneeded_s",  # seconds flying at them > max(0.40, recent speed + 0.15) m/s after < 0.30 m/s, any range
    "max_speed_mps",  # highest drone ground speed
    "governor_events",  # control ticks on which the governor intervened
    "start_failed",  # 1 if the lock never confirmed (excluded identically for every arm)
)
# Reported, not in the fitness: the share of the window in FOLLOW, the worst single overshoot and how many person
# events there were, the closest 3D approach of the drone to the head top.
BEHIND_EXTRA_METRICS = ("following", "overshoot_max_m", "overshoot_events", "min_dist3d_m")


def _runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(starts, ends) of the True runs of a boolean array, ends exclusive."""
    d = np.diff(mask.astype(np.int8), prepend=np.int8(0), append=np.int8(0))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def _recent_max(v: np.ndarray, k: int) -> np.ndarray:
    """out[i] = max(v[max(0, i - k + 1) : i + 1]): the most over the last k samples."""
    if v.size == 0 or k <= 1:
        return v.copy()
    padded = np.concatenate((np.full(k - 1, v[0]), v))
    return np.lib.stride_tricks.sliding_window_view(padded, k).max(axis=1)


def _f(info: Mapping, key: str, default: float) -> float:
    v = info.get(key, default)
    return default if v is None else float(v)


def empty_metrics(info: Mapping) -> dict[str, float]:
    """Every metric key for an episode with no trace (a failed start may end before the follow ran)."""
    m = dict.fromkeys(METRICS_BEHIND + BEHIND_EXTRA_METRICS, 0.0)
    for k in ("min_range_m", "min_dist3d_m", "alt_max_m"):
        m[k] = float("nan")
    m["collided"] = float(bool(info.get("collided", False)))
    m["start_failed"] = float(bool(info.get("start_failed", False)))
    return m


def _overshoot(t: np.ndarray, rng: np.ndarray, code: np.ndarray, speed: np.ndarray, i_lo: int, i_hi: int,
               ref_m: float) -> tuple[float, float, int]:
    """(sum, max, number of windows) of max(0, ref_m - smallest range) over the person's stop / slow-down / step back /
    turn around windows (each run + OVERSHOOT_AFTER_S, overlapping ones merged), restricted to samples [i_lo, i_hi)."""
    member = (code == EV_STOP) | (code == EV_TURN_AROUND) | (code == EV_STEP_BACK)
    n = len(code)
    for s, e in zip(*_runs(code == EV_SPEED)):  # a speed change is an event only when it slows the person down
        before, after = speed[s - 1] if s > 0 else speed[s], speed[e] if e < n else speed[e - 1]
        if after < before - SLOWDOWN_EPS_MPS:
            member[s:e] = True
    windows: list[list[float]] = []
    for s, e in zip(*_runs(member)):
        a, b = float(t[s]), float(t[e - 1]) + OVERSHOOT_AFTER_S
        if windows and a <= windows[-1][1] + _EPS:
            windows[-1][1] = max(windows[-1][1], b)
        else:
            windows.append([a, b])
    total = worst = 0.0
    count = 0
    for a, b in windows:
        lo = max(i_lo, int(np.searchsorted(t, a - _EPS, "left")))
        hi = min(i_hi, int(np.searchsorted(t, b + _EPS, "right")))
        if hi <= lo:
            continue  # wholly before the window (or after the trace)
        count += 1
        o = max(0.0, ref_m - float(rng[lo:hi].min()))
        total += o
        worst = max(worst, o)
    return total, worst, count


def episode_metrics(phys: Mapping[str, list], ctrl: Mapping[str, list], info: Mapping) -> dict[str, float]:
    """Every key of METRICS_BEHIND + BEHIND_EXTRA_METRICS as plain floats (module docstring for the definitions)."""
    d_m = float(info["distance_m"])
    alt_m = float(info["altitude_m"])
    t0 = _f(info, "metric_start_s", METRIC_START_S)
    ep_s = float(info["episode_s"])
    collided = bool(info.get("collided", False))
    tp = np.asarray(phys.get("t", ()), dtype=float)
    if tp.size == 0:
        return empty_metrics(info)
    t_stop = _f(info, "collision_t", float(tp[-1])) if collided else ep_s  # where the traces end
    dt = float(np.median(np.diff(tp[:200]))) if tp.size > 1 else PHYS_DT
    if not dt > 0.0:
        dt = PHYS_DT

    # ---- physics truth, the samples [i0, i1) of the window
    i0 = int(np.searchsorted(tp, t0 - _EPS, "left"))
    i1 = int(np.searchsorted(tp, ep_s + _EPS, "right"))
    n_ran = max(0, i1 - i0)
    n_rest = int(round(max(0.0, ep_s - max(t_stop, t0)) / dt)) if collided else 0
    n_full = max(n_ran + n_rest, int(round((ep_s - t0) / dt)), 1)

    def a(key: str, dtype: type = float) -> np.ndarray:
        return np.asarray(phys[key], dtype=dtype)

    x, y, z, vx, vy = a("x"), a("y"), a("z"), a("vx"), a("vy")
    px, py, ph, ps = a("px"), a("py"), a("p_heading"), a("p_speed")
    dx, dy = px - x, py - y
    rng = np.hypot(dx, dy)
    w = slice(i0, i1)
    rw, zw, psw = rng[w], z[w], ps[w]
    m: dict[str, float] = {"collided": float(collided), "start_failed": float(bool(info.get("start_failed", False)))}
    if n_ran:
        e = rw - d_m
        m["rms_range_err_m"] = math.sqrt(float(np.mean(e * e)))
        m["in_band"] = float(np.count_nonzero(np.abs(e) <= BAND_M + _EPS)) / n_full
        m["too_close_s"] = float(np.count_nonzero(rw < TOO_CLOSE_M)) * dt
        m["very_close_s"] = float(np.count_nonzero(rw < VERY_CLOSE_M)) * dt
        m["far_s"] = float(np.count_nonzero(rw > d_m + FAR_MARGIN_M)) * dt
        m["min_range_m"] = float(rw.min())
        hd = a("heading")[w]
        s, c = np.sin(hd), np.cos(hd)
        dxw, dyw = dx[w], dy[w]
        b = np.degrees(np.arctan2(dxw * c - dyw * s, dxw * s + dyw * c))
        m["rms_bearing_deg"] = math.sqrt((float(np.dot(b, b)) + n_rest * LOST_BEARING_DEG ** 2) / (n_ran + n_rest))
        m["in_view"] = float(np.count_nonzero(np.asarray(phys["in_view"], dtype=bool)[w])) / n_full
        m["alt_max_m"] = float(zw.max())
        m["ceiling_s"] = float(np.count_nonzero(zw > alt_m + CEILING_MARGIN_M)) * dt
        ea = zw - alt_m
        m["rms_alt_err_m"] = math.sqrt(float(np.mean(ea * ea)))
        vxw, vyw = vx[w], vy[w]
        k = max(1, int(round(ACCEL_WINDOW_S / dt)))
        if n_ran > k:
            ax, ay = vxw[k:] - vxw[:-k], vyw[k:] - vyw[:-k]
            m["accel_rms"] = math.sqrt(float(np.mean(ax * ax + ay * ay))) / (k * dt)
        else:
            m["accel_rms"] = 0.0
        gs = np.hypot(vxw, vyw)
        phw = ph[w]
        u = vxw * np.sin(phw) + vyw * np.cos(phw)
        to = (vxw * dxw + vyw * dyw) / np.maximum(rw, _EPS)
        walks = psw > WALKS_MPS
        track = walks & (rw < d_m + TRACK_MARGIN_M)
        de = u[track] - psw[track]
        m["speed_track_err"] = math.sqrt(float(np.dot(de, de)) / n_full)
        near = rw < d_m + NEAR_MARGIN_M
        m["excess_speed"] = float(np.sum(np.maximum(0.0, to - psw - EXCESS_SLACK_MPS)[near])) / n_full
        recent = _recent_max(ps, max(1, int(round(FAST_RECENT_S / dt))))[w]
        fast = (to > np.maximum(FAST_MPS, recent + EXCESS_SLACK_MPS)) & (recent < SLOW_PERSON_MPS)
        m["fast_unneeded_s"] = float(np.count_nonzero(fast)) * dt
        m["max_speed_mps"] = float(gs.max())
    else:  # a collision before the window opened: only what the charge says
        m.update(rms_range_err_m=0.0, in_band=0.0, too_close_s=0.0, very_close_s=0.0, far_s=0.0,
                 rms_bearing_deg=LOST_BEARING_DEG if n_rest else 0.0, in_view=0.0, alt_max_m=float("nan"),
                 ceiling_s=0.0, rms_alt_err_m=0.0, accel_rms=0.0, speed_track_err=0.0, excess_speed=0.0,
                 fast_unneeded_s=0.0, max_speed_mps=0.0, min_range_m=float("nan"))
    if collided:  # the collision sample itself, even if it came before the window
        m["min_range_m"] = float(rng[-1]) if not n_ran else min(m["min_range_m"], float(rng[-1]))
    h = info.get("person_height_m")
    if h is not None and n_ran:
        dz = zw - float(h)
        m["min_dist3d_m"] = float(np.sqrt(rw * rw + dz * dz).min())
    else:
        m["min_dist3d_m"] = float("nan")
    tot, worst, n_ev = _overshoot(tp, rng, np.asarray(phys["p_event"], dtype=np.int64), ps, i0, i1, d_m - BAND_M)
    m["overshoot_m"], m["overshoot_max_m"], m["overshoot_events"] = tot, worst, float(n_ev)

    m.update(_ctrl_metrics(ctrl, t0, ep_s, t_stop, collided))
    return {k: float(m[k]) for k in METRICS_BEHIND + BEHIND_EXTRA_METRICS}


def _ctrl_metrics(ctrl: Mapping[str, list], t0: float, ep_s: float, t_stop: float, collided: bool) -> dict:
    """losses, time_lost_s, wrong_lock_s, stick_*, governor_events and following from the control trace. Each tick
    holds until the next one (the last until t_stop, where the traces end)."""
    window_s = max(_EPS, ep_s - t0)
    tc = np.asarray(ctrl.get("t", ()), dtype=float)
    n = tc.size
    out = dict.fromkeys(("losses", "time_lost_s", "wrong_lock_s", "stick_fb", "stick_yaw", "stick_ud", "stick_lr",
                         "governor_events", "following"), 0.0)
    if n == 0:
        if collided:
            out["losses"], out["time_lost_s"] = 1.0, max(0.0, ep_s - max(t_stop, t0))
        return out
    t_end = max(t_stop, float(tc[-1]))
    ends = np.minimum(np.append(tc[1:], t_end), t_end)
    ov = np.maximum(0.0, ends - np.maximum(tc, t0))  # each tick's seconds inside the window
    follow = np.fromiter((s == FOLLOW_STATE for s in ctrl["state"]), dtype=bool, count=n)
    lock = np.asarray(ctrl["lock"], dtype=np.int64)
    seen = np.asarray(ctrl["seen"], dtype=bool)
    out["following"] = float(np.sum(ov[follow])) / window_s
    out["wrong_lock_s"] = float(np.sum(ov[lock == LOCK_SECOND]))
    in_w = tc >= t0 - _EPS
    out["governor_events"] = float(np.count_nonzero(np.asarray(ctrl["gov_event"], dtype=bool) & in_w))

    pair = follow[1:] & follow[:-1] & in_w[1:]
    n_pair = int(np.count_nonzero(pair))
    for key in ("fb", "yaw", "ud", "lr"):
        if n_pair:
            dv = np.abs(np.diff(np.asarray(ctrl[key], dtype=float)))
            out["stick_" + key] = float(np.sum(dv[pair])) / n_pair

    lost = ~(follow & seen & (lock == LOCK_TRUE))
    periods: list[tuple[float, float, bool]] = []  # (start, end, forced: the collision's rest of the episode)
    for s, e in zip(*_runs(lost)):
        tail = e >= n
        a, b = float(tc[s]), float(tc[e]) if not tail else t_end
        if collided and tail:
            b = ep_s
        periods.append((a, b, collided and tail))
    if collided and not (periods and periods[-1][2]):
        periods.append((t_stop, ep_s, True))
    losses, lost_s = 0, 0.0
    for a, b, forced in periods:
        inside = min(b, ep_s) - max(a, t0)
        if (forced or b - a > LOSS_MIN_S + _EPS) and (inside > 0.0 or (forced and a >= t0 - _EPS)):
            losses += 1
            lost_s += max(0.0, inside)
    out["losses"], out["time_lost_s"] = float(losses), lost_s
    return out
