"""Episode metrics from simulator truth: interfaces.METRICS plus min_dist3d_m and start_failed (DESIGN 8, 10).

    m = EpisodeMetrics(behind_m=1.0, min_range_m=0.7, t_metric=t0 + METRIC_START_S, corners=[(t_a, t_b), ...])
    m.phys(t, range_m, bearing_deg, in_view, below, alt_err_m, dist3d_m)   # every physics sub-step
    m.control(t, dt, state, d_yaw, yaw, gap, last_true)                    # every control step
    m.collided(range_m, dist3d_m, ticks_left, window_s_left)               # once, if a collision ends the episode
    m.result(n_phys_full, n_ctrl_full)                                     # the metrics dict

Only what happens after t_metric counts. Fractions (in_view, in_band, following) are over the full episode's
window. After a collision ("overhead": horizontal range < 0.3 m while above the head) the rest of the window counts
as lost, out of view, not following, at a true bearing of 90 deg (R10), so a collision never flatters an episode.

Definitions:
- losses: separate > 1 s gaps without a true detection (the caller passes `gap` per control step); corner_losses:
  those whose last true detection was inside one of the corner windows (a turn, up to 3 s after it).
- pitch_exits: the head leaving through the bottom edge (`below`) with the walker within 1.5 m.
- stick_change: mean |change of the sent yaw stick| over control steps in state "following" only.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

_EPS = 1e-9
LOST_BEARING_DEG = 90.0


class EpisodeMetrics:
    def __init__(self, behind_m: float, min_range_m: float, t_metric: float,
                 corners: Sequence[tuple[float, float]] = (), band_frac: float = 0.15,
                 pitch_exit_range_m: float = 1.5):
        self.behind, self.band, self.min_range_m = behind_m, band_frac * behind_m, min_range_m
        self.t_metric = t_metric
        self.corners = list(corners)
        self.pitch_exit_range = pitch_exit_range_m
        self.n_pm = self.n_run = self.n_view = self.n_band = self.pitch_exits = 0  # n_run: ticks that ran
        self.s_b2 = self.s_r2 = self.s_a2 = 0.0
        self.abs_b: list[float] = []
        self.min_range = self.min_d3 = math.inf
        self.too_close_s = 0.0
        self.n_cm = self.n_follow = self.losses = self.corner_losses = 0
        self.time_lost = self.s_dyaw = self.s_ayaw = 0.0
        self.collision = False
        self._was_in_view = True
        self._prev_gap = False

    def phys(self, t: float, range_m: float, bearing_deg: float, in_view: bool, below: bool, alt_err_m: float,
             dist3d_m: float, dt: float = 0.01) -> None:
        if t > self.t_metric + _EPS:
            self.n_pm += 1
            self.n_run += 1
            self.s_b2 += bearing_deg * bearing_deg
            self.abs_b.append(abs(bearing_deg))
            self.n_view += in_view
            if self._was_in_view and not in_view and below and range_m < self.pitch_exit_range:
                self.pitch_exits += 1
            e = range_m - self.behind
            self.s_r2 += e * e
            self.n_band += -self.band <= e <= self.band
            self.s_a2 += alt_err_m * alt_err_m
            if range_m < self.min_range:
                self.min_range = range_m
            if dist3d_m < self.min_d3:
                self.min_d3 = dist3d_m
            if range_m < self.min_range_m:
                self.too_close_s += dt
        self._was_in_view = in_view

    def control(self, t: float, dt: float, state: str, d_yaw: float, yaw: float, gap: bool, last_true: float) -> None:
        if t >= self.t_metric - _EPS:
            first = self.n_cm == 0
            self.n_cm += 1
            if state == "following":
                self.n_follow += 1
                self.s_dyaw += abs(d_yaw)
            self.s_ayaw += abs(yaw)
            if gap:
                self.time_lost += dt
                if not self._prev_gap or first:
                    self.losses += 1
                    self.corner_losses += any(a <= last_true <= b for a, b in self.corners)
        self._prev_gap = gap

    def collided(self, range_m: float, dist3d_m: float, ticks_left: int, window_s_left: float) -> None:
        """The rest of the metric window (ticks_left sub-steps, window_s_left s) is lost, out of view, at 90 deg."""
        self.collision = True
        self.min_range = min(self.min_range, range_m)
        self.min_d3 = min(self.min_d3, dist3d_m)
        if ticks_left > 0:
            self.n_pm += ticks_left
            self.s_b2 += LOST_BEARING_DEG * LOST_BEARING_DEG * ticks_left
            self.abs_b.extend([LOST_BEARING_DEG] * ticks_left)
        if window_s_left > 0.0:
            self.time_lost += window_s_left
            if not self._prev_gap or self.n_cm == 0:
                self.losses += 1

    def result(self, n_phys_full: int, n_ctrl_full: int) -> dict[str, float]:
        b = sorted(self.abs_b)
        n_pm, n_run = max(1, self.n_pm), max(1, self.n_run)
        return {
            "rms_bearing_deg": math.sqrt(self.s_b2 / n_pm),
            "p95_bearing_deg": b[min(len(b) - 1, int(0.95 * len(b)))] if b else 0.0,
            "in_view": self.n_view / max(1, n_phys_full),
            "following": self.n_follow / max(1, n_ctrl_full),
            "time_lost_s": self.time_lost,
            "losses": float(self.losses),
            "rms_range_err_m": math.sqrt(self.s_r2 / n_run),
            "in_band": self.n_band / max(1, n_phys_full),
            "min_range_m": self.min_range if self.min_range < math.inf else float("nan"),
            "too_close_s": self.too_close_s,
            "collision": float(self.collision),
            "rms_alt_err_m": math.sqrt(self.s_a2 / n_run),
            "stick_change": self.s_dyaw / max(1, self.n_follow),
            "mean_abs_yaw": self.s_ayaw / max(1, self.n_cm),
            "pitch_exits": float(self.pitch_exits),
            "corner_losses": float(self.corner_losses),
            "min_dist3d_m": self.min_d3 if self.min_d3 < math.inf else float("nan"),
        }
