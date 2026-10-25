"""Distance -> priority, with hysteresis and the time-to-contact override.

    P1  under 1.2 m         act now: about one white-cane length, under a second at walking speed
    P2  1.2 m to under 3 m  something is coming up: about two seconds of walking
    P3  3 m or more         it is there, in your line
    None                    nothing in the corridor

Raw distance from one camera is noisy (perception/near/README has the per-cue errors), so the state machine only
escalates after N_UP agreeing frames and only de-escalates after M_DOWN frames past a 10 % band beyond the
boundary. The exception is looming: when the time to contact is under TTC_P1 (1.5 s, the low end of the 1.5 to
2.5 s that pedestrian collision warnings target) the track goes to P1 at once, whatever the distance cues say.
Thresholds are multiplied by the class's size scale (priors.Prior.size_scale): a car is P1 from about 1.6 m.
`python -m perception.near --bands P1,P2` overrides them; evaluate.py scores at the bands its labels were made for.
"""

from __future__ import annotations

from dataclasses import dataclass

P1_M, P2_M = 1.2, 3.0
TTC_P1 = 1.5
TTC_RELEASE = 1.9
N_UP, M_DOWN = 2, 5
BAND = 0.10


def thresholds(size_scale: float = 1.0) -> tuple[float, float]:
    return P1_M * size_scale, P2_M * size_scale


def raw_priority(d: float | None, size_scale: float = 1.0) -> int | None:
    """The priority a distance alone gives (no hysteresis): ground truth for the evaluation uses this too."""
    if d is None:
        return None
    t1, t2 = thresholds(size_scale)
    return 1 if d < t1 else 2 if d < t2 else 3


@dataclass
class PriorityState:
    """One track's priority. update() once per frame with the filtered distance (None: out of the corridor)."""
    n_up: int = N_UP
    m_down: int = M_DOWN
    still: bool = False                 # a single still image: no frames to agree over, decide at once
    current: int | None = None
    pending: int | None = None
    count: int = 0
    by_ttc: bool = False

    def update(self, d: float | None, size_scale: float = 1.0, ttc: float | None = None) -> int | None:
        if ttc is not None and ttc < TTC_P1 and d is not None:
            self.current, self.pending, self.count, self.by_ttc = 1, None, 0, True
            return 1
        if self.by_ttc and (ttc is not None and ttc < TTC_RELEASE) and d is not None:
            return 1                                     # stay on P1 until the TTC is clearly released
        self.by_ttc = False
        target = raw_priority(d, size_scale)
        if self.still:
            self.current = target
            return target
        if target == self.current:
            self.pending, self.count = None, 0
            return self.current
        escalating = self.current is None or (target is not None and target < self.current)
        if not escalating and target is not None and d is not None and self.current is not None:
            # de-escalating: only count frames clearly past the boundary (BAND beyond it)
            t1, t2 = thresholds(size_scale)
            edge = t1 if self.current == 1 else t2
            if d < edge * (1 + BAND):
                self.pending, self.count = None, 0
                return self.current
        pend_esc = self.pending is not None and (self.current is None or self.pending < self.current)
        if escalating and pend_esc:
            # P3 -> P2 -> P1 in consecutive frames still agrees on "more urgent": commit the least urgent of them
            self.pending = max(self.pending, target)
        elif target != self.pending:
            self.pending, self.count = target, 0
        self.count += 1
        if self.count >= (self.n_up if escalating else self.m_down):
            self.current, self.pending, self.count = self.pending, None, 0
        return self.current
