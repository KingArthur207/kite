"""Learned-in-the-world priors: how tall and how wide things on a sidewalk are, and how sure we are of it.

The size cue (distance = f * H_real / h_px) is only as good as H_real. Each prior is a log-normal: a median and
a spread sigma in natural-log units (0.10 = about +-10 %), so the cue's variance in log-depth is just
sigma^2 plus the pixel term. A person's height is a tight prior (adults 1.62 to 1.78 m median by sex, CDC
anthropometry; children widen it); a pole's is nearly useless (a bollard to a street light), so its size cue
gets almost no weight and the ground and depth cues decide. Widths are view-dependent (a car is 1.8 m from
the front, 4.5 m from the side), so their spread is large unless the object looks the same from every side.

Thresholds scale with object size too (`size_scale`): a 1.5-tonne car at 3 m is more urgent than a chair at
3 m, and a big object's near face is closer than its box centre. The scale is (W*H / person's W*H) ** 0.25,
clamped to [0.85, 1.35]: P1 for a car starts at 2.6 m, for a pole at 1.8 m.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Prior:
    h: float            # median real height (m)
    sh: float           # log-sigma of the height
    w: float            # median real width as usually seen (m)
    sw: float           # log-sigma of the width
    moving: bool = False

    def size_scale(self) -> float:
        return min(1.35, max(0.85, ((self.w * self.h) / (0.5 * 1.7)) ** 0.25))


# Keys are earpiece kinds (perception/vocab.py, sim/standin_detect.py KIND) and detector labels; a label wins.
PRIORS: dict[str, Prior] = {
    "person": Prior(1.70, 0.08, 0.50, 0.25, True),
    "bike": Prior(1.05, 0.10, 1.10, 0.55, True),
    "bicycle": Prior(1.05, 0.10, 1.10, 0.55, True),
    "motorcycle": Prior(1.20, 0.12, 1.40, 0.55, True),
    "vehicle": Prior(1.50, 0.15, 2.50, 0.60, True),
    "car": Prior(1.50, 0.12, 2.50, 0.60, True),
    "bus": Prior(3.10, 0.08, 6.00, 0.70, True),
    "truck": Prior(3.00, 0.25, 5.00, 0.70, True),
    "dog": Prior(0.55, 0.35, 0.70, 0.50, True),
    "stroller": Prior(1.00, 0.12, 0.60, 0.30, True),
    "e_scooter": Prior(1.15, 0.10, 0.60, 0.55),
    "scooter": Prior(1.15, 0.10, 0.60, 0.55),
    "pole": Prior(3.00, 0.60, 0.20, 0.45),
    "bollard": Prior(0.90, 0.20, 0.25, 0.30),
    "sign_board": Prior(1.00, 0.25, 0.60, 0.30),
    "sign": Prior(0.60, 0.40, 0.60, 0.40),
    "stop sign": Prior(0.75, 0.10, 0.75, 0.10),
    "door": Prior(2.03, 0.06, 0.90, 0.20),           # US interior/entry doors: 80 in tall, 32-36 in wide
    "chair": Prior(0.85, 0.12, 0.50, 0.20),
    "bench": Prior(0.80, 0.15, 1.50, 0.50),
    "trash_can": Prior(1.00, 0.20, 0.60, 0.30),
    "bin": Prior(1.00, 0.20, 0.60, 0.30),
    "fire_hydrant": Prior(0.75, 0.15, 0.40, 0.25),
    "fire hydrant": Prior(0.75, 0.15, 0.40, 0.25),
    "parking meter": Prior(1.40, 0.15, 0.30, 0.30),
    "potted plant": Prior(0.80, 0.40, 0.50, 0.40),
    "plant": Prior(0.80, 0.40, 0.50, 0.40),
    "suitcase": Prior(0.65, 0.20, 0.45, 0.25),
    "construction": Prior(0.90, 0.30, 0.60, 0.50),
    "traffic cone": Prior(0.70, 0.15, 0.35, 0.15),
    "tree": Prior(3.00, 0.60, 0.40, 0.60),
    "street_furniture": Prior(1.10, 0.35, 0.60, 0.40),
    "overhead": Prior(0.60, 0.60, 1.00, 0.60),
}
UNKNOWN = Prior(1.00, 0.70, 0.60, 0.70)            # anything else: the size cue is nearly ignored


def prior_for(kind: str | None, label: str | None = None) -> Prior:
    for k in (label, kind):
        if k and k.lower() in PRIORS:
            return PRIORS[k.lower()]
    return UNKNOWN


def log_sigma(rel: float) -> float:
    """A relative spread (0.1 = 10 %) -> log-sigma."""
    return math.log1p(rel)
