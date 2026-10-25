"""What the walker hears for each body-camera priority (perception/near sends `priority` 1 to 3).

Each priority has its own fixed pitch, rate and level (sonar.PRIORITY_TONE), so the three are easy to tell apart
by ear and a farther priority is never quieter just because its stand-in distance is larger. Both engines.
"""

from __future__ import annotations

import pytest

from earpiece import rust_sonar
from earpiece import sonar as sonar_mod
from earpiece.config import SonarCfg
from earpiece.events import Obstacle

RATE = 48000
CANON = {1: (1.3, 1.2), 2: (3.0, 1.2), 3: (4.0, 1.2)}      # perception/near/output.py CANONICAL


class Rec:
    def __init__(self):
        self.now, self.rate, self.plays = 0, RATE, []

    def samples(self, s: float) -> int:
        return int(round(s * RATE))

    def play(self, cue, at=None, bus="cue", gain=1.0, tag=None, **_):
        self.plays.append((cue, at, bus, gain))


@pytest.fixture(autouse=True)
def cue_keys(monkeypatch):
    monkeypatch.setattr(sonar_mod, "_cue", lambda kind, pq, bq, rate: ("cue", kind, pq, bq))


ENGINES = ["python"] + (["rust"] if rust_sonar.available() else [])


def heard(engine: str, p: int, seconds: float = 4.0):
    e = Rec()
    s = sonar_mod.Sonar(e, SonarCfg()) if engine == "python" else rust_sonar.RustSonar(e, SonarCfg())
    d, cs = CANON[p]
    t, step, seq = 0.0, 512 / RATE, 0
    while t < seconds:
        seq += 1
        s.update_frame("near", [Obstacle("a", "person", 5.0, d, cs, 1.0, p)], seq=seq, now=t)
        s.tick(now=t)
        e.now += 512
        t += step
    beeps = [(cue[2], at / RATE, gain) for cue, at, bus, gain in e.plays if cue[0] == "cue"]
    return beeps[2:]                                        # past the start


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("p", [1, 2, 3])
def test_each_priority_has_its_own_pitch_rate_and_level(engine, p):
    pitch, every, level = sonar_mod.PRIORITY_TONE[p]
    b = heard(engine, p)
    assert len(b) >= 2
    assert {x[0] for x in b} == {int(round(pitch / 25.0) * 25)}
    gaps = [b2[1] - b1[1] for b1, b2 in zip(b, b[1:])]
    assert all(abs(g - every) < 0.002 for g in gaps), gaps
    assert all(abs(x[2] - level) < 1e-6 for x in b)


@pytest.mark.parametrize("engine", ENGINES)
def test_the_three_are_easy_to_tell_apart(engine):
    tones = {p: heard(engine, p) for p in (1, 2, 3)}
    pitch = {p: tones[p][0][0] for p in tones}
    gap = {p: tones[p][1][1] - tones[p][0][1] for p in tones}
    level = {p: tones[p][0][2] for p in tones}
    assert pitch[1] > pitch[2] > pitch[3]
    assert gap[2] / gap[1] > 2.5 and gap[3] / gap[2] > 2.0          # each clearly slower than the one above
    assert min(level.values()) >= 0.8                                # none fades away on a laptop speaker
