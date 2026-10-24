"""data/sample_walk.jsonl replayed into a live earpiece far faster than real time."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import wait_for
from earpiece.events import parse
from earpiece.sim import load_jsonl, schedule

ROOT = Path(__file__).resolve().parents[2]
WALK = ROOT / "data" / "sample_walk.jsonl"
sys.path.insert(0, str(ROOT / "scripts"))
import replay  # noqa: E402


def errors(app):
    return [m for m in app.logs if m.startswith(("[control]", "[net] handler error", "[bridge]"))]


def test_every_sample_event_is_valid():
    plan = schedule(load_jsonl(str(WALK)))
    assert len(plan) > 600
    for _, ev in plan:
        parse(dict(ev))


def test_replay_in_process_with_no_pacing(app):
    """All 673 events through handle() back to back: no handler raises, state stays bounded."""
    for _, ev in schedule(load_jsonl(str(WALK))):
        app.handle(parse(dict(ev)))
    assert not errors(app)
    with app.speaker._qlock:
        assert len(app.speaker._queue) <= app.speaker.MAX_QUEUE
    assert len(app.hazards) <= app.MAX_HAZARDS
    assert app.state()["context"] is not None


@pytest.mark.parametrize("speed", [10.0, 200.0])
def test_replay_script_over_udp_at_high_speed(app, speed):
    """scripts/replay.py at 10x (under the per-peer rate limit: every event lands) and 200x
    (over it: the excess is shed by the limiter, nothing else breaks)."""
    n = len(schedule(load_jsonl(str(WALK))))
    blocks0 = app.engine.stats["blocks"]
    assert replay.main([str(WALK), "--port", str(app.cfg.net.udp_port), "--speed", str(speed)]) == 0
    st = app.transport.stats
    assert wait_for(lambda: st["udp"] + st["limited"] >= n * 0.98, 5)     # loopback may drop a datagram
    assert st["bad"] == 0 and not errors(app)
    if speed == 10.0:
        assert st["limited"] == 0
    assert app.engine.stats["blocks"] > blocks0                              # audio kept running
    audio = app.output.recorded()
    assert np.all(np.isfinite(audio)) and np.abs(audio).max() <= app.engine.ceiling + 1e-6
    assert wait_for(lambda: not app.speaker.busy, 30)
