"""The speech scheduler under floods and preemption storms (offline fake voice, null output)."""

from __future__ import annotations

import random
import threading
import time

import pytest

from conftest import wait_for
from earpiece.config import SpeechCfg
from earpiece.mixer import Engine
from earpiece.output import NullOutput
from earpiece.speech import Speaker


@pytest.fixture
def spk(offline_speech):
    eng = Engine()
    out = NullOutput(eng, 1024, realtime=True)
    out.start()
    s = Speaker(eng, SpeechCfg(cache_dir=offline_speech / "cache"), key=None, log=lambda *_: None)
    s.start()
    yield s
    s.stop()
    out.stop()


def hold(spk, monkeypatch):
    """Stop the consumer taking anything off the queue (listening would still let critical through)."""
    monkeypatch.setattr(spk, "_next", lambda: None)


def test_a_flood_of_untimed_lines_stays_bounded(spk, monkeypatch):
    """'answer' and 'system' lines never expire on their own: without a cap, 5000 distinct ones
    from the network would queue for hours and make every say() slower."""
    hold(spk, monkeypatch)
    dropped = []
    longest = 0
    for i in range(5000):
        spk.say(f"status line number {i}", "system" if i % 2 else "answer",
                on_done=lambda uid, how: dropped.append(how))
        longest = max(longest, len(spk._queue))
    # the queue never grows past its cap, so each say() works on at most MAX_QUEUE lines: O(cap)
    # per call, not O(n). Counted, not timed, so a loaded machine can't fail it.
    assert longest == Speaker.MAX_QUEUE
    with spk._qlock:
        n = len(spk._queue)
        kept = [q.text for q in spk._queue]
    assert n == Speaker.MAX_QUEUE and spk.stats["shed"] == 5000 - n
    assert dropped.count("dropped") == 5000 - n            # every shed line is reported
    # the least important ("system") go first, then the oldest: the newest answers survive
    assert kept == [f"status line number {i}" for i in range(5000 - 2 * n, 5000, 2)]
    monkeypatch.undo()
    spk.cancel_all(keep_critical=False)
    assert wait_for(lambda: not spk.busy, 5)


def test_shedding_never_touches_critical_below_the_hard_cap(spk, monkeypatch):
    hold(spk, monkeypatch)
    for i in range(10):
        spk.say(f"Stop. Danger {i}.", "critical")
    for i in range(1000):
        spk.say(f"chatter {i}", ["info", "nav", "hazard", "answer"][i % 4], ttl=60)
    with spk._qlock:
        crit = [q.text for q in spk._queue if q.priority == "critical"]
        assert len(spk._queue) == Speaker.MAX_QUEUE
        # what survives the shedding is the most important
        assert [q.priority for q in spk._queue].count("hazard") == Speaker.MAX_QUEUE - 10
    assert crit == [f"Stop. Danger {i}." for i in range(10)]


def test_critical_flood_hits_the_hard_cap(spk, monkeypatch):
    hold(spk, monkeypatch)
    for i in range(1000):
        spk.say(f"Stop. Thing {i}.", "critical")
    with spk._qlock:
        assert len(spk._queue) == Speaker.MAX_QUEUE_CRITICAL


def test_preemption_storm(spk):
    """A long navigation line, then four threads firing hazards and criticals at it. No deadlock,
    the queue drains, criticals are never cut, the mixer ends silent."""
    spk.say("In two hundred meters, continue straight along Moss Street past the corner store", "nav", ttl=60)
    assert wait_for(lambda: spk._current is not None, 3)
    rng = random.Random(7)

    def storm(k):
        for i in range(60):
            p = rng.choice(["hazard", "hazard", "nav", "info", "critical"])
            text = "Stop." if p == "critical" else f"{p} {k}-{i % 9}"
            spk.say(text, p, ttl=1.5)
            if i % 10 == 0:
                spk.set_listening(i % 20 == 0)
            time.sleep(rng.uniform(0, 0.01))

    ths = [threading.Thread(target=storm, args=(k,)) for k in range(4)]
    for t in ths:
        t.start()
    for t in ths:
        t.join(10)
    assert not any(t.is_alive() for t in ths)                 # say() never blocked
    spk.set_listening(False)
    assert wait_for(lambda: not spk.busy, 30)
    hist = list(spk.history)
    assert any(h["text"] == "Stop." and h["result"] == "done" for h in hist)
    assert not [h for h in hist if h["priority"] == "critical" and h["result"] == "cut"]
    assert spk.stats["preempted"] >= 1
    assert wait_for(lambda: not spk.engine.speaking, 3)
    with spk._qlock:
        assert spk._queue == [] and spk._current is None
