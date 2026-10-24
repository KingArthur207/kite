"""The earpiece's shared state under hostile or runaway senders (null audio, offline voice)."""

from __future__ import annotations

import threading
import time

from conftest import wait_for
from earpiece.events import Msg


def test_control_loop_survives_hazards_added_from_another_thread(app):
    """The transport thread inserts hazards while the control loop iterates them. Iterating
    the live dict raised 'dictionary changed size during iteration' now and then, which
    skipped a whole control tick (sonar, watchdog, beacon). Probabilistic: a few seconds of
    churn hit it about once on a laptop before the fix."""
    t_end = time.monotonic() + 3.0
    i = 0
    while time.monotonic() < t_end:
        app._stop_since = time.monotonic() - 2        # keep _check_continue walking the hazards
        app._apply_hazard(f"k{i}", Msg("hazard", "x", kind="construction",
                                       state="wait" if i % 2 else None, ttl=0.5), time.monotonic())
        i += 1
    assert i > 1000
    assert not [m for m in app.logs if m.startswith("[control]")]


def test_cycling_place_names_never_spawn_a_thread_per_message(app):
    """Every new `place` in a position fix started a memory-lookup thread, with no limit."""
    peak = 0
    for i in range(500):
        app.handle(Msg("position", "phone", data={"lat": 37.8, "lng": -122.4, "place": f"Street {i}"}))
        peak = max(peak, sum(1 for t in threading.enumerate() if t.name == "place-notes"))
    assert peak <= 1
    assert len(app._noted_places) <= app.MAX_NOTED_PLACES
    assert wait_for(lambda: not any(t.name == "place-notes" for t in threading.enumerate()), 3)
    app.handle(Msg("position", "phone", data={"lat": 37.8, "lng": -122.4, "place": "Moss Street"}))
    assert "Moss Street" in app._noted_places            # lookups still happen once the worker is free


def test_hazard_table_is_bounded(app):
    """Hazards live up to 120 s and each one goes into every prompt Claude gets."""
    for i in range(1000):
        app.handle(Msg("hazard", "nav", kind="construction", key=f"site-{i}", ttl=120.0, priority="info"))
    assert len(app.hazards) == app.MAX_HAZARDS
    assert set(app.hazards) == {f"site-{i}" for i in range(1000 - app.MAX_HAZARDS, 1000)}   # newest kept
    assert len(app.context()["hazards"]) == app.MAX_HAZARDS


def test_inbound_memories_are_rate_limited(app):
    """Each `memory` message rewrites the local store and (with Moss keys) syncs to the cloud."""
    for i in range(100):
        app.handle(Msg("memory", "dashboard", kind="note", text=f"note {i}", data={}))
    assert app.memory.stats["writes"] == app.MEMORY_WRITES_PER_MIN
    assert any("over 30 a minute" in m for m in app.logs)


def spoken(app):
    return [h["text"] for h in app.speaker.history]


def walk(app):
    app.handle(Msg("hazard", "drone", kind="crosswalk", state="walk", bearing=0.0, distance=6.0))


def test_dont_walk_is_never_overtaken_by_a_walk_being_confirmed(app, monkeypatch):
    """The control loop confirms a pending "walk" while the transport thread may be applying a
    "don't walk". Without a lock, the loop could take the pending walk, the don't-walk could be
    applied, and then the stale walk on top: the walker hears "Walk signal is on." last."""
    orig = app._apply_hazard
    confirming = threading.Event()

    def slow_confirm(key, msg, now):
        if msg.state == "walk" and threading.current_thread().name == "control":
            confirming.set()
            time.sleep(0.2)                         # widen the window the race needs
        return orig(key, msg, now)

    monkeypatch.setattr(app, "_apply_hazard", slow_confirm)
    walk(app)
    assert confirming.wait(3)
    app.handle(Msg("hazard", "drone", kind="crosswalk", state="dont_walk", bearing=0.0, distance=6.0))
    time.sleep(0.3)
    assert app.hazards["crosswalk"]["state"] == "dont_walk"
    assert wait_for(lambda: "Don't walk." in spoken(app), 3)
    said = spoken(app)
    assert "Walk signal is on." not in said[said.index("Don't walk."):]


def test_perception_source_set_is_bounded(app):
    for i in range(1000):
        app.handle(Msg("obstacles", f"cam{i}", seq=1, objects=[]))
    assert len(app.perception_sources) == app.MAX_SOURCES
