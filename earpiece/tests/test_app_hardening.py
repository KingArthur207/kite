"""The earpiece's shared state under hostile or runaway senders (null audio, offline voice)."""

from __future__ import annotations

import time

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
