"""The dashboard's message channel is a public table: a flood of rows, or one very long row, must not take over
the walker's ear. Uses the fake SpacetimeDB from test_caregiver_inbox."""

from __future__ import annotations

from test_caregiver_inbox import FakeSpacetime, make, say

from earpiece import spacetime
from earpiece.spacetime import CAREGIVER_MAX_SPOKEN, CAREGIVER_MAX_TEXT, CAREGIVER_PREFIX, CAREGIVER_WINDOW_S


def started(fake):
    inbox, got, http = make(fake)
    inbox.poll(http)                       # startup: nothing earlier is news
    return inbox, got, http


def test_a_flood_is_capped_per_minute(monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(spacetime.time, "monotonic", lambda: clock[0])
    fake = FakeSpacetime()
    inbox, got, http = started(fake)
    for i in range(50):
        fake.add(10 + i, say(f"Message number {i}"))
    inbox.poll(http)
    assert len(got) == CAREGIVER_MAX_SPOKEN
    assert inbox.stats["flooded"] == 50 - CAREGIVER_MAX_SPOKEN
    clock[0] += CAREGIVER_WINDOW_S + 1     # a minute later the channel works again
    fake.add(100, say("I'm on my way."))
    inbox.poll(http)
    assert got[-1].text == CAREGIVER_PREFIX + "I'm on my way."


def test_a_message_longer_than_the_dashboard_allows_is_not_spoken():
    fake = FakeSpacetime()
    inbox, got, http = started(fake)
    fake.add(10, say("a " * (CAREGIVER_MAX_TEXT // 2 + 5)))
    fake.add(11, say("x" * CAREGIVER_MAX_TEXT))
    inbox.poll(http)
    assert [m.text for m in got] == [CAREGIVER_PREFIX + "x" * CAREGIVER_MAX_TEXT]
    assert inbox.stats["too_long"] == 1


def test_rejected_rows_do_not_use_up_the_budget():
    fake = FakeSpacetime()
    inbox, got, http = started(fake)
    for i in range(20):
        fake.add(10 + i, say("You can cross now"))       # the go-ahead filter drops these
    fake.add(40, say("Call me when you get there."))
    inbox.poll(http)
    assert [m.text for m in got] == [CAREGIVER_PREFIX + "Call me when you get there."]
    assert inbox.stats["flooded"] == 0
