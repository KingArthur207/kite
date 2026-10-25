"""phone.latest reads the public `latest` table, which anyone can push to: a junk row is skipped, never a crash
in kite.navigate and never a route from a NaN."""

from __future__ import annotations

import json

from phone.latest import fixes

COLS = ["key", "type", "source", "priority", "payload", "t"]
NOW = 1_800_000_000.0


def reply(*rows):
    return [{"schema": {"elements": [{"name": {"some": c}} for c in COLS]},
             "rows": [[f"position:{src}", "position", src, "nav", payload if isinstance(payload, str)
                       else json.dumps(payload), [int(t * 1e6)]] for src, payload, t in rows]}]


def test_good_rows_newest_first():
    got = fixes(reply(("phone-a", {"lat": 37.1, "lng": -122.2, "accuracy": 5}, NOW - 10),
                      ("phone-b", {"lat": 37.2, "lng": -122.3}, NOW - 2)), now=NOW)
    assert [f["source"] for f in got] == ["phone-b", "phone-a"]
    assert got[1]["accuracy"] == 5 and got[0]["age_s"] == 2


def test_junk_rows_are_skipped():
    junk = [
        ("phone-x", "not json", NOW), ("phone-x", [1, 2], NOW), ("phone-x", {"lat": "37", "lng": -122}, NOW),
        ("phone-x", {"lat": 37}, NOW), ("phone-x", {"lat": 91, "lng": 0}, NOW), ("phone-x", {"lat": 0, "lng": 181}, NOW),
        ("phone-x", {"lat": True, "lng": 0}, NOW), ("phone-x", '{"lat": NaN, "lng": 0}', NOW),
        ("drone", {"lat": 1, "lng": 2}, NOW),
    ]
    good = ("phone-ok", {"lat": 1.5, "lng": 2.5, "source": "spoofed", "t": 0}, NOW - 1)
    got = fixes(reply(*junk, good), now=NOW)
    assert len(got) == 1 and got[0]["source"] == "phone-ok" and got[0]["t"] == NOW - 1


def test_bad_accuracy_is_dropped_not_trusted():
    got = fixes(reply(("phone-a", {"lat": 1, "lng": 2, "accuracy": -3}, NOW)), now=NOW)
    assert got and "accuracy" not in got[0]


def test_malformed_replies():
    for bad in ([], [{}], [{"schema": {"elements": []}, "rows": [5]}], None, "x"):
        assert fixes(bad, now=NOW) == []
