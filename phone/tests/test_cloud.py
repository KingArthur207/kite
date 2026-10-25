"""phone.cloud: the Kite app's fixes from SpacetimeDB's `event` table to the earpiece. A fake SQL endpoint and a local
UDP socket stand in for SpacetimeDB and the earpiece; no network, no Bluetooth."""

from __future__ import annotations

import io
import json
import socket

import pytest

from phone import cloud
from phone.cloud import CloudFeed, rows, to_fix
from phone.link import PhoneLink

COLS = ["id", "type", "source", "priority", "payload", "t"]


def app(fix_t, lat=37.7783521, lng=-122.4087683, **kw):
    return {"type": "position", "source": "phone", "lat": lat, "lng": lng, "fix_t": fix_t, **kw}


def mirror(lat=37.7783521, lng=-122.4087683):
    """What the earpiece pushes back: the validated event, no fix_t."""
    return {"type": "position", "source": "phone", "lat": lat, "lng": lng, "seq": 4}


def reply(*events):
    return [{"schema": {"elements": [{"name": {"some": c}} for c in COLS]},
             "rows": [[rid, "position", "phone", "", p if isinstance(p, str) else json.dumps(p), [0]]
                      for rid, p in events]}]


class FakeDB:
    """The `event` table behind the SQL endpoint: answers `id > N` the way SpacetimeDB does."""

    def __init__(self, *events):
        self.events = list(events)
        self.queries: list[str] = []

    def add(self, *events):
        self.events += events

    def __call__(self, q: str):
        self.queries.append(q)
        after = int(q.rsplit("id > ", 1)[1])
        return reply(*[e for e in self.events if e[0] > after])


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def make(db, clock=None):
    clock = clock or Clock()
    sent = []
    link = PhoneLink(send=sent.append, clock=clock)
    return CloudFeed(link, db, clock=clock), link, sent, clock


def test_rows_mark_the_earpiece_copy_and_junk_as_none():
    got = rows(reply((3, app(1000)), (1, mirror()), (2, "not json"), (4, [1, 2]), (5, app("1000")),
                     (6, app(float("nan"))), (7, app(True))))
    assert [r[0] for r in got] == [1, 2, 3, 4, 5, 6, 7]
    assert [r[1] is not None for r in got] == [False, False, True, False, False, False, False]


def test_rows_survive_a_malformed_reply():
    assert rows(None) == [] and rows([]) == [] and rows([{"rows": []}]) == []
    assert rows([{"schema": {"elements": [{"name": {"some": "id"}}]}, "rows": [["x"], [True], "row"]}]) == []


def test_to_fix_maps_the_payload_to_the_phone_format():
    assert to_fix(app(1000, accuracy=4.8, speed=1.2, course=90)) == {
        "t": 1000, "lat": 37.7783521, "lng": -122.4087683, "acc": 4.8, "speed": 1.2, "course": 90}


def test_the_first_poll_only_notes_where_the_log_ends():
    db = FakeDB((10, app(1000)), (11, app(2000)))
    feed, link, sent, _ = make(db)
    assert feed.poll() == 0
    assert feed.after == 11 and link.fix is None
    assert "id > 0" in db.queries[0]


def test_new_app_rows_reach_the_earpiece_and_the_mirror_never_comes_back():
    db = FakeDB((10, app(1000)))
    feed, link, sent, clock = make(db)
    feed.poll()
    db.add((12, app(2000, lat=37.78, accuracy=5.0)))
    assert feed.poll() == 1
    assert "id > 10" in db.queries[-1]
    link.tick()
    positions = [m for m in sent if m["type"] == "position"]
    assert positions == [{"type": "position", "source": "phone", "lat": 37.78, "lng": -122.4087683, "accuracy": 5.0}]
    db.add((13, mirror(lat=37.78)))                 # the earpiece's copy of what it was just given
    clock.t += 1
    assert feed.poll() == 0
    link.tick()
    assert len([m for m in sent if m["type"] == "position"]) == 1
    assert feed.stats == {"rows": 2, "fed": 1, "skipped": 1}


def test_a_heartbeat_keeps_the_link_up_without_a_second_position():
    db = FakeDB()
    feed, link, sent, clock = make(db)
    feed.poll()
    db.add((1, app(1000)))
    feed.poll()
    link.tick()
    for i in range(2, 6):                           # the app resends the same fix every 2 s while standing still
        clock.t += 2
        db.add((i, app(1000)))
        feed.poll()
        link.tick()
    assert link.link_ok
    assert len([m for m in sent if m["type"] == "position"]) == 1
    assert not [m for m in sent if m["type"] == "say"]


def test_silence_says_lost_and_the_next_fix_says_back():
    db = FakeDB()
    feed, link, sent, clock = make(db)
    feed.poll()
    db.add((1, app(1000)))
    feed.poll()
    link.tick()
    clock.t += 6
    feed.poll()
    link.tick()
    assert [m["text"] for m in sent if m["type"] == "say"] == ["Kite has lost the phone. Directions are paused."]
    db.add((2, app(9000)))
    feed.poll()
    link.tick()
    assert [m["text"] for m in sent if m["type"] == "say"][-1] == "The phone is back."


def test_a_freshly_published_database_is_picked_up():
    db = FakeDB((500, app(1000)))
    feed, link, sent, clock = make(db)
    feed.poll()
    db.events = [(1, app(5000))]                    # republished: ids start again from 1
    clock.t += cloud.REBASE_S
    feed.poll()
    assert feed.after == 1
    db.add((2, app(6000)))
    assert feed.poll() == 1


def test_a_bad_source_never_reaches_the_query():
    with pytest.raises(ValueError):
        CloudFeed(PhoneLink(), FakeDB(), source="phone' OR '1'='1")


def test_run_sends_position_events_to_the_earpiece_over_udp(monkeypatch):
    ear = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ear.bind(("127.0.0.1", 0))
    ear.settimeout(2)
    monkeypatch.setattr(cloud, "POLL_S", 0)
    db = FakeDB((7, app(1000)))
    polls = {"n": 0}

    def query(q):
        polls["n"] += 1
        if polls["n"] == 2:
            db.add((8, app(2000, lat=37.79, accuracy=3.0)))
        return db(q)

    out = io.StringIO()
    args = cloud.parser().parse_args(["--ear-host", "127.0.0.1", "--ear-port", str(ear.getsockname()[1])])
    try:
        assert cloud.run(args, query=query, stream=out, stop=lambda: polls["n"] >= 4) == 0
        got = json.loads(ear.recv(4096))
    finally:
        ear.close()
    assert got["type"] == "position" and got["source"] == "phone"
    assert (got["lat"], got["accuracy"]) == (37.79, 3.0)
    assert "1 fixes from the app, 1 positions sent" in out.getvalue()


def test_run_says_once_when_spacetimedb_is_unreachable(monkeypatch):
    monkeypatch.setattr(cloud, "POLL_S", 0)
    n = {"i": 0}

    def down(q):
        n["i"] += 1
        raise OSError("no route")

    out = io.StringIO()
    cloud.run(cloud.parser().parse_args(["--no-send"]), query=down, stream=out, stop=lambda: n["i"] >= 3)
    assert out.getvalue().count("can't reach SpacetimeDB") == 1
