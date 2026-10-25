"""PhoneLink and PhoneHub without a radio: reassembly, duplicate fixes, malformed input, loss / recovery
announcements and the earpiece being down, all on an injected clock."""

import json
import threading
from types import SimpleNamespace

import pytest

from phone.fix import MAX_BUFFER, MAX_LINE
from phone.link import BACK_TEXT, LOST_TEXT, NO_GPS_TEXT, PhoneHub, PhoneLink, short_id

FIX = {"t": 1761330000123, "lat": 37.778352, "lng": -122.408768, "acc": 4.8}


def line(**kw) -> bytes:
    return (json.dumps({**FIX, **kw}, separators=(",", ":")) + "\n").encode()


@pytest.fixture
def rig():
    """A link on a fake clock that records what it sends and logs."""
    clock = SimpleNamespace(t=100.0)
    sent, logs = [], []
    link = PhoneLink(send=sent.append, clock=lambda: clock.t, log=logs.append)
    return SimpleNamespace(link=link, clock=clock, sent=sent, logs=logs)


def positions(sent):
    return [m for m in sent if m["type"] == "position"]


def says(sent):
    return [m["text"] for m in sent if m["type"] == "say"]


# ---------------------------------------------------------------- reassembly
def test_a_fix_written_one_byte_at_a_time_completes_once(rig):
    data = line()
    done = [rig.link.on_chunk(data[i:i + 1]) for i in range(len(data))]
    assert sum(done) == 1 and done[-1] == 1 and rig.link.stats["messages"] == 1
    assert rig.link.fix.lat == FIX["lat"]


def test_one_write_can_hold_several_fixes_and_blank_lines(rig):
    assert rig.link.on_chunk(line(t=1) + b"\n\r\n" + line(t=2, lat=37.1)) == 2
    assert rig.link.stats["messages"] == 2 and rig.link.fix.lat == 37.1


def test_chunk_accepts_bytearray_and_memoryview(rig):
    """CoreBluetooth and bless hand writes over as bytearray / NSData-backed buffers."""
    assert rig.link.on_chunk(bytearray(line(t=1))) == 1
    assert rig.link.on_chunk(memoryview(line(t=2))) == 1


# ---------------------------------------------------------------- malformed input
def test_garbage_between_fixes_is_counted_and_dropped(rig):
    chunk = b"garbage\n" + line(t=1) + b'{"lat":"north","lng":1}\n' + b"[1,2]\n" + b"\xff\xfe\n" + line(t=2)
    assert rig.link.on_chunk(chunk) == 6                       # six lines completed ...
    assert rig.link.stats == {"messages": 2, "bad": 4, "forwarded": 0, "lost": 0}   # ... two of them fixes
    assert rig.link.fix.t == 2


def test_an_overlong_line_is_bad_and_the_next_fix_still_arrives(rig):
    assert rig.link.on_chunk(b"x" * (MAX_LINE + 1) + b"\n" + line()) == 2
    assert rig.link.stats["bad"] == 1 and rig.link.stats["messages"] == 1


def test_a_lost_newline_resets_the_buffer_instead_of_growing_forever(rig):
    rig.link.on_chunk(b"{" * (MAX_BUFFER + 1))                 # no terminator in 4 KB
    assert rig.link._asm.overflows == 1 and len(rig.link._asm._buf) == 0
    assert rig.link.on_chunk(line()) == 1 and rig.link.stats["messages"] == 1


@pytest.mark.parametrize("raw", [None, 5, [1], "", "{}x", b"{" * 3000])
def test_on_message_never_raises(rig, raw):
    assert rig.link.on_message(raw) is None
    assert rig.link.stats["bad"] == 1 and rig.link.last_msg is None


def test_bad_messages_do_not_keep_the_link_alive(rig):
    rig.link.on_chunk(line())
    rig.link.tick()
    rig.clock.t = 106.0
    rig.link.on_chunk(b"garbage\n" * 3)
    rig.link.tick()
    assert says(rig.sent) == [LOST_TEXT]


# ---------------------------------------------------------------- duplicates and forwarding
def test_a_repeated_fix_is_forwarded_once_but_still_counts_as_the_link_being_up(rig):
    rig.link.on_chunk(line(t=1))
    rig.link.tick()
    rig.clock.t = 104.0
    rig.link.on_chunk(line(t=1))                               # the same fix again (retransmit / app resend)
    rig.link.tick()
    rig.clock.t = 108.0                                        # 8 s after the first, 4 s after the repeat
    rig.link.tick()
    assert len(positions(rig.sent)) == 1 and says(rig.sent) == []
    assert rig.link.stats["messages"] == 2 and rig.link.link_ok is True
    rig.link.on_chunk(line(t=2, lat=37.9))                     # a new fix is forwarded again
    rig.link.tick()
    assert [p["lat"] for p in positions(rig.sent)] == [FIX["lat"], 37.9]


def test_a_fix_without_a_timestamp_is_never_taken_for_a_duplicate(rig):
    for i in range(3):
        rig.clock.t = 100.0 + i
        rig.link.on_chunk(line(t=None, lat=37.0 + i / 10))
        rig.link.tick()
    assert [p["lat"] for p in positions(rig.sent)] == [37.0, 37.1, 37.2]


def test_forwarding_is_rate_limited_and_sends_the_newest_fix(rig):
    rig.link.on_chunk(line(t=1, lat=37.1))
    rig.link.tick(100.0)
    rig.link.on_chunk(line(t=2, lat=37.2))
    rig.link.on_chunk(line(t=3, lat=37.3))
    rig.link.tick(100.2)                                       # inside 1 / FORWARD_HZ: held
    assert [p["lat"] for p in positions(rig.sent)] == [37.1]
    rig.link.tick(100.5)
    assert [p["lat"] for p in positions(rig.sent)] == [37.1, 37.3]
    rig.link.tick(101.5)                                       # nothing new: nothing sent
    assert rig.link.stats["forwarded"] == 2


def test_a_heartbeat_without_coordinates_keeps_the_link_but_forwards_nothing(rig):
    rig.link.on_chunk(line(t=None, lat=None, lng=None, heading=90))
    assert rig.link.tick() == [] and rig.link.link_ok is True
    assert rig.link.fix.heading == 90.0 and rig.link.last_gps is None


def test_forwarded_position_is_what_the_earpiece_accepts(rig):
    from earpiece.events import parse
    rig.link.on_chunk(line(speed=1.3, course=450.0))
    (pos,) = rig.link.tick()
    m = parse(pos)
    assert m.source == "phone" and m.data["course"] == 90.0 and "walker" not in pos


# ---------------------------------------------------------------- loss, recovery, no GPS
def test_loss_and_recovery_are_announced_once_each(rig):
    rig.link.on_chunk(line(t=1))
    rig.link.tick()
    for t in (106.0, 107.0, 120.0):                            # quiet: said once, not every tick
        rig.clock.t = t
        rig.link.tick()
    rig.link.on_chunk(line(t=2))
    rig.link.tick()
    rig.clock.t = 121.0
    rig.link.tick()
    assert says(rig.sent) == [LOST_TEXT, BACK_TEXT]
    lost = next(m for m in rig.sent if m.get("text") == LOST_TEXT)
    assert (lost["priority"], lost["key"], lost["source"]) == ("system", "phone-link", "phone")
    assert rig.link.stats["lost"] == 1
    assert rig.logs[0] == "[phone] connected" and rig.logs[1].startswith("[phone] lost: nothing for 6.0 s")
    assert rig.logs[2] == "[phone] back"


def test_the_first_connection_is_not_announced_as_back(rig):
    rig.link.on_chunk(line())
    assert says(rig.link.tick()) == []


def test_no_gps_is_said_once_and_again_only_after_gps_came_back(rig):
    rig.link.on_chunk(line(t=1))
    rig.link.tick()
    for i in range(1, 40):                                     # heartbeats keep the link up, GPS is gone
        rig.clock.t = 100.0 + i
        rig.link.on_chunk(line(t=None, lat=None, lng=None))
        rig.link.tick()
    assert says(rig.sent) == [NO_GPS_TEXT]
    rig.link.on_chunk(line(t=2))
    rig.link.tick()
    for i in range(40, 60):
        rig.clock.t = 100.0 + i
        rig.link.on_chunk(line(t=None, lat=None, lng=None))
        rig.link.tick()
    assert says(rig.sent) == [NO_GPS_TEXT, NO_GPS_TEXT]


@pytest.mark.parametrize("kw", [{"walker": False}, {"announce": False}])
def test_a_silent_link_still_forwards_but_never_speaks(kw):
    clock, sent = SimpleNamespace(t=100.0), []
    link = PhoneLink(send=sent.append, clock=lambda: clock.t, **kw)
    link.on_chunk(line(t=1))
    link.tick()
    clock.t = 110.0
    link.tick()
    assert says(sent) == [] and len(positions(sent)) == 1 and link.stats["lost"] == 1
    assert positions(sent)[0].get("walker") == kw.get("walker")


def test_the_earpiece_being_down_never_stops_the_link(rig):
    def down(msg):
        raise ConnectionRefusedError("earpiece is not running")
    rig.link.send = down
    rig.link.on_chunk(line(t=1))
    out = rig.link.tick()
    assert len(positions(out)) == 1 and rig.link.stats["forwarded"] == 1
    assert rig.logs[-1] == "[phone] send failed: ConnectionRefusedError: earpiece is not running"
    rig.link.send = rig.sent.append
    rig.link.on_chunk(line(t=2))
    rig.clock.t = 101.0
    assert len(positions(rig.link.tick())) == 1 and len(rig.sent) == 1


# ---------------------------------------------------------------- ages and status
def test_ages_and_status(rig):
    assert (rig.link.link_age(), rig.link.fix_age()) == (None, None)
    assert rig.link.status() == {"ok": False, "n": 0, "fwd": 0, "fix_age": None}
    rig.link.on_chunk(line(t=1))
    rig.link.tick()
    rig.clock.t = 103.26
    rig.link.on_chunk(line(t=None, lat=None, lng=None))
    assert rig.link.link_age() == 0.0 and rig.link.fix_age() == pytest.approx(3.26)
    assert rig.link.status() == {"ok": True, "n": 2, "fwd": 1, "fix_age": 3.3}
    assert rig.link.fix_age(now=50.0) == 0.0                   # a clock behind the fix is never a negative age
    assert rig.link.alive(105.0) and not rig.link.alive(108.3)


# ---------------------------------------------------------------- threads
def test_writes_from_several_threads_are_all_counted():
    """Transports call on_chunk / on_message from their own threads; no message or bad count may be lost."""
    link = PhoneLink(clock=lambda: 100.0)
    n_threads, n_each = 6, 200
    start = threading.Barrier(n_threads)

    def writer(k):
        start.wait(timeout=5)
        for i in range(n_each):
            link.on_message(line(t=k * 10_000 + i).strip())
            link.on_message(b"garbage")

    threads = [threading.Thread(target=writer, args=(k,)) for k in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert not any(t.is_alive() for t in threads)
    assert link.stats["messages"] == link.stats["bad"] == n_threads * n_each


# ---------------------------------------------------------------- hub
def test_short_id_is_stable_hex_and_empty_without_a_central():
    assert short_id("UUID-1") == short_id("UUID-1") != short_id("UUID-2")
    assert int(short_id("UUID-1"), 16) >= 0 and len(short_id("UUID-1")) == 4
    assert short_id(None) == short_id("") == ""


def test_hub_walker_link_follows_the_walker():
    clock = SimpleNamespace(t=100.0)
    hub = PhoneHub(clock=lambda: clock.t)
    assert hub.walker_link() is None
    hub.on_chunk(line(), central="A")
    hub.tick()
    assert hub.walker_link() is hub.links[short_id("A")] and hub.walker_link().walker is True


def test_hub_a_new_phone_after_the_walker_was_lost_says_back_once():
    """The walker heard 'lost'; a phone that was never connected takes over: the hub says 'back', exactly once."""
    clock, sent = SimpleNamespace(t=100.0), []
    hub = PhoneHub(send=sent.append, clock=lambda: clock.t)
    hub.on_chunk(line(t=1), central="A")
    hub.tick()
    clock.t = 110.0
    hub.tick()
    assert says(sent) == [LOST_TEXT]
    hub.on_chunk(line(t=2), central="B")
    hub.tick()
    clock.t = 111.0
    hub.tick()
    assert hub.walker == short_id("B") and says(sent) == [LOST_TEXT, BACK_TEXT]
    back = [m for m in sent if m.get("text") == BACK_TEXT][0]
    assert back["source"] == f"phone-{short_id('B')}"


def test_hub_back_announcement_survives_the_earpiece_being_down():
    clock, logs, calls = SimpleNamespace(t=100.0), [], []

    def flaky(msg):
        calls.append(msg)
        if msg.get("text") == BACK_TEXT:
            raise OSError("earpiece gone")

    hub = PhoneHub(send=flaky, clock=lambda: clock.t, log=logs.append)
    hub.on_chunk(line(t=1), central="A")
    hub.tick()
    clock.t = 110.0
    hub.tick()
    hub.on_chunk(line(t=2), central="B")
    out = hub.tick()
    assert BACK_TEXT in says(out) and "[phone] send failed: OSError: earpiece gone" in logs
    assert hub.walker == short_id("B") and len(positions(out)) == 1
