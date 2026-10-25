"""Phone link over Bluetooth, without a radio: parsing, chunk reassembly, the GATT write path, the server build."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from phone import FIX_CHAR_UUID, SERVICE_UUID, STATUS_CHAR_UUID
from phone.ble import GattHandlers, build_server
from phone.fake_phone import chunks, fake_fix
from phone.fix import BadMessage, LineAssembler, format_line, parse_fix, to_position
from phone.link import BACK_TEXT, LOST_TEXT, PhoneHub, short_id

GPS_ONLY = {"t": 1761330000123, "lat": 37.778352, "lng": -122.408768, "acc": 4.8, "alt": 12.3,
            "speed": None, "course": None}


def test_gps_only_fix_has_no_heading_and_the_line_does_not_mention_a_compass():
    fix = parse_fix(json.dumps(GPS_ONLY))
    assert (fix.lat, fix.lng, fix.acc, fix.speed, fix.course, fix.heading) == (37.778352, -122.408768, 4.8,
                                                                             None, None, None)
    pos = to_position(fix)
    assert pos == {"type": "position", "source": "phone", "lat": 37.778352, "lng": -122.408768, "accuracy": 4.8}
    line = format_line(fix, 0.5, 0.5, True)
    assert "compass" not in line and "heading" not in line
    assert line.startswith("37.778352, -122.408768  +/-5 m  |  - m/s")


def test_heading_when_the_phone_sends_one():
    fix = parse_fix({**GPS_ONLY, "heading": 268.9, "heading_acc": 10})
    assert to_position(fix)["heading"] == 268.9
    assert "heading 268.9 W" in format_line(fix, 0.5, 0.5, True)


def test_gps_course_is_the_heading_while_walking():
    fix = parse_fix({**GPS_ONLY, "speed": 1.3, "course": 90.0})
    assert to_position(fix)["heading"] == 90.0
    assert "1.3 m/s toward E" in format_line(fix, 0.5, 0.5, True)


@pytest.mark.parametrize("raw", [b"not json", b"[1,2]", b"\xff\xfe", b"{" * 2000, '{"lat":"north"}'])
def test_garbage_is_a_bad_message(raw):
    with pytest.raises(BadMessage):
        parse_fix(raw)


def test_assembler_joins_chunks_and_splits_messages():
    a = LineAssembler()
    one, two = fake_fix(0, 37.0, -122.0), fake_fix(1, 37.0, -122.0)
    parts = chunks(one + two, 20)
    out = [line for p in parts for line in a.feed(p)]
    assert out == [one.strip(), two.strip()]
    assert a.feed(b"partial") == [] and a.feed(b"\n") == [b"partial"]


def test_gatt_write_of_fix_chunks_reaches_the_link_and_forwards_a_position():
    sent = []
    clock = SimpleNamespace(t=100.0)
    hub = PhoneHub(send=sent.append, clock=lambda: clock.t)
    h = GattHandlers(hub)
    fix_char = SimpleNamespace(uuid=FIX_CHAR_UUID.upper())    # backends hand UUIDs over in either case
    for part in chunks((json.dumps(GPS_ONLY) + "\n").encode(), 20):
        h.write(fix_char, bytearray(part))                    # no central (WinRT): one shared link, "phone"
    h.write(SimpleNamespace(uuid=STATUS_CHAR_UUID), bytearray(b"x\n"))
    link = hub.links[""]
    assert (h.writes, h.ignored, link.stats["messages"], link.stats["bad"]) == (len(chunks(
        (json.dumps(GPS_ONLY) + "\n").encode(), 20)), 1, 1, 0)
    hub.tick(100.1)
    assert sent == [{"type": "position", "source": "phone", "lat": 37.778352, "lng": -122.408768, "accuracy": 4.8,
                     "walker": True}]


def _phones(sent, clock, **kw):
    hub = PhoneHub(send=sent.append, clock=lambda: clock.t, **kw)
    return hub, GattHandlers(hub), SimpleNamespace(uuid=FIX_CHAR_UUID)


def _feed(h, char, central, lat, t, clock):
    msg = (json.dumps({**GPS_ONLY, "t": t, "lat": lat, "speed": 1.3, "course": 90.0}) + "\n").encode()
    return [(central, p) for p in chunks(msg, 20)]


def test_two_interleaved_centrals_give_two_clean_sources_and_one_walker():
    sent, clock = [], SimpleNamespace(t=100.0)
    hub, h, char = _phones(sent, clock)
    a, b = "UUID-EDDY", "UUID-ARTHUR"
    pa, pb = _feed(h, char, a, 37.1, 1, clock), _feed(h, char, b, 37.2, 1, clock)
    for i in range(max(len(pa), len(pb))):                    # chunk by chunk, alternating phones
        for c, part in (pa[i:i + 1] + pb[i:i + 1]):
            h.write(char, bytearray(part), central=c)
    hub.tick(100.1)
    sa, sb = f"phone-{short_id(a)}", f"phone-{short_id(b)}"
    assert len(short_id(a)) == 4 and sa != sb
    pos = {m["source"]: m for m in sent if m["type"] == "position"}
    assert set(pos) == {sa, sb}
    assert (pos[sa]["lat"], pos[sa]["walker"]) == (37.1, True)            # first to connect guides
    assert (pos[sb]["lat"], pos[sb]["walker"]) == (37.2, False)
    assert pos[sa]["speed"] == 1.3 and pos[sa]["course"] == 90.0
    assert all(x.stats["bad"] == 0 for x in hub.links.values())
    from earpiece.events import parse
    m = parse(pos[sb])
    assert m.source == sb and m.data["walker"] is False and m.data["course"] == 90.0


def test_walker_fails_over_silently_and_only_the_walker_announces():
    sent, clock = [], SimpleNamespace(t=100.0)
    hub, h, char = _phones(sent, clock, link_stale_s=5.0)
    a, b = "A", "B"
    for c, p in _feed(h, char, a, 37.1, 1, clock) + _feed(h, char, b, 37.2, 1, clock):
        h.write(char, bytearray(p), central=c)
    hub.tick(100.1)
    assert hub.walker == short_id(a)
    for t in range(2, 20):                                    # B keeps talking, A goes quiet
        clock.t = 100.0 + t
        for c, p in _feed(h, char, b, 37.2, t, clock):
            h.write(char, bytearray(p), central=c)
        hub.tick()
    assert hub.walker == short_id(b)
    says = [m for m in sent if m["type"] == "say"]
    assert says == []                                         # handover: directions never stopped
    for t in range(20, 30):                                   # now B goes quiet too: B (walker) is lost
        clock.t = 100.0 + t
        hub.tick()
    says = [m for m in sent if m["type"] == "say"]
    assert [(m["source"], m["text"]) for m in says] == [(f"phone-{short_id(b)}", LOST_TEXT)]
    for c, p in _feed(h, char, a, 37.1, 30, clock):           # A comes back: it takes over and says so
        h.write(char, bytearray(p), central=c)
    hub.tick()
    assert hub.walker == short_id(a)
    assert [m["text"] for m in sent if m["type"] == "say"] == [LOST_TEXT, BACK_TEXT]


def test_corebluetooth_write_handler_passes_the_central_through():
    from phone.ble import _cb_write_requests
    got = []
    req = SimpleNamespace(central=lambda: SimpleNamespace(identifier=lambda: SimpleNamespace(UUIDString=lambda: "C")),
                          characteristic=lambda: SimpleNamespace(UUID=lambda: SimpleNamespace(UUIDString=lambda: "F")),
                          value=lambda: b"x")
    pm = SimpleNamespace(respondToRequest_withResult_=lambda r, code: got.append(("respond", code)))
    _cb_write_requests(SimpleNamespace(write_request_func=lambda u, v, central: got.append((u, v, central))),
                       pm, [req])
    assert got == [("F", b"x", "C"), ("respond", 0)]


def test_walker_flag_picks_that_phone():
    sent, clock = [], SimpleNamespace(t=100.0)
    hub, h, char = _phones(sent, clock, walker=f"phone-{short_id('B')}")
    for c, p in _feed(h, char, "A", 37.1, 1, clock) + _feed(h, char, "B", 37.2, 1, clock):
        h.write(char, bytearray(p), central=c)
    hub.tick(100.1)
    assert hub.walker == short_id("B")


def test_earpiece_guides_only_from_the_walker_phone():
    from earpiece.events import parse
    calls = []
    app = SimpleNamespace(bridges=[], position=None, route=SimpleNamespace(position=lambda p, now: calls.append(p)),
                          _noted_places=set())
    from earpiece.app import Earpiece
    Earpiece.handle(app, parse({"type": "position", "source": "phone-aaaa", "lat": 1.0, "lng": 2.0,
                                "walker": False}))
    assert calls == [] and app.position is None
    Earpiece.handle(app, parse({"type": "position", "source": "phone-bbbb", "lat": 3.0, "lng": 4.0,
                                "walker": True, "speed": 1.2}))
    assert app.position == {"lat": 3.0, "lng": 4.0, "speed": 1.2} and calls == [app.position]


class FakeServer:
    def __init__(self, name, loop):
        self.name, self.services, self.chars = name, [], {}

    async def add_new_service(self, uuid):
        self.services.append(uuid)

    async def add_new_characteristic(self, service, uuid, props, value, perms):
        self.chars[uuid] = (service, props, value, perms)


def test_build_server_registers_kite_service_and_characteristics():
    from bless import GATTCharacteristicProperties as P

    h = GattHandlers(PhoneHub())
    server = asyncio.run(build_server(h, server_cls=FakeServer))
    assert server.name == "Kite" and server.services == [SERVICE_UUID]
    assert server.write_request_func == h.write and server.read_request_func == h.read
    _, fix_props, _, _ = server.chars[FIX_CHAR_UUID]
    assert fix_props & P.write and fix_props & P.write_without_response
    _, st_props, st_value, _ = server.chars[STATUS_CHAR_UUID]
    assert st_props & P.read and st_props & P.notify
    assert st_value is None          # CoreBluetooth rejects a cached value on a notify characteristic
