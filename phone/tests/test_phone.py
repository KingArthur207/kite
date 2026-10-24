"""Phone link over Bluetooth, without a radio: parsing, chunk reassembly, the GATT write path, the server build."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from phone import FIX_CHAR_UUID, SERVICE_UUID, STATUS_CHAR_UUID
from phone.ble import GattHandlers, build_server
from phone.fake_phone import chunks, fake_fix
from phone.fix import BadMessage, LineAssembler, format_line, parse_fix, to_position
from phone.link import PhoneLink

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
    link = PhoneLink(send=sent.append, clock=lambda: clock.t)
    h = GattHandlers(link)
    fix_char = SimpleNamespace(uuid=FIX_CHAR_UUID.upper())    # backends hand UUIDs over in either case
    for part in chunks((json.dumps(GPS_ONLY) + "\n").encode(), 20):
        h.write(fix_char, bytearray(part))
    h.write(SimpleNamespace(uuid=STATUS_CHAR_UUID), bytearray(b"x\n"))
    assert (h.writes, h.ignored, link.stats["messages"], link.stats["bad"]) == (len(chunks(
        (json.dumps(GPS_ONLY) + "\n").encode(), 20)), 1, 1, 0)
    link.tick(100.1)
    assert sent == [{"type": "position", "source": "phone", "lat": 37.778352, "lng": -122.408768, "accuracy": 4.8}]


class FakeServer:
    def __init__(self, name, loop):
        self.name, self.services, self.chars = name, [], {}

    async def add_new_service(self, uuid):
        self.services.append(uuid)

    async def add_new_characteristic(self, service, uuid, props, value, perms):
        self.chars[uuid] = (service, props, value, perms)


def test_build_server_registers_kite_service_and_characteristics():
    from bless import GATTCharacteristicProperties as P

    h = GattHandlers(PhoneLink())
    server = asyncio.run(build_server(h, server_cls=FakeServer))
    assert server.name == "Kite" and server.services == [SERVICE_UUID]
    assert server.write_request_func == h.write and server.read_request_func == h.read
    _, fix_props, _, _ = server.chars[FIX_CHAR_UUID]
    assert fix_props & P.write and fix_props & P.write_without_response
    _, st_props, st_value, _ = server.chars[STATUS_CHAR_UUID]
    assert st_props & P.read and st_props & P.notify
    assert st_value is None          # CoreBluetooth rejects a cached value on a notify characteristic
