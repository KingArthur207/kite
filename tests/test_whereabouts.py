"""Phone-fix text the kite window prints. No Tk and no network."""

from run.whereabouts import fix_from_position, format_fix, parse_fix

SAMPLE = (
    '{"kind":"location_fix","lat":37.8719,"lon":-122.2585,"accuracy_m":12,'
    '"altitude_m":15.2,"speed_mps":1.1,"heading_deg":274,"timestamp":"2026-10-24T01:58:00Z"}'
)


def test_parse_fix_reads_the_phone_message():
    fix = parse_fix(SAMPLE)
    assert fix is not None
    assert fix["lat"] == 37.8719
    assert fix["lon"] == -122.2585
    assert fix["accuracy_m"] == 12
    assert fix["altitude_m"] == 15.2
    assert fix["speed_mps"] == 1.1
    assert fix["heading_deg"] == 274
    assert fix["timestamp"] == "2026-10-24T01:58:00Z"


def test_parse_fix_rejects_other_board_traffic():
    assert parse_fix("hello from the board") is None
    assert parse_fix('{"kind":"heartbeat","lat":1,"lon":2}') is None
    assert parse_fix('{"kind":"location_fix","lat":true,"lon":2}') is None
    assert parse_fix('{"kind":"location_fix","lat":91,"lon":0}') is None
    assert parse_fix("") is None
    assert parse_fix(None) is None


def test_format_fix_says_when_the_board_has_it():
    fix = parse_fix(SAMPLE)
    assert fix is not None
    fix["accepted"] = True
    fix["age_s"] = 4
    assert format_fix(fix) == "gps: 37.87190, -122.25850  ±12m  alt 15m  1.1m/s  hdg 274  live"
    fix["accepted"] = False
    fix["age_s"] = 20
    assert format_fix(fix).endswith("20s old  on the way")


def test_a_spacetime_position_is_the_same_fix_the_map_draws():
    fix = fix_from_position({
        "source": "phone", "lat": 37.8719, "lng": -122.2585, "accuracy": 12,
        "alt": 15.2, "speed": 1.1, "course": 274, "t": 1761263880,
    })
    assert fix is not None
    assert fix["lat"] == 37.8719
    assert fix["lon"] == -122.2585
    assert fix["accuracy_m"] == 12
    assert fix["timestamp"] == "2025-10-23T23:58:00Z"
    assert fix_from_position({"lat": 91, "lng": 0}) is None
    assert fix_from_position(None) is None
