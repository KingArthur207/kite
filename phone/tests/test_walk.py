"""The walk preview asks Google once, then only after a real diversion, and never twice inside 20 seconds."""

import math

import phone  # noqa: F401  (puts the earpiece package on sys.path)
from earpiece.route import Route, Step
from phone.walk_server import WalkSession

LAT, LNG = 37.8667, -122.2577


def _east(lat: float, lng: float, meters: float) -> float:
    return lng + meters / (111320 * math.cos(math.radians(lat)))


def _route_from(lat: float, lng: float) -> Route:
    a = (lat, lng)
    b = (lat + 0.0005, lng)   # ~56 m north
    c = (lat + 0.0010, lng)   # ~111 m north
    steps = [
        Step("Head north on Bowditch Street", "DEPART", 56, a, b, [a, b]),
        Step("Continue north", "STRAIGHT", 55, b, c, [b, c]),
    ]
    return Route(steps, 111, 90, c, "Cloyne Court Hotel")


def test_route_is_fetched_once_and_rerouted_only_after_a_real_diversion():
    calls = []

    def compute(origin, destination, label=None, **_kwargs):
        calls.append((origin, destination, label))
        return _route_from(origin[0], origin[1])

    session = WalkSession(compute=compute)
    session.start(LAT, LNG, "Cloyne Court Hotel, 2600 Ridge Road, Berkeley, CA", now=1000)
    assert len(calls) == 1

    session.tick(LAT + 0.00015, LNG, now=1010)   # ~17 m north, still on the line
    view = session.tick(LAT + 0.00030, LNG, now=1020)
    assert len(calls) == 1
    assert view["progress_m"] > 10
    assert view["on_route"] is True
    assert abs(view["puck"][1] - LNG) < 1e-5

    off = _east(LAT + 0.00030, LNG, 40)
    session.tick(LAT + 0.00030, off, now=1030)
    held = session.tick(LAT + 0.00030, off, now=1033)
    assert len(calls) == 1
    assert held["google_calls"] == 1
    assert held["on_route"] is False

    rerouted = session.tick(LAT + 0.00030, off, now=1039)
    assert len(calls) == 2
    assert rerouted["google_calls"] == 2
    assert calls[1][0][0] == LAT + 0.00030

    # The new route starts where the walker is. Step off that one immediately.
    # RouteFollower will not ask again until this has lasted 8 seconds, and the
    # session will not spend a Google call until 20 seconds after the last one.
    off2 = _east(LAT + 0.00030, off, 40)
    session.tick(LAT + 0.00030, off2, now=1040)
    session.tick(LAT + 0.00030, off2, now=1043)
    assert session.google_calls == 2
    blocked = session.tick(LAT + 0.00030, off2, now=1048)
    assert session.google_calls == 2
    assert blocked["holding"] is True
    still = session.tick(LAT + 0.00030, off2, now=1058)
    assert still["google_calls"] == 2
    assert still["holding"] is True
