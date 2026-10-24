import json
import math
import os
import random
import time
from pathlib import Path

import pytest

from conftest import wait_for
from earpiece import geo
from earpiece import route as R

DATA = Path(__file__).parent / "data"

# A live Routes API v2 response (Lyon St and Bay St -> Palace of Fine Arts), trimmed to the field mask.
LYON = {"routes": [{"distanceMeters": 225, "duration": "182s", "legs": [{"steps": [
    {"distanceMeters": 12, "polyline": {"encodedPolyline": "caveFdqjjVBA"},
     "startLocation": {"latLng": {"latitude": 37.8012983, "longitude": -122.4477054}},
     "endLocation": {"latLng": {"latitude": 37.8012801, "longitude": -122.4476964}},
     "navigationInstruction": {"maneuver": "DEPART", "instructions": "Head northwest toward Lyon St"}},
    {"distanceMeters": 87, "polyline": {"encodedPolyline": "_aveFbqjjVq@@KB]@H`@yAR"},
     "startLocation": {"latLng": {"latitude": 37.8012801, "longitude": -122.4476964}},
     "endLocation": {"latLng": {"latitude": 37.8021368, "longitude": -122.4480085}},
     "navigationInstruction": {"maneuver": "NAME_CHANGE", "instructions": "Continue onto Lyon St\nTake the stairs"}},
    {"distanceMeters": 126, "polyline": {"encodedPolyline": "kfveF`sjjV?^Sp@Wb@c@`@g@T]D"},
     "startLocation": {"latLng": {"latitude": 37.8021368, "longitude": -122.4480085}},
     "endLocation": {"latLng": {"latitude": 37.8028923, "longitude": -122.4489099}},
     "navigationInstruction": {"maneuver": "TURN_LEFT", "instructions": "Turn left\nDestination will be on the left"}},
]}]}]}


def marina() -> R.Route:
    """A recorded 957 m walk with 8 steps: a 12 m step, an instruction-less crossing, a 14 m last step."""
    d = json.loads((DATA / "route_marina_walk.json").read_text())
    return R.parse_routes(d["response"], "Filbert Street", tuple(d["destination"]))


class Rec:
    """Collects a follower's callbacks."""

    def __init__(self):
        self.said: list[tuple[str, str, float]] = []
        self.beacons: list[tuple[float, float, str | None]] = []
        self.reroutes = 0
        self.arrived: list[str | None] = []

    def follower(self, route: R.Route, **kw) -> R.RouteFollower:
        return R.RouteFollower(route, lambda t, k, ttl: self.said.append((t, k, ttl)),
                               lambda b, d, lab: self.beacons.append((b, d, lab)),
                               self._reroute, self.arrived.append, **kw)

    def _reroute(self):
        self.reroutes += 1

    def keys(self) -> list[str]:
        return [k for _, k, _ in self.said]


def walk(f: R.RouteFollower, seed: int = 0, heading: str | None = "phone", speed: float = 1.2,
         noise: float = 1.0, detour=None, max_t: float = 2000.0) -> list[float]:
    """Walk the route at `speed`, one fix a second, with phone-like GPS error: a slowly
    drifting bias (+-5-10 m), small white noise and the occasional 10 m spike.
    `noise` scales the error; `detour(t)` may return a sideways offset in meters.
    Returns progress after each fix."""
    rng = random.Random(seed)
    s = t = bx = by = 0.0
    progress = []
    while not f.done and t < max_t:
        x, y = f.point_at(s)
        (px, py), (qx, qy) = f.point_at(min(s, f.length - 1.0)), f.point_at(min(s, f.length - 1.0) + 1.0)
        dx, dy = qx - px, qy - py                               # walking direction here
        n = math.hypot(dx, dy) or 1.0
        if detour is not None:
            off = detour(t)
            x, y = x - dy / n * off, y + dx / n * off
        if noise:
            bx, by = 0.97 * bx + rng.gauss(0, 0.8 * noise), 0.97 * by + rng.gauss(0, 0.8 * noise)
            wx, wy = rng.gauss(0, noise), rng.gauss(0, noise)
            if rng.random() < 0.03:
                wx += rng.choice((-10.0, 10.0)) * noise
            x, y = x + bx + wx, y + by + wy
        hd = None
        if heading == "phone":
            hd = (f.frame.heading_of(dx, dy) + rng.gauss(0, 8)) % 360
        lat, lng = f.frame.to_latlng(x, y)
        f.update(lat, lng, hd, now=t)
        progress.append(f.progress)
        t += 1.0
        s = min(s + speed, f.length)
    return progress


# ====================================================================== geo
def test_haversine_and_bearing_known_values():
    assert geo.haversine_m(37.8, -122.44, 37.8, -122.44) == 0
    assert geo.haversine_m(0, 0, 1, 0) == pytest.approx(111195, rel=1e-4)        # one degree of latitude
    assert geo.haversine_m(37.80, -122.44, 37.81, -122.44) == pytest.approx(1111.95, abs=0.5)
    # San Francisco City Hall -> Los Angeles City Hall, about 559 km
    assert geo.haversine_m(37.7793, -122.4193, 34.0537, -118.2428) == pytest.approx(559_000, rel=0.005)
    assert geo.bearing_deg(37.8, -122.44, 37.81, -122.44) == pytest.approx(0.0, abs=1e-9)
    assert geo.bearing_deg(37.8, -122.44, 37.8, -122.43) == pytest.approx(90.0, abs=0.01)
    assert geo.bearing_deg(37.81, -122.44, 37.80, -122.44) == pytest.approx(180.0, abs=1e-9)
    assert geo.bearing_deg(37.8, -122.43, 37.8, -122.44) == pytest.approx(270.0, abs=0.01)
    assert geo.bearing_deg(0, 0, 1, 1) == pytest.approx(45.0, abs=0.01)


def test_relative_bearing_is_in_the_walkers_frame():
    assert geo.relative_bearing(10, 350) == pytest.approx(20)
    assert geo.relative_bearing(350, 10) == pytest.approx(-20)
    assert geo.relative_bearing(90, 0) == pytest.approx(90)      # target east, facing north: on the right
    assert geo.relative_bearing(270, 0) == pytest.approx(-90)
    assert abs(geo.relative_bearing(180, 0)) == pytest.approx(180)


@pytest.mark.parametrize("rotation", [0.0, 25.0, -110.0])
def test_local_frame_round_trip_under_a_centimetre_over_a_kilometre(rotation):
    f = geo.LocalFrame(37.8013, -122.4477, rotation)
    for b in range(0, 360, 15):
        for r in (1.0, 250.0, 1000.0):
            x, y = r * math.sin(math.radians(b)), r * math.cos(math.radians(b))
            lat, lng = f.to_latlng(x, y)
            x2, y2 = f.to_xy(lat, lng)
            assert math.hypot(x2 - x, y2 - y) < 0.01
            assert geo.haversine_m(37.8013, -122.4477, lat, lng) == pytest.approx(r, rel=0.005)
    for lat, lng in [(37.8, -122.44), (37.81, -122.455), (37.795, -122.437)]:
        assert f.to_latlng(*f.to_xy(lat, lng)) == pytest.approx((lat, lng), abs=1e-9)


def test_local_frame_axes_and_headings():
    north = geo.LocalFrame(37.8, -122.44)
    lat, lng = north.to_latlng(0, 100)
    assert geo.bearing_deg(37.8, -122.44, lat, lng) == pytest.approx(0, abs=0.01)
    east = geo.LocalFrame(37.8, -122.44, rotation_deg=90)        # sim +y points east
    x, y = east.to_xy(*north.to_latlng(100, 0))
    assert (x, y) == pytest.approx((0, 100), abs=0.01)
    assert east.heading_of(0, 1) == pytest.approx(90)
    assert east.heading_of(-1, 0) == pytest.approx(0)
    for h in (0, 33, 180, 271):
        assert east.heading_of(*east.direction_of(h)) == pytest.approx(h)


def test_local_frame_from_landmarks_recovers_the_sim_mapping():
    truth = geo.LocalFrame(37.8005, -122.4470, rotation_deg=-32.5)
    marks = [(ll, truth.to_xy(*ll)) for ll in [(37.8013, -122.4477), (37.8021, -122.4480), (37.8029, -122.4489)]]
    f = geo.LocalFrame.from_landmarks(marks)
    assert f.rotation_deg == pytest.approx(-32.5, abs=0.01)
    assert geo.haversine_m(f.anchor_lat, f.anchor_lng, truth.anchor_lat, truth.anchor_lng) < 0.05
    assert f.fit_rms_m < 0.05
    one = geo.LocalFrame.from_landmarks(marks[:1], rotation_deg=-32.5)
    assert one.to_xy(37.8021, -122.4480) == pytest.approx(truth.to_xy(37.8021, -122.4480), abs=0.05)


def test_heading_from_track():
    f = geo.LocalFrame(37.8, -122.44)
    north = [(t * 0.1, *f.to_latlng(0, 1.2 * t * 0.1)) for t in range(40)]     # 10 Hz, walking north
    assert geo.heading_from_track(north) == pytest.approx(0, abs=0.5)
    east = [(t, *f.to_latlng(1.3 * t, 0)) for t in range(6)]                   # 1 Hz GPS, walking east
    assert geo.heading_from_track(east) == pytest.approx(90, abs=0.5)
    rng = random.Random(3)
    still = [(t, *f.to_latlng(rng.uniform(-1, 1), rng.uniform(-1, 1))) for t in range(10)]
    assert geo.heading_from_track(still) is None                              # GPS wander, not walking
    assert geo.heading_from_track(north[:1]) is None
    assert geo.heading_from_track([]) is None


# ====================================================================== parsing and wording
def test_decode_polyline_matches_googles_reference_example():
    pts = R.decode_polyline("_p~iF~ps|U_ulLnnqC_mqNvxq`@")
    assert pts == pytest.approx([(38.5, -120.2), (40.7, -120.95), (43.252, -126.453)])


def test_parse_live_response_and_make_it_speakable():
    r = R.parse_routes(LYON, "Palace of Fine Arts")
    assert len(r.steps) == 3 and r.distance_m == 225 and r.duration_s == 182
    assert r.destination == pytest.approx((37.8028923, -122.4489099))
    assert [s.maneuver for s in r.steps] == ["DEPART", "NAME_CHANGE", "TURN_LEFT"]
    assert R.sentences(r.steps[1].instruction) == ["Continue onto Lyon Street", "Take the stairs"]
    assert R.sentences(r.steps[2].instruction) == ["Turn left", "Destination on your left"]
    pts, starts = r.polyline()
    assert starts[0] == 0 and starts == sorted(starts) and pts[starts[2]] == pytest.approx(r.steps[2].start, abs=1e-5)
    with pytest.raises(R.RouteError):
        R.parse_routes({})


@pytest.mark.parametrize("raw,spoken", [
    ("Head north on Lyon St toward Bay St/Marina Blvd", "Head north on Lyon Street toward Bay Street and Marina Boulevard"),
    ("Turn left onto St Francis Blvd", "Turn left onto St Francis Boulevard"),
    ("Turn right onto N Point St", "Turn right onto North Point Street"),
    ("Slight left onto Richardson Ave", "Bear left onto Richardson Avenue"),
    ("Turn slight right onto Marina Blvd", "Bear right onto Marina Boulevard"),
    ("Destination will be on the right", "Destination on your right"),
])
def test_speakable(raw, spoken):
    assert R.speakable(raw) == spoken


def test_fit_keeps_cues_short_and_never_dangling():
    long = "In 30 meters, bear left onto Golden Gate Promenade toward the Marina Green Drive parking lot."
    assert R.fit(long) == "In 30 meters, bear left onto Golden Gate Promenade."
    t = R.fit("one two three four five six seven eight nine ten eleven onto twelve thirteen")
    assert len(t.split()) <= 12 and not t.rstrip(".").endswith("onto")


# ====================================================================== following a route
def assert_clean_walk(rec: Rec, f: R.RouteFollower, progress: list[float]):
    keys = rec.keys()
    announced = [m.index for m in f.maneuvers]
    assert keys[0] == "route:start"
    for i in announced:
        assert keys.count(f"route:now:{i}") == 1, (i, keys)
        assert keys.count(f"route:pre:{i}") <= 1
    order = [int(k.split(":")[2]) for k in keys if k.startswith("route:now:")]
    assert order == announced                                    # in order, none skipped
    pre_then_now = [k for k in keys if k.startswith(("route:pre:", "route:now:")) and not k.endswith("arrive")]
    for i in announced:                                          # a heads-up always comes before its "now"
        if f"route:pre:{i}" in pre_then_now:
            assert pre_then_now.index(f"route:pre:{i}") < pre_then_now.index(f"route:now:{i}")
    assert all(b >= a for a, b in zip(progress, progress[1:]))  # never goes backwards
    assert rec.arrived == ["Filbert Street"] and rec.reroutes == 0
    assert all(len(t.split()) <= R.MAX_WORDS for t, _, _ in rec.said), rec.said


@pytest.mark.parametrize("seed", range(6))
def test_jittery_walk_announces_each_maneuver_once_in_order(seed):
    rec = Rec()
    f = rec.follower(marina())
    assert [m.index for m in f.maneuvers] == [1, 2, 4, 5, 6, 7]  # step 3 has no instruction: silent
    progress = walk(f, seed)
    assert_clean_walk(rec, f, progress)
    assert keys_with(rec, "route:now:2") == ["Now turn left onto Chestnut Street."]
    assert keys_with(rec, "route:now:5") == ["Continue onto Lombard Street."]
    assert keys_with(rec, "route:pre:arrive") and "on your left" in keys_with(rec, "route:pre:arrive")[0]
    walk(f, seed + 100)                                          # still walking about after arriving
    assert rec.arrived == ["Filbert Street"]


@pytest.mark.parametrize("seed", range(3))
def test_jittery_walk_with_heading_from_the_track_only(seed):
    rec = Rec()
    f = rec.follower(marina())
    assert_clean_walk(rec, f, walk(f, seed, heading=None))
    assert len(rec.beacons) > 500                                # the track gives a heading while walking


def keys_with(rec: Rec, key: str) -> list[str]:
    return [t for t, k, _ in rec.said if k == key]


def test_heads_up_wording_and_timing():
    rec = Rec()
    f = rec.follower(marina())
    walk(f, noise=0)
    pre = keys_with(rec, "route:pre:2")
    assert pre in (["In 30 meters, turn left onto Chestnut Street."], ["In 25 meters, turn left onto Chestnut Street."])
    ttl = [ttl for _, k, ttl in rec.said if k == "route:pre:2"][0]
    assert 3 <= ttl <= 10                                        # dropped if it can't be said before the turn
    assert keys_with(rec, "route:now:1") == ["Now bear left onto Richardson Avenue."]


def test_live_example_says_the_stairs_and_the_destination_side():
    rec = Rec()
    r = R.parse_routes(LYON, "Palace of Fine Arts")
    f = rec.follower(r)
    f.update(*r.steps[0].start, 350.0, now=-1.0)                 # standing at the corner, facing up Lyon St
    walk(f, noise=0)
    texts = [t for t, _, _ in rec.said]
    assert texts[0] == "Head toward Lyon Street, straight ahead."     # walker-frame, not "northwest"
    assert "Continue onto Lyon Street. Take the stairs." in texts
    assert "Now turn left." in texts
    assert any(t.startswith("Palace of Fine Arts is") and t.endswith("on your left.") for t in texts)
    assert rec.arrived == ["Palace of Fine Arts"]


def test_start_without_heading_keeps_googles_wording():
    rec = Rec()
    f = rec.follower(marina())
    lat, lng = marina().steps[0].start
    f.update(lat, lng, None, now=0.0)
    assert rec.said[0][0] == "Head south toward Richardson Avenue." and rec.beacons == []


def test_off_route_for_eight_seconds_asks_for_a_new_route_once():
    rec = Rec()
    f = rec.follower(marina())
    walk(f, 1, detour=lambda t: 40.0 if t >= 150 else 0.0, max_t=200)
    assert rec.reroutes == 1 and f.rerouting
    assert keys_with(rec, "route:off-route") == ["You're off the route. Finding a new one."]
    assert rec.arrived == []
    n = len(rec.said)
    walk(f, 2, max_t=30)                                         # rerouting: silent until replaced
    assert len(rec.said) == n and rec.reroutes == 1


def test_short_excursions_and_spikes_are_not_off_route():
    rec = Rec()
    f = rec.follower(marina())
    # 5 s across the street and back, then single 60 m spikes: never "off the route"
    progress = walk(f, 4, detour=lambda t: 40.0 if 150 <= t < 155 else (60.0 if int(t) % 97 == 0 and t > 0 else 0.0))
    assert rec.reroutes == 0
    assert_clean_walk(rec, f, progress)


def test_far_start_does_not_loop_reroutes():
    """Google may start the route at a footpath 40 m away: guide there, don't reroute."""
    rec = Rec()
    f = rec.follower(marina())
    x, y = f.point_at(0)
    for t in range(30):
        f.update(*f.frame.to_latlng(x + 40, y), 270.0, now=float(t))
    assert rec.reroutes == 0 and len(rec.beacons) > 0
    assert abs(rec.beacons[-1][0]) < 60                          # pointing west, back to the route


def test_no_heading_means_no_beacon():
    rec = Rec()
    f = rec.follower(marina())
    x, y = f.point_at(0)
    rng = random.Random(5)
    for t in range(20):                                          # standing still, GPS wandering
        f.update(*f.frame.to_latlng(x + rng.uniform(-1, 1), y + rng.uniform(-1, 1)), None, now=float(t))
    assert rec.beacons == [] and rec.keys() == ["route:start"]
    f.update(*f.frame.to_latlng(x, y), 180.0, now=21.0)          # the phone reports a heading
    b, d, label = rec.beacons[-1]
    assert abs(b) < 30 and 900 < d < 1000 and label == "Filbert Street"


def test_beacon_points_in_the_walkers_frame():
    rec = Rec()
    f = rec.follower(marina())
    lat, lng = f.frame.to_latlng(*f.point_at(20))
    ahead = f.frame.heading_of(*[b - a for a, b in zip(f.point_at(20), f.point_at(30))])
    f.update(lat, lng, ahead, now=0.0)
    assert abs(rec.beacons[-1][0]) < 5
    f.update(lat, lng, (ahead + 90) % 360, now=1.0)              # turned to face right: beacon on the left
    assert rec.beacons[-1][0] == pytest.approx(-90, abs=5)
    f.update(lat, lng, (ahead + 180) % 360, now=2.0)
    assert abs(rec.beacons[-1][0]) > 170


def test_beacon_aims_at_the_corner_not_across_it():
    rec = Rec()
    f = rec.follower(marina())
    corner = f.maneuvers[1]                                      # the left onto Chestnut
    p = f.point_at(corner.s - 12)
    heading = f.frame.heading_of(*[b - a for a, b in zip(p, f.point_at(corner.s))])
    for t, s in enumerate(range(0, int(corner.s) - 11, 2)):
        f.update(*f.frame.to_latlng(*f.point_at(s)), heading, now=float(t))
    assert abs(rec.beacons[-1][0]) < 5                           # still straight on toward the corner


def test_inaccurate_fixes_are_ignored():
    rec = Rec()
    f = rec.follower(marina())
    lat, lng = f.frame.to_latlng(*f.point_at(400))
    f.update(lat, lng, 0.0, now=0.0, accuracy=120.0)
    assert rec.said == [] and f.progress == 0


def test_straight_line_fallback():
    rec = Rec()
    r = R.straight_route((37.8013, -122.4477), (37.8029, -122.4489), "Palace of Fine Arts")
    f = rec.follower(r)
    walk(f, noise=0)
    assert rec.arrived == ["Palace of Fine Arts"] and f.maneuvers == []


# ====================================================================== the app's live route
class FakeRoutes:
    def __init__(self, route: R.Route | None = None, fail: Exception | None = None, delay: float = 0.0):
        self.route, self.fail, self.delay = route, fail, delay
        self.calls: list[tuple] = []

    def __call__(self, origin, destination, key=None, label=None, timeout=6.0):
        self.calls.append((tuple(origin), tuple(destination), label))
        time.sleep(self.delay)
        if self.fail:
            raise self.fail
        r = self.route
        return R.Route(r.steps, r.distance_m, r.duration_s, r.destination, label)


def test_guide_routes_off_thread_and_emits_ordinary_messages(monkeypatch):
    fake = FakeRoutes(marina(), delay=0.1)
    monkeypatch.setattr(R, "compute_route", fake)
    out, events = [], []
    g = R.RouteGuide(out.append, event=lambda k, d: events.append(d), log=lambda s: None)
    lat, lng = marina().steps[0].start
    t0 = time.monotonic()
    g.start((37.7975, -122.4405), "Filbert Street")                # no position yet: waits for one
    assert fake.calls == [] and g.summary() == {"destination": "Filbert Street", "state": "routing"}
    g.position({"lat": lat, "lng": lng, "heading": 180.0})
    assert time.monotonic() - t0 < 0.05                            # the network never blocks
    assert wait_for(lambda: any(m.type == "guidance" for m in out), 3)
    assert fake.calls[0][1] == (37.7975, -122.4405)
    assert out[0].type == "guidance" and out[0].priority == "nav" and out[0].source == "route"
    assert any(m.type == "beacon" and m.label == "Filbert Street" for m in out)
    assert events[0]["state"] == "started" and len(events[0]["points"]) > 10
    g.cancel()
    n = len(out)
    g.position({"lat": lat, "lng": lng, "heading": 180.0})
    assert len(out) == n and not g.active and events[-1]["state"] == "cancelled"


def test_guide_falls_back_to_a_straight_beacon_and_retries(monkeypatch):
    fake = FakeRoutes(fail=R.RouteError("Routes API 503 UNAVAILABLE"))
    monkeypatch.setattr(R, "compute_route", fake)
    monkeypatch.setattr(R.RouteGuide, "RETRY_S", 0.0)
    out = []
    g = R.RouteGuide(out.append, log=lambda s: None)
    g.position({"lat": 37.8013, "lng": -122.4477, "heading": 300.0})
    g.start((37.8029, -122.4489), "Palace of Fine Arts")
    assert wait_for(lambda: any(m.type == "beacon" for m in out), 3)
    says = [m.text for m in out if m.type == "say"]
    assert len(says) == 1 and says[0].startswith("I couldn't get walking directions.")
    assert len(says[0].split()) <= R.MAX_WORDS
    fake.fail, fake.route = None, R.parse_routes(LYON)             # the network is back
    assert wait_for(lambda: (g.position({"lat": 37.8013, "lng": -122.4477, "heading": 300.0}) or
                             any(m.type == "guidance" for m in out)), 3)
    assert len(fake.calls) >= 2


def test_guide_without_a_key_does_not_retry(monkeypatch):
    monkeypatch.delenv("GOOGLE_MAPS_API_KEY", raising=False)
    monkeypatch.setattr(R.RouteGuide, "RETRY_S", 0.0)
    out = []
    g = R.RouteGuide(out.append, log=lambda s: None)
    g.position({"lat": 37.8013, "lng": -122.4477})
    g.start((37.8029, -122.4489), "Palace of Fine Arts")
    assert wait_for(lambda: any(m.type == "say" for m in out), 3)
    for _ in range(5):
        g.position({"lat": 37.8013, "lng": -122.4477})
        time.sleep(0.02)
    assert g._attempts == 1 and g.follower is not None           # straight-line beacon instead


def test_guide_reroutes_from_the_latest_position(monkeypatch):
    fake = FakeRoutes(marina())
    monkeypatch.setattr(R, "compute_route", fake)
    out = []
    g = R.RouteGuide(out.append, log=lambda s: None)
    lat, lng = marina().steps[0].start
    g.position({"lat": lat, "lng": lng, "heading": 180.0}, now=0.0)
    g.start((37.7975, -122.4405), "Filbert Street")
    assert wait_for(lambda: g.follower is not None, 3)
    f = g.follower
    x, y = f.point_at(0)
    for t in range(1, 15):                                         # walk off to the east and keep going
        lat, lng = f.frame.to_latlng(x + 30 + 3 * t, y)
        g.position({"lat": lat, "lng": lng, "heading": 90.0}, now=float(t))
    assert wait_for(lambda: len(fake.calls) == 2, 3)
    assert fake.calls[1][0] != fake.calls[0][0]                    # from where the walker is now
    assert [m.text for m in out].count("You're off the route. Finding a new one.") == 1
    assert wait_for(lambda: g.follower is not None and g.follower is not f, 3)


@pytest.mark.skipif(not os.environ.get("KITE_LIVE_TESTS"), reason="set KITE_LIVE_TESTS=1")
def test_live_routes_api():
    r = R.compute_route((37.8013, -122.4477), (37.8029, -122.4489), label="Palace of Fine Arts")
    assert 50 < r.distance_m < 1500 and len(r.steps) >= 2
    assert r.steps[0].maneuver == "DEPART" and all(s.points for s in r.steps)
    by_address = R.compute_route((37.8013, -122.4477), "Palace of Fine Arts, San Francisco")
    assert by_address.distance_m < 3000
    rec = Rec()
    f = rec.follower(r)
    walk(f, noise=0)
    assert rec.arrived == ["Palace of Fine Arts"] and rec.reroutes == 0
