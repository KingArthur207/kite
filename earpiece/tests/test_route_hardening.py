"""Routing under real-world mess: GPS jitter and drift, wrong turns and U-turns, flaky or
rate-limited Google, and the wording the walker hears. No network: Google is faked."""

import random
import re
import time
from types import SimpleNamespace

import httpx
import pytest

from conftest import wait_for
from earpiece import geo
from earpiece import route as R
from earpiece.events import Msg, distance_phrase
from test_route import FakeRoutes, Rec, marina, walk

STREETS = ["Oak", "Pine", "Bay", "Lyon", "Union", "Pierce", "Scott", "Fillmore", "Steiner", "Baker"]
SUFFIX = ["St", "Ave", "Blvd", "Way", "Pl"]
TURNS = {-90: ("TURN_LEFT", "Turn left onto"), 90: ("TURN_RIGHT", "Turn right onto"),
         -45: ("TURN_SLIGHT_LEFT", "Slight left onto"), 45: ("TURN_SLIGHT_RIGHT", "Slight right onto"),
         0: ("NAME_CHANGE", "Continue onto"), 15: ("NAME_CHANGE", "Continue onto")}


def synthetic_route(seed: int) -> R.Route:
    """A random but plausible walk: 2 to 6 straight blocks joined by turns, with Google-style
    abbreviations in the instructions so the wording rules are exercised too."""
    rng = random.Random(seed)
    frame = geo.LocalFrame(37.80 + rng.uniform(-0.01, 0.01), -122.44 + rng.uniform(-0.01, 0.01))
    x = y = 0.0
    heading = rng.uniform(0, 360)
    steps = []
    for i in range(rng.randint(2, 6)):
        length = rng.uniform(35, 150)
        name = f"{rng.choice(STREETS)} {rng.choice(SUFFIX)}"
        if i == 0:
            man, text = "DEPART", f"Head {rng.choice(['north', 'south', 'east', 'northwest'])} on {name}"
        else:
            turn = rng.choice(list(TURNS))
            heading += turn
            man, lead = TURNS[turn]
            text = f"{lead} {name}"
        dx, dy = frame.direction_of(heading % 360)
        n = max(2, int(length // 10))
        pts = [frame.to_latlng(x + dx * length * k / n, y + dy * length * k / n) for k in range(n + 1)]
        steps.append(R.Step(text, man, length, pts[0], pts[-1], pts))
        x, y = x + dx * length, y + dy * length
    if rng.random() < 0.5:
        steps[-1].instruction += f"\nDestination will be on the {rng.choice(['left', 'right'])}"
    return R.Route(steps, sum(s.distance_m for s in steps), None, steps[-1].end, "Test Place")


# ---------------------------------------------------------------------- wording rules
_ABBREV = re.compile(r"\b(?:St|Ave|Blvd|Rd|Dr|Ln|Pl|Hwy)\b\.?")
_IMPERIAL = re.compile(r"\b(?:ft|feet|foot|yards?|miles?|km|kilometers?|metres?)\b", re.I)
_COMPASS = re.compile(r"\b(?:north|south|east|west)(?:east|west)?\b", re.I)


def assert_speakable(text: str, ttl: float, heading_known: bool):
    assert len(text.split()) <= R.MAX_WORDS, text
    assert text.endswith((".", "?")), text
    assert not _ABBREV.search(text), text                       # TTS would say "saint" or "S T"
    assert not _IMPERIAL.search(text), text                     # meters, always spelled "meters"
    for v in re.findall(r"(\d+(?:\.\d+)?) meters?\b", text):
        f = float(v)
        assert f >= 1 and (f < 10 or f % 5 == 0), text          # "in 25 meters", never "in 23 meters"
    assert "1 meters" not in text and " 0 meters" not in text, text
    for h in re.findall(r"at (\d+) o'clock", text):
        assert 1 <= int(h) <= 12, text
    if heading_known:
        assert not _COMPASS.search(text), text                  # walker's frame, not the map's
    assert 0 < ttl <= 30, (text, ttl)


def assert_clean(rec: Rec, f: R.RouteFollower, heading_known: bool, label="Test Place"):
    keys = rec.keys()
    assert len(keys) == len(set(keys)), keys                    # nothing said twice: no flapping
    order = [int(k.split(":")[2]) for k in keys if k.startswith("route:now:")]
    assert order == [m.index for m in f.maneuvers], keys        # every maneuver, once, in order
    assert not any(":uturn:" in k for k in keys), keys
    assert rec.arrived == [label] and rec.reroutes == 0
    for text, _, ttl in rec.said:
        assert_speakable(text, ttl, heading_known)


# ====================================================================== property tests
@pytest.mark.parametrize("seed", range(24))
@pytest.mark.parametrize("heading", ["phone", None])
def test_random_routes_random_walks(seed, heading):
    rec = Rec()
    f = rec.follower(synthetic_route(seed))
    progress = walk(f, seed, heading=heading, speed=random.Random(seed).uniform(0.8, 1.6))
    assert all(b >= a for a, b in zip(progress, progress[1:]))
    assert_clean(rec, f, heading_known=heading == "phone")


@pytest.mark.parametrize("seed", range(8))
def test_recorded_route_with_heavy_drift(seed):
    """Phone GPS on a bad day: bias wandering 8-12 m and spikes twice as big."""
    rec = Rec()
    f = rec.follower(marina())
    walk(f, seed, noise=1.6)
    assert_clean(rec, f, heading_known=True, label="Filbert Street")


@pytest.mark.parametrize("seed", range(6))
def test_dawdling_and_standing_still_never_rewinds_or_repeats(seed):
    """Stops at random for 5-40 s (waiting at a light, talking to someone), facing anywhere."""
    rng = random.Random(seed)
    rec = Rec()
    f = rec.follower(marina())
    s = t = 0.0
    while not f.done and t < 3000:
        if rng.random() < 0.02:
            face = rng.uniform(0, 360)
            for _ in range(rng.randint(5, 40)):
                x, y = f.point_at(s)
                f.update(*f.frame.to_latlng(x + rng.gauss(0, 3), y + rng.gauss(0, 3)), face, now=t)
                t += 1
        x, y = f.point_at(s)
        (px, py), (qx, qy) = f.point_at(min(s, f.length - 1)), f.point_at(min(s, f.length - 1) + 1)
        f.update(*f.frame.to_latlng(x + rng.gauss(0, 2), y + rng.gauss(0, 2)),
                 f.frame.heading_of(qx - px, qy - py), now=t)
        s, t = min(s + 1.2, f.length), t + 1
    assert_clean(rec, f, heading_known=True, label="Filbert Street")


# ====================================================================== wrong way
def walk_profile(f: R.RouteFollower, path, seed=0, noise=1.0, t0=0.0):
    """Feed fixes along path(t) -> (x, y) in the follower's frame, heading = direction of motion."""
    rng = random.Random(seed)
    t = t0
    prev = path(t - 1)
    while True:
        p = path(t)
        if p is None or f.done:
            return t
        hd = f.frame.heading_of(p[0] - prev[0], p[1] - prev[1]) if p != prev else None
        f.update(*f.frame.to_latlng(p[0] + rng.gauss(0, noise), p[1] + rng.gauss(0, noise)),
                 hd if hd is None else (hd + rng.gauss(0, 8)) % 360, now=t)
        prev, t = p, t + 1


def test_turning_back_is_noticed_once_before_any_reroute():
    rec = Rec()
    f = rec.follower(marina())
    turn_at = 110.0

    def there_and_back(t):                       # 110 m on, then 45 m back, then on to the end
        if t <= 92:
            return f.point_at(1.2 * t)
        if t <= 92 + 38:
            return f.point_at(turn_at - 1.2 * (t - 92))
        s = turn_at - 45 + 1.2 * (t - 130)
        return f.point_at(s) if s <= f.length + 3 else None
    walk_profile(f, there_and_back, seed=3)
    uturns = [(t, k) for t, k, _ in rec.said if ":uturn:" in k]
    assert [k for _, k in uturns] == ["route:uturn:1"]
    assert uturns[0][0] == "You're walking away from the route. Turn around."
    assert rec.reroutes == 0 and rec.arrived == ["Filbert Street"]


def test_wrong_way_from_the_start_is_noticed_before_the_reroute():
    rec = Rec()
    f = rec.follower(marina())
    (ax, ay), (bx, by) = f.point_at(0), f.point_at(5)
    ux, uy = (bx - ax) / 5, (by - ay) / 5
    walk_profile(f, lambda t: (ax - ux * 1.2 * t, ay - uy * 1.2 * t) if t <= 40 else None, seed=1)
    keys = rec.keys()
    assert "route:uturn:1" in keys
    if rec.reroutes:
        assert keys.index("route:uturn:1") < keys.index("route:off-route")


def test_a_second_u_turn_gets_its_own_cue():
    rec = Rec()
    f = rec.follower(marina())

    def zigzag(t):
        legs = [(0, 1.2, 80), (96, -1.2, 66), (66, 1.2, 280), (269, -1.2, 255), (255, 1.2, 290)]
        s, start = None, 0.0
        for a, v, b in legs:
            dur = abs(b - a) / 1.2
            if t <= start + dur:
                s = a + v * (t - start)
                break
            start += dur
        return None if s is None else f.point_at(s)
    walk_profile(f, zigzag, seed=5)
    assert [k for k in rec.keys() if ":uturn:" in k] == ["route:uturn:1", "route:uturn:2"]


@pytest.mark.parametrize("seed", range(5))
def test_standing_still_facing_backwards_is_not_a_u_turn(seed):
    rec = Rec()
    f = rec.follower(marina())
    walk_profile(f, lambda t: f.point_at(1.2 * t) if t <= 80 else None, seed=seed)
    x, y = f.point_at(96)
    back = (f.frame.heading_of(*[b - a for a, b in zip(f.point_at(97), f.point_at(96))]))
    rng = random.Random(seed)
    for t in range(81, 200):                                     # GPS wanders, phone faces back the way
        jump = rng.choice((-10.0, 10.0)) if rng.random() < 0.03 else 0.0
        f.update(*f.frame.to_latlng(x + rng.gauss(0, 2.5) + jump, y + rng.gauss(0, 2.5)), back, now=float(t))
    assert not any(":uturn:" in k for k in rec.keys()) and rec.reroutes == 0


# ====================================================================== arriving
def test_one_lucky_fix_at_the_destination_is_not_arriving():
    rec = Rec()
    f = rec.follower(marina())
    walk_profile(f, lambda t: f.point_at(1.2 * t) if t <= 100 else None, noise=0)
    dlat, dlng = f.route.destination
    f.update(dlat, dlng, 180.0, now=101.0)                       # a 800 m GPS jump onto the door
    assert rec.arrived == [] and not f.done
    walk_profile(f, lambda t: f.point_at(min(f.length, 1.2 * t)) if t <= 1000 else None, t0=102, noise=0)
    assert rec.arrived == ["Filbert Street"]


def test_walking_past_the_destination_still_arrives_once():
    rec = Rec()
    f = rec.follower(marina())
    (ax, ay), (bx, by) = f.point_at(f.length - 5), f.point_at(f.length)
    ux, uy = (bx - ax) / 5, (by - ay) / 5

    def past(t):
        s = 1.2 * t
        if s <= f.length:
            return f.point_at(s)
        return (bx + ux * (s - f.length), by + uy * (s - f.length)) if s < f.length + 40 else None
    walk_profile(f, past, seed=2)
    assert rec.arrived == ["Filbert Street"] and rec.reroutes == 0


# ====================================================================== RouteGuide: Google failing
def test_http_failures_are_classified():
    assert isinstance(R.error_for(429, "RESOURCE_EXHAUSTED"), R.RouteQuota)
    assert isinstance(R.error_for(403, "PERMISSION_DENIED"), R.RouteDenied)
    assert isinstance(R.error_for(400, "INVALID_ARGUMENT"), R.RouteDenied)
    assert isinstance(R.error_for(503, "UNAVAILABLE"), R.RouteUnavailable)
    assert isinstance(R.error_for(500, "INTERNAL"), R.RouteUnavailable)
    with pytest.raises(R.NoRoute):
        R.parse_routes({"routes": []})


@pytest.mark.parametrize("boom,cls", [
    (httpx.ReadTimeout("slow"), R.RouteUnavailable),
    (httpx.ConnectError("no wifi"), R.RouteUnavailable),
    (SimpleNamespace(status_code=429, reason_phrase="Too Many Requests",
                     json=lambda: {"error": {"status": "RESOURCE_EXHAUSTED"}}), R.RouteQuota),
    (SimpleNamespace(status_code=403, reason_phrase="Forbidden",
                     json=lambda: {"error": {"status": "PERMISSION_DENIED"}}), R.RouteDenied),
    (SimpleNamespace(status_code=502, reason_phrase="Bad Gateway", json=lambda: (_ for _ in ()).throw(ValueError())),
     R.RouteUnavailable),
])
def test_fetch_failures_never_leak_the_key(monkeypatch, boom, cls):
    def post(*a, **k):
        if isinstance(boom, Exception):
            raise boom
        return boom
    monkeypatch.setattr(httpx, "post", post)
    with pytest.raises(cls) as e:
        R.fetch_routes((37.8, -122.44), (37.81, -122.44), "SECRET-KEY-123", timeout=0.1)
    assert "SECRET-KEY-123" not in str(e.value) and "SECRET" not in repr(e.value)


def guide(monkeypatch, fake, **attrs):
    monkeypatch.setattr(R, "compute_route", fake)
    out, events = [], []
    g = R.RouteGuide(out.append, event=lambda k, d: events.append(d), log=lambda s: None)
    for k, v in attrs.items():
        setattr(g, k, v)
    return g, out, events


START = {"lat": 37.8013, "lng": -122.4477, "heading": 300.0}


@pytest.mark.parametrize("err,retry", [(R.RouteQuota("429"), 60.0), (R.RouteUnavailable("timeout"), 15.0),
                                       (R.RouteDenied("403"), None), (R.NoRoute("none"), None),
                                       (R.NoKey("no key"), None)])
def test_each_failure_retries_on_its_own_schedule(monkeypatch, err, retry):
    g, out, events = guide(monkeypatch, FakeRoutes(fail=err))
    g.position(START)
    t0 = time.monotonic()
    g.start((37.8029, -122.4489), "Palace of Fine Arts")
    assert wait_for(lambda: any(e.get("state") == "fallback" for e in events), 3)
    failed = [e for e in events if e.get("state") == "failed"][0]
    assert failed["retry_s"] == retry
    if retry is None:
        assert not g._pending
    else:
        assert g._pending and g._retry_at - t0 == pytest.approx(retry, abs=0.5)
    says = [m.text for m in out if m.type == "say"]
    lead = "No walking route found." if isinstance(err, R.NoRoute) else "No directions right now."
    assert says == [f"{lead} The beacon points to Palace of Fine Arts."]
    assert any(m.type == "beacon" and m.label == "Palace of Fine Arts" for m in out)   # never just silence


def test_no_directions_text_stays_short():
    for label in [None, "Walgreens", "Palace of Fine Arts", "The Big Long Name Of A Very Specific Bakery"]:
        for e in (None, R.NoRoute("x"), R.RouteQuota("y")):
            t = R.no_directions_text(label, e)
            assert len(t.split()) <= R.MAX_WORDS and t.endswith("."), t
    assert R.no_directions_text("The Big Long Name Of A Very Specific Bakery").endswith("your destination.")


def test_failures_are_explained_once_per_destination(monkeypatch):
    fake = FakeRoutes(marina())
    g, out, events = guide(monkeypatch, fake, REROUTE_MIN_S=0.0)
    lat, lng = marina().steps[0].start
    g.position({"lat": lat, "lng": lng, "heading": 180.0})
    g.start((37.7975, -122.4405), "Filbert Street")
    assert wait_for(lambda: g.follower is not None and not g.follower.straight, 3)
    fake.fail = R.RouteUnavailable("timeout")
    for _ in range(3):                                           # three reroutes fail in a row
        g._reroute(g._gen)
        assert wait_for(lambda: g.follower is not None and not g._busy, 3)
    assert [m.text for m in out if m.type == "say"] == ["No directions right now. The beacon points to Filbert Street."]
    g.start((37.7975, -122.4405), "Filbert Street")              # a new request may say it again
    assert wait_for(lambda: len([m for m in out if m.type == "say"]) == 2, 3)


# ====================================================================== RouteGuide: rate limits
def test_reroutes_are_rate_limited_and_the_beacon_holds_meanwhile(monkeypatch):
    fake = FakeRoutes(marina())
    g, out, events = guide(monkeypatch, fake)
    lat, lng = marina().steps[0].start
    g.position({"lat": lat, "lng": lng, "heading": 180.0})
    g.start((37.7975, -122.4405), "Filbert Street")
    assert wait_for(lambda: g.follower is not None, 3)
    g.follower.on_reroute()                                      # first reroute: straight away
    assert wait_for(lambda: len(fake.calls) == 2 and g.follower is not None and not g.follower.straight, 3)
    n = len(out)
    g.follower.on_reroute()                                      # 2nd within 20 s: held
    assert events[-1]["state"] == "holding" and g.follower.straight
    for _ in range(5):
        g.position({"lat": lat + 0.0003, "lng": lng, "heading": 180.0})
    time.sleep(0.1)
    assert len(fake.calls) == 2                                  # no request until it's allowed
    assert any(m.type == "beacon" and m.label == "Filbert Street" for m in out[n:])   # still pointing
    g._retry_at = 0.0                                            # time passes
    g.position({"lat": lat, "lng": lng, "heading": 180.0})
    assert wait_for(lambda: len(fake.calls) == 3, 3)
    assert g.stats["reroutes"] == 2 and g.stats["reroutes_delayed"] == 1


def test_reroutes_per_window_are_capped(monkeypatch):
    fake = FakeRoutes(marina())
    g, out, events = guide(monkeypatch, fake, REROUTE_MIN_S=0.0, REROUTE_MAX=3)
    lat, lng = marina().steps[0].start
    g.position({"lat": lat, "lng": lng, "heading": 180.0})
    g.start((37.7975, -122.4405), "Filbert Street")
    for i in range(3):
        assert wait_for(lambda: g.follower is not None and not g.follower.straight and not g._busy, 3)
        g.follower.on_reroute()
    assert wait_for(lambda: len(fake.calls) == 4, 3)
    assert wait_for(lambda: g.follower is not None and not g._busy, 3)
    g.follower.on_reroute()                                      # the 4th in ten minutes waits
    waits = [e["wait_s"] for e in events if e.get("state") == "rerouting"]
    assert waits[:3] == [0.0, 0.0, 0.0] and waits[3] > 500
    time.sleep(0.1)
    assert len(fake.calls) == 4


def test_off_route_line_is_not_repeated_within_a_minute(monkeypatch):
    g, out, _ = guide(monkeypatch, FakeRoutes(marina()))
    g.destination = (37.0, -122.0)
    for gen in (1, 2, 3):
        g._gen = gen
        g._guidance("You're off the route. Finding a new one.", f"route{gen}:off-route", 10.0, gen)
    assert [m.text for m in out] == ["You're off the route. Finding a new one."]
    g._off_route_said -= 61
    g._guidance("You're off the route. Finding a new one.", "route3:off-route", 10.0, 3)
    assert len(out) == 2


# ====================================================================== nav speech
def test_route_speech_is_nav_priority_with_expiry(monkeypatch):
    fake = FakeRoutes(marina())
    g, out, _ = guide(monkeypatch, fake)
    lat, lng = marina().steps[0].start
    g.position({"lat": lat, "lng": lng, "heading": 180.0})
    g.start((37.7975, -122.4405), "Filbert Street")
    assert wait_for(lambda: g.follower is not None, 3)
    f = g.follower
    s = 0.0
    while g.active and s < f.length + 5:
        x, y = f.point_at(s)
        x2, y2 = f.point_at(s + 1)
        g.position({"lat": f.frame.to_latlng(x, y)[0], "lng": f.frame.to_latlng(x, y)[1],
                    "heading": f.frame.heading_of(x2 - x, y2 - y) if (x2, y2) != (x, y) else 180.0})
        s += 1.2
    guidance = [m for m in out if m.type == "guidance"]
    assert len(guidance) >= 8 and guidance[-1].text == "You have arrived at Filbert Street."
    for m in guidance:
        assert m.priority == "nav" and m.ttl is not None and 0 < m.ttl <= 30, m
        assert_speakable(m.text, m.ttl, heading_known=True)
    assert all(m.priority == "answer" for m in out if m.type == "say")


def test_a_late_route_cue_is_dropped_not_said_late(app):
    app.speaker.listening = True                                 # the wearer is talking
    app.handle(Msg("guidance", "route", text="Now turn left onto Bay Street.", key="route9:now:3",
                   ttl=0.3, priority="nav"))
    time.sleep(0.6)
    app.speaker.listening = False
    app.speaker._kick()
    assert wait_for(lambda: app.speaker.stats["dropped_ttl"] >= 1, 3)
    time.sleep(0.2)
    assert "Now turn left onto Bay Street." not in [h["text"] for h in app.speaker.history]


def test_a_hazard_goes_before_queued_directions(app):
    app.speaker.paused = True
    app.handle(Msg("guidance", "route", text="In 25 meters, turn left onto Bay Street.", key="route9:pre:3",
                   ttl=10.0, priority="nav"))
    app.say("Car ahead.", "hazard", ttl=5)
    app.speaker.paused = False
    app.speaker._kick()
    assert wait_for(lambda: len(app.speaker.history) >= 2, 5)
    texts = [h["text"] for h in app.speaker.history]
    assert texts.index("Car ahead.") < texts.index("In 25 meters, turn left onto Bay Street.")


@pytest.mark.parametrize("m,spoken", [(0.4, "very close"), (1.0, "1 meter"), (4.26, "4.5 meters"),
                                      (12.4, "10 meters"), (23.0, "25 meters"), (98.0, "100 meters"),
                                      (143.0, "140 meters")])
def test_distance_phrases(m, spoken):
    assert distance_phrase(m) == spoken


def test_heads_up_distances_are_rounded_like_a_person_would():
    for seed in range(10):
        rec = Rec()
        f = rec.follower(synthetic_route(seed))
        walk(f, seed, noise=0)
        for text, key, _ in rec.said:
            if ":pre:" in key and not key.endswith("arrive"):
                assert re.match(r"^In (?:[1-9]\d*) meters, ", text), text
