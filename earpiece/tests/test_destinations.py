
import pytest

from conftest import wait_for
from earpiece import destinations as dst


@pytest.mark.parametrize("q,kind,what", [
    ("the end of the street", "relative", "end_of_street"), ("end of street", "relative", "end_of_street"),
    ("the end of this street please", "relative", "end_of_street"), ("the next corner", "relative", "next_intersection"),
    ("the crosswalk", "relative", "nearest_crosswalk"), ("across the street", "relative", "cross_street"),
    ("back the way I came", "relative", "back"), ("the entrance", "relative", "nearest_entrance"),
    ("the nearest pharmacy", "nearby", "pharmacy"), ("a coffee shop nearby", "nearby", "coffee shop"),
    ("bathroom near me", "nearby", "bathroom"), ("closest bus stop", "nearby", "bus stop"),
    ("Palace of Fine Arts", "named", "Palace of Fine Arts"), ("the corner store", "named", "corner store"),
    ("crosswalk on Lombard", "named", "crosswalk on Lombard"),
])
def test_classify(q, kind, what):
    d = dst.classify(q)
    assert d.kind == kind and (d.relation or d.query) == what


def test_resolve_fills_place_and_distance():
    fake = lambda q, lat, lng, nearby: [{"name": "Walgreens", "address": "2141 Chestnut St, SF",
                                         "lat": 37.8004, "lng": -122.4385}]
    d = dst.resolve(dst.classify("the nearest pharmacy"), 37.8013, -122.4477, search=fake)
    assert d.name == "Walgreens" and 700 < d.distance_m < 900
    r = d.request()
    assert r["target_type"] == "nearby" and r["lat"] == 37.8004


class _Resp:
    def __init__(self, status=200, body=None, bad_json=False):
        self.status_code, self._body, self._bad = status, body, bad_json

    def raise_for_status(self):
        import httpx
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(f"{self.status_code}", request=httpx.Request("POST", "https://x"),
                                        response=httpx.Response(self.status_code))

    def json(self):
        if self._bad:
            raise ValueError("not JSON")
        return self._body


def _fake_post(monkeypatch, resp):
    import httpx
    calls = []

    def post(url, **kw):
        calls.append({"url": url, **kw})
        if isinstance(resp, Exception):
            raise resp
        return resp
    monkeypatch.setattr(httpx, "post", post)
    return calls


def test_places_search_without_a_key_never_touches_the_network(monkeypatch):
    monkeypatch.delenv("GOOGLE_MAPS_API_KEY", raising=False)
    calls = _fake_post(monkeypatch, _Resp(200, {}))
    with pytest.raises(RuntimeError, match="GOOGLE_MAPS_API_KEY"):
        dst.places_search("pharmacy", 37.8, -122.4, True)
    assert calls == []


def test_places_search_sends_a_bounded_request_with_a_timeout(monkeypatch):
    calls = _fake_post(monkeypatch, _Resp(200, {"places": []}))
    assert dst.places_search("pharmacy", 37.8, -122.4, True, key="k", timeout=1.5) == []
    c = calls[0]
    assert c["timeout"] == 1.5 and c["headers"]["X-Goog-Api-Key"] == "k"
    assert c["json"]["maxResultCount"] == 3 and c["json"]["rankPreference"] == "DISTANCE"
    assert c["json"]["locationBias"]["circle"]["radius"] == 1500.0
    dst.places_search("Palace of Fine Arts", None, None, False, key="k")
    assert "locationBias" not in calls[1]["json"] and "rankPreference" not in calls[1]["json"]


@pytest.mark.parametrize("body", [
    {}, {"places": None}, {"places": []},
    {"places": [{}]},                                                   # nothing usable
    {"places": [{"displayName": {"text": "No location"}}]},            # no coordinates
    {"places": [{"displayName": None, "location": {"latitude": 1, "longitude": 2}}]},   # no name
    {"places": [{"displayName": {}, "location": None}]},
])
def test_places_search_drops_results_it_cannot_navigate_to(monkeypatch, body):
    _fake_post(monkeypatch, _Resp(200, body))
    assert dst.places_search("x", 37.8, -122.4, False, key="k") == []


def test_places_search_keeps_good_results_next_to_junk(monkeypatch):
    body = {"places": [{"displayName": {"text": "Walgreens"}, "shortFormattedAddress": "Chestnut St",
                        "location": {"latitude": 37.8, "longitude": -122.43}}, {}]}
    _fake_post(monkeypatch, _Resp(200, body))
    assert dst.places_search("x", 37.8, -122.4, False, key="k") == [
        {"name": "Walgreens", "address": "Chestnut St", "lat": 37.8, "lng": -122.43}]


@pytest.mark.parametrize("resp", [_Resp(403, {}), _Resp(500, {}), "timeout", _Resp(200, bad_json=True)])
def test_places_search_failures_raise_for_the_caller_to_handle(monkeypatch, resp):
    import httpx
    _fake_post(monkeypatch, httpx.ReadTimeout("slow") if resp == "timeout" else resp)
    with pytest.raises((httpx.HTTPError, ValueError)):
        dst.places_search("x", 37.8, -122.4, False, key="k")


def test_resolve_leaves_relative_destinations_and_misses_alone():
    rel = dst.classify("the next corner")
    assert dst.resolve(rel, 37.8, -122.4, search=lambda *a: pytest.fail("relative needs no search")) is rel
    named = dst.resolve(dst.classify("Nowhere Plaza"), 37.8, -122.4, search=lambda *a: [])
    assert named.name is None and named.lat is None and named.distance_m is None
    far = dst.resolve(dst.classify("Palace of Fine Arts"), None, None,
                      search=lambda *a: [{"name": "Palace", "address": None, "lat": 37.80, "lng": -122.45}])
    assert far.name == "Palace" and far.distance_m is None          # no fix, so no distance


def test_haversine():
    assert dst.haversine_m(37.8, -122.44, 37.8, -122.44) == 0
    assert 1100 < dst.haversine_m(37.80, -122.44, 37.81, -122.44) < 1120


def test_app_relative_and_nearby_destinations(app):
    events = []
    app.transport.publish = lambda o: events.append(o)
    app.places_search = lambda q, lat, lng, nearby: [{"name": "Walgreens", "address": "2141 Chestnut St, SF",
                                                      "lat": 37.8004, "lng": -122.4385}]
    app.heard("bring me to the end of the street")
    assert wait_for(lambda: "Okay, I'll guide you to the end of the street." in
                    [h["text"] for h in app.speaker.history], 3)
    req = [e for e in events if e["type"] == "voice_request"][-1]
    assert req["target_type"] == "relative" and req["relation"] == "end_of_street"
    app.handle(__import__("earpiece.events", fromlist=["parse"]).parse(
        {"type": "position", "lat": 37.8013, "lng": -122.4477, "place": "Lyon Street"}))
    app.heard("take me to the nearest pharmacy")
    assert wait_for(lambda: any(t.startswith("The nearest pharmacy is Walgreens on 2141 Chestnut St")
                                for t in [h["text"] for h in app.speaker.history]), 12)
    req = [e for e in events if e["type"] == "voice_request"][-1]
    assert req["name"] == "Walgreens" and req["target_type"] == "nearby"


def test_a_places_timeout_or_miss_is_spoken_and_handed_to_the_route_planner(app):
    """Google slow or down: the walker hears why, and the request still goes out unresolved."""
    import httpx

    from earpiece.events import parse
    events = []
    app.transport.publish = lambda o: events.append(o)
    app.handle(parse({"type": "position", "lat": 37.8013, "lng": -122.4477}))

    def slow(q, lat, lng, nearby):
        raise httpx.ReadTimeout("places took too long")
    app.places_search = slow
    app.heard("take me to the nearest pharmacy")
    assert wait_for(lambda: any(h["text"].startswith("I couldn't look that up")
                                for h in app.speaker.history), 12)
    req = [e for e in events if e["type"] == "voice_request"][-1]
    assert req["target_type"] == "nearby" and "name" not in req and "places took" not in str(req)
    app.places_search = lambda *a: []
    app.heard("take me to Nowhere Plaza")
    assert wait_for(lambda: any(h["text"] == "I couldn't find Nowhere Plaza." for h in app.speaker.history), 12)


@pytest.mark.skipif(not __import__("os").environ.get("KITE_LIVE_TESTS"), reason="set KITE_LIVE_TESTS=1")
def test_live_places():
    d = dst.resolve(dst.classify("the nearest pharmacy"), 37.8013, -122.4477)
    assert d.name and d.distance_m < 3000
