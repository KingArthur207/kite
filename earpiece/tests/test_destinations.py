import time

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


@pytest.mark.skipif(not __import__("os").environ.get("KITE_LIVE_TESTS"), reason="set KITE_LIVE_TESTS=1")
def test_live_places():
    d = dst.resolve(dst.classify("the nearest pharmacy"), 37.8013, -122.4477)
    assert d.name and d.distance_m < 3000
