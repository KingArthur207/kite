"""kite.navigate: parsing what the walker said, the Places / Routes replies, and the spoken plan or error. Google is
faked at urllib.request.urlopen: no test reaches the network."""

import io
import json
import ssl
import sys
import urllib.error
import urllib.request

import pytest

import kite.navigate as nav
from kite.navigate import Place, Plan, choose, decode_polyline, destination_text

HOME = (37.8667, -122.2577)


def test_destination_text_strips_the_ask_and_spots_nearest():
    assert destination_text("Take me to the nearest pharmacy.") == ("pharmacy", True)
    assert destination_text("take me to Doe Library") == ("Doe Library", False)
    assert destination_text("navigate to coffee near me") == ("coffee", True)


def test_decode_polyline_matches_googles_example():
    assert decode_polyline("_p~iF~ps|U_ulLnnqC_mqNvxq`@") == [(38.5, -120.2), (40.7, -120.95), (43.252, -126.453)]


def test_choose_rejects_an_out_of_range_answer():
    p = Plan("clarify", "x", "Which one?", (0.0, 0.0), choices=[Place("a", None, 0, 0)])
    assert choose(p, 2).status == "error"


# ---------------------------------------------------------------- helpers: a fake Google
def _no_network(*a, **kw):
    raise AssertionError("a test tried to reach the network")


@pytest.fixture
def google(monkeypatch):
    """A fake Google: urlopen answers from `replies` by URL (an exception is raised) and records every request."""
    state = {"requests": [], "replies": {}}

    def urlopen(req, timeout=None, context=None):
        state["requests"].append(req)
        reply = state["replies"][req.full_url]
        if isinstance(reply, BaseException):
            raise reply
        return io.BytesIO(json.dumps(reply).encode())

    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "test-key-123")
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return state


def _place(name, lat, lng, addr=None):
    return {"displayName": {"text": name}, "location": {"latitude": lat, "longitude": lng},
            **({"shortFormattedAddress": addr} if addr else {})}


ROUTE = {"routes": [{"distanceMeters": 412, "duration": "331s", "polyline": {"encodedPolyline": "_p~iF~ps|U_ulLnnqC"},
                     "legs": [{"steps": [{"navigationInstruction": {"instructions": "Head north\non Bowditch St"}},
                                         {"navigationInstruction": {"instructions": "  "}},
                                         {},
                                         {"navigationInstruction": {"instructions": "Turn left"}}]}]}]}


# ---------------------------------------------------------------- parsing and geometry
@pytest.mark.parametrize("said,want", [
    ("Please bring me to a nearby cafe!", ("cafe", True)),
    ("directions to the Campanile", ("Campanile", False)),
    ("pharmacy around here", ("pharmacy", True)),
    ("  go   to   the   closest  bus stop ", ("bus stop", True)),
    ("Doe Library", ("Doe Library", False)),
    ("", ("", False)),
])
def test_destination_text_more_phrasings(said, want):
    """The ask is stripped however it is phrased; 'nearest / nearby / around here' marks a category search."""
    assert destination_text(said) == want


def test_haversine_one_degree_of_latitude():
    """~111.2 km per degree of latitude, zero for the same point."""
    assert nav.haversine_m(0, 0, 1, 0) == pytest.approx(111195, rel=1e-3)
    assert nav.haversine_m(*HOME, *HOME) == 0.0


def test_how_far_rounds_for_speech():
    """Tens of metres under a kilometre, tenths of a kilometre above; nothing when unknown."""
    assert nav._how_far(None) == ""
    assert nav._how_far(234) == ", 230 meters away"
    assert nav._how_far(1500) == ", 1.5 kilometers away"


# ---------------------------------------------------------------- the key and the HTTP layer
def test_key_from_the_environment_then_the_repo_env_file(monkeypatch, tmp_path):
    """GOOGLE_MAPS_API_KEY wins; else the repo's .env (quotes stripped); else a clear error."""
    monkeypatch.setattr(nav, "__file__", str(tmp_path / "kite" / "navigate.py"))  # the 'repo' is tmp_path
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "from-env")
    assert nav._key() == "from-env"
    monkeypatch.delenv("GOOGLE_MAPS_API_KEY")
    with pytest.raises(RuntimeError, match="no GOOGLE_MAPS_API_KEY"):
        nav._key()
    (tmp_path / ".env").write_text("OTHER=1\n  GOOGLE_MAPS_API_KEY='from-file'\n")
    assert nav._key() == "from-file"


def test_ssl_context_with_and_without_certifi(monkeypatch):
    """certifi's CA bundle when installed, the system store otherwise."""
    assert isinstance(nav._ssl(), ssl.SSLContext)
    monkeypatch.setitem(sys.modules, "certifi", None)
    assert isinstance(nav._ssl(), ssl.SSLContext)


def test_post_sends_the_key_in_a_header_never_the_url(google):
    """POST JSON with X-Goog-Api-Key and the field mask; the key is not in the URL."""
    google["replies"][nav.PLACES_URL] = {"places": []}
    assert nav._post(nav.PLACES_URL, {"a": 1}, "places.location") == {"places": []}
    (req,) = google["requests"]
    assert req.get_method() == "POST" and json.loads(req.data) == {"a": 1}
    assert req.get_header("X-goog-api-key") == "test-key-123" and "test-key-123" not in req.full_url
    assert req.get_header("X-goog-fieldmask") == "places.location"


@pytest.mark.parametrize("body,why", [(b'{"error": {"status": "PERMISSION_DENIED"}}', "PERMISSION_DENIED"),
                                      (b"<html>quota</html>", "Forbidden"), (b"{}", "Forbidden")])
def test_post_turns_an_http_error_into_a_short_reason(google, body, why):
    """Google's error status when it sent JSON, the HTTP reason otherwise; never the key."""
    google["replies"][nav.ROUTES_URL] = urllib.error.HTTPError(nav.ROUTES_URL, 403, "Forbidden", {}, io.BytesIO(body))
    with pytest.raises(RuntimeError) as e:
        nav._post(nav.ROUTES_URL, {}, "x")
    assert str(e.value) == f"Google said 403 {why}" and "test-key-123" not in str(e.value)


# ---------------------------------------------------------------- Places and Routes replies
def test_search_places_keeps_the_closest_three_valid_places(google):
    """Places without a name or a location are dropped; the nearest MAX_CHOICES come back, closest first."""
    google["replies"][nav.PLACES_URL] = {"places": [
        _place("far", 37.90, -122.2577), _place("near", 37.8670, -122.2577, "1 A St, Berkeley"),
        {"displayName": {"text": "nowhere"}}, {"location": {"latitude": 37.8668, "longitude": -122.2577}},
        _place("mid", 37.87, -122.2577), _place("farther", 37.95, -122.2577)]}
    got = nav.search_places("coffee", HOME, nearby=True)
    assert [p.name for p in got] == ["near", "mid", "far"]
    assert got[0].address == "1 A St, Berkeley" and got[0].distance_m == pytest.approx(33.4, abs=1.0)
    body = json.loads(google["requests"][0].data)
    assert body["rankPreference"] == "DISTANCE" and body["locationBias"]["circle"]["radius"] == nav.NEARBY_RADIUS_M
    google["replies"][nav.PLACES_URL] = {}
    assert nav.search_places("Doe Library", HOME, nearby=False) == []
    body = json.loads(google["requests"][1].data)
    assert "rankPreference" not in body and body["locationBias"]["circle"]["radius"] == nav.NAMED_RADIUS_M


def test_walking_route_parses_steps_distance_and_duration(google):
    """Multi-line instructions join with '. ', empty steps drop out, '331s' -> 331.0, the polyline decodes."""
    google["replies"][nav.ROUTES_URL] = ROUTE
    pts, steps, dist, dur = nav.walking_route(HOME, Place("Doe", None, 37.8723, -122.2595))
    assert pts == [(38.5, -120.2), (40.7, -120.95)]
    assert steps == ["Head north. on Bowditch St", "Turn left"]
    assert (dist, dur) == (412.0, 331.0)
    body = json.loads(google["requests"][0].data)
    assert body["travelMode"] == "WALK" and body["destination"]["location"]["latLng"]["latitude"] == 37.8723


def test_walking_route_without_distance_or_duration_or_any_route(google):
    """Missing / odd fields read as None; no route at all is an error."""
    google["replies"][nav.ROUTES_URL] = {"routes": [{"duration": "soon"}]}
    assert nav.walking_route(HOME, Place("x", None, 0, 0)) == ([], [], None, None)
    google["replies"][nav.ROUTES_URL] = {"routes": []}
    with pytest.raises(RuntimeError, match="no walking route"):
        nav.walking_route(HOME, Place("x", None, 0, 0))


# ---------------------------------------------------------------- the phone's position
def test_phone_position_through_the_real_phone_latest_import(monkeypatch, capsys):
    """kite.navigate imports phone.latest lazily: its newest fix is the origin; no reachable server -> None."""
    import phone.latest

    def offline(*a, **kw):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(urllib.request, "urlopen", offline)
    assert nav.phone_position() is None
    assert "no phone position" in capsys.readouterr().out
    monkeypatch.setattr(phone.latest, "latest", lambda: [{"lat": 1.5, "lng": 2.5}, {"lat": 9.0, "lng": 9.0}])
    assert nav.phone_position() == (1.5, 2.5)
    monkeypatch.setattr(phone.latest, "latest", lambda: [])
    assert nav.phone_position() is None


def test_phone_position_when_the_phone_package_is_missing(monkeypatch):
    """A drone-only install without phone/ just has no position."""
    monkeypatch.setitem(sys.modules, "phone.latest", None)
    assert nav.phone_position() is None


# ---------------------------------------------------------------- plan() and choose()
def test_plan_says_what_is_wrong_before_any_lookup(monkeypatch):
    """Nothing understood -> ask where to; no origin and no phone fix -> say so; neither hits the network."""
    monkeypatch.setattr(urllib.request, "urlopen", _no_network)
    monkeypatch.setattr(nav, "phone_position", lambda: None)
    assert nav.plan("   ?").say == "Where would you like to go?"
    p = nav.plan("take me to Doe Library")
    assert p.status == "error" and "location" in p.say


def test_plan_uses_the_phone_when_no_origin_is_given(google, monkeypatch):
    """No --from: the phone's newest fix is the origin."""
    monkeypatch.setattr(nav, "phone_position", lambda: HOME)
    google["replies"][nav.PLACES_URL] = {"places": []}
    assert nav.plan("take me to the moon").origin == HOME


@pytest.mark.parametrize("failure", [urllib.error.URLError("no network"), TimeoutError("timed out"),
                                     urllib.error.HTTPError(nav.PLACES_URL, 500, "Server Error", {}, io.BytesIO())])
def test_plan_speaks_an_error_when_the_lookup_fails(google, failure, capsys):
    """Offline, a timeout or a Google error: a spoken 'can't look up places', not a crash."""
    google["replies"][nav.PLACES_URL] = failure
    p = nav.plan("take me to the nearest pharmacy", HOME)
    assert (p.status, p.say, p.origin) == ("error", "I can't look up places right now.", HOME)
    assert "[navigate]" in capsys.readouterr().out


def test_plan_without_a_key_fails_softly(monkeypatch, tmp_path):
    """No API key anywhere: the same spoken error, and no request is made."""
    monkeypatch.delenv("GOOGLE_MAPS_API_KEY", raising=False)
    monkeypatch.setattr(nav, "__file__", str(tmp_path / "kite" / "navigate.py"))
    monkeypatch.setattr(urllib.request, "urlopen", _no_network)
    assert nav.plan("take me to Doe Library", HOME).say == "I can't look up places right now."


def test_plan_with_no_match(google):
    """Nothing found nearby: say so with the query."""
    google["replies"][nav.PLACES_URL] = {"places": []}
    p = nav.plan("take me to the nearest unicorn", HOME)
    assert (p.status, p.say) == ("error", "I couldn't find unicorn near you.")


def test_nearest_goes_straight_to_the_closest_and_routes(google):
    """'nearest X' never asks which one: it routes to the closest match and speaks distance and minutes."""
    google["replies"][nav.PLACES_URL] = {"places": [_place("Far Rx", 37.90, -122.2577),
                                                    _place("Near Rx", 37.8700, -122.2577)]}
    google["replies"][nav.ROUTES_URL] = ROUTE
    p = nav.plan("take me to the nearest pharmacy", HOME)
    assert p.status == "route" and p.destination.name == "Near Rx"
    assert p.say == "Going to Near Rx, 410 meters away, about 6 minutes."
    assert p.steps == ["Head north. on Bowditch St", "Turn left"] and p.points and p.duration_s == 331.0


def test_a_named_place_with_several_matches_asks_which_one_then_routes_the_pick(google):
    """Several branches: 'Which one? 1: ... on <street>. 2: ...' and choose() routes the answer."""
    google["replies"][nav.PLACES_URL] = {"places": [
        _place("Starbucks", 37.8700, -122.2577, "2128 Oxford St, Berkeley"), _place("Starbucks", 37.8750, -122.2577)]}
    p = nav.plan("take me to Starbucks", HOME)
    assert p.status == "clarify" and len(p.choices) == 2
    assert p.say == "Which one? 1: Starbucks on 2128 Oxford St. 2: Starbucks."
    google["replies"][nav.ROUTES_URL] = {"routes": [{"duration": "20s"}]}
    r = choose(p, 2)
    assert r.status == "route" and r.destination is p.choices[1]
    assert r.points == [HOME, (37.8750, -122.2577)]  # no polyline: a straight line for the map
    assert r.say == "Going to Starbucks, about 1 minutes."
    assert json.loads(json.dumps(r.to_dict()))["destination"]["name"] == "Starbucks"


def test_one_named_match_routes_and_a_route_failure_is_spoken(google):
    """A single match routes directly; when Routes fails the walker hears the place was found but not the way."""
    google["replies"][nav.PLACES_URL] = {"places": [_place("Doe Library", 37.8723, -122.2595)]}
    google["replies"][nav.ROUTES_URL] = urllib.error.URLError("down")
    p = nav.plan("take me to Doe Library", HOME)
    assert (p.status, p.say) == ("error", "I found Doe Library but can't get walking directions.")


def test_choose_needs_a_pending_question_and_an_origin():
    """choose() on a route / error plan, or without an origin, asks again instead of routing."""
    c = [Place("a", None, 0, 0)]
    assert choose(Plan("route", "x", "", HOME, choices=c), 1).say == "Sorry, which one?"
    assert choose(Plan("clarify", "x", "", None, choices=c), 1).status == "error"
    assert choose(Plan("clarify", "x", "", HOME, choices=c), 0).choices == c


# ---------------------------------------------------------------- the CLI
def test_main_routes_from_the_given_origin_and_prints_the_steps(google, capsys):
    """--from lat,lng is the origin; the plain output lists the steps and the path; exit 0."""
    google["replies"][nav.PLACES_URL] = {"places": [_place("Doe Library", 37.8723, -122.2595)]}
    google["replies"][nav.ROUTES_URL] = ROUTE
    assert nav.main(["take me to Doe Library", "--from", "37.8667,-122.2577"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("[route] Going to Doe Library") and "  - Turn left" in out and "path: 2 points, 412 m" in out
    body = json.loads(google["requests"][0].data)
    assert body["locationBias"]["circle"]["center"] == {"latitude": 37.8667, "longitude": -122.2577}


def test_main_lists_the_choices_and_pick_answers_them(google, capsys):
    """Without --pick it prints the numbered choices; --pick 2 --json routes the second and prints JSON."""
    google["replies"][nav.PLACES_URL] = {"places": [_place("Starbucks", 37.8700, -122.2577, "Oxford St"),
                                                    _place("Starbucks", 37.8750, -122.2577)]}
    assert nav.main(["Starbucks", "--from", "37.8667,-122.2577"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("[clarify] Which one?") and "  1. Starbucks  Oxford St, 370 meters away" in out
    google["replies"][nav.ROUTES_URL] = ROUTE
    assert nav.main(["Starbucks", "--from", "37.8667,-122.2577", "--pick", "2", "--json"]) == 0
    p = json.loads(capsys.readouterr().out)
    assert p["status"] == "route" and p["destination"]["lat"] == 37.875


def test_main_exits_1_on_an_error_and_2_without_text(monkeypatch, capsys):
    """An error plan exits 1; no destination text at all is an argparse usage error."""
    monkeypatch.setattr(nav, "phone_position", lambda: None)
    assert nav.main(["take me home"]) == 1
    assert capsys.readouterr().out.startswith("[error] I don't have your location")
    with pytest.raises(SystemExit) as e:
        nav.main([])
    assert e.value.code == 2
