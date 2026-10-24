"""kite.sources.streetview against a fake Street View API (no network, no key): stepping along a route, pose and
heading math, gap and repeat skipping, the Google-only side retry, request counting, the cache, no files left."""

import json
import math
import os
import urllib.parse

import cv2
import numpy as np
import pytest

from kite.gps import latlng_to_sim, sim_to_latlng
from kite.sources import open_source, parse_spec
from kite.sources import streetview as sv

KEY = "test-key-not-real"
# a 20 m leg due north, then 10 m due east, from the sim origin
ROUTE = [sim_to_latlng(0, 0), sim_to_latlng(0, 20), sim_to_latlng(10, 20)]


class FakeAPI:
    """Panoramas every 5 m along the route's x = 0 / y = 20 lines (Google's), none for 7 < y < 13 on the first leg
    (a gap), and optionally a contributed photosphere that is nearest to every step."""

    def __init__(self, contributed: bool = False):
        self.calls: list[tuple[str, dict]] = []
        self.contributed = contributed

    def __call__(self, url: str) -> bytes:
        base, _, query = url.partition("?")
        q = dict(urllib.parse.parse_qsl(query))
        assert q.pop("key") == KEY
        kind = "metadata" if base.endswith("/metadata") else "image"
        self.calls.append((kind, q))
        if kind == "image":
            w, h = (int(v) for v in q["size"].split("x"))
            img = np.full((h, w, 3), 128, np.uint8)
            return cv2.imencode(".jpg", img)[1].tobytes()
        lat, lng = (float(v) for v in q["location"].split(","))
        x, y = latlng_to_sim(lat, lng)
        if self.contributed and abs(x) < 1 and y < 18:  # right on the walk line: the photosphere answers
            return json.dumps({"status": "OK", "pano_id": "user", "copyright": "© Someone",
                               "location": {"lat": lat, "lng": lng}, "date": "2021-08"}).encode()
        if y < 18:
            if 7 < y < 13:
                return json.dumps({"status": "ZERO_RESULTS"}).encode()
            px, py = 0.0, round(y / 5) * 5.0
        else:
            px, py = round(x / 5) * 5.0, 20.0
        plat, plng = sim_to_latlng(px, py)
        return json.dumps({"status": "OK", "pano_id": f"p{px:g}_{py:g}", "copyright": "© 2026 Google",
                           "location": {"lat": plat, "lng": plng}, "date": "2026-09"}).encode()


@pytest.fixture
def api(monkeypatch):
    fake = FakeAPI()
    monkeypatch.setattr(sv, "_http_get", fake)
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", KEY)
    return fake


def frames(src):
    out = []
    while True:
        ok, img = src.read()
        if not ok:
            return out
        out.append((img, dict(src.pose), dict(src.pano)))


def test_walk_steps_every_step_m_with_segment_headings():
    steps = sv.walk(ROUTE, 2.5)
    along = [s[3] for s in steps]
    assert along[0] == 0 and along[-1] == pytest.approx(30, abs=0.01)
    assert np.allclose(np.diff(along), 2.5, atol=0.01)
    assert {round(s[2]) for s in steps if s[3] < 20} == {0}  # north
    assert {round(s[2]) for s in steps if s[3] > 20} == {90}  # east
    x, y = latlng_to_sim(*steps[-1][:2])
    assert (x, y) == pytest.approx((10, 20), abs=0.01)


def test_route_frames_poses_gaps_repeats_and_request_count(api, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with sv.StreetViewSource(ROUTE, pitch=-10, hfov_deg=60, speed=0.5) as src:
        got = frames(src)
        assert src.camera.width == 640 and src.camera.height == 480
        assert src.camera.fx == pytest.approx(320 / math.tan(math.radians(30)))
    panos = [p["id"] for _, _, p in got]
    assert len(panos) == len(set(panos))  # each panorama once per heading
    assert "p0_10" not in panos  # inside the gap
    assert panos[:2] == ["p0_0", "p0_5"] and panos[-1] == "p10_20"
    for img, pose, pano in got:
        assert img.shape == (480, 640, 3)
        assert (pose["x"], pose["y"]) == pytest.approx(latlng_to_sim(pano["lat"], pano["lng"]), abs=0.01)
        assert pose["heading"] == (0.0 if pano["along_m"] < 20 else 90.0)
        assert pose["t"] == pytest.approx(pano["along_m"] / 0.5)
        assert pose["pitch"] == -10 and pose["cam_height"] == sv.SV_CAM_HEIGHT_M
    n_steps = len(sv.walk(ROUTE, sv.STEP_M))
    assert src.gaps == 3 and src.repeats == n_steps - len(got) - 3  # y = 7.5, 10, 12.5
    # one metadata call per step, two more per gap (the google_only retries either side)
    assert src.requests == {"metadata": n_steps + 2 * src.gaps, "image": len(got), "cached": 0}
    assert [k for k, _ in api.calls].count("image") == len(got)
    image_q = next(q for k, q in api.calls if k == "image")
    assert image_q["pitch"] == "-10" and image_q["fov"] == "60" and image_q["size"] == "640x480"
    assert list(tmp_path.iterdir()) == []  # nothing written without a cache


def test_google_only_retries_beside_a_contributed_photosphere(api):
    api.contributed = True
    with sv.StreetViewSource(ROUTE) as src:
        got = frames(src)
    assert got and all(p["copyright"].startswith("© 2026 Google") for _, _, p in got)
    with sv.StreetViewSource(ROUTE, google_only=False) as src:
        anyp = frames(src)
    assert anyp[0][2]["id"] == "user" and anyp[0][1]["cam_height"] == sv.CONTRIBUTED_CAM_HEIGHT_M


def test_cache_reuses_responses_and_must_be_in_the_temp_dir(api, tmp_path):
    cache = tmp_path / "sv"
    with sv.StreetViewSource(ROUTE, cache_dir=cache) as a:
        first = frames(a)
    calls = len(api.calls)
    with sv.StreetViewSource(ROUTE, cache_dir=cache) as b:
        again = frames(b)
    assert len(api.calls) == calls and b.requests["image"] == b.requests["metadata"] == 0
    assert [p for _, _, p in again] == [p for _, _, p in first]
    with pytest.raises(ValueError, match="cache_dir"):
        sv.StreetViewSource(ROUTE, cache_dir=os.path.expanduser("~/kite-sv-cache"))


def test_single_view_spec_and_open_source(api):
    lat, lng = sim_to_latlng(0, 5)
    assert parse_spec("streetview:route") == ("streetview", "route")
    with open_source(f"streetview:{lat},{lng},45", pitch=0) as src:
        got = frames(src)
    assert len(got) == 1 and got[0][1]["heading"] == 45 and got[0][2]["id"] == "p0_5"
    assert src.name.startswith("streetview:")
    with pytest.raises(ValueError):
        open_source("streetview:not,a,view")


def test_errors_never_show_the_key(monkeypatch):
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", KEY)
    monkeypatch.setattr(sv, "_http_get", lambda url: json.dumps({"status": "REQUEST_DENIED"}).encode())
    with pytest.raises(RuntimeError) as e:
        sv.StreetViewSource(ROUTE).read()
    assert KEY not in str(e.value) and "REQUEST_DENIED" in str(e.value)
    monkeypatch.delenv("GOOGLE_MAPS_API_KEY")
    with pytest.raises(RuntimeError, match="GOOGLE_MAPS_API_KEY"):
        sv.StreetViewSource(ROUTE)


def test_scope_is_moss_street_only():
    from kite.sources.streetview import _require_scope, in_scope
    assert in_scope(37.778352, -122.408768)          # the Moss/Howard zebra
    assert in_scope(37.7771792, -122.4073136)        # Moss at Folsom
    assert not in_scope(37.8013, -122.4477)          # Lyon St, far away
    import pytest
    with pytest.raises(ValueError):
        _require_scope(37.7749, -122.4194)           # Civic Center


def test_view_jpeg_checks_the_panorama_first(monkeypatch):
    """The sim's Street View walk asks for a panorama by id: metadata first, in SCOPE and Google's, else no image."""
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", KEY)
    where = {"moss": (37.7779, -122.4082, "© 2026 Google"), "far": (37.7749, -122.4194, "© 2026 Google"),
             "user": (37.7779, -122.4082, "© Someone")}
    calls = []

    def fake(url):
        q = dict(urllib.parse.parse_qsl(url.partition("?")[2]))
        calls.append(q)
        if "/metadata" in url:
            lat, lng, cr = where[q["pano"]]
            return json.dumps({"status": "OK", "pano_id": q["pano"], "copyright": cr,
                               "location": {"lat": lat, "lng": lng}}).encode()
        return b"\xff\xd8jpeg"

    monkeypatch.setattr(sv, "_http_get", fake)
    src = sv.StreetViewSource(single=(37.7779, -122.4082, 0), pitch=-10, hfov_deg=69)
    jpeg, meta = src.view_jpeg("moss", 375)
    assert jpeg == b"\xff\xd8jpeg" and meta["pano_id"] == "moss"
    assert calls[-1]["heading"] == "15.0" and calls[-1]["pitch"] == "-10" and calls[-1]["fov"] == "69"
    src.view_jpeg("moss", 30)
    assert src.requests == {"metadata": 1, "image": 2, "cached": 0}       # metadata once per panorama
    with pytest.raises(ValueError, match="outside"):
        src.view_jpeg("far", 0)
    with pytest.raises(ValueError, match="not Google"):
        src.view_jpeg("user", 0)
    assert src.requests["image"] == 2
