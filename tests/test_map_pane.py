"""Map tile math. Imports the pure functions only and does not open a Tk window."""

import math

from run.map_pane import latlng_from_world, meters_per_pixel, tile_url, tile_xy, world_px


def test_tile_xy_origin():
    # Zoom 0 is one tile for the whole world. (0, 0) sits in it.
    assert tile_xy(0, 0, 0) == (0, 0)

    # Zoom 1 is a 2×2 grid. Longitude 0 is the left edge of the eastern column;
    # latitude 0 is the top edge of the southern row. floor() of that boundary is (1, 1).
    n = 2 ** 1
    x = math.floor((0.0 + 180.0) / 360.0 * n)
    mercator = math.asinh(math.tan(math.radians(0.0)))
    y = math.floor((1.0 - mercator / math.pi) / 2.0 * n)
    assert (x, y) == (1, 1)
    assert tile_xy(0, 0, 1) == (x, y)


def test_tile_xy_clamped():
    # One tile covers the world, whatever the coordinate is.
    assert tile_xy(37.8, -122.2, 0) == (0, 0)
    # Longitude ±180 is the edge of the square. 180 would be index 2**z, which is out of range.
    assert tile_xy(0, 180, 2)[0] == 3
    assert tile_xy(0, -180, 2)[0] == 0
    n = 2 ** 5
    for lat, lng in ((89.9, 400.0), (-89.9, -400.0)):
        x, y = tile_xy(lat, lng, 5)
        assert 0 <= x < n
        assert 0 <= y < n


def test_berkeley_round_trip():
    lat, lng, zoom = 37.8667, -122.2577, 16
    x, y = world_px(lat, lng, zoom)
    lat2, lng2 = latlng_from_world(x, y, zoom)
    assert abs(lat2 - lat) < 1e-6
    assert abs(lng2 - lng) < 1e-6


def test_tile_url_is_mapbox_streets_and_needs_a_token():
    assert tile_url(16, 1, 2, "") is None
    url = tile_url(16, 10512, 25344, "pk.test")
    assert url is not None
    assert url.startswith("https://api.mapbox.com/styles/v1/mapbox/streets-v12/tiles/256/16/10512/25344?")
    assert "access_token=pk.test" in url
    assert "openstreetmap.org" not in url


def test_meters_per_pixel_equator_zoom_zero():
    # 2 * pi * 6378137 / 256 ≈ 156543.03 m at the equator, zoom 0.
    assert abs(meters_per_pixel(0, 0) - 156543) < 1
