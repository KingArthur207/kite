"""
Google reps at Cal Hacks gave us permission (Sat Oct 24, ~3 AM) to use Street View, including
detection on the frames, only for Moss St between Folsom and Howard: see docs/google-permission.md.
SCOPE (below) enforces that; nothing is stored, committed or used for training.
Google Street View along the planned route as a frame source: real imagery of our corner before the Tello shoot.

    from kite.sources import open_source
    with open_source("streetview:route", pitch=0) as src:            # sim/public/osm/route.json, drone-like
        while True:
            ok, bgr = src.read()
            ...
    open_source("streetview:37.77811,-122.40848,315")                # one view: lat, lng, compass heading

    python -m kite.sources.streetview [route | lat,lng,heading] [--pitch -10] [--fov 69] [--cache-dir /tmp/sv]

The route is walked in steps of step_m (2.5 m). Each step asks the metadata endpoint (free) for the nearest outdoor
panorama within radius_m. The nearest is often a user-contributed photosphere ("source=outdoor" lets them through,
and on Moss St they are 2021 photos INSIDE a shop), which hides Google's own panorama in the roadway a few metres
away; so with google_only (default) a non-Google answer is retried at SIDE_M either side of the step. No Google
panorama at all is a gap and the step is skipped, and a step on the panorama and heading of the previous frame is
skipped too (same picture). Then one Static image of that panorama is fetched (billed), heading = the walking
direction of that route segment.

Pose: x, y = the panorama's lat/lng in sim metres (kite.gps.latlng_to_sim), heading = compass degrees,
cam_height = SV_CAM_HEIGHT_M for Google's panoramas (the car roof: between the drone's 2 m and the chest's 1.2 m,
and in the roadway, not on the sidewalk) or CONTRIBUTED_CAM_HEIGHT_M for user photospheres (google_only=False),
pitch as asked (+ = up), t = metres along the route / speed. .pano: id, date, copyright, lat, lng, step, route
metres and metres from the step of the frame just read. .requests: metadata, image and cached counts.

Google's terms: images are not stored. Frames live in memory only; cache_dir (repeat runs without paying again)
must be in the temp directory, and you delete it when done. The key comes from GOOGLE_MAPS_API_KEY (or key=) and is
never printed: request errors are reported without the URL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from itertools import pairwise
from pathlib import Path

import numpy as np

from ..camera import Camera
from ..gps import compass, latlng_to_sim, sim_to_latlng
from .pose import empty_pose

API = "https://maps.googleapis.com/maps/api/streetview"
ROUTE_JSON = Path(__file__).resolve().parents[2] / "sim" / "public" / "osm" / "route.json"
STEP_M = 2.5
RADIUS_M = 15  # a panorama further than this from a step is a gap (Howard's car panoramas are ~11 m off its sidewalk)
SV_CAM_HEIGHT_M = 2.5  # ASSUMED: Google's car camera, roughly 2.5 m above the road
CONTRIBUTED_CAM_HEIGHT_M = 1.6  # ASSUMED: a user-contributed photosphere (copyright not Google), hand or monopod
SIDE_M = 4.0  # google_only: retry this far left and right of the step (Moss's roadway is 6 m wide)
SIZE = (640, 480)  # the Static API's largest size without a premium plan is 640 px a side; 4:3 like the Tello
TELLO_HFOV_DEG = 55.1  # load_camera(): fx 920 at 960 wide
SPEED_MPS = 0.6  # the walker's pace at the shoot (spec section 8): sets pose t
TIMEOUT_S = 15.0
TMP_ROOTS = (Path(tempfile.gettempdir()).resolve(), Path("/tmp").resolve())


# Moss St from Folsom to Howard plus the corner, with about 30 m of margin (docs/google-permission.md).
SCOPE = (37.7768, 37.7788, -122.4095, -122.4068)   # lat min, lat max, lng min, lng max


def in_scope(lat: float, lng: float) -> bool:
    lat0, lat1, lng0, lng1 = SCOPE
    return lat0 <= lat <= lat1 and lng0 <= lng <= lng1


def _require_scope(lat: float, lng: float) -> None:
    if not in_scope(lat, lng):
        raise ValueError(f"streetview: {lat:.5f},{lng:.5f} is outside the one street Google allowed us (Moss St)")


def _http_get(url: str) -> bytes:
    """One GET. Tests replace this. Errors never carry the URL (it has the key in it)."""
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_S) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Street View request failed: HTTP {e.code} {e.reason}") from None
    except (urllib.error.URLError, OSError) as e:
        raise RuntimeError(f"Street View request failed: {getattr(e, 'reason', e)}") from None


def load_route(path: str | Path = ROUTE_JSON) -> list[tuple[float, float]]:
    return [(p["lat"], p["lng"]) for p in json.loads(Path(path).read_text())["points"]]


def walk(points: list[tuple[float, float]], step_m: float = STEP_M) -> list[tuple[float, float, float, float]]:
    """A lat/lng polyline -> (lat, lng, compass heading of its segment, metres along) every step_m, ending on the
    last point. Distances on kite.gps's tangent plane around the first point."""
    origin = points[0]
    xy = [latlng_to_sim(lat, lng, origin) for lat, lng in points]
    out, along, carry = [], 0.0, 0.0  # carry: metres into the current segment where the next step lands
    for (x0, y0), (x1, y1) in pairwise(xy):
        seg = math.hypot(x1 - x0, y1 - y0)
        if seg == 0:
            continue
        head = float(compass(x1 - x0, y1 - y0))
        s = carry
        while s < seg - 1e-9:
            f = s / seg
            out.append((x0 + f * (x1 - x0), y0 + f * (y1 - y0), head, along + s))
            s += step_m
        carry, along = s - seg, along + seg
    if len(xy) > 1 and out:
        out.append((*xy[-1], out[-1][2], along))
    return [(*sim_to_latlng(x, y, origin), h, a) for x, y, h, a in out]


class StreetViewSource:
    """Street View frames along a route (points: lat/lng list) or one view (single: (lat, lng, heading)). Same shape
    as the other sources in kite.sources.open: read() -> (ok, bgr), .pose, .camera, .name, release()."""

    def __init__(self, points: list[tuple[float, float]] | None = None, single: tuple[float, float, float] | None = None,
                 pitch: float = 0.0, hfov_deg: float = TELLO_HFOV_DEG, size: tuple[int, int] = SIZE,
                 step_m: float = STEP_M, radius_m: float = RADIUS_M, speed: float = SPEED_MPS,
                 google_only: bool = True, cache_dir: str | Path | None = None, key: str | None = None):
        self.key = key or os.environ.get("GOOGLE_MAPS_API_KEY")
        if not self.key:
            raise RuntimeError("streetview: set GOOGLE_MAPS_API_KEY (the repo .env has it)")
        if not 10 <= hfov_deg <= 120:
            raise ValueError("streetview: hfov_deg must be 10 to 120 (the Static API's fov range)")
        if not -90 <= pitch <= 90:
            raise ValueError("streetview: pitch must be -90 to 90")
        self.pitch, self.hfov_deg, self.size = float(pitch) + 0.0, float(hfov_deg), (int(size[0]), int(size[1]))
        self.radius_m, self.speed, self.google_only = radius_m, speed, google_only
        if single is not None:
            lat, lng, head = single
            self.steps = [(lat, lng, float(head) % 360, 0.0)]
            self.name = f"streetview:{lat},{lng},{head:g}"
        else:
            self.steps = walk(points or load_route(), step_m)
            self.name = "streetview:route"
        self.cache = None
        if cache_dir is not None:
            self.cache = Path(cache_dir).expanduser().resolve()
            if not any(self.cache.is_relative_to(r) for r in TMP_ROOTS):
                raise ValueError(f"streetview: cache_dir must be under {TMP_ROOTS[0]} or /tmp (Google's terms: no "
                                 "long-term storage)")
            self.cache.mkdir(parents=True, exist_ok=True)
        self.camera = Camera.from_hfov(*self.size, self.hfov_deg, f"Street View Static fov {self.hfov_deg:g} deg")
        self.pose: dict | None = None
        self.pano: dict | None = None
        self.requests = {"metadata": 0, "image": 0, "cached": 0}
        self.gaps = 0  # steps with no (Google, if google_only) panorama within radius_m
        self.repeats = 0  # steps on the same panorama and heading as the frame before
        self.frames = 0
        self._i = 0
        self._last: tuple[str, float] | None = None
        self._open = True

    # ---------------------------------------------------------------- HTTP
    def _get(self, kind: str, params: dict) -> bytes:
        url = f"{API}{'/metadata' if kind == 'metadata' else ''}?{urllib.parse.urlencode({**params, 'key': self.key})}"
        path = None
        if self.cache is not None:
            path = self.cache / f"{kind}-{hashlib.sha256(urllib.parse.urlencode(params).encode()).hexdigest()[:24]}"
            if path.is_file():
                self.requests["cached"] += 1
                return path.read_bytes()
        self.requests[kind] += 1
        data = _http_get(url)
        if path is not None:
            path.write_bytes(data)
        return data

    def _metadata(self, lat: float, lng: float) -> dict | None:
        _require_scope(lat, lng)
        m = json.loads(self._get("metadata", {"location": f"{lat:.7f},{lng:.7f}", "radius": round(self.radius_m),
                                              "source": "outdoor"}))
        status = m.get("status")
        if status == "OK":
            return m
        if status in ("ZERO_RESULTS", "NOT_FOUND"):
            return None
        raise RuntimeError(f"Street View metadata: {status} {m.get('error_message', '')}".strip())

    def _find(self, lat: float, lng: float, heading: float) -> dict | None:
        """Metadata of the panorama for a step: the nearest one, or with google_only the nearest Google one at the
        step or SIDE_M to its left or right."""
        x, y = latlng_to_sim(lat, lng)
        a = math.radians(heading + 90)  # to the right of the walking direction
        for side in ((0.0, -SIDE_M, SIDE_M) if self.google_only else (0.0,)):
            m = self._metadata(*sim_to_latlng(x + side * math.sin(a), y + side * math.cos(a)))
            if m is not None and (not self.google_only or _is_google(m)):
                return m
        return None

    def _image(self, pano_id: str, heading: float) -> np.ndarray:
        import cv2

        data = self._get("image", {"pano": pano_id, "heading": f"{heading:.1f}", "pitch": f"{self.pitch:g}",
                                   "fov": f"{self.hfov_deg:g}", "size": f"{self.size[0]}x{self.size[1]}",
                                   "return_error_code": "true"})
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("Street View image: the response is not an image")
        return img

    # ---------------------------------------------------------------- source
    def read(self) -> tuple[bool, np.ndarray | None]:
        while self._open and self._i < len(self.steps):
            i = self._i
            self._i += 1
            lat, lng, head, along = self.steps[i]
            m = self._find(lat, lng, head)
            if m is None:
                self.gaps += 1
                continue
            pid, google = m["pano_id"], _is_google(m)
            if self._last == (pid, round(head, 1)):
                self.repeats += 1
                continue
            img = self._image(pid, head)
            self._last = (pid, round(head, 1))
            plat, plng = m["location"]["lat"], m["location"]["lng"]
            x, y = latlng_to_sim(plat, plng)
            self.pose = empty_pose(along / self.speed, x=round(x, 2), y=round(y, 2), heading=round(head, 1),
                                   cam_height=SV_CAM_HEIGHT_M if google else CONTRIBUTED_CAM_HEIGHT_M, pitch=self.pitch, heading_ref="compass")
            self.pano = {"id": pid, "date": m.get("date"), "copyright": m.get("copyright"), "lat": plat, "lng": plng, "step": i,
                         "along_m": round(along, 1), "offset_m": round(_metres(lat, lng, plat, plng), 1)}
            if img.shape[1::-1] != (self.camera.width, self.camera.height):
                self.camera = self.camera.scaled(img.shape[1], img.shape[0])
            self.frames += 1
            return True, img
        return False, None

    def isOpened(self) -> bool:
        return self._open

    def summary(self) -> str:
        r = self.requests
        return (f"[streetview] {self.frames} frames from {len(self.steps)} steps ({self.gaps} gaps, {self.repeats} "
                f"repeats); requests: {r['image']} image (billed), {r['metadata']} metadata (free), "
                f"{r['cached']} from cache")

    def release(self) -> None:
        if self._open:
            self._open = False
            print(self.summary(), file=sys.stderr, flush=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def _is_google(meta: dict) -> bool:
    return "Google" in (meta.get("copyright") or "")


def _metres(lat0, lng0, lat1, lng1) -> float:
    return math.hypot(*latlng_to_sim(lat1, lng1, (lat0, lng0)))


def parse(arg: str) -> dict:
    """'route' / '' -> the planned route; 'lat,lng,heading' -> one view. Returns StreetViewSource kwargs."""
    arg = arg.strip().lower()
    if arg in ("", "route"):
        return {}
    try:
        lat, lng, head = (float(v) for v in arg.split(","))
    except ValueError:
        raise ValueError(f"streetview:{arg}: use streetview:route or streetview:<lat>,<lng>,<heading>") from None
    return {"single": (lat, lng, head)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m kite.sources.streetview", description="List the Street View frames "
                                "along the route (or one view) with their poses; prints the request count.")
    p.add_argument("where", nargs="?", default="route", help="route (default) or lat,lng,heading")
    p.add_argument("--pitch", type=float, default=0.0, help="deg, + = up (0 drone-like, about -10 chest-like)")
    p.add_argument("--fov", type=float, default=TELLO_HFOV_DEG, help="horizontal FOV, deg")
    p.add_argument("--step", type=float, default=STEP_M, help="metres between steps")
    p.add_argument("--radius", type=float, default=RADIUS_M, help="a panorama further than this is a gap, m")
    p.add_argument("--any-pano", action="store_true", help="also use user-contributed photospheres")
    p.add_argument("--cache-dir", help="reuse responses between runs; must be under the temp dir, delete it after")
    a = p.parse_args(argv)
    try:
        with StreetViewSource(**parse(a.where), pitch=a.pitch, hfov_deg=a.fov, step_m=a.step,
                              radius_m=a.radius, google_only=not a.any_pano,
                              cache_dir=a.cache_dir) as src:
            while True:
                ok, _ = src.read()
                if not ok:
                    break
                pv, po = src.pano, src.pose
                print(f"step {pv['step']:3d}  {pv['along_m']:6.1f} m  t {po['t']:6.1f} s  {pv['lat']:.6f},"
                      f"{pv['lng']:.6f} ({pv['offset_m']} m off)  heading {po['heading']:5.1f}  {pv['date']}  "
                      f"{po['cam_height']} m  {pv['copyright']}")
    except (RuntimeError, ValueError) as e:
        print(e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
