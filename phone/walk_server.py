"""Local walking preview.

The page is Mapbox. Google Routes is asked once when navigation starts, and again only
after the walker has been off the route long enough for RouteFollower to call it, and
even then at most once every 20 seconds and five times in ten minutes.

The Mapbox token and the Google key stay in the repo-root .env. Neither is written
into the page or into a log. The page fetches the Mapbox token from /api/config to
draw the map (a public pk. token, made to be used in browsers: restrict it to this
URL in the Mapbox account). The Google key never leaves this process.

It listens on 127.0.0.1 only, and because any web page can send requests to
127.0.0.1, it also serves only requests whose Host names this computer (a page
that rebinds its own domain to 127.0.0.1 can't read the walker's position) and
takes a POST only as application/json from its own page (another site's form or
fetch can't spend Google Routes calls).

    python -m phone.walk_server
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import phone  # noqa: F401  (puts the earpiece package on sys.path)
from earpiece.route import (
    NoKey,
    RouteDenied,
    RouteError,
    RouteFollower,
    RouteQuota,
    compute_route,
    no_directions_text,
)

ROOT = Path(__file__).resolve().parent.parent
PAGE = Path(__file__).resolve().parent / "walk.html"
HOST = "127.0.0.1"
PORT = 8735

REROUTE_MIN_S = 20.0
REROUTE_MAX = 5
REROUTE_WINDOW_S = 600.0
UNAVAILABLE_S = 15.0
QUOTA_S = 60.0
MAX_BODY = 65536
MAX_DESTINATION = 200
LOOPBACK = ("127.0.0.1", "localhost", "::1")


def _host_name(header: str | None) -> str:
    name = (header or "").strip().lower()
    if name.startswith("["):
        return name[1:].split("]", 1)[0]
    return name.rsplit(":", 1)[0] if name.count(":") == 1 else name


def _number(v, lo: float, hi: float, name: str) -> float | None:
    """None stays None; anything else must be a finite number in [lo, hi]."""
    if v is None:
        return None
    if isinstance(v, bool):
        raise ValueError(f"{name} must be a number")
    f = float(v)
    if not math.isfinite(f) or not lo <= f <= hi:
        raise ValueError(f"{name} is out of range")
    return f


def load_env() -> None:
    path = ROOT / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value:
            os.environ.setdefault(key, value)


def _round_pt(p) -> list[float]:
    return [round(float(p[0]), 6), round(float(p[1]), 6)]


class WalkSession:
    """One navigation. `compute` is injectable so tests never touch the network."""

    def __init__(self, compute=None):
        self.compute = compute or compute_route
        self._lock = threading.RLock()
        self.reset()

    def reset(self) -> None:
        self.follower: RouteFollower | None = None
        self.route = None
        self.guidance = ""
        self.google_calls = 0
        self.holding = False
        self.hold_s = 0.0
        self.arrived = False
        self.label = ""
        self.destination_query = ""
        self.last: dict | None = None
        self.ignored_accuracy = False
        self._pending = False
        self._reroute_at: list[float] = []
        self._retry_at = 0.0
        self._fatal = False
        self._error = ""

    def start(self, lat, lng, destination, accuracy=None, heading=None, now=None) -> dict:
        now = time.monotonic() if now is None else now
        with self._lock:
            self.reset()
            self.destination_query = destination
            self.label = "Cloyne Court Hotel" if "cloyne" in destination.lower() else destination
            self._remember(lat, lng, accuracy, heading)
            try:
                route = self._fetch((float(lat), float(lng)), destination)
            except RouteError as e:
                self._fail(e, now, initial=True)
                return self._view(ok=False)
            self._begin(route)
        return self.tick(lat, lng, accuracy, heading, now)

    def tick(self, lat, lng, accuracy=None, heading=None, now=None) -> dict:
        now = time.monotonic() if now is None else now
        with self._lock:
            if self.route is None and self.follower is None and not self._error:
                return {"ok": False, "error": "Navigation has not started.", "navigating": False, "google_calls": self.google_calls}
            self._remember(lat, lng, accuracy, heading)
            if accuracy is not None and float(accuracy) > RouteFollower.MAX_ACCURACY_M:
                self.ignored_accuracy = True
                return self._view()
            self.ignored_accuracy = False
            if self.holding and self._pending and now >= self._retry_at and not self._fatal:
                self._reroute(now)
            follower = self.follower
            if follower is not None and not follower.rerouting and not follower.done:
                follower.update(float(lat), float(lng), heading, now, None if accuracy is None else float(accuracy))
            if self._pending and not self._fatal:
                self._consider_reroute(now)
            return self._view()

    def state(self) -> dict:
        with self._lock:
            if self.route is None and not self._error:
                return {"ok": False, "error": "Navigation has not started.", "navigating": False, "google_calls": 0}
            return self._view(ok=self.route is not None)

    def _remember(self, lat, lng, accuracy, heading) -> None:
        self.last = {"lat": float(lat), "lng": float(lng), "accuracy": accuracy, "heading": heading}

    def _fetch(self, origin, destination):
        self.google_calls += 1
        return self.compute(origin, destination, label=self.label)

    def _begin(self, route) -> None:
        self.route = route
        self.arrived = False
        self.holding = False
        self.hold_s = 0.0
        self._fatal = False
        self._error = ""

        def on_guidance(text, _key, _ttl):
            self.guidance = text

        def on_beacon(_bearing, _dist, _label):
            return None

        def on_reroute():
            self._pending = True

        def on_arrived(label):
            self.arrived = True
            who = label or self.label
            self.guidance = f"You have arrived at {who}."

        self.follower = RouteFollower(
            route, on_guidance, on_beacon, on_reroute, on_arrived, label=self.label)
        self._pending = False

    def _fail(self, err: RouteError, now: float, initial: bool = False) -> None:
        self.guidance = no_directions_text(self.label, err)
        self._error = self.guidance
        if isinstance(err, (NoKey, RouteDenied)):
            self._fatal = True
            self._pending = False
            self.holding = False
            return
        if initial and self.route is None:
            self._fatal = isinstance(err, RouteError) and not isinstance(err, (RouteQuota,))
            # A first failure has nothing to follow. Unavailable and quota can be retried by pressing start again.
            self._pending = False
            self.holding = False
            return
        delay = QUOTA_S if isinstance(err, RouteQuota) else UNAVAILABLE_S
        self.holding = True
        self.hold_s = delay
        self._retry_at = now + delay
        self._pending = True

    def _wait(self, now: float) -> float:
        self._reroute_at = [t for t in self._reroute_at if now - t < REROUTE_WINDOW_S]
        wait = 0.0
        if self._reroute_at:
            wait = max(wait, self._reroute_at[-1] + REROUTE_MIN_S - now)
        if len(self._reroute_at) >= REROUTE_MAX:
            wait = max(wait, self._reroute_at[0] + REROUTE_WINDOW_S - now)
        return max(0.0, wait)

    def _consider_reroute(self, now: float) -> None:
        wait = self._wait(now)
        if wait > 0:
            self.holding = True
            self.hold_s = wait
            self._retry_at = now + wait
            self.guidance = "You're off the route. Next route request is held."
            return
        self._reroute(now)

    def _reroute(self, now: float) -> None:
        self._pending = False
        self.holding = False
        self.hold_s = 0.0
        self._reroute_at.append(now)
        last = self.last
        try:
            route = self._fetch((last["lat"], last["lng"]), self.destination_query)
        except RouteError as e:
            self._fail(e, now)
            return
        self._begin(route)
        self.follower.update(last["lat"], last["lng"], last.get("heading"), now,
                             None if last.get("accuracy") is None else float(last["accuracy"]))

    def _split(self, progress: float) -> tuple[list, list, list]:
        """Traveled, route-remaining, and the puck, in [lat, lng]."""
        follower = self.follower
        pts = self.route.points()
        cum = follower._cum
        here = follower.frame.to_latlng(*follower.point_at(progress))
        traveled = [_round_pt(p) for p, s in zip(pts, cum) if s <= progress]
        remaining = [_round_pt(p) for p, s in zip(pts, cum) if s >= progress]
        here_r = _round_pt(here)
        if not traveled or traveled[-1] != here_r:
            traveled.append(here_r)
        if not remaining or remaining[0] != here_r:
            remaining.insert(0, here_r)
        return traveled, remaining, here_r

    def _view(self, ok: bool = True) -> dict:
        last = self.last or {"lat": None, "lng": None, "accuracy": None}
        raw = None if last["lat"] is None else _round_pt((last["lat"], last["lng"]))
        follower = self.follower
        summary = follower.summary() if follower is not None else {}
        on_route = bool(
            follower is not None and not self.holding and not self.ignored_accuracy
            and not follower.off_route and not follower.rerouting and not follower.done)
        traveled, route_remaining, snapped = ([], [], None)
        if follower is not None and self.route is not None:
            traveled, route_remaining, snapped = self._split(follower.progress)
        if on_route and snapped is not None:
            puck = snapped
        else:
            puck = raw
        dest = _round_pt(self.route.destination) if self.route is not None else None
        if self.holding and raw is not None and dest is not None:
            remaining = [raw, dest]
        else:
            remaining = route_remaining
        body = {
            "ok": ok and self.route is not None,
            "navigating": self.route is not None and not self.arrived,
            "label": self.label,
            "guidance": self.guidance,
            "distance_m": None if self.route is None else round(self.route.distance_m),
            "duration_s": None if self.route is None else self.route.duration_s,
            "remaining_m": summary.get("remaining_m"),
            "progress_m": None if follower is None else round(follower.progress),
            "next": summary.get("next"),
            "next_in_m": summary.get("next_in_m"),
            "off_route": bool(follower and follower.off_route),
            "rerouting": bool(follower and follower.rerouting),
            "arrived": self.arrived,
            "holding": self.holding,
            "hold_s": round(self.hold_s, 1),
            "google_calls": self.google_calls,
            "puck": puck,
            "raw": raw,
            "accuracy_m": None if last.get("accuracy") is None else round(float(last["accuracy"]), 1),
            "on_route": on_route,
            "ignored_accuracy": self.ignored_accuracy,
            "traveled": traveled,
            "remaining": remaining,
            "route_remaining": route_remaining,
            "destination": dest,
        }
        if not body["ok"]:
            body["error"] = self._error or "Navigation has not started."
        return body


SESSION = WalkSession()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        print("[walk] " + (fmt % args), flush=True)

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict) -> None:
        self._send(code, json.dumps(payload).encode("utf-8"), "application/json")

    def _local(self) -> bool:
        """Only requests that name this computer: a DNS-rebinding page sends its own domain as Host."""
        if _host_name(self.headers.get("Host")) in LOOPBACK:
            return True
        self.close_connection = True          # the body (if any) is never read
        self._json(403, {"ok": False, "error": "This server is for this computer only."})
        return False

    def _same_origin(self) -> bool:
        """A POST must be JSON from this page. Another site can't send application/json without a CORS
        preflight, which this server never approves, and a browser that sends Origin must send ours."""
        ctype = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        origin = self.headers.get("Origin")
        host = (self.headers.get("Host") or "").strip().lower()
        if ctype == "application/json" and (origin is None or origin.strip().lower() == f"http://{host}"):
            return True
        self.close_connection = True
        self._json(403, {"ok": False, "error": "Send JSON from the walk page."})
        return False

    def _read_json(self) -> dict | None:
        try:
            n = int(self.headers.get("Content-Length", "0") or 0)
        except ValueError:
            n = -1
        if n < 0 or n > MAX_BODY:
            self.close_connection = True
        if n < 0:
            self._json(400, {"ok": False, "error": "Bad Content-Length."})
            return None
        if n > MAX_BODY:
            self._json(413, {"ok": False, "error": "Request is too large."})
            return None
        raw = self.rfile.read(n) if n else b"{}"
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            data = None
        if not isinstance(data, dict):
            self._json(400, {"ok": False, "error": "Expected a JSON object."})
            return None
        return data

    def do_GET(self):
        if not self._local():
            return
        path = self.path.split("?", 1)[0]
        if path in ("/", "/walk.html"):
            if not PAGE.is_file():
                self._json(404, {"ok": False, "error": "walk.html is missing."})
                return
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            return
        if path == "/api/config":
            self._json(200, {"mapboxToken": os.environ.get("MAPBOX_TOKEN", "")})
            return
        if path == "/api/state":
            self._json(200, SESSION.state())
            return
        self._json(404, {"ok": False, "error": "Not found."})

    def do_POST(self):
        if not self._local():
            return
        path = self.path.split("?", 1)[0]
        if path not in ("/api/start", "/api/tick"):
            self._json(404, {"ok": False, "error": "Not found."})
            return
        if not self._same_origin():
            return
        data = self._read_json()
        if data is None:
            return
        try:
            lat = _number(data["lat"], -90.0, 90.0, "lat")
            lng = _number(data["lng"], -180.0, 180.0, "lng")
            if lat is None or lng is None:
                raise ValueError("lat and lng are required.")
        except (KeyError, TypeError, ValueError, OverflowError):
            self._json(400, {"ok": False, "error": "lat and lng are required."})
            return
        try:
            accuracy = _number(data.get("accuracy"), 0.0, 1e6, "accuracy")
            heading = _number(data.get("heading"), -360.0, 720.0, "heading")
        except (TypeError, ValueError, OverflowError):
            self._json(400, {"ok": False, "error": "accuracy and heading must be numbers."})
            return
        if path == "/api/start":
            destination = str(data.get("destination") or "").strip()
            if not destination:
                self._json(400, {"ok": False, "error": "A destination is required."})
                return
            if len(destination) > MAX_DESTINATION:
                self._json(400, {"ok": False, "error": "The destination is too long."})
                return
            self._json(200, SESSION.start(lat, lng, destination, accuracy, heading))
            return
        self._json(200, SESSION.tick(lat, lng, accuracy, heading))


def main() -> None:
    load_env()
    port = PORT
    args = __import__("sys").argv[1:]
    if len(args) >= 2 and args[0] == "--port":
        port = int(args[1])
    httpd = ThreadingHTTPServer((HOST, port), Handler)
    print(f"Walk preview at http://{HOST}:{port}/walk.html", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
