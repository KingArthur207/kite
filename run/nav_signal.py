"""Live phone GPS and a walking route for the kite desktop window.

Voice is not built. A caller submits a text command, the same words a voice command would eventually produce.

    nav = NavSignal()          # tests pass plan_fn, choose_fn, latest_fn; the defaults hit the network
    nav.start()                # newest phone fix, about once a second
    nav.submit("to Cloyne Court", app.log)
    nav.snapshot()             # fix, route, say, step, distance_m, choosing
"""

from __future__ import annotations

import math
import re
import threading
import time

POLL_S = 1.0
OFF_ROUTE_M = 25.0
OFF_ROUTE_S = 8.0
REROUTE_MIN_S = 20.0
REROUTE_MAX = 5
REROUTE_WINDOW_S = 600.0

_DEST = re.compile(r"^destination\s*:\s*(.*)$", re.IGNORECASE)
_FIND = re.compile(r"^find\s+me(?:\s+(.*))?$", re.IGNORECASE)
_NEAR = re.compile(
    r"^(?:the\s+|a\s+|an\s+|any\s+)?(?:nearest|closest|nearby|close\s+by|local)\b"
    r"|\b(?:nearby|near\s+me|close\s+by|around\s+here)\s*$",
    re.IGNORECASE,
)


def nav_phrase(text: str) -> str | None:
    """The place words for plan(), or None when this line is not navigation.

    ``destination: Cloyne Court`` is that place. ``find me bubble tea`` is a nearby
    search (``nearest`` is added when the line does not already say so). An empty
    string means the prefix was there and the place was blank.
    """
    line = " ".join(text.strip().split())
    dest = _DEST.match(line)
    if dest:
        return _unquote(dest.group(1))
    find = _FIND.match(line)
    if not find:
        return None
    body = _unquote(find.group(1) or "")
    if not body or _NEAR.search(body):
        return body
    return "nearest " + body


def _unquote(text: str) -> str:
    body = text.strip()
    if len(body) >= 2 and body[0] == body[-1] and body[0] in "\"'":
        body = body[1:-1].strip()
    return body


def meters_from_path(lat: float, lng: float, points) -> float:
    """Metres from a fix to the nearest point on the path. One point, or none, is on the path."""
    if not points or len(points) < 2:
        return 0.0
    cos = math.cos(math.radians(lat))
    best = float("inf")
    for a, b in zip(points, points[1:]):
        ax = (a[1] - lng) * 111_320.0 * cos
        ay = (a[0] - lat) * 111_320.0
        bx = (b[1] - lng) * 111_320.0 * cos
        by = (b[0] - lat) * 111_320.0
        dx, dy = bx - ax, by - ay
        span = dx * dx + dy * dy
        t = 0.0 if span == 0 else max(0.0, min(1.0, (-ax * dx + -ay * dy) / span))
        best = min(best, math.hypot(ax + t * dx, ay + t * dy))
    return best


def _positive_int(text: str) -> int | None:
    """The answer to "which one?" (1, 2, 3, ...), or None when text is anything else."""
    s = text.strip()
    if not s.isascii() or not s.isdigit():
        return None
    n = int(s)
    return n if n > 0 else None


class NavSignal:
    """The newest phone fix and the walking route (or the pending "which one?") for the map."""

    def __init__(self, plan_fn=None, choose_fn=None, latest_fn=None):
        self._plan_fn = plan_fn
        self._choose_fn = choose_fn
        self._latest_fn = latest_fn
        self._lock = threading.Lock()
        self._gate = threading.Lock()  # start/stop only; the poll thread never takes this
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fix: dict | None = None
        self._route: list | None = None
        self._say = ""
        self._step = ""
        self._distance_m = None
        self._pending = None  # the clarify plan choose() still needs
        self._phrase: str | None = None
        self._off_since: float | None = None
        self._rerouted: list[float] = []
        self._log = None

    def start(self) -> None:
        """Read the newest phone fix about once a second until stop()."""
        with self._gate:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._watch_phone, name="nav-gps", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Stop the phone poll. Safe to call when it was never started."""
        with self._gate:
            self._stop.set()
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=POLL_S + 1.0)

    def submit(self, text: str, log) -> None:
        """Resolve text on a daemon thread. log(message, kind) is App.log and is safe from that thread."""
        threading.Thread(target=self.resolve, args=(text, log), name="nav-cmd", daemon=True).start()

    def note_fix(self, lat, lng) -> None:
        """Remember a fix the window already has, so a command does not wait for the next poll."""
        try:
            lat_f, lng_f = float(lat), float(lng)
        except (TypeError, ValueError):
            return
        if not math.isfinite(lat_f) or not math.isfinite(lng_f):
            return
        with self._lock:
            fix = dict(self._fix or {})
            fix["lat"] = lat_f
            fix["lng"] = lng_f
            self._fix = fix

    def resolve(self, text: str, log) -> None:
        """Apply text now. A positive integer answers a pending clarify; anything else is planned from the fix."""
        self._log = log
        with self._lock:
            pending = self._pending
            origin = self._origin()
        number = _positive_int(text)
        if pending is not None and number is not None:
            plan = self._choose()(pending, number)
        else:
            plan = self._plan()(text, origin)
            if getattr(plan, "status", None) == "route":
                with self._lock:
                    self._phrase = text
                    self._off_since = None
        say, kind = self._store(plan)
        log(say, kind)

    def waiting_for_choice(self) -> bool:
        """True after a clarify, until a route replaces it."""
        with self._lock:
            return self._pending is not None

    def snapshot(self) -> dict:
        """fix, route, say, step, distance_m, choosing. Safe from the window thread."""
        with self._lock:
            return {
                "fix": None if self._fix is None else dict(self._fix),
                "route": None if self._route is None else list(self._route),
                "say": self._say,
                "step": self._step,
                "distance_m": self._distance_m,
                "choosing": self._pending is not None,
            }

    def _plan(self):
        if self._plan_fn is None:  # imported here so importing this module does not touch the network
            from kite.navigate import plan as plan_fn
            self._plan_fn = plan_fn
        return self._plan_fn

    def _choose(self):
        if self._choose_fn is None:
            from kite.navigate import choose as choose_fn
            self._choose_fn = choose_fn
        return self._choose_fn

    def _latest(self):
        if self._latest_fn is None:
            from phone.latest import latest as latest_fn
            self._latest_fn = latest_fn
        return self._latest_fn

    def _phrase_if_off_route(self) -> str | None:
        """The phrase to plan again, or None. Caller holds self._lock."""
        route = self._route
        if self._pending is not None or not self._phrase or not route or len(route) < 2:
            self._off_since = None
            return None
        origin = self._origin()
        if origin is None:
            return None
        if meters_from_path(origin[0], origin[1], route) <= OFF_ROUTE_M:
            self._off_since = None
            return None
        now = time.monotonic()
        if self._off_since is None:
            self._off_since = now
            return None
        if now - self._off_since < OFF_ROUTE_S:
            return None
        self._rerouted = [t for t in self._rerouted if now - t < REROUTE_WINDOW_S]
        if self._rerouted and now - self._rerouted[-1] < REROUTE_MIN_S:
            return None
        if len(self._rerouted) >= REROUTE_MAX:
            return None
        self._rerouted.append(now)
        self._off_since = None
        return self._phrase

    def _origin(self) -> tuple[float, float] | None:
        """(lat, lng) from the current fix. Caller holds self._lock."""
        fix = self._fix
        if not isinstance(fix, dict):
            return None
        lat, lng = fix.get("lat"), fix.get("lng")
        if lat is None or lng is None:
            return None
        try:
            return (float(lat), float(lng))
        except (TypeError, ValueError):
            return None

    def _watch_phone(self) -> None:
        while not self._stop.is_set():
            self._pull_fix()
            self._stop.wait(POLL_S)

    def _pull_fix(self) -> None:
        """Store the first row. A failure or an empty read leaves the fix already stored."""
        try:
            rows = self._latest()()
            if not rows:
                return
            row = rows[0]
            if not isinstance(row, dict):
                return
            row = dict(row)
        except Exception:
            return
        phrase = None
        origin = None
        with self._lock:
            self._fix = row
            phrase = self._phrase_if_off_route()
            origin = self._origin()
        if phrase and origin is not None and self._log is not None:
            plan = self._plan()(phrase, origin)
            say, kind = self._store(plan)
            self._log(say, kind)

    def _store(self, plan) -> tuple[str, str]:
        """A route replaces the path. A clarify only asks. An error only changes what we say."""
        say = plan.say or ""
        kind = "error" if plan.status == "error" else "recv"
        with self._lock:
            self._say = say
            if plan.status == "route":
                steps = plan.steps or []
                self._route = list(plan.points or [])
                self._step = steps[0] if steps else ""
                self._distance_m = plan.distance_m
                self._pending = None
            elif plan.status == "clarify":
                self._pending = plan
        return say, kind
