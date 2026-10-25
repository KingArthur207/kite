"""NavSignal turns a text command into a walking route. No network, no Tk, no microphone."""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "run")]

from nav_signal import NavSignal, meters_from_path, nav_phrase


class _Plan:
    def __init__(self, status, say, points=None, steps=None, distance_m=None):
        self.status = status
        self.say = say
        self.points = points
        self.steps = steps
        self.distance_m = distance_m


def _boom(*_args, **_kwargs):
    raise AssertionError("not used")


def _until(ready, what, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if ready():
            return
        time.sleep(0.01)
    raise AssertionError(what)


def test_route_stores_the_points_and_say():
    points = [(37.8736, -122.2570), (37.8754, -122.2588)]
    seen = {}

    def plan_fn(text, origin):
        seen["text"] = text
        seen["origin"] = origin
        return _Plan("route", "Going to Cloyne Court, 420 meters away.", points=points,
                     steps=["Head north on Ridge Road"], distance_m=420)

    lines = []
    nav = NavSignal(plan_fn=plan_fn, choose_fn=_boom, latest_fn=_boom)
    assert nav.snapshot() == {
        "fix": None, "route": None, "say": "", "step": "", "distance_m": None, "choosing": False}
    nav.resolve("to Cloyne Court", lambda message, kind: lines.append((message, kind)))
    snap = nav.snapshot()
    assert seen == {"text": "to Cloyne Court", "origin": None}
    assert snap["route"] == points
    assert snap["say"] == "Going to Cloyne Court, 420 meters away."
    assert snap["step"] == "Head north on Ridge Road"
    assert snap["distance_m"] == 420
    assert snap["choosing"] is False
    assert nav.waiting_for_choice() is False
    assert lines == [("Going to Cloyne Court, 420 meters away.", "recv")]


def test_clarify_waits_for_a_choice_and_does_not_set_a_route():
    asked = _Plan("clarify", "Which one? 1: Cloyne Court. 2: Cloyne Court Hotel.")
    lines = []
    nav = NavSignal(plan_fn=lambda text, origin: asked, choose_fn=_boom, latest_fn=_boom)
    nav.resolve("to Cloyne Court", lambda message, kind: lines.append((message, kind)))
    snap = nav.snapshot()
    assert nav.waiting_for_choice() is True
    assert snap["route"] is None
    assert snap["choosing"] is True
    assert snap["say"] == asked.say
    assert snap["step"] == ""
    assert snap["distance_m"] is None
    assert lines == [(asked.say, "recv")]


def test_a_number_answers_the_pending_clarify():
    asked = _Plan("clarify", "Which one? 1: Cloyne Court. 2: Cloyne Court Hotel.")
    points = [(37.875, -122.259), (37.876, -122.260)]
    calls = []

    def plan_fn(text, origin):
        calls.append((text, origin))
        return asked

    def choose_fn(plan, n):
        calls.append(("choose", plan, n))
        return _Plan("route", "Going to Cloyne Court Hotel.", points=points, steps=["Turn left"], distance_m=90)

    lines = []
    nav = NavSignal(plan_fn=plan_fn, choose_fn=choose_fn, latest_fn=_boom)
    nav.resolve("to Cloyne Court", lambda message, kind: lines.append((message, kind)))
    assert nav.waiting_for_choice() is True
    nav.resolve("2", lambda message, kind: lines.append((message, kind)))
    assert calls == [("to Cloyne Court", None), ("choose", asked, 2)]
    snap = nav.snapshot()
    assert snap["route"] == points
    assert snap["say"] == "Going to Cloyne Court Hotel."
    assert snap["step"] == "Turn left"
    assert snap["distance_m"] == 90
    assert nav.waiting_for_choice() is False
    assert snap["choosing"] is False
    assert lines[-1] == ("Going to Cloyne Court Hotel.", "recv")


def test_an_error_sets_say_and_leaves_the_route():
    points = [(37.87, -122.26), (37.88, -122.25)]
    calls = {"n": 0}

    def plan_fn(text, origin):
        calls["n"] += 1
        if calls["n"] == 1:
            return _Plan("route", "Going to Cloyne Court.", points=points, steps=["Head north"], distance_m=200)
        return _Plan("error", "I can't look up places right now.")

    lines = []
    nav = NavSignal(plan_fn=plan_fn, choose_fn=_boom, latest_fn=_boom)
    nav.resolve("to Cloyne Court", lambda message, kind: lines.append((message, kind)))
    nav.resolve("to nowhere", lambda message, kind: lines.append((message, kind)))
    snap = nav.snapshot()
    assert snap["say"] == "I can't look up places right now."
    assert snap["route"] == points
    assert snap["step"] == "Head north"
    assert snap["distance_m"] == 200
    assert nav.waiting_for_choice() is False
    assert lines[-1] == ("I can't look up places right now.", "error")


def test_a_failed_phone_read_keeps_the_previous_fix():
    good = {"source": "phone", "lat": 37.8756, "lng": -122.2603, "accuracy": 4}
    other = {"source": "phone-old", "lat": 0.0, "lng": 0.0}
    mode = {"fail": True}
    calls = {"n": 0}
    seen = {}

    def latest():
        calls["n"] += 1
        if mode["fail"]:
            raise OSError("can't reach SpacetimeDB")
        return [good, other]

    def plan_fn(text, origin):
        seen["origin"] = origin
        return _Plan("route", "Going.", points=[(37.88, -122.25)], steps=["Head north"], distance_m=12)

    nav = NavSignal(plan_fn=plan_fn, choose_fn=_boom, latest_fn=latest)
    try:
        nav.start()
        nav.start()
        _until(lambda: calls["n"] >= 1, "the phone poll never ran")
        alive = [t for t in threading.enumerate() if t.name == "nav-gps" and t.is_alive()]
        assert len(alive) == 1 and alive[0].daemon
        assert nav.snapshot()["fix"] is None
        nav.stop()
        assert not any(t.name == "nav-gps" and t.is_alive() for t in threading.enumerate())

        mode["fail"] = False
        before = calls["n"]
        nav.start()
        _until(lambda: calls["n"] > before and nav.snapshot()["fix"] is not None, "no phone fix arrived")
        assert nav.snapshot()["fix"] == good
        nav.resolve("to Cloyne Court", lambda *_: None)
        assert seen["origin"] == (37.8756, -122.2603)

        mode["fail"] = True
        before = calls["n"]
        _until(lambda: calls["n"] > before, "the failing read never happened")
        fix = nav.snapshot()["fix"]
        assert fix["lat"] == 37.8756 and fix["lng"] == -122.2603 and fix["source"] == "phone"
    finally:
        nav.stop()


def test_submit_calls_resolve_on_a_daemon_thread():
    called = {}
    done = threading.Event()

    class Spy(NavSignal):
        def resolve(self, text, log):
            called["thread"] = threading.current_thread()
            super().resolve(text, log)

    def plan_fn(text, origin):
        return _Plan("route", "Going to Cloyne Court.", points=[(1.0, 2.0)], steps=["Walk"], distance_m=5)

    def log(message, kind):
        called["line"] = (message, kind)
        done.set()

    nav = Spy(plan_fn=plan_fn, choose_fn=_boom, latest_fn=_boom)
    nav.submit("to Cloyne Court", log)
    assert done.wait(2)
    assert called["thread"].daemon
    assert called["thread"].name == "nav-cmd"
    assert called["thread"] is not threading.current_thread()
    assert called["line"] == ("Going to Cloyne Court.", "recv")
    assert nav.snapshot()["route"] == [(1.0, 2.0)]
    assert nav.snapshot()["say"] == "Going to Cloyne Court."


def test_console_prefixes():
    assert nav_phrase("takeoff") is None
    assert nav_phrase("land") is None
    assert nav_phrase('destination: "Cloyne Court"') == "Cloyne Court"
    assert nav_phrase("destination: Doe Library") == "Doe Library"
    assert nav_phrase("destination:") == ""
    assert nav_phrase("find me bubble tea") == "nearest bubble tea"
    assert nav_phrase("find me the nearest bubble tea shop") == "the nearest bubble tea shop"
    assert nav_phrase("find me coffee nearby") == "coffee nearby"
    assert nav_phrase("find me") == ""


def test_a_fix_on_the_segment_is_not_off_the_path():
    # A point halfway along a short north-south segment is on the path.
    assert meters_from_path(37.87005, -122.26, [(37.87, -122.26), (37.8701, -122.26)]) < 5
    # A hundred metres east of that segment is off it.
    assert meters_from_path(37.87005, -122.2588, [(37.87, -122.26), (37.8701, -122.26)]) > 80


def test_import_does_not_open_navigate_phone_or_earpiece():
    script = (
        "import os, sys\n"
        "from pathlib import Path\n"
        "root = Path(os.environ['NAV_SIGNAL_ROOT'])\n"
        "sys.path[:0] = [str(root), str(root / 'run')]\n"
        "import nav_signal\n"
        "bad = [n for n in sys.modules if n in ('kite.navigate', 'phone.latest') "
        "or n == 'earpiece' or n.startswith('earpiece.')]\n"
        "assert not bad, bad\n"
    )
    env = os.environ.copy()
    env["NAV_SIGNAL_ROOT"] = str(ROOT)
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, cwd=str(ROOT))
    assert proc.returncode == 0, proc.stdout + proc.stderr
