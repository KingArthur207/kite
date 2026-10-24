"""python -m kite.preflight: every check with fakes (a real busy UDP port, fake drones with good and bad batteries, no
video, slow video), the GO / NO-GO decision, exit codes and the JSON output. No drone, no network."""

import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from collections import namedtuple

import numpy as np
import pytest

from kite import preflight as pf
from kite.config import SafetyCfg
from kite.drone import SafetyGovernor
from kite.drone.tello import TelloDrone
from kite.types import Frame, Telemetry

FAST = {"first_frame_s": 0.3, "measure_s": 0.3}


@pytest.fixture(autouse=True)
def quick_queries(monkeypatch):
    monkeypatch.setattr(pf, "RTT_GAP_S", 0.01)
    monkeypatch.setattr(pf, "RTT_TIMEOUT_S", 0.2)


def statuses(checks):
    return {c.name: c.status for c in checks}


def free_udp_port():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# ------------------------------------------------------------------ laptop checks
def test_deps_present_and_missing():
    assert pf.check_deps().status == pf.GO
    c = pf.check_deps(("numpy", "kite_no_such_module"))
    assert c.status == pf.NO_GO and "kite_no_such_module" in c.detail


def test_ffmpeg_missing_only_warns(monkeypatch):
    monkeypatch.setattr(pf.shutil, "which", lambda name: None)
    assert pf.check_ffmpeg().status == pf.WARN


@pytest.mark.parametrize("free_gb, want", [(1.0, pf.NO_GO), (3.0, pf.WARN), (40.0, pf.GO)])
def test_disk_thresholds(tmp_path, monkeypatch, free_gb, want):
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(pf.shutil, "disk_usage", lambda p: usage(1e12, 0, free_gb * 1e9))
    disk, writable = pf.check_disk(str(tmp_path / "recordings"))
    assert disk.status == want and disk.data["free_gb"] == free_gb
    assert writable.status == pf.GO and (tmp_path / "recordings").is_dir()
    assert not list((tmp_path / "recordings").iterdir())  # the probe file is gone


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write anywhere")
def test_recordings_not_writable(tmp_path):
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o500)
    try:
        assert statuses(pf.check_disk(str(ro)))["recordings folder"] == pf.NO_GO
    finally:
        ro.chmod(0o700)


def test_ports_free():
    port = free_udp_port()
    assert pf.check_ports((port,)).status == pf.GO


def test_port_held_in_this_process():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
        c = pf.check_ports((port,))
    assert c.status == pf.NO_GO and c.detail.startswith(f"{port} held by")


@pytest.mark.skipif(shutil.which("lsof") is None, reason="needs lsof to name the program")
def test_port_held_by_another_program_is_named():
    """A stale recorder / phone-app bridge: another process binds the port; the check names its pid."""
    code = ("import socket, sys, time\n"
            "s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.bind(('', 0))\n"
            "print(s.getsockname()[1], flush=True); time.sleep(30)\n")
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    try:
        port = int(proc.stdout.readline())
        c = pf.check_ports((port,))
        assert c.status == pf.NO_GO
        assert f"pid {proc.pid}" in c.detail and f"kill {proc.pid}" in c.detail
        assert c.data["busy"][str(port)]
    finally:
        proc.kill()
        proc.wait()


def pmset(source, pct, state):
    return f"Now drawing from '{source}'\n -InternalBattery-0 (id=1)\t{pct}%; {state}; 0:20 remaining present: true\n"


PMSET_LOW = pmset("Battery Power", 9, "discharging")
PMSET_MID = pmset("Battery Power", 30, "discharging")
PMSET_AC_LOW = pmset("AC Power", 9, "charging")
PMSET_FULL = pmset("AC Power", 100, "charged")


def fake_run(outputs):
    """_run replacement: the first key that is a prefix of the command's words wins."""
    def run(cmd, timeout=5.0):
        for key, out in outputs.items():
            if list(cmd[:len(key)]) == list(key):
                return out
        return ""
    return run


@pytest.mark.parametrize("text, want", [(PMSET_LOW, pf.NO_GO), (PMSET_MID, pf.WARN), (PMSET_AC_LOW, pf.WARN),
                                        (PMSET_FULL, pf.GO), ("Now drawing from 'AC Power'\n", pf.SKIP)])
def test_laptop_battery(monkeypatch, text, want):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(pf, "_run", fake_run({("pmset",): text}))
    assert pf.check_laptop_battery().status == want


PORTS = "Hardware Port: Ethernet\nDevice: en3\n\nHardware Port: Wi-Fi\nDevice: en1\nEthernet Address: aa\n"


def wifi_outputs(ssid_line, ip, airport="You are not associated with an AirPort network.\n"):
    return {("networksetup", "-listallhardwareports"): PORTS,
            ("ipconfig", "getsummary", "en1"): f"<dictionary> {{\n  BSSID : <redacted>\n{ssid_line}\n}}\n",
            ("networksetup", "-getairportnetwork", "en1"): airport,
            ("ipconfig", "getifaddr", "en1"): ip}


@pytest.mark.parametrize("ssid_line, ip, airport, quick, want, method", [
    ("  SSID : TELLO-5A1B2C", "192.168.10.2\n", "", False, pf.GO, "ssid"),
    ("  SSID : HomeNet", "192.168.4.10\n", "", False, pf.NO_GO, "ssid"),
    ("  SSID : HomeNet", "192.168.4.10\n", "", True, pf.WARN, "ssid"),
    ("", "", "Current Wi-Fi Network: TELLO-ABCDEF\n", False, pf.GO, "ssid"),  # older macOS: networksetup shows it
    ("  SSID : <redacted>", "192.168.10.3\n", None, False, pf.GO, "ip"),  # macOS hides the SSID: judge the address
    ("  SSID : <redacted>", "192.168.4.123\n", None, False, pf.NO_GO, "ip"),
    ("  SSID : <redacted>", "", None, False, pf.NO_GO, "ip"),  # Wi-Fi not joined
])
def test_wifi(monkeypatch, ssid_line, ip, airport, quick, want, method):
    monkeypatch.setattr(sys, "platform", "darwin")
    outputs = wifi_outputs(ssid_line, ip) if airport is None else wifi_outputs(ssid_line, ip, airport)
    monkeypatch.setattr(pf, "_run", fake_run(outputs))
    c = pf.check_wifi(quick=quick)
    assert c.status == want and c.data["method"] == method and c.data["device"] == "en1"
    assert c.data["on_tello"] == (want == pf.GO)


def test_camera_json(tmp_path):
    assert pf.check_camera_json(tmp_path / "tello_camera.json").status == pf.WARN
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"fx": 921.4, "fy": 920.0}))
    c = pf.check_camera_json(good)
    assert c.status == pf.GO and "921" in c.detail
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert pf.check_camera_json(bad).status == pf.WARN


@pytest.mark.parametrize("tty, want", [(True, pf.GO), (False, pf.NO_GO)])
def test_tty(monkeypatch, tty, want):
    class Stdin:
        def isatty(self):
            return tty
    monkeypatch.setattr(sys, "stdin", Stdin())
    assert pf.check_tty().status == want


# ------------------------------------------------------------------ drone checks
class TimedSource:
    """Frames at `fps` from `first_after_s` on (None: never), sequence numbers derived from the clock."""

    def __init__(self, fps=30.0, first_after_s=0.05, size=(960, 720)):
        self.fps, self.first_after_s, self.t0 = fps, first_after_s, time.monotonic()
        self.image = np.zeros((size[1], size[0], 3), np.uint8)

    def read(self):
        dt = time.monotonic() - self.t0
        if self.first_after_s is None or dt < self.first_after_s:
            return None
        return Frame(self.image, time.time(), int((dt - self.first_after_s) * self.fps), "fake")

    def wait_first(self, timeout_s=10.0):
        t_end = time.monotonic() + timeout_s
        while time.monotonic() < t_end:
            f = self.read()
            if f is not None:
                return f
            time.sleep(0.005)
        return None


class FakeRawTello:
    """The djitellopy surface the preflight uses: state dicts (a new one per packet) and query replies."""

    def __init__(self, state_hz=10.0, reply_s=0.02, bat=90, temph=55):
        self.state_hz, self.reply_s = state_hz, reply_s
        self.base = {"bat": bat, "templ": temph - 2, "temph": temph, "h": 0, "tof": 10}
        self.udp = {"responses": [], "state": {}}
        self.sent = []
        self._tick, self._state = -1, dict(self.base)

    def get_current_state(self):
        if self.state_hz:
            tick = int(time.monotonic() * self.state_hz)
            if tick != self._tick:
                self._tick, self._state = tick, dict(self.base)
        return self._state

    def get_own_udp_object(self):
        return self.udp

    def send_command_without_return(self, cmd):
        self.sent.append(cmd)
        if self.reply_s is not None:
            threading.Timer(self.reply_s, lambda: self.udp["responses"].append(str(self.base["bat"]).encode())).start()


class FakeLinkDrone:
    """connect / telemetry / frame_source / close, like TelloDrone in dry run."""

    def __init__(self, bat=90, temph=55, fps=30.0, first_after_s=0.05, size=(960, 720), state_hz=10.0,
                 reply_s=0.02, connect_error=None):
        self.tello = FakeRawTello(state_hz, reply_s, bat, temph)
        self.bat, self.connect_error = bat, connect_error
        self.source_args = (fps, first_after_s, size)
        self.connected = self.closed = False

    def connect(self):
        if self.connect_error:
            raise RuntimeError(self.connect_error)
        self.connected = True

    def telemetry(self):
        return Telemetry(t=time.time(), battery_pct=self.bat)

    def frame_source(self):
        return TimedSource(*self.source_args)

    def close(self):
        self.closed = True


def run_drone(measure_s=FAST["measure_s"], **kw):
    drone = FakeLinkDrone(**kw)
    checks = pf.drone_checks(lambda: drone, first_frame_s=FAST["first_frame_s"], measure_s=measure_s)
    return statuses(checks), checks, drone


def test_healthy_drone_is_go_and_link_closed():
    st, checks, drone = run_drone(measure_s=0.6)
    assert drone.closed
    assert all(s == pf.GO for s in st.values()), checks
    by = {c.name: c for c in checks}
    assert 25 <= by["video"].data["fps"] <= 33
    assert 7 <= by["telemetry"].data["hz"] <= 13
    assert 15 <= by["command latency"].data["median_ms"] <= 120
    assert by["drone battery"].data["expected_min"] == pytest.approx((90 - 20) / pf.DRAIN_PCT_PER_MIN, abs=0.06)
    assert drone.tello.sent == ["battery?"] * pf.RTT_SAMPLES  # read-only queries, nothing else


@pytest.mark.parametrize("bat, want", [(40, pf.NO_GO), (55, pf.WARN), (85, pf.GO)])
def test_drone_battery(bat, want):
    st, checks, _ = run_drone(bat=bat)
    assert st["drone battery"] == want


def test_no_battery_figure_is_no_go():
    assert pf._battery(None).status == pf.NO_GO


@pytest.mark.parametrize("temph, want", [(60, pf.GO), (75, pf.WARN), (90, pf.NO_GO)])
def test_drone_temperature(temph, want):
    assert run_drone(temph=temph)[0]["drone temperature"] == want


def test_no_video_is_no_go():
    st, _, drone = run_drone(first_after_s=None)
    assert st["video"] == pf.NO_GO and drone.closed


@pytest.mark.parametrize("fps, want", [(12.0, pf.WARN), (2.0, pf.NO_GO)])
def test_slow_video(fps, want):
    assert run_drone(fps=fps)[0]["video"] == want


def test_wrong_frame_size_warns():
    assert run_drone(size=(1280, 720))[0]["frame size"] == pf.WARN


@pytest.mark.parametrize("hz, want", [(0.0, pf.NO_GO), (3.0, pf.WARN)])
def test_slow_telemetry(hz, want):
    assert run_drone(measure_s=0.5, state_hz=hz)[0]["telemetry"] == want  # 0.5 s holds at least one 3 Hz packet


def test_no_command_replies_is_no_go():
    st, _, _ = run_drone(reply_s=None)
    assert st["command latency"] == pf.NO_GO


def test_connect_failure_is_no_go_and_still_closes():
    st, checks, drone = run_drone(connect_error="Did not receive a state packet from the Tello")
    assert st["drone link"] == pf.NO_GO and "state packet" in checks[0].detail
    assert drone.closed and "drone link closed" in st


def test_default_drone_is_a_dry_run(monkeypatch):
    """The real path builds the Tello with make_tello(cfg, dry_run=True): motion can never be sent."""
    import kite.drone

    seen = {}

    def fake_make_tello(cfg, dry_run=False):
        seen["dry_run"] = dry_run
        return FakeLinkDrone()

    monkeypatch.setattr(kite.drone, "make_tello", fake_make_tello)
    pf.drone_checks(**FAST)
    assert seen == {"dry_run": True}


def test_through_the_real_drone_layer_sends_no_motion():
    """TelloDrone(dry_run) inside SafetyGovernor, as kite.record builds it: the raw state / command channel is found
    through both wrappers and only 'command' and the read-only queries reach the drone."""
    raw = FakeRawTello()
    raw.connect = lambda: raw.sent.append("command")
    raw.end = lambda: raw.sent.append("<end>")
    inner = TelloDrone(tello=raw, video=False, dry_run=True)
    inner.frame_source = lambda: TimedSource()  # video=False: stand in for the UDP reader
    checks = pf.drone_checks(lambda: SafetyGovernor(inner, SafetyCfg()), **FAST)
    assert all(c.status == pf.GO for c in checks), checks
    assert raw.sent == ["command"] + ["battery?"] * pf.RTT_SAMPLES + ["<end>"]


# ------------------------------------------------------------------ run / verdict / main
def all_go(monkeypatch, on_tello=True):
    monkeypatch.setattr(pf, "check_deps", lambda: pf.Check("python deps", pf.GO, "ok"))
    monkeypatch.setattr(pf, "check_ffmpeg", lambda: pf.Check("ffmpeg", pf.GO, "ok"))
    monkeypatch.setattr(pf, "check_disk", lambda out: [pf.Check("disk", pf.GO, "ok")])
    monkeypatch.setattr(pf, "check_ports", lambda: pf.Check("udp ports", pf.GO, "ok"))
    monkeypatch.setattr(pf, "check_laptop_battery", lambda: pf.Check("laptop battery", pf.GO, "ok"))
    monkeypatch.setattr(pf, "check_wifi", lambda quick=False: pf.Check(
        "tello wi-fi", pf.GO if on_tello else (pf.WARN if quick else pf.NO_GO), "ok", {"on_tello": on_tello}))
    monkeypatch.setattr(pf, "check_camera_json", lambda: pf.Check("camera calibration", pf.WARN, "missing"))
    monkeypatch.setattr(pf, "check_tty", lambda: pf.Check("terminal", pf.GO, "ok"))
    monkeypatch.setattr(pf, "MEASURE_S", 0.3)
    monkeypatch.setattr(pf, "FIRST_FRAME_S", 0.3)


def test_run_on_tello_wifi_checks_the_drone(monkeypatch):
    all_go(monkeypatch)
    checks = pf.run(make_drone=lambda: FakeLinkDrone())
    assert "drone battery" in statuses(checks) and pf.verdict(checks) == pf.GO


def test_run_skips_the_drone(monkeypatch):
    all_go(monkeypatch, on_tello=False)
    st = statuses(pf.run(make_drone=lambda: pytest.fail("must not connect")))
    assert st["drone"] == pf.SKIP and st["tello wi-fi"] == pf.NO_GO
    st = statuses(pf.run(quick=True, make_drone=lambda: pytest.fail("must not connect")))
    assert st["drone"] == pf.SKIP and st["tello wi-fi"] == pf.WARN


def test_run_drone_flag_forces_it_and_busy_ports_block_it(monkeypatch):
    all_go(monkeypatch, on_tello=False)
    assert "drone battery" in statuses(pf.run(drone=True, make_drone=lambda: FakeLinkDrone()))
    monkeypatch.setattr(pf, "check_ports", lambda: pf.Check("udp ports", pf.NO_GO, "8889 held"))
    st = statuses(pf.run(drone=True, make_drone=lambda: pytest.fail("must not connect")))
    assert st["drone"] == pf.SKIP


def test_main_go_prints_next_command(monkeypatch, capsys):
    all_go(monkeypatch)
    monkeypatch.setattr(pf, "drone_checks", lambda make_drone=None: [pf.Check("drone battery", pf.GO, "90%")])
    assert pf.main([]) == 0
    out = capsys.readouterr().out
    assert "GO (warnings: camera calibration). Next:" in out
    assert "python -m kite.record --source tello --out recordings --fly --speed 50" in out


def test_main_no_go_exit_code(monkeypatch, capsys):
    all_go(monkeypatch)
    monkeypatch.setattr(pf, "drone_checks", lambda make_drone=None: [pf.Check("drone battery", pf.NO_GO, "40%")])
    assert pf.main([]) == 1
    out = capsys.readouterr().out
    assert "NO-GO: drone battery." in out and "kite.record" not in out


def test_main_json(monkeypatch, capsys):
    all_go(monkeypatch, on_tello=False)
    assert pf.main(["--quick", "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["verdict"] == "GO" and doc["next_command"].startswith("python -m kite.record --source tello")
    names = {c["name"]: c["status"] for c in doc["checks"]}
    assert names["drone"] == "SKIP" and names["tello wi-fi"] == "WARN" and names["camera calibration"] == "WARN"
    assert pf.main(["--json"]) == 1
    doc = json.loads(capsys.readouterr().out)
    assert doc["verdict"] == "NO-GO" and doc["next_command"] is None


def test_quick_and_drone_conflict():
    with pytest.raises(SystemExit) as e:
        pf.main(["--quick", "--drone"])
    assert e.value.code == 2
