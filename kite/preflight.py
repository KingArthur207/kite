"""Pre-flight check for the shoot: one command at the corner, one GO / NO-GO line per check, exit 1 on NO-GO.

    python -m kite.preflight            # laptop checks, plus the drone checks when the laptop is on the Tello Wi-Fi
    python -m kite.preflight --quick    # laptop checks only (no drone, the Wi-Fi check only warns)
    python -m kite.preflight --drone    # force the drone checks
    python -m kite.preflight --json     # machine-readable results on stdout
    python -m kite.preflight --video-backend opencv   # check another Tello video reader (default pyav)

Laptop: python deps, ffmpeg, free disk and a writable recordings folder, the Tello UDP ports free (8889 commands,
8890 state, 11111 video; the program holding one is named), laptop battery, the Tello Wi-Fi, the camera calibration
file and an interactive terminal (the --fly keys need one).

Drone: opened through kite's drone layer with dry_run=True, so NO motion command can be sent. Reads the battery and
temperature, waits for the first video frame, measures the frame rate, frame size and telemetry rate over a few
seconds and the command round trip, then closes the link at once (streaming drains the battery and heats the drone
on the ground). The drone part takes about 6 s with a healthy drone, 15 s at most.

Exit code: 0 for GO (warnings allowed), 1 for NO-GO.
"""

from __future__ import annotations

import argparse
import importlib
import ipaddress
import json
import os
import re
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .camera import DEFAULT_PATH as CAMERA_JSON
from .config import VIDEO_BACKENDS, add_video_link_args, video_link_argv

GO, WARN, NO_GO, SKIP = "GO", "WARN", "NO-GO", "SKIP"
NEXT_COMMAND = "python -m kite.record --source tello --out {out} --fly --speed 50"

DEPS = ("cv2", "av", "djitellopy", "numpy")
TELLO_PORTS = (8889, 8890, 11111)  # commands / state / video, all bound on the laptop side
TELLO_NET = ipaddress.ip_network("192.168.10.0/24")  # the Tello's DHCP range; the drone itself is 192.168.10.1
TELLO_SSID_PREFIX = "TELLO-"

DISK_NO_GO_GB, DISK_WARN_GB = 2.0, 5.0  # a 5-minute take is about 0.7 GB of MJPG
LAPTOP_NO_GO_PCT, LAPTOP_WARN_PCT = 15.0, 40.0  # the laptop records the whole take, unplugged on the sidewalk

DRONE_NO_GO_PCT = 50.0  # one 5-minute take on one battery
AUTO_LAND_PCT = 20.0  # SafetyCfg.min_battery_pct: the governor lands here
DRAIN_PCT_PER_MIN = 8.0  # estimate: DJI's 13 min per pack, a little less with video and manoeuvres (to measure)
TAKE_MIN = 5.0
TEMP_WARN_C, TEMP_NO_GO_C = 70.0, 85.0  # estimates: the Tello heats up sitting on the ground with video on

FIRST_FRAME_S = 10.0  # wait this long for the first frame
MEASURE_S = 3.0  # window for fps and telemetry rate
FRAME_SIZE = (960, 720)
FPS_WARN, FPS_NO_GO = 20.0, 5.0  # the governor hovers after 1 s without a new frame
TELEMETRY_WARN_HZ = 5.0
RTT_SAMPLES, RTT_TIMEOUT_S, RTT_WARN_MS = 3, 1.0, 200.0
RTT_GAP_S = 0.1  # the Tello drops commands sent closer together than this
LINK_TARGET_S = 15.0


@dataclass
class Check:
    name: str
    status: str  # GO | WARN | NO-GO | SKIP
    detail: str
    data: dict = field(default_factory=dict)


def _run(cmd: list[str], timeout: float = 5.0) -> str:
    """stdout of a system command, "" if it is missing, fails or hangs (every probe here is best effort)."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


# ------------------------------------------------------------------ laptop checks
def check_deps(modules=DEPS) -> Check:
    found, missing = [], []
    for name in modules:
        try:
            mod = importlib.import_module(name)
        except Exception as e:  # an ImportError, or a broken binary wheel raising something else
            missing.append(f"{name} ({type(e).__name__}: {e})")
            continue
        version = getattr(mod, "__version__", None)
        if version is None:
            try:
                from importlib.metadata import version as dist_version

                version = dist_version(name)
            except Exception:
                version = "?"
        found.append(f"{name} {version}")
    if missing:
        return Check("python deps", NO_GO, "cannot import " + ", ".join(missing) + ': run pip install -e ".[dev]"',
                     {"missing": missing})
    return Check("python deps", GO, ", ".join(found))


def check_ffmpeg() -> Check:
    path = shutil.which("ffmpeg")
    if path:
        return Check("ffmpeg", GO, path)
    return Check("ffmpeg", WARN, "not on PATH: recording works without it, but converting video.mkv for the edit "
                 "needs it (brew install ffmpeg)")


def check_disk(out: str) -> list[Check]:
    try:
        os.makedirs(out, exist_ok=True)
    except OSError as e:
        return [Check("recordings folder", NO_GO, f"cannot create {out}: {e}")]
    free_gb = shutil.disk_usage(out).free / 1e9
    data = {"free_gb": round(free_gb, 2)}
    takes = free_gb / 0.7
    if free_gb < DISK_NO_GO_GB:
        disk = Check("disk", NO_GO, f"{free_gb:.1f} GB free in {out}: under {DISK_NO_GO_GB:g} GB (~0.7 GB per "
                     f"5-minute take); free some space", data)
    elif free_gb < DISK_WARN_GB:
        disk = Check("disk", WARN, f"{free_gb:.1f} GB free in {out} (~{takes:.0f} five-minute takes); "
                     f"{DISK_WARN_GB:g} GB is comfortable", data)
    else:
        disk = Check("disk", GO, f"{free_gb:.1f} GB free in {out} (~{takes:.0f} five-minute takes)", data)
    try:
        with tempfile.NamedTemporaryFile(dir=out, prefix=".preflight_") as f:
            f.write(b"x" * 4096)
            f.flush()
        writable = Check("recordings folder", GO, f"{out}/ is writable")
    except OSError as e:
        writable = Check("recordings folder", NO_GO, f"cannot write in {out}: {e}")
    return [disk, writable]


def port_holders(ports) -> dict[int, list[str]]:
    """{port: ["Python (pid 123)", ...]} for processes with a socket BOUND on that local UDP port (lsof)."""
    args = ["lsof", "-nP", "-Fpcn"]
    for port in ports:
        args.append(f"-iUDP:{port}")
    holders: dict[int, list[str]] = {p: [] for p in ports}
    pid = cmd = None
    for line in _run(args).splitlines():
        if line.startswith("p"):
            pid, cmd = line[1:], None
        elif line.startswith("c"):
            cmd = line[1:]
        elif line.startswith("n") and pid is not None:
            local = line[1:].split("->", 1)[0]  # a socket that only SENDS to the port is listed too: skip it
            for p in ports:
                who = f"{cmd or '?'} (pid {pid})"
                if local.endswith(f":{p}") and who not in holders[p]:
                    holders[p].append(who)
    return holders


def _bind_fails(port: int) -> str | None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.bind(("", port))
        except OSError as e:
            return str(e)
    return None


def check_ports(ports=TELLO_PORTS) -> Check:
    """A plain bind (what djitellopy and the video reader do) plus lsof to name the program in the way."""
    errors = {p: _bind_fails(p) for p in ports}
    holders = port_holders(ports)
    busy = {p: holders[p] or ["another program"] for p in ports if errors[p] or holders[p]}
    if not busy:
        return Check("udp ports", GO, ", ".join(map(str, ports)) + " free")
    parts = [f"{p} held by {', '.join(who)}" for p, who in busy.items()]
    pids = sorted({w.rsplit("pid ", 1)[1].rstrip(")") for who in busy.values() for w in who if "pid " in w})
    fix = f"quit it (kill {' '.join(pids)})" if pids else "close the Tello phone app and other Tello scripts"
    return Check("udp ports", NO_GO, "; ".join(parts) + f": {fix}", {"busy": {str(p): w for p, w in busy.items()}})


def parse_pmset(text: str) -> dict | None:
    m = re.search(r"(\d+)%;\s*([^;]+);", text)
    if not m:
        return None
    return {"pct": float(m.group(1)), "state": m.group(2).strip(), "ac": "'AC Power'" in text}


def check_laptop_battery() -> Check:
    if sys.platform != "darwin":
        return Check("laptop battery", SKIP, "macOS only")
    b = parse_pmset(_run(["pmset", "-g", "batt"]))
    if b is None:
        return Check("laptop battery", SKIP, "no battery reported by pmset")
    pct, state = b["pct"], b["state"]
    where = "on AC" if b["ac"] else "on battery"
    detail = f"{pct:.0f}%, {state}, {where}"
    if pct < LAPTOP_NO_GO_PCT and not b["ac"]:
        return Check("laptop battery", NO_GO, f"{detail}: charge it, the laptop records the whole take", b)
    if pct < LAPTOP_WARN_PCT:
        return Check("laptop battery", WARN, f"{detail}: under {LAPTOP_WARN_PCT:.0f}%, charge before the shoot", b)
    return Check("laptop battery", GO, detail, b)


def wifi_device() -> str:
    ports = _run(["networksetup", "-listallhardwareports"])
    m = re.search(r"Hardware Port: (?:Wi-Fi|AirPort)\s*\nDevice: (\S+)", ports)
    return m.group(1) if m else "en0"


def wifi_ssid(dev: str) -> str | None:
    """SSID, or None when macOS hides it (it redacts SSIDs for apps without Location permission)."""
    m = re.search(r"^\s*SSID : (.+)$", _run(["ipconfig", "getsummary", dev]), re.M)
    if m and "redacted" not in m.group(1):
        return m.group(1).strip()
    m = re.search(r"Current Wi-Fi Network: (.+)$", _run(["networksetup", "-getairportnetwork", dev]), re.M)
    if m:
        return m.group(1).strip()
    return None


def check_wifi(quick: bool = False) -> Check:
    """The SSID when macOS shows it, else the 192.168.10.x address the Tello's DHCP hands out."""
    if sys.platform != "darwin":
        return Check("tello wi-fi", SKIP, "macOS only")
    dev = wifi_device()
    ssid = wifi_ssid(dev)
    ip = _run(["ipconfig", "getifaddr", dev]).strip()
    data = {"device": dev, "ssid": ssid, "ip": ip or None, "on_tello": False}
    miss = WARN if quick else NO_GO  # --quick is the laptop-only check: fine on normal Wi-Fi
    try:
        in_range = bool(ip) and ipaddress.ip_address(ip) in TELLO_NET
    except ValueError:
        in_range = False
    if ssid is not None:
        data["method"] = "ssid"
        if ssid.upper().startswith(TELLO_SSID_PREFIX):
            data["on_tello"] = True
            return Check("tello wi-fi", GO, f"{dev} on {ssid}" + (f" ({ip})" if ip else ""), data)
        return Check("tello wi-fi", miss, f"{dev} on '{ssid}', not a {TELLO_SSID_PREFIX}xxxxxx network: join the "
                     f"drone's Wi-Fi", data)
    data["method"] = "ip"  # macOS redacted the SSID
    if not ip:
        return Check("tello wi-fi", miss, f"{dev} has no IPv4 address: power the Tello on and join its "
                     f"{TELLO_SSID_PREFIX}xxxxxx Wi-Fi", data)
    if in_range:
        data["on_tello"] = True
        return Check("tello wi-fi", GO, f"{dev} at {ip}, the Tello's 192.168.10.x range (macOS hides the SSID)", data)
    return Check("tello wi-fi", miss, f"{dev} at {ip}, not the Tello's 192.168.10.x: join the "
                 f"{TELLO_SSID_PREFIX}xxxxxx Wi-Fi (macOS hides the SSID, so this is judged by the address)", data)


AWDL_FIX = ("sudo ifconfig awdl0 down (back on reboot), and keep AirDrop at 'No One' and Handoff off "
            "(System Settings > General > AirDrop & Handoff) so it stays down")


def check_awdl() -> Check:
    """macOS AWDL (AirDrop, Handoff, Continuity, AirPlay) hops the Wi-Fi radio off the Tello's channel about once a
    second. The first real capture stalled 0.15-0.46 s about every second and got ~2 usable frames/s (31 KB/s)."""
    if sys.platform != "darwin":
        return Check("wi-fi radio", SKIP, "macOS only")
    out = _run(["ifconfig", "awdl0"])
    if not out:
        return Check("wi-fi radio", GO, "no awdl0 interface")
    active = "status: active" in out
    if "<UP" in out and active:
        return Check("wi-fi radio", WARN, f"AWDL (AirDrop/Handoff) is active: it pulls the radio off the Tello's channel "
                     f"about once a second and the video drops to a few broken frames. Fix: {AWDL_FIX}",
                     {"awdl0": "active"})
    return Check("wi-fi radio", GO, "AWDL (AirDrop/Handoff) is down", {"awdl0": "up" if "<UP" in out else "down"})


def check_camera_json(path: Path = CAMERA_JSON) -> Check:
    if not path.is_file():
        return Check("camera calibration", WARN, f"{path.name} missing: perception uses the published fx estimate "
                     "(python -m kite.calibrate --source tello)")
    try:
        cam = json.loads(path.read_text())
        return Check("camera calibration", GO, f"{path.name}: fx {float(cam['fx']):.0f} px")
    except (ValueError, KeyError, TypeError) as e:
        return Check("camera calibration", WARN, f"{path.name} unreadable ({e}): perception uses the published "
                     "estimate")


def check_tty() -> Check:
    if sys.stdin.isatty():
        return Check("terminal", GO, "interactive (the --fly keys work)")
    return Check("terminal", NO_GO, "stdin is not a terminal: run the recorder in Terminal itself, not through a "
                 "pipe or an IDE runner, or the --fly keys do nothing")


# ------------------------------------------------------------------ drone checks
def _raw_tello(drone):
    """djitellopy.Tello behind any wrapper (SafetyGovernor.inner -> TelloDrone.tello), else None."""
    d = drone
    while d is not None and hasattr(d, "inner"):
        d = d.inner
    return getattr(d, "tello", None)


def round_trip_ms(tello, cmd: str = "battery?", timeout_s: float | None = None) -> float | None:
    """Time from sending a read-only query to its reply (djitellopy's own sender polls every 100 ms: too coarse)."""
    timeout_s = RTT_TIMEOUT_S if timeout_s is None else timeout_s
    replies = tello.get_own_udp_object()["responses"]
    replies.clear()
    t0 = time.perf_counter()
    tello.send_command_without_return(cmd)
    while time.perf_counter() - t0 < timeout_s:
        if replies:
            replies.pop(0)
            return (time.perf_counter() - t0) * 1000.0
        time.sleep(0.001)
    return None


def expected_minutes(battery_pct: float) -> float:
    return max(0.0, (battery_pct - AUTO_LAND_PCT) / DRAIN_PCT_PER_MIN)


def _battery(pct: float | None) -> Check:
    if pct is None:
        return Check("drone battery", NO_GO, "no battery figure in the state stream")
    mins = expected_minutes(pct)
    data = {"pct": pct, "expected_min": round(mins, 1)}
    detail = f"{pct:.0f}%: about {mins:.1f} min of flight before the {AUTO_LAND_PCT:.0f}% auto-land (estimate)"
    if pct < DRONE_NO_GO_PCT:
        return Check("drone battery", NO_GO, f"{detail}; under {DRONE_NO_GO_PCT:.0f}%, not enough for the take", data)
    if mins < TAKE_MIN:
        return Check("drone battery", WARN, f"{detail}; a {TAKE_MIN:.0f}-minute take is tight, keep it short", data)
    return Check("drone battery", GO, detail, data)


def _temperature(state: dict) -> Check:
    hi = state.get("temph")
    if hi is None:
        return Check("drone temperature", SKIP, "no temperature in the state stream")
    lo = state.get("templ", hi)
    data = {"templ": lo, "temph": hi}
    detail = f"{lo}-{hi} C"
    if hi >= TEMP_NO_GO_C:
        return Check("drone temperature", NO_GO, f"{detail}: too hot, power off and let it cool (shade, 5 min)", data)
    if hi >= TEMP_WARN_C:
        return Check("drone temperature", WARN, f"{detail}: warm, keep the time on the ground short", data)
    return Check("drone temperature", GO, detail, data)


def _measure(src, tello, first, seconds: float) -> tuple[float, float]:
    """(video fps, telemetry Hz) over one window: distinct frame seqs, and distinct state dicts (djitellopy stores a
    new dict per state packet)."""
    frames = states = 0
    last_seq = first.seq
    last_state = tello.get_current_state() if tello is not None else None
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        f = src.read()
        if f is not None and f.seq != last_seq:
            frames, last_seq = frames + 1, f.seq
        if tello is not None:
            st = tello.get_current_state()
            if st is not last_state:
                states, last_state = states + 1, st
        time.sleep(0.002)
    dt = time.monotonic() - t0
    return frames / dt, states / dt


def _reader_stats(src) -> dict:
    try:
        return src.stats() if hasattr(src, "stats") else {}
    except Exception:
        return {}


def _no_frame_detail(stats: dict, first_frame_s: float) -> str:
    """Why no frame: nothing arrived, or packets arrived and did not decode (the raw reader counts both)."""
    backend = stats.get("backend", "?")
    if stats.get("datagrams") == 0:
        return (f"no frame within {first_frame_s:g} s ({backend} reader): no video packet reached the laptop. Close "
                "the Tello phone app, power-cycle the drone; macOS: Local Network access for the terminal app")
    if stats.get("datagrams"):
        return (f"no frame within {first_frame_s:g} s ({backend} reader): {stats['datagrams']} video packets "
                f"({stats.get('bytes', 0) // 1024} KB) arrived but none decoded (decode errors "
                f"{stats.get('decode_errors')}, skipped {stats.get('skipped_bytes')} bytes waiting for a keyframe). "
                "Capture them for analysis: python -m kite.video_doctor")
    return (f"no frame within {first_frame_s:g} s ({backend} reader): close the Tello phone app, power-cycle the "
            "drone; macOS: Local Network access for the terminal app. Then: python -m kite.video_doctor")


def _reader_counters(stats: dict, seconds: float) -> str:
    """One line of the raw reader's counters, to tell a weak link (few packets) from a decoder problem (packets but
    few frames, errors, resyncs)."""
    if "datagrams" not in stats:
        return ""
    sec = max(seconds, 1e-6)
    keys = ("frames_decoded", "corrupt_frames", "decode_errors", "resyncs", "queue_drops", "skipped_bytes", "keyframe_seen")
    return (f"; reader: {stats['datagrams'] / sec:.0f} packets/s, {stats.get('bytes', 0) / sec / 1024:.0f} KB/s, "
            + ", ".join(f"{k} {stats.get(k)}" for k in keys) + (f", last error {stats['last_error']}" if stats.get("last_error") else ""))


def _video_checks(drone, tello, first_frame_s: float, measure_s: float) -> list[Check]:
    """First frame, fps, frame size and (in the same window) the telemetry rate."""
    try:
        src = drone.frame_source()
    except Exception as e:
        return [Check("video", NO_GO, f"no video reader: {e}")]
    t_wait = time.monotonic()
    first = src.wait_first(first_frame_s)
    if first is None:
        stats = _reader_stats(src)
        why = getattr(src, "diagnosis", lambda: None)()  # the socket reader says which case it is
        detail = _no_frame_detail(stats, first_frame_s) + (f". Reader: {why}" if why else "")
        return [Check("video", NO_GO, detail + "; try another --video-backend", {"reader": stats})]
    wait = time.monotonic() - t_wait
    fps, hz = _measure(src, tello, first, measure_s)
    out = []
    stats = _reader_stats(src)
    data = {"first_frame_s": round(wait, 2), "fps": round(fps, 1), "reader": stats}
    detail = f"first frame after {wait:.1f} s, {fps:.1f} fps over {measure_s:g} s"
    if stats.get("backend"):
        detail += f" ({stats['backend']} reader)"
    since_start = time.monotonic() - t_wait
    if fps < FPS_NO_GO:
        out.append(Check("video", NO_GO, f"{detail}: frozen, the governor would keep hovering"
                         + _reader_counters(stats, since_start), data))
    elif fps < FPS_WARN:
        out.append(Check("video", WARN, f"{detail}: under {FPS_WARN:.0f} fps (weak link? move closer, away from "
                         "other 2.4 GHz Wi-Fi)" + _reader_counters(stats, since_start), data))
    else:
        out.append(Check("video", GO, detail, data))
    if "datagrams" in stats:  # kite's raw reader counts packets; the others don't
        out.append(_link_check(stats))
    size = (first.width, first.height)
    if size == FRAME_SIZE:
        out.append(Check("frame size", GO, f"{size[0]}x{size[1]}"))
    else:
        out.append(Check("frame size", WARN, f"{size[0]}x{size[1]}, expected {FRAME_SIZE[0]}x{FRAME_SIZE[1]}: the "
                         "camera calibration will not match", {"size": list(size)}))
    if tello is None:
        out.append(Check("telemetry", SKIP, "no raw state reader on this drone"))
    elif hz == 0:
        out.append(Check("telemetry", NO_GO, f"no new state packet in {measure_s:g} s: the governor would keep "
                         "hovering (allow incoming connections for Python)", {"hz": 0.0}))
    elif hz < TELEMETRY_WARN_HZ:
        out.append(Check("telemetry", WARN, f"{hz:.1f} Hz, under {TELEMETRY_WARN_HZ:g} Hz", {"hz": round(hz, 1)}))
    else:
        out.append(Check("telemetry", GO, f"{hz:.1f} Hz", {"hz": round(hz, 1)}))
    return out


def _link_check(stats: dict) -> Check:
    """One line on the radio link (packet rate, gaps, loss estimated from the H.264 stream), from the raw reader's
    counters (plain socket counters: the same on Windows, macOS and Linux)."""
    from .sources.tello_raw import link_verdict

    grade, line = link_verdict(stats)
    return Check("video link", GO if grade == "GOOD" else WARN, line, {"grade": grade})


def _latency_check(tello) -> Check:
    if tello is None:
        return Check("command latency", SKIP, "no raw command channel on this drone")
    samples = []
    for _ in range(RTT_SAMPLES):
        samples.append(round_trip_ms(tello))
        time.sleep(RTT_GAP_S)
    got = [s for s in samples if s is not None]
    data = {"samples_ms": [None if s is None else round(s, 1) for s in samples]}
    if not got:
        return Check("command latency", NO_GO, f"no reply to {RTT_SAMPLES} queries: the drone stopped answering", data)
    med = statistics.median(got)
    data["median_ms"] = round(med, 1)
    lost = len(samples) - len(got)
    detail = f"{med:.0f} ms round trip (median of {len(got)})" + (f", {lost} lost" if lost else "")
    return Check("command latency", WARN if med > RTT_WARN_MS or lost else GO, detail, data)


def drone_checks(make_drone=None, first_frame_s: float | None = None, measure_s: float | None = None,
                 video_backend: str | None = None, link_args=None) -> list[Check]:
    """Connect (dry run: no motion), read and measure, close. make_drone: a factory for tests. video_backend: the
    Tello video reader to check (None: DroneCfg's, the one kite.record uses; "auto" = raw on macOS, pyav elsewhere).
    link_args: parsed --video-bitrate / --video-resolution / --video-fps (kite.config.add_video_link_args)."""
    first_frame_s = FIRST_FRAME_S if first_frame_s is None else first_frame_s
    measure_s = MEASURE_S if measure_s is None else measure_s
    if make_drone is None:
        from .config import Config, video_link_args
        from .drone import make_tello

        cfg = Config()
        if video_backend:
            cfg.drone.video_backend = video_backend
        video_link_args(cfg.drone, link_args)

        def make_drone():
            return make_tello(cfg, dry_run=True)  # dry run: takeoff / rc / moves are never sent

    out: list[Check] = []
    t0 = time.monotonic()
    drone = make_drone()
    try:
        try:
            drone.connect()
        except Exception as e:
            out.append(Check("drone link", NO_GO, f"cannot connect: {e}. Power-cycle the Tello, rejoin its Wi-Fi, "
                             "close the phone app"))
            return out
        out.append(Check("drone link", GO, f"connected in {time.monotonic() - t0:.1f} s (dry run: no motion sent)"))
        tello = _raw_tello(drone)
        state = (tello.get_current_state() or {}) if tello is not None else {}
        out.append(_battery(drone.telemetry().battery_pct))
        out.append(_temperature(state))

        out += _video_checks(drone, tello, first_frame_s, measure_s)
        out.append(_latency_check(tello))
    finally:
        try:
            drone.close()  # streamoff + release the ports right away: video drains the battery on the ground
        finally:
            link_s = time.monotonic() - t0
            status = GO if link_s <= LINK_TARGET_S else WARN
            out.append(Check("drone link closed", status, f"link open {link_s:.1f} s (target < {LINK_TARGET_S:g} s); "
                             "power the drone off if the take is not starting within a few minutes",
                             {"link_s": round(link_s, 1)}))
    return out


# ------------------------------------------------------------------ run
def run(out: str = "recordings", quick: bool = False, drone: bool = False, make_drone=None,
        on_check=None, video_backend: str | None = None, link_args=None) -> list[Check]:
    """All checks in order. The drone checks run when the laptop is on the Tello Wi-Fi or drone=True, never with
    quick."""
    checks: list[Check] = []

    def add(c: Check | list[Check]) -> None:
        for one in c if isinstance(c, list) else [c]:
            checks.append(one)
            if on_check is not None:
                on_check(one)

    add(check_deps())
    add(check_ffmpeg())
    add(check_disk(out))
    ports = check_ports()
    add(ports)
    add(check_laptop_battery())
    add(check_awdl())
    wifi = check_wifi(quick=quick)
    add(wifi)
    add(check_camera_json())
    add(check_tty())

    if quick:
        add(Check("drone", SKIP, "--quick: drone not checked"))
    elif not drone and not wifi.data.get("on_tello"):
        add(Check("drone", SKIP, "not on the Tello Wi-Fi (--drone forces the drone checks)"))
    elif ports.status == NO_GO:
        add(Check("drone", SKIP, "the Tello ports are busy (see udp ports)"))
    else:
        if on_check is not None:
            on_check(None)  # "connecting" notice
        add(drone_checks(make_drone, video_backend=video_backend, link_args=link_args))
    return checks


def verdict(checks: list[Check]) -> str:
    return NO_GO if any(c.status == NO_GO for c in checks) else GO


def _line(c: Check) -> str:
    return f"  {c.status:<6} {c.name:<19} {c.detail}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m kite.preflight", description="GO / NO-GO check before the Tello shoot.")
    p.add_argument("--out", default="recordings", help="the recordings folder the take will be written to")
    p.add_argument("--quick", action="store_true", help="laptop checks only, no drone")
    p.add_argument("--drone", action="store_true", help="run the drone checks even when not on the Tello Wi-Fi")
    p.add_argument("--json", action="store_true", help="print machine-readable results")
    p.add_argument("--video-backend", choices=VIDEO_BACKENDS, default=None,
                   help="Tello H.264 reader to check (default pyav; opt-in: raw, socket, auto = raw on macOS)")
    add_video_link_args(p)
    args = p.parse_args(argv)
    if args.quick and args.drone:
        p.error("--quick and --drone contradict each other")

    def show(c: Check | None) -> None:
        if c is None:
            print("  ...    connecting to the Tello (dry run, no motion), about 6 s", flush=True)
        else:
            print(_line(c), flush=True)

    if not args.json:
        print(f"kite preflight {time.strftime('%Y-%m-%d %H:%M:%S')}")
    checks = run(out=args.out, quick=args.quick, drone=args.drone,
                 on_check=None if args.json else show, video_backend=args.video_backend, link_args=args)
    v = verdict(checks)
    nxt = NEXT_COMMAND.format(out=args.out) + (f" --video-backend {args.video_backend}" if args.video_backend else "")
    nxt += video_link_argv(args)
    if args.json:
        print(json.dumps({"verdict": v, "checks": [asdict(c) for c in checks],
                          "next_command": nxt if v == GO else None}, indent=2))
    else:
        bad = [c.name for c in checks if c.status == NO_GO]
        warns = [c.name for c in checks if c.status == WARN]
        print()
        if bad:
            print(f"NO-GO: {', '.join(bad)}. Fix and rerun: python -m kite.preflight")
        else:
            print("GO" + (f" (warnings: {', '.join(warns)})" if warns else "") + ". Next:")
            print(f"  {nxt}")
    return 0 if v == GO else 1


if __name__ == "__main__":
    sys.exit(main())
