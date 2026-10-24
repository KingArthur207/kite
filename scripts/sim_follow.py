"""Fly the real follow code (kite.follow through kite.record's Recorder and the SafetyGovernor) against the 3D sim's
drone on the Moss route, and score how well it keeps the walker in view.

    .venv/bin/python scripts/sim_follow.py --speeds 0.5 0.7 1.0          # starts its own relay and vite
    .venv/bin/python scripts/sim_follow.py --speeds 0.7 --headed         # watch it
    .venv/bin/python scripts/sim_follow.py --score recordings/sim_follow/<take>   # re-score a finished run

Needs: playwright (+ `playwright install chromium`), websockets, ultralytics + data/models/yolo11n.pt (or
KITE_YOLO_WEIGHTS), matplotlib. Ports: --vite-port 6699 and --relay-port 6721 by default, so it never talks to a
relay or sim someone else is running; the page's earpiece socket is off.

Each run is a scripted pilot doing what the shoot runbook does: take off (t), climb to the walker's head top + 0.3 m
(r), press g until follow engages, then tell the walker to go. The walker (?walker=route) walks the route at the given
speed: up Moss's NE sidewalk, 3 s at the curb, left across the zebra, along Howard. When it is done the pilot lands.
The recorder writes its usual folder (telemetry.jsonl carries the sim's truth: drone and walker poses) and this script
scores it: standoff distance, lag, walker in view (truth projected into the drone camera), follow lost events,
oscillation, and a top-down trace PNG.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SIM = ROOT / "sim"
sys.path.insert(0, str(ROOT))

WALKER_HEAD_M = 1.78        # the sim walker's model ("man" in sim/src/actors.ts)
ABOVE_M = 0.3               # FollowCfg.above_m: the pilot climbs to head top + this before pressing g
CURB_S = 3.0


# ------------------------------------------------------------------ scripted pilot
class Pilot:
    """keys for Recorder: t, r (climb) and g, then walker go, then end the run when the route is done."""

    def __init__(self, drone, send_go, max_s: float):
        self.drone, self.send_go, self.max_s = drone, send_go, max_s
        self.rec = self.follow = None
        self.phase, self.t_phase, self.t_key = "wait", time.time(), 0.0
        self.t0 = time.time()
        self.log: list[tuple[float, str]] = []

    def _to(self, phase: str) -> None:
        self.phase, self.t_phase = phase, time.time()
        self.log.append((time.time(), phase))
        print(f"  [pilot] {phase}", flush=True)

    def read(self) -> str:
        now = time.time()
        st = self.drone.inner.get_current_state()
        h = (st.get("h") or 0) / 100
        if now - self.t0 > self.max_s and self.rec.end_reason == "":
            self.rec.end_reason = "pilot: run too long"
        if now - self.t_key < 0.4:
            return ""
        key = ""
        if self.phase == "wait" and self.follow.ready and self.rec.last_frame_t and now - self.t_phase > 2:
            self._to("takeoff")
            key = "t"
        elif self.phase == "takeoff" and self.drone.flying and not self.drone.busy():
            self._to("climb")
        elif self.phase == "takeoff" and now - self.t_phase > 15:
            self._to("wait")   # takeoff refused or failed: try again
        elif self.phase == "climb":
            if h < WALKER_HEAD_M + ABOVE_M - 0.08:   # the climb carries on ~0.05 m after the stick drops
                key = "r"
            else:
                key = " "                             # hover
                self._to("settle")
        elif self.phase == "settle" and now - self.t_phase > 2.0:
            self._to("engage")
        elif self.phase == "engage":
            if self.follow.active:
                self._to("walk")
                self.send_go()
            else:
                key = "g"
        elif self.phase == "walk":
            if st.get("walker_state") == "done":
                self._to("done")
            elif not self.follow.active:   # cancelled (safety hover): a pilot presses g again
                key = "g"
        elif self.phase == "done" and now - self.t_phase > 3:
            self.rec.end_reason = "route done"
        if key:
            self.t_key = now
        return key


def fly(speed: float, args) -> Path:
    from kite.config import Config
    from kite.drone import SafetyGovernor
    from kite.drone.sim import SIM_CAMERA, SimDrone
    from kite.follow import FollowCfg, FollowMode, YoloPersonDetector
    from kite.record import Recorder, RecordCfg

    cfg = Config()
    if args.max_rc:
        cfg.safety.max_rc = args.max_rc
    relay = f"ws://127.0.0.1:{args.relay_port}"
    drone = SafetyGovernor(SimDrone(f"{relay}/sub"), cfg.safety)
    drone.connect()
    frames = drone.frame_source()
    if frames.wait_first(15) is None:
        raise SystemExit("no drone frames from the sim")
    fcfg = FollowCfg(video_lag_s=cfg.drone.video_lag_s)
    follow = FollowMode(frames, lambda: YoloPersonDetector(args.weights), fcfg, SIM_CAMERA, max_rc=cfg.safety.max_rc)
    follow.start_tracker()
    pilot = Pilot(drone, lambda: drone.inner.publish({"type": "walker_cmd", "cmd": "go"}), max_s=120 + 110 / speed)
    rcfg = RecordCfg(out_dir=str(args.out), fly=True)
    meta = {"source": "sim", "source_kind": f"sim_follow_{speed:g}", "walk_m_s": speed, "url": args.page_url,
            "follow_cfg": fcfg.__dict__, "safety_cfg": cfg.safety.__dict__}
    rec = Recorder(frames, drone, rcfg, meta, keys=pilot, safety=cfg.safety, follow=follow)
    pilot.rec, pilot.follow = rec, follow
    follow.log = rec._event
    try:
        out = rec.run()
    finally:
        follow.close()
        drone.close()
    (out / "pilot.json").write_text(json.dumps(pilot.log))
    return out


# ------------------------------------------------------------------ scoring
def _jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _reversals(xs: list[int]) -> int:
    signs = [np.sign(x) for x in xs if x != 0]
    return int(sum(a != b for a, b in zip(signs, signs[1:])))


def _seg_dist(px, py, segs: np.ndarray) -> np.ndarray:
    a, b = segs[:, :2], segs[:, 2:]
    ab = b - a
    t = np.clip(((px - a[:, 0]) * ab[:, 0] + (py - a[:, 1]) * ab[:, 1]) / np.maximum((ab ** 2).sum(1), 1e-9), 0, 1)
    return np.hypot(a[:, 0] + t * ab[:, 0] - px, a[:, 1] + t * ab[:, 1] - py)


def _inside(px, py, poly: list) -> bool:
    inside, n = False, len(poly)
    for i in range(n):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
        if (y1 > py) != (y2 > py) and px < x1 + (py - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def score(take: Path, plot: Path | None = None) -> dict:
    from kite.drone.sim import SIM_CAMERA as cam

    meta = json.loads((take / "meta.json").read_text())
    tel = [r for r in _jsonl(take / "telemetry.jsonl") if "walker_x" in r]
    ev = _jsonl(take / "events.jsonl")
    moss = json.loads((SIM / "public" / "osm" / "moss.json").read_text())
    walk = [r for r in tel if r.get("walker_state") in ("walking", "curb")]
    if not walk:
        raise SystemExit(f"{take}: the walker never walked")
    t0, t1 = walk[0]["t_mono"], walk[-1]["t_mono"]
    # truth during the walk
    T = np.array([r["t_mono"] - t0 for r in walk])
    dx = np.array([r["x"] for r in walk]); dy = np.array([r["y"] for r in walk])
    wx = np.array([r["walker_x"] for r in walk]); wy = np.array([r["walker_y"] for r in walk])
    alt = np.array([r["alt"] for r in walk]); hd = np.radians([r["heading"] for r in walk])
    pitch = np.radians([r.get("pitch", 0) for r in walk])
    standoff = np.hypot(wx - dx, wy - dy)
    # walker in view: head top projected into the drone camera (yaw + pitch; roll ignored, occlusion ignored)
    ex, ey = wx - dx, wy - dy
    fwd = ex * np.sin(hd) + ey * np.cos(hd)
    lat = ex * np.cos(hd) - ey * np.sin(hd)
    up = WALKER_HEAD_M - alt
    elev = np.arctan2(up, np.maximum(fwd, 1e-3)) - pitch     # above the optical axis
    col = cam.cx + cam.fx * lat / np.maximum(fwd, 1e-3)
    row = cam.cy - cam.fy * np.tan(elev)
    bearing = np.degrees(np.arctan2(lat, fwd))
    # in view: in front, horizontally inside the frame (box centre), and the head top above the bottom edge
    in_view = (fwd > 0.2) & (col > 0) & (col < cam.width) & (row < cam.height)
    # follow decisions in the walk window
    fol = [e for e in ev if e["event"] == "follow" and t0 <= e["t_mono"] <= t1]
    states = [e["state"] for e in fol]
    lost_events = sum(1 for a, b in zip(["following"] + states, states) if b == "lost" and a != "lost")
    lost_s = sum(1 for s in states if s == "lost") / 10.0          # control ticks are 10 Hz
    cancels = [e for e in ev if e["event"] == "follow off" and t0 <= e["t_mono"] <= t1]
    minutes = max(1e-6, (t1 - t0) / 60)
    rcs = [e["rc"] for e in fol if e["state"] == "following"]
    # buildings
    segs = np.array([[*b["pts"][k], *b["pts"][k + 1]] for b in moss["buildings"] for k in range(len(b["pts"]) - 1)])
    clear = np.array([_seg_dist(x, y, segs).min() for x, y in zip(dx, dy)])
    inside = np.array([any(_inside(x, y, b["pts"]) for b in moss["buildings"]) for x, y in zip(dx[::5], dy[::5])])
    dt = np.diff(T, append=T[-1])
    with open(take / "frames.csv") as fh:
        ft = np.array([float(r["t_mono"]) for r in csv.DictReader(fh)])
    ft = ft[(ft >= t0) & (ft <= t1)]
    gaps = np.diff(ft) if len(ft) > 1 else np.zeros(0)
    res = {
        "take": str(take), "walk_m_s": meta.get("walk_m_s"), "walk_s": round(t1 - t0, 1),
        "standoff_mean_m": round(float(standoff.mean()), 2), "standoff_median_m": round(float(np.median(standoff)), 2),
        "standoff_p95_m": round(float(np.percentile(standoff, 95)), 2),
        "standoff_min_m": round(float(standoff.min()), 2), "standoff_max_m": round(float(standoff.max()), 2),
        "max_lag_s": round(float(standoff.max() / max(0.1, meta.get("walk_m_s") or 1)), 1),
        "in_view_pct": round(100 * float((in_view * dt).sum() / max(1e-6, dt.sum())), 1),
        "out_of_view_longest_s": round(float(_longest(~in_view, T)), 1),
        "lost_events": lost_events, "lost_s": round(lost_s, 1), "follow_cancelled": len(cancels),
        "bearing_rms_deg": round(float(np.sqrt(np.mean(bearing[in_view] ** 2))) if in_view.any() else float("nan"), 1),
        "yaw_reversals_per_min": round(_reversals([r[3] for r in rcs]) / minutes, 1),
        "fb_reversals_per_min": round(_reversals([r[1] for r in rcs]) / minutes, 1),
        "alt_mean_m": round(float(alt.mean()), 2),
        "building_clearance_min_m": round(float(clear.min()), 2),
        "inside_building_s": round(float(inside.sum() * 5 * np.median(np.diff(T))), 1),
        "max_rc": meta.get("safety_cfg", {}).get("max_rc"),
        "frame_gaps_over_0_5s": int((gaps > 0.5).sum()), "frame_gap_max_s": round(float(gaps.max(initial=0)), 2),
        "drone_fps": round(len(ft) / max(1e-6, t1 - t0), 1),
        "safety_hovers": sum(1 for e in ev if "-> hover" in e["event"] and t0 <= e["t_mono"] <= t1),
        "end_reason": meta.get("summary", {}).get("end_reason"),
    }
    if plot is not None:
        _plot(plot, moss, dx, dy, wx, wy, T, in_view, res)
    return res


def _longest(mask: np.ndarray, T: np.ndarray) -> float:
    best, start = 0.0, None
    for i, m in enumerate(mask):
        if m and start is None:
            start = T[i]
        if (not m or i == len(mask) - 1) and start is not None:
            best, start = max(best, T[i] - start), None
    return best


def _plot(path: Path, moss: dict, dx, dy, wx, wy, T, in_view, res: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    route = np.array(moss["route"])
    lo = np.minimum(route.min(0), [min(dx.min(), wx.min()), min(dy.min(), wy.min())]) - 8
    hi = np.maximum(route.max(0), [max(dx.max(), wx.max()), max(dy.max(), wy.max())]) + 8
    fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=130)
    for b in moss["buildings"]:
        ax.add_patch(Polygon(b["pts"], closed=True, fc="#e4e1da", ec="#b9b4aa", lw=0.6))
    for c in moss.get("curbs", []):
        ax.plot([c[0], c[2]], [c[1], c[3]], color="#cfcac0", lw=0.6)
    ax.plot(route[:, 0], route[:, 1], "--", color="#9a9a9a", lw=1, label="route")
    ax.plot(wx, wy, color="#2b6cb0", lw=2, label="walker")
    ax.plot(dx, dy, color="#dd6b20", lw=1.4, label="drone")
    out = ~in_view
    ax.scatter(dx[out], dy[out], s=6, color="#c53030", zorder=5, label="walker out of view")
    for t in range(0, int(T[-1]) + 1, 5):   # who was where every 5 s
        i = int(np.searchsorted(T, t))
        if i < len(T):
            ax.plot([dx[i], wx[i]], [dy[i], wy[i]], color="#555", lw=0.5)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
    ax.set_xlabel("m east of the Moss zebra"); ax.set_ylabel("m north")
    ax.set_title(f"Follow on Moss St, walker {res['walk_m_s']:g} m/s: standoff median {res['standoff_median_m']} m, "
                 f"max {res['standoff_max_m']} m\nwalker in view {res['in_view_pct']}%, follow lost "
                 f"{res['lost_events']}x ({res['lost_s']} s)  (lines join drone and walker every 5 s)", fontsize=9)
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------ servers
def wait_http(url: str, timeout: float) -> None:
    end = time.time() + timeout
    while time.time() < end:
        try:
            urllib.request.urlopen(url, timeout=1)
            return
        except Exception:
            time.sleep(0.3)
    raise SystemExit(f"not reachable: {url}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--speeds", type=float, nargs="+", default=[0.5, 0.7, 1.0])
    ap.add_argument("--vite-port", type=int, default=6699)
    ap.add_argument("--relay-port", type=int, default=6721)
    ap.add_argument("--url", help="a running sim (default: start vite on --vite-port)")
    ap.add_argument("--world", default="moss")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, default=ROOT / "recordings" / "sim_follow")
    ap.add_argument("--docs", type=Path, default=SIM / "docs", help="where the trace PNGs go")
    ap.add_argument("--tag", default="", help="suffix for the PNG/JSON names (e.g. the controller)")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--headed", action="store_true", help="show the page (and render its screen view)")
    ap.add_argument("--max-rc", type=int, default=0, help="override SafetyCfg.max_rc (default: the config's, 50)")
    ap.add_argument("--score", type=Path, nargs="*", help="only score these finished takes")
    args = ap.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""

    if args.score:
        for take in args.score:
            print(json.dumps(score(take, take / "trace.png"), indent=1))
        return 0

    from playwright.sync_api import sync_playwright

    procs = [subprocess.Popen([sys.executable, str(SIM / "relay.py"), "--port", str(args.relay_port)])]
    url = args.url
    if not url:
        url = f"http://127.0.0.1:{args.vite_port}"
        procs.append(subprocess.Popen(["npx", "vite", "--port", str(args.vite_port), "--strictPort", "--host",
                                       "127.0.0.1"], cwd=SIM, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    results = []
    try:
        wait_http(url, 30)
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=not args.headed, args=[
                "--use-angle=metal", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
            for v in args.speeds:
                page = browser.new_page(viewport={"width": 960, "height": 720})
                page.on("pageerror", lambda e: print(f"[page error] {e}", file=sys.stderr))
                args.page_url = (f"{url}/?world={args.world}&drone=sim&walker=route&walk={v:g}&curb={CURB_S:g}"
                                 f"&seed={args.seed}&relay=ws://127.0.0.1:{args.relay_port}/pub&ear="
                                 + ("" if args.headed else "&screen=0&chest=0"))
                page.goto(args.page_url)
                print(f"== walker {v:g} m/s: {args.page_url}", flush=True)
                take = fly(v, args)
                page.close()
                res = score(take, args.docs / f"follow_trace_{v:g}{suffix}.png")
                results.append(res)
                print(json.dumps(res, indent=1), flush=True)
            browser.close()
    finally:
        for pr in procs:
            pr.terminate()
    (args.docs / f"follow_eval{suffix}.json").write_text(json.dumps(results, indent=1) + "\n")
    print(table(results))
    return 0


def table(results: list[dict]) -> str:
    cols = [("walk_m_s", "walker m/s"), ("standoff_median_m", "standoff median m"), ("standoff_p95_m", "p95 m"),
            ("standoff_min_m", "min m"), ("standoff_max_m", "max m (lag)"), ("in_view_pct", "walker in view %"),
            ("out_of_view_longest_s", "longest out of view s"), ("lost_events", "follow lost"),
            ("lost_s", "lost s"), ("follow_cancelled", "cancelled"), ("yaw_reversals_per_min", "yaw reversals/min"),
            ("bearing_rms_deg", "bearing rms deg"), ("building_clearance_min_m", "closest wall m")]
    lines = ["| " + " | ".join(c[1] for c in cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[k]) for k, _ in cols) + " |" for r in results]
    return "\n".join(lines)


if __name__ == "__main__":
    os.environ.setdefault("YOLO_OFFLINE", "1")
    sys.exit(main())
