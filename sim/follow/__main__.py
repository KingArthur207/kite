"""Run the rendered follow sim (see sim/follow/__init__.py). Exit code 0 unless the walk ended in a collision."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from kite.camera import Camera
from kite.config import SafetyCfg
from kite.drone import SafetyGovernor
from kite.follow import FollowCfg, FollowController, WalkerTracker, YoloPersonDetector, detector_available

from .scene import DelayedVideo, OracleDetector, Pose, Walker, moss_walk, render, walker_geom
from .sim_tello import SimTello

PHYS_HZ, FRAME_HZ, CONTROL_HZ = 100, 30, 10
COLLISION_M = 0.5  # drone (horizontal) closer than this to the walker's head axis


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    # at max_rc 50 the drone tops out near 0.48 m/s (kite's Tello model: 0.96 m/s at full stick), so a faster walker
    # outpaces it and follow hovers instead of chasing: raise --max-rc with --speed
    p.add_argument("--speed", type=float, default=0.4, help="walking speed, m/s (default 0.4)")
    p.add_argument("--seconds", type=float, default=None, help="walk this long, then land (default: the whole walk)")
    p.add_argument("--max-rc", type=int, default=SafetyCfg().max_rc, help="SafetyCfg.max_rc (default 50)")
    p.add_argument("--lag", type=float, default=0.25, help="video lag, s (DroneCfg.video_lag_s)")
    p.add_argument("--detector", choices=["oracle", "yolo"], default="oracle")
    p.add_argument("--miss", type=float, default=0.05, help="oracle: chance a frame misses the walker")
    p.add_argument("--predict", action="store_true", help="FollowCfg.predict: steer on the walker prediction")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="folder for follow.mp4, ticks.csv, summary.json")
    p.add_argument("--preview", action="store_true", help="live window (q quits)")
    p.add_argument("--realtime", action="store_true", help="pace the sim at wall-clock speed (implied by --preview)")
    a = p.parse_args(argv)

    rng = np.random.default_rng(a.seed)
    cam = Camera()
    walker = Walker(moss_walk(a.speed))
    fcfg = FollowCfg(video_lag_s=a.lag, predict=a.predict)
    hold_h = walker.height_m + fcfg.above_m  # follow altitude: above_m over the head top
    clock = {"t": 0.0}
    video = DelayedVideo(a.lag)
    tello = SimTello(lambda: clock["t"], video, x=walker.x, y=walker.y - fcfg.behind_m, heading_deg=0.0)
    gov = SafetyGovernor(tello, replace(SafetyCfg(), max_rc=a.max_rc), clock=lambda: clock["t"])
    if a.detector == "yolo":
        why = detector_available()
        if why:
            print(f"yolo detector unavailable: {why}")
            return 2
        yolo = YoloPersonDetector()
        detect = lambda f: yolo(np.ascontiguousarray(f.image))  # noqa: E731
    else:
        detect = OracleDetector(rng, miss=a.miss)
    tracker, ctrl = WalkerTracker(fcfg), FollowController(fcfg, cam, max_rc=a.max_rc)

    out = Path(a.out) if a.out else None
    writer = rows = None
    if out:
        out.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out / "follow.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FRAME_HZ,
                                 (cam.width, cam.height))
        rows = []
    walk_end = a.seconds if a.seconds is not None else walker.duration + 3.0  # + takeoff and climb time
    end_t = walk_end + 20.0  # time to land; the loop ends as soon as it is down
    phase, following, last_seq = "takeoff", False, -1
    track = cmd = None
    ranges, errs, states, in_view = [], [], {}, []
    collision_t = None
    gov.connect()
    gov.takeoff()
    n_phys = 0
    wall0 = time.monotonic()
    while clock["t"] < end_t:
        n_phys += 1
        dt = 1.0 / PHYS_HZ
        clock["t"] = t = n_phys * dt
        if following:
            walker.step(dt)
        tello.step(dt)
        m = tello.model
        pose = Pose(m.x, m.y, m.z, math.degrees(m.heading), m.pitch_deg)
        if n_phys % (PHYS_HZ // FRAME_HZ) == 0:  # ~30 fps (33 / 34 ms)
            video.push(render(cam, pose, walker), t, walker_geom(cam, pose, walker) if m.z > 0.05 else None)
        video.advance(t)
        if n_phys % (PHYS_HZ // CONTROL_HZ):
            continue
        # ---- control tick (10 Hz), as kite.record runs follow
        frame = video.read()
        action = gov.check(t, frame.t if frame is not None else None)
        if phase == "takeoff" and not gov.busy() and gov.flying:
            gov.move("up", int(round((hold_h - m.z) * 100)))  # a discrete climb to the follow altitude
            phase = "climb"
        elif phase == "climb" and not gov.busy():
            ctrl.reset(gov.telemetry().height_m)  # 'g': follow from here, holding this altitude
            phase, following = "follow", True
        elif phase == "follow" and t >= walk_end:
            gov.land()
            phase, following = "land", False
        elif phase == "land" and not gov.flying and not gov.busy():
            break
        if following and frame is not None and frame.seq != last_seq:
            last_seq = frame.seq
            track = tracker.update(detect(frame), frame.t, frame.width, frame.height)
        if following:
            cmd = ctrl.update(track, gov.telemetry(), t)
            if action == "ok":
                gov.rc(*cmd.rc)
            states[cmd.state] = states.get(cmd.state, 0) + 1
            rng_true = math.hypot(walker.x - m.x, walker.y - m.y)
            ranges.append(rng_true)
            target = fcfg.behind_m + fcfg.headway_s * walker.v
            errs.append(rng_true - target)
            in_view.append(frame is not None and frame.geom is not None and frame.geom[1] < cam.height)
            if rng_true < COLLISION_M and collision_t is None:
                collision_t = t
            if rows is not None:
                rows.append([f"{t:.1f}", f"{walker.v:.2f}", f"{rng_true:.2f}",
                             "" if cmd.range_m is None else f"{cmd.range_m:.2f}", cmd.state, *cmd.rc,
                             f"{m.z:.2f}", action])
        if writer is not None or a.preview:
            img = _overlay(frame, cmd, track, t, phase, gov) if frame is not None else None
            if img is not None and writer is not None:
                writer.write(img)
            if img is not None and a.preview:
                cv2.imshow("kite follow sim", img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        if a.preview or a.realtime:
            lag = t - (time.monotonic() - wall0)
            if lag > 0:
                time.sleep(lag)
    if writer is not None:
        writer.release()
    if a.preview:
        cv2.destroyAllWindows()
    r = np.array(ranges) if ranges else np.array([math.nan])
    summary = {
        "sim_s": round(clock["t"], 1), "walker_speed_m_s": a.speed, "max_rc": a.max_rc, "detector": a.detector,
        "range_m": {"mean": round(float(np.nanmean(r)), 2), "min": round(float(np.nanmin(r)), 2),
                    "max": round(float(np.nanmax(r)), 2)},
        "range_error_rms_m": round(float(np.sqrt(np.nanmean(np.square(errs)))) if errs else math.nan, 2),
        "walker_in_frame_pct": round(100.0 * float(np.mean(in_view)) if in_view else 0.0, 1),
        "states_pct": {k: round(100.0 * v / max(1, sum(states.values())), 1) for k, v in sorted(states.items())},
        "collision_t": collision_t, "landed": not gov.flying,
        "governor_events": [e for _, e in gov.events][-8:], "drone_log": [c for _, c in tello.log],
    }
    print(json.dumps(summary, indent=1))
    if out:
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        with open(out / "ticks.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["t", "walker_v", "range_true", "range_est", "state", "lr", "fb", "ud", "yaw", "alt", "gov"])
            w.writerows(rows)
        print(f"-> {out}/follow.mp4, ticks.csv, summary.json")
    return 1 if collision_t is not None else 0


def _overlay(frame, cmd, track, t, phase, gov) -> np.ndarray:
    img = frame.image.copy()
    if track is not None:
        x1, y1, x2, y2 = (int(v) for v in track.box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 0), 2)
    tel = gov.telemetry()
    line = f"t {t:5.1f}s  {phase}"
    if cmd is not None:
        line += (f"  {cmd.state}  range {cmd.range_m if cmd.range_m is None else round(cmd.range_m, 2)} m"
                 f"  rc {cmd.rc}")
    line2 = f"alt {tel.height_m:.2f} m  bat {tel.battery_pct:.0f}%  pitch {tel.pitch_deg:+.0f}"
    for i, s in enumerate((line, line2)):
        cv2.putText(img, s, (10, 26 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, s, (10, 26 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


if __name__ == "__main__":
    sys.exit(main())
