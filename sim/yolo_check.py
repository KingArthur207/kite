"""Does YOLO see the sim's glTF cars and people? Dev check, not part of the sim.

    cd sim && npx vite --port 5199 --strictPort &
    uv run --no-project --with playwright --with ultralytics --with opencv-python-headless \
        python sim/yolo_check.py --url http://localhost:5199 --out sim/docs

Opens the sim in headless Chromium with its clock frozen, mocks the relay and earpiece sockets
(so it never touches a shared relay), walks the walker in steps and grabs drone + chest frames.
On Moss it walks up Moss's NE sidewalk to Howard and watches the zebra; `?world=placeholder`
in --url keeps the old walk north. The default world is Google 3D tiles when sim/.env.local has a
key, and the capture frames show tiles; `?detector=osm` keeps them on OSM Moss (tiles on screen
only) and `?world=moss` is OSM everywhere. It waits for tiles to load at every stop, and fails if
the sim requested a tile outside the Moss geofence or (with ?detector=osm) drew a tile into a
capture frame. Drone frames go through the Tello camera model (tello_cam.py, as sim/frames.py does) unless
the URL has ?cam=raw or --cam raw (--cam both scores the same frames off, then on); only its per-frame look applies here, no latency or dropped frames (frames are
grabbed with the walker stopped, so there is no motion blur either). --frames DIR saves the clean drone frames for
scripts/sim_real_gap.py. Each frame comes with the actors' boxes at render time (window.__kiteActors), projected
with the camera model from config.ts. Prints per-camera, per-class recall of visible instances.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from playwright.sync_api import sync_playwright
from ultralytics import YOLO

from tello_cam import TelloCam, load_model

FX, W, H = 1000.0, 960, 720
HEAD_TOP, DRONE_ABOVE, DRONE_BEHIND, CHEST_H, CHEST_OUT = 1.7, 0.3, 1.0, 1.2, 0.3
COCO = {0: "person", 2: "car", 5: "car", 7: "car"}  # bus and truck count as car
MIN_BOX_H = 30          # px: smaller instances are too far to expect a detection
IOU_MATCH = 0.3

# Tag every frame with the actors' boxes at render time: JSON appended after the JPEG bytes
# (decoders ignore trailing data). main.ts calls toBlob right after the capture render.
MARK = b"KITEGT"
INIT = """
(() => {
  const toBlob = HTMLCanvasElement.prototype.toBlob;
  HTMLCanvasElement.prototype.toBlob = function (cb, ...rest) {
    const gt = JSON.stringify(window.__kiteActors ? window.__kiteActors() : []);
    return toBlob.call(this, (b) => cb(b && new Blob([b, "KITEGT" + gt], { type: b.type })), ...rest);
  };
})();
"""


def cam_pose(pose: dict, cam: int):
    h = math.radians(pose["heading"])
    f = np.array([math.sin(h), 0, -math.cos(h)])
    w = np.array([pose["x"], 0, -pose["y"]])
    if cam == 0:
        p = w - f * DRONE_BEHIND
        p[1] = HEAD_TOP + DRONE_ABOVE
    else:
        p = w + f * CHEST_OUT
        p[1] = CHEST_H
    return p, f, np.array([math.cos(h), 0, math.sin(h)])


def project(a: dict, pose: dict, cam: int):
    p, f, r = cam_pose(pose, cam)
    lo, hi = a["min"], a["max"]
    us, vs = [], []
    # Moss-world cars send their oriented box corners; an axis-aligned box of a car at 45 degrees is far too big.
    corners = a.get("corners") or [(cx, cy, cz) for cx in (lo[0], hi[0]) for cy in (lo[1], hi[1]) for cz in (lo[2], hi[2])]
    for c in corners:
        d = np.array(c) - p
        z = d @ f
        if z < 0.3:
            return None
        us.append(W / 2 + FX * (d @ r) / z)
        vs.append(H / 2 - FX * d[1] / z)
    box = [min(us), min(vs), max(us), max(vs)]
    clipped = [max(box[0], 0), max(box[1], 0), min(box[2], W), min(box[3], H)]
    area = lambda b: max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    if area(box) == 0 or area(clipped) / area(box) < 0.6 or clipped[3] - clipped[1] < MIN_BOX_H:
        return None
    dist = float(np.linalg.norm((np.array(lo) + np.array(hi)) / 2 - p))
    return clipped, dist


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def cover(a, b):  # fraction of a covered by b
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / ((a[2] - a[0]) * (a[3] - a[1]) + 1e-9)


OCCLUDED = defaultdict(int)  # kind -> in frame but behind tile geometry (tiles in capture only)


def visible(actors, pose, cam):
    gts = []
    for a in actors:
        if not (pr := project(a, pose, cam)):
            continue
        if a.get("occ") and a["occ"][cam]:
            OCCLUDED[a["kind"]] += 1
            continue
        gts.append((a["kind"], *pr, a.get("name", "")))
    gts.sort(key=lambda g: g[2])
    keep = []
    for g in gts:
        if not any(cover(g[1], kg[1]) > 0.6 for kg in keep):
            keep.append(g)
    return keep


def grab(url: str, per_cam: int):
    frames = []  # (cam, jpeg, actors, pose)
    state = {"pose": None}

    def on_relay(ws):
        def msg(m):
            if isinstance(m, str):
                d = json.loads(m)
                if d.get("type") == "pose" and d.get("camera") == "drone":
                    state["pose"] = d
            elif state["pose"] and state.get("still"):
                i = m.rfind(MARK)
                frames.append((m[0], bytes(m[1:i]), json.loads(m[i + len(MARK):]), dict(state["pose"])))
        ws.on_message(msg)

    with sync_playwright() as pw:
        b = pw.chromium.launch(args=["--use-angle=metal", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"])
        page = b.new_page(viewport={"width": 1280, "height": 800})
        page.on("console", lambda m: m.type in ("error", "warning") and print("[page]", m.text))
        page.add_init_script(INIT)
        page.route_web_socket("ws://127.0.0.1:5020/pub", on_relay)
        page.route_web_socket("ws://127.0.0.1:5011/", lambda ws: None)
        page.route_web_socket("ws://127.0.0.1:5011", lambda ws: None)
        page.clock.install()
        page.goto(url)
        page.wait_for_function("window.__kiteActors && window.__kiteActors().length >= 12", timeout=30000)
        page.clock.run_for(3000)

        def settle(max_s: float = 15):
            # Tiles world: tile fetches run in real time, so tick the frozen clock until they land.
            if not page.evaluate("!!window.__kiteTiles"):
                return
            for _ in range(int(max_s / 0.2)):
                page.clock.run_for(100)
                page.wait_for_timeout(100)
                if page.evaluate("window.__kiteTiles.tilesStats().progress >= 1"):
                    return

        settle(40)
        # Stops along the sidewalk; at each, look ahead and then turned toward the road.
        steps = [(2.5, 0), (0, 30), (0, -30), (3, 0), (0, 40), (0, -40), (4, 0), (0, 0), (5, 0), (0, 0), (5, 0), (0, 0), (8, 0)]
        if "world=placeholder" not in url:
            # Moss: the road is on the left. Walk up Moss to the Howard curb, turn left to face
            # the zebra and watch cars turn in from Howard.
            steps = [(a, -t) for a, t in steps] + [(34, 0), (0, -90)] + [(0, 0)] * 16
        for walk_s, turn_deg in steps:
            state["still"] = False
            if walk_s:
                page.keyboard.down("ArrowUp"); page.clock.run_for(int(walk_s * 1000)); page.keyboard.up("ArrowUp")
            if turn_deg:
                key = "ArrowRight" if turn_deg > 0 else "ArrowLeft"
                page.keyboard.down(key); page.clock.run_for(int(abs(turn_deg) / 90 * 1000)); page.keyboard.up(key)
            settle()
            page.clock.run_for(450)                     # let a pose message for the stopped walker go out
            page.wait_for_timeout(300)
            state["still"] = True
            for _ in range(per_cam):
                page.clock.run_for(130)
                page.wait_for_timeout(120)              # toBlob is real-time async
            page.wait_for_timeout(300)
        if page.evaluate("!!window.__kiteTiles"):
            st = page.evaluate("window.__kiteTiles.tilesStats()")
            print(f"tiles: {st['requests']} requests, fence violations {st['fenceViolations']}, capture leaks {st['captureLeaks']}")
            assert st["fenceViolations"] == 0, "requested a Google tile outside the Moss geofence"
            assert st["captureLeaks"] == 0, "?detector=osm drew a Google tile into a capture frame"
        b.close()
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:5199")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--per-stop", type=int, default=4)
    ap.add_argument("--out", default="sim/docs")
    ap.add_argument("--dump", default="", help="also write every raw frame here")
    ap.add_argument("--cam", choices=["tello", "raw", "both"], default=None,
                    help="Tello camera model on the drone frames (default: tello, raw if the URL has cam=raw); "
                         "both scores the same grab twice, model off then on")
    ap.add_argument("--frames", default="", help="write the drone frames the detector saw here (no boxes drawn)")
    args = ap.parse_args()
    cam_mode = args.cam or ("raw" if "cam=raw" in args.url else "tello")
    frames = grab(args.url, args.per_stop)
    yolo = YOLO(args.model)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"{len(frames)} frames, model {args.model}, visible = box >= {MIN_BOX_H}px tall, match IoU >= {IOU_MATCH}")
    print("  frames per camera:", {CAMS[c]: sum(1 for f in frames if f[0] == c) for c in CAMS})
    if OCCLUDED:
        print("  skipped, behind 3D tiles:", dict(OCCLUDED))
    for mode in (["raw", "tello"] if cam_mode == "both" else [cam_mode]):
        evaluate(frames, yolo, mode, args, out, mode if cam_mode == "both" else "")


CAMS = {0: "drone", 1: "chest"}


def evaluate(frames, yolo, mode: str, args, out: Path, sub: str) -> None:
    model_cfg = load_model()
    tello = {c: TelloCam(model_cfg, seed=c) for c, name in CAMS.items() if mode == "tello" and name in model_cfg["cameras"]}
    stats = defaultdict(lambda: [0, 0])   # (cam, cls) -> [hit, total]
    by_dist = defaultdict(lambda: [0, 0])
    by_model = defaultdict(lambda: [0, 0])
    confs = defaultdict(list)            # (cam, cls) -> confidence of each matched detection
    best = {}
    for n, (cam, jpeg, actors, pose) in enumerate(frames):
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if cam in tello:
            img = tello[cam].process(img, t=n / 8)
        if args.frames and cam == 0:
            Path(args.frames, sub).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(Path(args.frames, sub, f"{n:04d}.png")), img)
        res = yolo(img, verbose=False, conf=0.25)[0]
        dets = [(COCO[int(c)], b.tolist(), float(s)) for b, c, s in zip(res.boxes.xyxy, res.boxes.cls, res.boxes.conf) if int(c) in COCO]
        gts = visible(actors, pose, cam)
        hits = 0
        for k, box, dist, name in gts:
            match = [s for dk, db, s in dets if dk == k and iou(box, db) >= IOU_MATCH]
            ok = bool(match)
            if match:
                confs[(cam, k)].append(max(match))
            hits += ok
            stats[(cam, k)][0] += ok; stats[(cam, k)][1] += 1
            band = "<10m" if dist < 10 else "10-20m" if dist < 20 else ">20m"
            by_dist[(cam, k, band)][0] += ok; by_dist[(cam, k, band)][1] += 1
            by_model[name][0] += ok; by_model[name][1] += 1
            # ground truth: thin white if found, red if missed
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 255) if ok else (0, 0, 255), 1)
        for k, db, s in dets:
            col = (0, 200, 255) if k == "car" else (60, 255, 60)
            cv2.rectangle(img, (int(db[0]), int(db[1])), (int(db[2]), int(db[3])), col, 2)
            cv2.putText(img, f"{k} {s:.2f}", (int(db[0]), int(db[1]) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        if args.dump:
            Path(args.dump, sub).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(Path(args.dump, sub, f"{n:03d}_{cam}_{hits}of{len(gts)}.jpg")), img)
        score = (len({g[0] for g in gts}), hits)
        if score > best.get(cam, ((0, 0), None))[0]:
            best[cam] = (score, img)

    print(f"\ncamera model on drone frames: {mode}")
    for (cam, k), (h, t) in sorted(stats.items()):
        c = confs[(cam, k)]
        print(f"  {CAMS[cam]:5s} {k:6s} {h}/{t} = {h / max(t, 1):.0%}" + (f"   mean conf {np.mean(c):.2f}" if c else ""))
    for (cam, k, band), (h, t) in sorted(by_dist.items()):
        print(f"  {CAMS[cam]:5s} {k:6s} {band:6s} {h}/{t} = {h / max(t, 1):.0%}")
    print("  by model:", ", ".join(f"{m} {h}/{t}" for m, (h, t) in sorted(by_model.items())))
    for cam, (_, img) in best.items():
        name = f"yolo_{CAMS[cam]}{'_' + sub if sub else ''}.jpg"
        cv2.imwrite(str(out / name), cv2.resize(img, (640, 480)), [cv2.IMWRITE_JPEG_QUALITY, 80])


if __name__ == "__main__":
    main()
