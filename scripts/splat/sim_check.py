"""Does the sim draw the splat, on screen and in the detector's frames? Dev check.

    cd sim && npx vite --port 6399 --strictPort &          # any free port
    python sim/relay.py --port 6421 &
    uv run --no-project --with playwright --with ultralytics --with opencv-python-headless \
        --with websockets --with numpy python -I scripts/splat/sim_check.py \
        --url 'http://127.0.0.1:6399/?world=splat&src=/local/moss.splat' --relay 6421 --out data/splat/check

Opens the page in headless Chromium (Metal), waits for the splat, walks the demo route for
--seconds, measures the screen fps, saves screenshots, reads drone and chest frames from the
relay with sim/frames.py SimFrameSource and runs YOLO (weights/yolo11n.pt) on them.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sim"))
from frames import SimFrameSource  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--relay", type=int, default=6421)
ap.add_argument("--out", default=str(ROOT / "data/splat/check"))
ap.add_argument("--seconds", type=float, default=20)
ap.add_argument("--weights", default=str(ROOT / "weights/yolo11n.pt"))
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
url = a.url + ("&" if "?" in a.url else "?") + f"relay=ws://127.0.0.1:{a.relay}/pub&ear="

with sync_playwright() as p:
    b = p.chromium.launch(args=["--use-angle=metal", "--ignore-gpu-blocklist", "--enable-gpu"])
    page = b.new_page(viewport={"width": 1280, "height": 800})
    page.on("console", lambda m: m.text.startswith("[splat]") and print("page:", m.text))
    page.goto(url)
    page.wait_for_function("window.__kiteSplat && window.__kiteSplat.ready", timeout=180_000)
    time.sleep(2)
    page.screenshot(path=str(out / "screen_start.png"))
    srcs = {cam: SimFrameSource(cam, url=f"ws://127.0.0.1:{a.relay}/sub") for cam in ("drone", "chest")}
    page.keyboard.down("ArrowUp")          # walk the route while frames come in
    fps = page.evaluate("""() => new Promise((res) => { let n = 0; const t0 = performance.now();
        const f = () => { n++; if (performance.now() - t0 < 3000) requestAnimationFrame(f); else res(n / 3); };
        requestAnimationFrame(f); })""")
    print(f"screen fps while walking: {fps:.1f}")
    frames: dict[str, list] = {"drone": [], "chest": []}
    t_end = time.time() + a.seconds
    while time.time() < t_end:
        for cam, s in srcs.items():
            ok, img = s.read()
            if ok:
                frames[cam].append(img)
    page.keyboard.up("ArrowUp")
    page.screenshot(path=str(out / "screen_walk.png"))
    b.close()

from ultralytics import YOLO  # noqa: E402

model = YOLO(a.weights)
for cam, imgs in frames.items():
    print(f"{cam}: {len(imgs)} frames in {a.seconds:.0f} s ({len(imgs) / a.seconds:.1f} fps)")
    if not imgs:
        continue
    cv2.imwrite(str(out / f"frame_{cam}.jpg"), imgs[len(imgs) // 2])
    hits: Counter = Counter()
    with_any = 0
    for i, img in enumerate(imgs):
        r = model(img, verbose=False, classes=[0, 2, 5, 7])[0]
        names = [model.names[int(c)] for c in r.boxes.cls]
        hits.update(names)
        with_any += bool(names)
        if names and not (out / f"yolo_{cam}.jpg").exists():
            cv2.imwrite(str(out / f"yolo_{cam}.jpg"), r.plot())
    print(f"  YOLO: {dict(hits)} across {len(imgs)} frames, {with_any} frames with a detection")
