"""Record a cinematic sim shot (sim/src/shots.ts) to MP4 for the demo video.

    uv run --no-project --with playwright python scripts/record_shot.py corner
    uv run --no-project --with playwright python scripts/record_shot.py follow --world moss
    uv run --no-project --with playwright python scripts/record_shot.py all

Opens the sim at ?shot=<name> in headless Chromium (Metal GPU), steps it frame by frame through
window.kiteShot.frame(1/fps) (the page's clock is this script's fixed step, never the wall clock), writes
each frame as a PNG to a temp dir, encodes H.264 (crf 18) with ffmpeg, deletes the frames and saves a
640 px still. Output: out/shots/<shot>-<world>.mp4 and .jpg (gitignored). Shots on the tiles world show
Google Photorealistic 3D Tiles: they are for the demo video only and must never be committed.

Without --url it starts its own vite on --port (default 6999) and stops it afterward. Tiles need
VITE_GOOGLE_MAPS_API_KEY in sim/.env.local (sh sim/tiles_env.sh).
"""

from __future__ import annotations

import argparse
import base64
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
SIM = ROOT / "sim"
SHOTS = ["establishing", "follow", "corner", "explainer"]


def wait_http(url: str, timeout: float) -> None:
    end = time.time() + timeout
    while time.time() < end:
        try:
            urllib.request.urlopen(url, timeout=1)
            return
        except Exception:
            time.sleep(0.3)
    raise SystemExit(f"sim not reachable at {url}")


def record(browser, url: str, name: str, world: str | None, fps: int, out_dir: pathlib.Path,
           max_seconds: float | None, preview: list[float] | None = None) -> dict:
    page = browser.new_page(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: m.type == "error" and errors.append(m.text))
    q = f"?shot={name}" + (f"&world={world}" if world else "")
    page.goto(url + "/" + q)
    page.wait_for_function("window.kiteShot && window.kiteShot.ready()", timeout=300_000, polling=500)
    info = page.evaluate("({ world: kiteShot.world, duration: kiteShot.duration, still: kiteShot.still })")
    duration = min(info["duration"], max_seconds or info["duration"])
    n = round(duration * fps)
    still_i = min(n - 1, round(info["still"] * fps))
    stem = f"{name}-{info['world']}"
    mp4, still = out_dir / f"{stem}.mp4", out_dir / f"{stem}.jpg"
    t0 = time.time()
    if preview:
        # Same stepping, but render only at the preview times (tiles load late, so the look is rougher).
        want = {min(n - 1, round(t * fps)) for t in preview}
        for i in range(max(want) + 1):
            data = page.evaluate("([dt, r]) => kiteShot.frame(dt, r)", [0 if i == 0 else 1 / fps, i in want])
            if data:
                (out_dir / f"{stem}-t{i / fps:05.2f}.png").write_bytes(base64.b64decode(data.split(",", 1)[1]))
        page.close()
        return {"shot": name, "world": info["world"], "preview": sorted(i / fps for i in want), "errors": errors[:10]}
    tmp = pathlib.Path(tempfile.mkdtemp(prefix=f"kite_shot_{name}_"))
    try:
        for i in range(n):
            data = page.evaluate("dt => kiteShot.frame(dt)", 0 if i == 0 else 1 / fps)
            (tmp / f"f{i:05d}.png").write_bytes(base64.b64decode(data.split(",", 1)[1]))
            if i % (fps * 2) == 0:
                el = time.time() - t0
                print(f"  {name}: frame {i}/{n}  ({i / el if el else 0:.1f} fps)", file=sys.stderr)
        stats = page.evaluate("kiteShot.stats()")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(tmp / "f%05d.png"),
                        "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", str(mp4)], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp / f"f{still_i:05d}.png"),
                        "-vf", "scale=640:-2", "-q:v", "3", str(still)], check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        page.close()
    tiles = stats.get("tiles") or {}
    if stats["fenceViolations"] or tiles.get("fenceViolations"):
        errors.append(f"geofence: camera {stats['fenceViolations']}, tile requests {tiles.get('fenceViolations')}")
    return {"shot": name, "world": info["world"], "mp4": str(mp4), "still": str(still),
            "frames": n, "seconds": round(n / fps, 2), "record_s": round(time.time() - t0, 1),
            "tile_requests": tiles.get("requests"), "tile_fence_violations": tiles.get("fenceViolations"),
            "camera_fence_violations": stats["fenceViolations"], "errors": errors[:10]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shot", choices=SHOTS + ["all"])
    ap.add_argument("--world", help="override the shot's world: tiles, moss, real, photo")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--max-seconds", type=float, help="record only the first N seconds (quick checks)")
    ap.add_argument("--preview", help="comma-separated shot seconds: save PNG stills there instead of an MP4")
    ap.add_argument("--url", help="running sim, e.g. http://127.0.0.1:6999 (default: start vite)")
    ap.add_argument("--port", type=int, default=6999)
    ap.add_argument("--out", default=str(ROOT / "out" / "shots"))
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found")
    from playwright.sync_api import sync_playwright

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    vite = None
    url = args.url
    if not url:
        url = f"http://127.0.0.1:{args.port}"
        vite = subprocess.Popen(["npx", "vite", "--port", str(args.port), "--strictPort", "--host", "127.0.0.1"],
                                cwd=SIM, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    failed = False
    try:
        wait_http(url, 30)
        with sync_playwright() as p:
            browser = p.chromium.launch(args=["--use-angle=metal", "--ignore-gpu-blocklist", "--enable-gpu-rasterization"])
            for name in SHOTS if args.shot == "all" else [args.shot]:
                preview = [float(x) for x in args.preview.split(",")] if args.preview else None
                r = record(browser, url, name, args.world, args.fps, out_dir, args.max_seconds, preview)
                failed |= bool(r.get("camera_fence_violations") or r.get("tile_fence_violations"))
                print(json.dumps(r))
            browser.close()
    finally:
        if vite is not None:
            vite.terminate()
    if failed:
        raise SystemExit("geofence violated; do not use these files")


if __name__ == "__main__":
    main()
