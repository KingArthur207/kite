"""Headless sim metrics: open the sim with ?metrics=1 and print the results JSON.

    uv run --with playwright python scripts/sim_metrics.py --mode blind --trials 20
    uv run --with playwright python scripts/sim_metrics.py --mode oracle --url http://localhost:5173

Without --url it starts its own vite on --port (default 5287) and stops it afterward.
First time only: `uv run --with playwright playwright install chromium`.
Kite mode runs in real time (~50 s a walk) and needs the earpiece and the detector running.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
import urllib.request

SIM = pathlib.Path(__file__).resolve().parent.parent / "sim"


def wait_http(url: str, timeout: float) -> None:
    end = time.time() + timeout
    while time.time() < end:
        try:
            urllib.request.urlopen(url, timeout=1)
            return
        except Exception:
            time.sleep(0.3)
    raise SystemExit(f"sim not reachable at {url}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["blind", "oracle", "kite"], default="blind")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--url", help="running sim, e.g. http://localhost:5173 (default: start vite)")
    ap.add_argument("--port", type=int, default=5287)
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright

    vite = None
    url = args.url
    if not url:
        url = f"http://127.0.0.1:{args.port}"
        vite = subprocess.Popen(["npx", "vite", "--port", str(args.port), "--strictPort", "--host", "127.0.0.1"],
                                cwd=SIM, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_http(url, 30)
        timeout_s = args.trials * 200 if args.mode == "kite" else 300
        with sync_playwright() as p:
            browser = p.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
            page = browser.new_page()
            page.on("pageerror", lambda e: print(f"[page error] {e}", file=sys.stderr))
            page.goto(f"{url}/?metrics=1&mode={args.mode}&trials={args.trials}&seed={args.seed}")
            page.wait_for_function("window.kiteMetrics && window.kiteMetrics.done", timeout=timeout_s * 1000)
            print(json.dumps(page.evaluate("window.kiteMetrics")))
            browser.close()
    finally:
        if vite is not None:
            vite.terminate()


if __name__ == "__main__":
    main()
