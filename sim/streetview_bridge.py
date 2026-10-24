"""Street View walk -> detector frames. The page (sim ?world=streetview) sends its view to the relay; this fetches the
Static API image for that panorama and heading and publishes it as the sim's drone and chest frames.

    python sim/streetview_bridge.py [--relay ws://127.0.0.1:5020]

In:  {"type": "sv_view", "pano", "lat", "lng", "heading"} on relay /sub (the page resends it every 2 s).
Out: on relay /pub, FPS times a second per camera, the latest images (1 byte camera id + JPEG, like the sim) and
     a `pose` per camera with x, y (sim metres of the panorama), heading, cam_height, pitch (+ = up) and fx.
     The stand-in detector reads them unchanged (sim/frames.py).

Each new view costs one metadata request per panorama (free) and two Static images (billed): drone at pitch 0 and
hfov 51.3 (the sim cameras' 2*atan(480/1000)), chest at pitch -10 and hfov 69 (a phone's 1x camera).
kite/sources/streetview.py checks every panorama against SCOPE (Moss St only, docs/google-permission.md) and that
it is Google's. Images live in memory only (the last LRU_VIEWS views, so walking back is free); nothing is written.
The key comes from GOOGLE_MAPS_API_KEY in the repo .env and is never printed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kite.gps import latlng_to_sim  # noqa: E402
from kite.sources.streetview import SV_CAM_HEIGHT_M, StreetViewSource, in_scope  # noqa: E402

SIM_HFOV = math.degrees(2 * math.atan(480 / 1000))  # sim/src/config.ts: FX 1000 at 960 wide
CAMS = {0: ("drone", 0.0, SIM_HFOV), 1: ("chest", -10.0, 69.0)}  # id: (name, pitch + = up, hfov)
FPS = 5  # per camera: the stand-in's 3-frame onset holds in 0.6 s
LRU_VIEWS = 32


def load_key() -> None:
    """GOOGLE_MAPS_API_KEY from the repo .env (in a worktree, the main checkout's), unless already set."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for env in (ROOT / ".env", ROOT.parents[2] / ".env" if ROOT.parent.name == "worktrees" else None):
        if env is not None and env.is_file():
            load_dotenv(env, override=False)


class Bridge:
    def __init__(self):
        self.srcs = {cid: StreetViewSource(single=(37.778, -122.408, 0), pitch=p, hfov_deg=f)
                     for cid, (_, p, f) in CAMS.items()}
        self.srcs[1]._pano_meta = self.srcs[0]._pano_meta  # one metadata request per panorama, not per camera
        self.want: tuple[str, float] | None = None  # (pano, heading) the page shows
        self.have: tuple[str, float] | None = None
        self.views: OrderedDict = OrderedDict()  # (pano, heading) -> [frame bytes, pose json] per camera
        self.changed = asyncio.Event()

    def on_message(self, msg) -> None:
        if isinstance(msg, bytes):
            return
        try:
            d = json.loads(msg)
        except json.JSONDecodeError:
            return
        if not isinstance(d, dict) or d.get("type") != "sv_view" or not isinstance(d.get("pano"), str):
            return
        try:
            lat, lng, heading = float(d["lat"]), float(d["lng"]), float(d["heading"])
        except (KeyError, TypeError, ValueError):
            return
        if not in_scope(lat, lng):  # the page's fence; the panorama's own location is checked again below
            print(f"[sv-bridge] ignoring a view outside Moss St: {lat:.5f},{lng:.5f}", flush=True)
            return
        want = (d["pano"], round(heading) % 360)
        if want != self.want:
            self.want = want
            self.changed.set()

    def fetch(self, pano: str, heading: float) -> list:
        out = []
        for cid, (name, pitch, _) in CAMS.items():
            src = self.srcs[cid]
            jpeg, m = src.view_jpeg(pano, heading)
            x, y = latlng_to_sim(m["location"]["lat"], m["location"]["lng"])
            pose = {"type": "pose", "camera": name, "x": round(x, 2), "y": round(y, 2), "heading": heading,
                    "cam_height": SV_CAM_HEIGHT_M, "pitch": pitch, "fx": round(src.camera.fx, 1),
                    "source": "streetview", "pano": pano}
            out.append([bytes([cid]) + jpeg, json.dumps(pose)])
        return out

    async def fetcher(self) -> None:
        while True:
            await self.changed.wait()
            self.changed.clear()
            want = self.want
            if want is None or want == self.have:
                continue
            if want not in self.views:
                try:
                    self.views[want] = await asyncio.to_thread(self.fetch, *want)
                except (RuntimeError, ValueError) as e:
                    print(f"[sv-bridge] {e}", flush=True)
                    continue
                while len(self.views) > LRU_VIEWS:
                    self.views.popitem(last=False)
                r = self.srcs[0].requests, self.srcs[1].requests
                print(f"[sv-bridge] {want[0]} heading {want[1]}: requests metadata {r[0]['metadata'] + r[1]['metadata']}"
                      f", static {r[0]['image'] + r[1]['image']}", flush=True)
            self.views.move_to_end(want)
            self.have = want

    async def listen(self, url: str) -> None:
        while True:
            try:
                async with websockets.connect(url, max_size=8 * 1024 * 1024) as ws:
                    async for msg in ws:
                        self.on_message(msg)
            except (OSError, websockets.ConnectionClosed):
                await asyncio.sleep(1.0)

    async def publish(self, url: str) -> None:
        while True:
            try:
                async with websockets.connect(url, max_size=8 * 1024 * 1024) as ws:
                    while True:
                        for frame, pose in self.views.get(self.have, []) if self.have else []:
                            await ws.send(pose)
                            await ws.send(frame)
                        await asyncio.sleep(1 / FPS)
            except (OSError, websockets.ConnectionClosed):
                await asyncio.sleep(1.0)


async def run(relay: str) -> None:
    b = Bridge()
    print(f"[sv-bridge] {relay}/sub sv_view -> {relay}/pub drone + chest frames at {FPS} fps", flush=True)
    await asyncio.gather(b.fetcher(), b.listen(f"{relay}/sub"), b.publish(f"{relay}/pub"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--relay", default="ws://127.0.0.1:5020", help="relay base URL (no /pub or /sub)")
    a = ap.parse_args()
    load_key()
    try:
        asyncio.run(run(a.relay.rstrip("/")))
    except RuntimeError as e:  # no key
        sys.exit(str(e))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
