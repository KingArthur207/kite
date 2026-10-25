"""Grok Imagine: the street scenes we can't film, run through Kite's real pipeline.

Some of the moments Kite exists for can't be staged safely or on demand: a cyclist
cutting across a crosswalk, a car turning into the walker's path, someone stepping
out of a doorway. Grok Imagine generates each one as it would look from the walker's
chest camera (a still with grok-imagine-image, then a short clip with
grok-imagine-video), and Kite's own detector and earpiece play it, so a judge at our
table hears Kite react to a street that was never there.

    python scripts/imagine_scenes.py list
    python scripts/imagine_scenes.py generate cyclist          # still + clip into data/imagined/
    python scripts/imagine_scenes.py generate --all
    python scripts/imagine_scenes.py rehearse cyclist          # detector on the clip, no audio: did Kite warn?
    scripts/demo.sh live data/imagined/cyclist.mp4             # hear it: detector -> earpiece -> AirPods

Needs XAI_API_KEY in the repo-root .env. Every generation is written to
data/imagined/manifest.json with its model and prompt, so the set is reproducible.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "imagined"
API = "https://api.x.ai/v1"
IMAGE_MODEL = "grok-imagine-image"
VIDEO_MODEL = "grok-imagine-video-1.5-lite"

CHEST = ("Photo from a phone worn on a pedestrian's chest at 1.2 m height, looking straight ahead, "
         "slight wide-angle lens, realistic, natural colors, San Francisco, overcast morning. ")

# name -> (start frame, motion, what Kite should do)
SCENES = {
    "cyclist": (CHEST + "The walker stands at the curb of a zebra crosswalk. A cyclist is about to ride across "
                "the crosswalk from the left, about 4 meters ahead.",
                "The cyclist rides across the crosswalk from left to right in front of the camera. "
                "The camera moves slowly forward toward the curb.",
                "beeps from the left that speed up; 'Stop.' if the bike crosses the walking corridor"),
    "car_turning": (CHEST + "The walker is on the curb at a crosswalk. A silver sedan on the right is starting "
                    "a right turn into the crosswalk, about 6 meters away.",
                    "The car turns right through the crosswalk, crossing in front of the camera. "
                    "The camera stays still at the curb.",
                    "a car warning from the right before it enters the walker's path"),
    "doorway": (CHEST + "A narrow sidewalk beside shop fronts. A man is stepping out of a shop doorway on the "
                "right, about 3 meters ahead, into the walker's path.",
                "The man walks out of the doorway and steps into the middle of the sidewalk, toward the camera. "
                "The camera walks slowly forward.",
                "person beeps from the right, then 'Stop.' when he is close and in the corridor"),
    "oncoming": (CHEST + "A busy sidewalk. A woman looking at her phone is walking straight toward the camera, "
                 "about 7 meters ahead.",
                 "The woman keeps walking straight at the camera while looking at her phone. "
                 "The camera walks forward toward her.",
                 "beeps from ahead that speed up as she closes in, then 'Stop.'"),
    "scooter": (CHEST + "A sidewalk with an electric rental scooter lying across it, about 3 meters ahead, "
                "partly blocking the path.",
                "The camera walks slowly forward along the sidewalk toward the scooter lying on the ground.",
                "a low obstacle ahead; the chest camera's job, the drone can't see it"),
}


def rel(p: Path) -> str:
    """A path for messages: relative to the repo when it's inside it."""
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def key() -> str:
    k = os.environ.get("XAI_API_KEY")
    if not k:
        env = ROOT / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("XAI_API_KEY="):
                    k = line.split("=", 1)[1].strip()
    if not k:
        sys.exit("XAI_API_KEY is not set (repo-root .env)")
    return k


def call(method: str, path: str, body: dict | None = None, timeout: float = 120) -> dict:
    req = urllib.request.Request(API + path, method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Authorization": f"Bearer {key()}", "Content-Type": "application/json",
                                          "User-Agent": "kite-imagine/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:   # nosec B310: fixed https host
        return json.loads(r.read())


MAX_CLIP = 200 << 20          # a 6 s clip is a few MB; anything this big is not one


class _HttpsOnlyRedirects(urllib.request.HTTPRedirectHandler):
    """urllib follows redirects to any scheme; a clip URL must stay on https all the way."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        from urllib.parse import urlparse
        if urlparse(newurl).scheme != "https":
            raise ValueError(f"refusing a redirect to a non-https URL: {newurl[:80]}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(url: str, max_bytes: int = MAX_CLIP) -> bytes:
    """Fetch a generated clip. Only https, redirects included: the URL comes back from the API,
    so refuse anything else, and stop reading past max_bytes."""
    from urllib.parse import urlparse
    if urlparse(url).scheme != "https":
        raise ValueError(f"refusing to download a non-https URL: {url[:80]}")
    opener = urllib.request.build_opener(_HttpsOnlyRedirects)
    with opener.open(url, timeout=120) as r:   # nosec B310: https checked above and on every redirect
        data = r.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"clip is over {max_bytes >> 20} MB: not a generated scene")
    return data


def manifest() -> dict:
    p = OUT / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_manifest(m: dict) -> None:
    (OUT / "manifest.json").write_text(json.dumps(m, indent=2) + "\n")


def generate(name: str, duration: int, force: bool) -> None:
    still_prompt, motion, expect = SCENES[name]
    OUT.mkdir(parents=True, exist_ok=True)
    jpg, mp4 = OUT / f"{name}.jpg", OUT / f"{name}.mp4"
    m = manifest()
    if force or not jpg.exists():
        print(f"[{name}] still: {IMAGE_MODEL}")
        r = call("POST", "/images/generations", {"model": IMAGE_MODEL, "prompt": still_prompt, "n": 1,
                                                  "response_format": "b64_json"})
        jpg.write_bytes(base64.b64decode(r["data"][0]["b64_json"]))
    if force or not mp4.exists():
        print(f"[{name}] clip: {VIDEO_MODEL}, {duration} s (async)")
        data_uri = "data:image/jpeg;base64," + base64.b64encode(jpg.read_bytes()).decode()
        job = call("POST", "/videos/generations", {"model": VIDEO_MODEL, "prompt": motion,
                                                    "image": {"url": data_uri}, "duration": duration})
        rid = job["request_id"]
        t0 = time.monotonic()
        while True:
            time.sleep(5)
            st = call("GET", f"/videos/{rid}")
            status = st.get("status")
            if status == "done":
                mp4.write_bytes(download(st["video"]["url"]))
                break
            if status in ("failed", "expired") or time.monotonic() - t0 > 600:
                sys.exit(f"[{name}] video {status or 'timed out'}: {json.dumps(st)[:300]}")
            print(f"[{name}]   {status} ({time.monotonic() - t0:.0f} s)")
    m[name] = {"image_model": IMAGE_MODEL, "video_model": VIDEO_MODEL, "duration_s": duration,
               "still_prompt": still_prompt, "motion_prompt": motion, "kite_should": expect,
               "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    save_manifest(m)
    print(f"[{name}] -> {rel(jpg)}, {rel(mp4)}")


DETECT = ["uv", "run", "--no-project", "--with", "ultralytics", "--with", "lap", "--with", "websockets",
          "--with", "python-dotenv", "--with", "numpy", "--with", "scipy", "--with", "opencv-python-headless",
          "python", str(ROOT / "sim" / "standin_detect.py")]


def rehearse(name: str) -> dict:
    """Run Kite's detector over the clip with no audio and summarise what the walker would have heard."""
    mp4, jsonl = OUT / f"{name}.mp4", OUT / f"{name}.rehearsal.jsonl"
    if not mp4.exists():
        sys.exit(f"{rel(mp4)} is missing: run `generate {name}` first")
    subprocess.run(DETECT + ["--source", str(mp4), "--camera", "chest", "--pitch", "0", "--jsonl", str(jsonl),
                             "--no-send", "--fast"], check=True, cwd=ROOT)
    events = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
    obs = [e for e in events if e.get("type") == "obstacles" and e.get("objects")]
    kinds = sorted({o.get("kind") for e in obs for o in e["objects"]})
    near = [(e["t"], o) for e in obs for o in e["objects"] if (o.get("distance") or 99) <= 1.5]
    # then the real earpiece: what would the walker have heard?
    wav = OUT / f"{name}.heard.wav"
    py = ROOT / "earpiece" / ".venv" / "bin" / "python"
    subprocess.run([str(py), "-m", "earpiece", "render", "--jsonl", str(jsonl), str(wav)], check=True,
                   cwd=ROOT / "earpiece", stdout=subprocess.DEVNULL)
    caps = wav.with_suffix(".captions.jsonl")
    said = [json.loads(line) for line in caps.read_text().splitlines() if line.strip()] if caps.exists() else []
    summary = {"scene": name, "frames_with_objects": len(obs), "kinds": kinds,
               "first_seen_s": round(obs[0]["t"] - events[0]["t"], 2) if obs else None,
               "closest_m": round(min((o.get("distance") or 99) for e in obs for o in e["objects"]), 2) if obs else None,
               "would_stop": bool(near) or any(s.get("text", "").startswith("Stop") for s in said),
               "kite_said": [f'{s["t"]:.1f}s {s["text"]}' for s in said],
               "kite_should": SCENES[name][2]}
    (OUT / f"{name}.rehearsal.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    g = sub.add_parser("generate")
    g.add_argument("names", nargs="*")
    g.add_argument("--all", action="store_true")
    g.add_argument("--duration", type=int, default=6, choices=range(1, 16), metavar="1-15",
                   help="clip length in seconds (each second costs credits)")
    g.add_argument("--force", action="store_true")
    r = sub.add_parser("rehearse")
    r.add_argument("names", nargs="*")
    r.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if a.cmd == "list":
        for n, (_, _, expect) in SCENES.items():
            have = "ready" if (OUT / f"{n}.mp4").exists() else "not generated"
            print(f"{n:12s} {have:14s} Kite should: {expect}")
        return
    names = list(SCENES) if a.all else a.names
    unknown = [n for n in names if n not in SCENES]
    if unknown or not names:
        sys.exit(f"pick from: {', '.join(SCENES)}")
    for n in names:
        generate(n, a.duration, a.force) if a.cmd == "generate" else rehearse(n)


if __name__ == "__main__":
    main()
