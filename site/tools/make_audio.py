"""Render the website's audio from the real earpiece code (run with the earpiece venv).

    earpiece/.venv/bin/python site/tools/make_audio.py

Writes site/assets/audio/*.m4a (AAC via macOS afconvert) and site/assets/walk.json
(waveform peaks + the timeline of what Kite said, for the player).
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "earpiece"))

from earpiece import sounds, spatial, sim                      # noqa: E402
from earpiece.__main__ import write_wav                          # noqa: E402
from earpiece.app import Earpiece                               # noqa: E402
from earpiece.config import Cfg                                 # noqa: E402
from earpiece.events import Obstacle, parse                     # noqa: E402
from earpiece.mixer import Engine                               # noqa: E402
from earpiece.sonar import Sonar                                # noqa: E402

OUT = ROOT / "site" / "public" / "assets" / "audio"
RATE = 48000


def m4a(name: str, stereo: np.ndarray) -> None:
    wav = OUT / f"{name}.wav"
    write_wav(str(wav), stereo, RATE)
    subprocess.run(["afconvert", "-f", "m4af", "-d", "aac", "-b", "128000", str(wav), str(OUT / f"{name}.m4a")],
                   check=True)
    wav.unlink()


def pad(x: np.ndarray, s: float = 0.15) -> np.ndarray:
    z = np.zeros((int(s * RATE), 2), np.float32)
    return np.concatenate([z, x, z])


def cues() -> None:
    k = sounds.proximity_cue("person", 900)
    for name, az in (("dir-left", -70), ("dir-ahead", 0), ("dir-right", 70)):
        x = np.concatenate([spatial.render(k, az, RATE), np.zeros((int(0.25 * RATE), 2), np.float32)] * 3)
        m4a(name, pad(x * 0.9))
    for kind, az in (("person", -25), ("vehicle", 40), ("pole", 0), ("curb", -10)):
        x = np.concatenate([spatial.render(sounds.proximity_cue(kind, 900), az, RATE),
                            np.zeros((int(0.22 * RATE), 2), np.float32)] * 3)
        m4a(f"kind-{kind}", pad(x * 0.9))
    for name, fn, az in (("beacon-on", sounds.beacon_on_course, 0),
                         ("beacon-side", lambda: sounds.beacon_off_course(-60), -60),
                         ("beacon-behind", sounds.beacon_behind, 170)):
        x = np.concatenate([spatial.render(fn(), az, RATE), np.zeros((int(0.5 * RATE), 2), np.float32)] * 3)
        m4a(name, pad(x * 0.9))
    # a person walking straight at you: the real sonar, start to "critical"
    e = Engine(RATE, master_gain=0.95)
    s = Sonar(e, Cfg().sonar)
    out, t, fr = [], 0.0, 0
    while t < 4.6:
        if t * 15 >= fr:
            fr += 1
            s.update_frame("cam", [Obstacle("p", "person", -18 + 4 * t, max(0.45, 5.2 - 1.15 * t), None, 0.9)],
                           seq=fr, now=t)
        s.tick(now=t)
        out.append(e.render(1024))
        t += 1024 / RATE
    m4a("approach", pad(np.concatenate(out)))


def site_walk():
    """The demo walk, ordered for a listener: route first, then Moss's remembered curb,
    then a pedestrian, a pole, the crosswalk, the turn."""
    for t, m in sim.timeline("walk"):
        if m["type"] == "position":
            continue                    # re-timed below
        if m["type"] == "obstacles" and t < 6.0:
            continue                    # give the first instruction room to finish
        if m["type"] == "beacon" and t > 32.0:
            continue
        yield t, m
    for i in range(18):                 # keep the obstacle feed alive to the end (no "detection paused")
        yield 40.0 + i / 15, {"type": "obstacles", "source": "webcam", "seq": 100000 + i, "objects": []}
    yield 33.0, {"type": "position", "place": "Lyon Street and Bay Street", "lat": 37.8013, "lng": -122.4477}


def walk() -> dict:
    cfg = Cfg()
    app = Earpiece(cfg, output="null", record=True, network=False, log=lambda *_: None)
    app.start(prewarm=True, announce=False)
    # real use: Kite boots (Moss index load, phrase cache warm-up) before anyone walks
    tw = time.monotonic()
    while app.memory.status not in ("moss", "local") and time.monotonic() - tw < 30:
        time.sleep(0.2)
    time.sleep(6.0)
    app.output.frames.clear()
    starts: dict[str, float] = {}
    orig = app.speaker._speak

    async def traced(u):
        starts.setdefault(u.text + u.uid, time.time())
        return await orig(u)

    app.speaker._speak = traced
    t0 = time.monotonic()
    w0 = time.time()
    stop = threading.Event()

    def feed():
        for t, m in sorted(site_walk(), key=lambda x: x[0]):
            d = t0 + t - time.monotonic()
            if d > 0 and stop.wait(d):
                return
            app.handle(parse(m))

    threading.Thread(target=feed, daemon=True).start()
    dur = 41.0                                   # the scripted walk plus the arrival at Bay Street
    while time.monotonic() - t0 < dur:
        time.sleep(0.1)
    stop.set()
    app.speaker.wait_idle(6)
    audio = app.output.recorded()[: int(dur * RATE)]
    said = []
    begun = sorted(starts.items(), key=lambda kv: kv[1])
    for h in app.speaker.history:                          # history stamps the END of each line
        if h["result"] != "done" or h["t"] - w0 >= dur:
            continue
        st = max((v for k, v in begun if k.startswith(h["text"]) and v <= h["t"]), default=h["t"] - 1.5)
        said.append({"t": round(st - w0, 2), "end": round(h["t"] - w0, 2),
                     "text": h["text"], "priority": h["priority"]})
    app.stop()
    m4a("walk", audio)
    env = np.abs(audio).max(axis=1)
    n = 480
    peaks = [round(float(env[i * len(env) // n:(i + 1) * len(env) // n].max()), 3) for i in range(n)]
    return {"duration": dur, "peaks": peaks, "said": said,
            "beeps": app.sonar.stats["beeps"], "criticals": app.sonar.stats["criticals"]}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    cues()
    info = walk()
    (ROOT / "site" / "public" / "assets" / "walk.json").write_text(json.dumps(info))
    print(json.dumps({k: v for k, v in info.items() if k != "peaks"}, indent=1))
