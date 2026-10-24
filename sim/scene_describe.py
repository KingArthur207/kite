"""Claude scene descriptions for the earpiece: one camera in, `scene` events out.

    uv run --with anthropic --with opencv-python-headless --with websockets --with python-dotenv \
        --with numpy python sim/scene_describe.py --source sim-drone

Every --interval seconds (and right after the wearer asks "what's in front of me?") one downscaled
JPEG goes to Claude, and the 1-2 sentence answer goes to the earpiece as {"type":"scene",...} with
source "scene-chest" or "scene-drone". The earpiece doesn't speak scenes; its brain uses the latest
one (under 90 s old) to answer questions. Frames stay in memory and are never written to disk.

--source: sim-chest | sim-drone | <camera index> | <video file> | <kite.record folder>
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "earpiece"), str(ROOT / "sim")]

from earpiece.client import EarpieceClient  # noqa: E402

MODEL = "claude-haiku-5-5"  # fast vision model; --model to override
WIDTH, QUALITY = 768, 70
TIMEOUT_S = 4.0
# Same as earpiece/brain.py _AROUND (copied: importing the brain pulls in scipy via events.py).
_AROUND = re.compile(r"\b(?:what'?s|what is|anything|anyone|who'?s|who is)\b.*\b(?:around|ahead|in front|near|nearby|close)\b")

SYSTEM_PROMPT = """You describe a camera view for a blind pedestrian. The image comes from a camera on \
their chest or from a small drone just above and behind them, looking the way they walk.

In one or two short spoken sentences (under 35 words), say what is ahead: the path or sidewalk, people, \
vehicles, obstacles, curbs, steps, crossings, and any sign text you can actually read. Use "ahead", \
"left", "right" and rough distances in meters.

Rules:
- Never make a safety judgment. Never say it is safe or clear to cross or to walk; just describe what is there.
- Only describe what is visible in this image. Don't guess about anything out of frame or unclear.
- Don't describe faces, and don't guess anyone's identity, age, gender or race. "A person" is enough.
- No lists, no markdown."""


def jpeg_b64(img: np.ndarray, width: int = WIDTH, quality: int = QUALITY) -> str:
    """Downscale and JPEG-encode in memory (never on disk)."""
    h, w = img.shape[:2]
    if w > width:
        img = cv2.resize(img, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encode failed")
    return base64.standard_b64encode(buf.tobytes()).decode()


def describe(client, img: np.ndarray, model: str = MODEL, timeout: float = TIMEOUT_S) -> str:
    r = client.with_options(timeout=timeout, max_retries=0).messages.create(
        model=model,
        max_tokens=1024,
        output_config={"effort": "low"},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": jpeg_b64(img)}},
            {"type": "text", "text": "What is ahead of me?"},
        ]}],
    )
    if r.stop_reason == "refusal":
        return ""
    return " ".join(b.text.strip() for b in r.content if b.type == "text").strip()


class SceneDescriber:
    """At most one Claude call in flight; a new one starts when `interval` has passed or on force."""

    def __init__(self, client, ear: EarpieceClient, model: str = MODEL, interval: float = 6.0,
                 timeout: float = TIMEOUT_S, log=print):
        self.client, self.ear, self.model = client, ear, model
        self.interval, self.timeout, self.log = interval, timeout, log
        self._busy = threading.Lock()
        self._last = -1e9

    def tick(self, img: np.ndarray | None, now: float, force: bool = False, wait: bool = False) -> bool:
        """Start a description if one is due. Returns False if skipped (no frame, busy, too soon)."""
        if img is None or (not force and now - self._last < self.interval):
            return False
        if not self._busy.acquire(blocking=False):
            return False                      # previous call still running: skip, don't queue
        self._last = now
        t = threading.Thread(target=self._run, args=(img,), daemon=True)
        t.start()
        if wait:
            t.join()
        return True

    def _run(self, img: np.ndarray) -> None:
        t0 = time.monotonic()
        try:
            text = describe(self.client, img, self.model, self.timeout)
            if text:
                self.ear.scene(text)
            self.log(f"[scene] {time.monotonic() - t0:.1f}s {text!r}")
        except Exception as e:
            self.log(f"[scene] skipped ({type(e).__name__}: {e})")
        finally:
            self._busy.release()


def asks_whats_ahead(text: str) -> bool:
    low = text.lower()
    return bool(_AROUND.search(low)) or "describe" in low or "see" in low


def listen_for_questions(url: str, on_question, log=print) -> None:
    """Earpiece WebSocket: each `heard` event (the wearer's transcribed speech) that asks what's
    around triggers on_question(). Reconnects forever."""
    from websockets.sync.client import connect
    while True:
        try:
            with connect(url) as ws:
                log(f"[scene] listening for questions on {url}")
                for raw in ws:
                    m = json.loads(raw)
                    if m.get("type") == "heard" and asks_whats_ahead(m.get("text", "")):
                        on_question()
        except Exception:
            time.sleep(1.0)


def open_source(spec: str):
    """Returns (camera name, latest() -> newest BGR frame or None)."""
    if spec in ("sim-chest", "sim-drone"):
        from frames import SimFrameSource
        return spec[4:], _latest_of(SimFrameSource(spec[4:]))
    if spec.isdigit():
        from kite.sources.opencv_sources import WebcamSource
        src = WebcamSource(int(spec)).start()
        return "chest", lambda: (f := src.read()) and f.image
    p = Path(spec)
    if (p / "frames.csv").is_file():
        from kite.sources.recording import RecordingSource
        return "drone", _latest_of(RecordingSource(p, realtime=True))
    if p.is_file():
        from kite.sources.opencv_sources import VideoFileSource
        src = VideoFileSource(p, loop=True).start()
        return "drone", lambda: (f := src.read()) and f.image
    raise SystemExit(f"--source {spec!r}: not sim-chest, sim-drone, a camera index, a video file or a kite.record folder")


def _latest_of(src):
    """Drain a blocking (ok, img) reader in a thread and keep only the newest frame."""
    slot: list = [None]

    def pump():
        while True:
            ok, img = src.read()
            if ok:
                slot[0] = img
            elif hasattr(src, "seek"):           # recording ended: loop it
                src.seek(0)
    threading.Thread(target=pump, daemon=True).start()
    return lambda: slot[0]


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    host = os.environ.get("KITE_EARPIECE_HOST", "127.0.0.1")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--camera", choices=["chest", "drone"], help="override the scene source name")
    ap.add_argument("--interval", type=float, default=6.0, help="seconds between descriptions")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--host", default=host)
    ap.add_argument("--port", type=int, default=int(os.environ.get("KITE_EARPIECE_PORT", "5010")))
    ap.add_argument("--ws", default=None, help="earpiece WebSocket for questions (default ws://HOST:5011)")
    a = ap.parse_args()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY missing (put it in the repo-root .env)")

    import anthropic
    cam, latest = open_source(a.source)
    cam = a.camera or cam
    ear = EarpieceClient(a.host, a.port, source=f"scene-{cam}")
    sd = SceneDescriber(anthropic.Anthropic(), ear, a.model, a.interval)
    asked = threading.Event()
    threading.Thread(target=listen_for_questions, args=(a.ws or f"ws://{a.host}:5011", asked.set),
                     daemon=True).start()
    print(f"[scene] {a.source} -> scene-{cam} at {a.host}:{a.port}, every {a.interval:g}s, {a.model}")
    while True:
        force = asked.wait(0.1)
        if force:
            asked.clear()
        sd.tick(latest(), time.monotonic(), force=force)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
