"""Replay a JSONL of Kite events into the earpiece (and optionally SpacetimeDB).

    python scripts/replay.py data/sample_walk.jsonl               # to KITE_EARPIECE_HOST
    python scripts/replay.py out.jsonl --spacetime --speed 2 --loop

One event per line, in the earpiece wire format (earpiece/earpiece/events.py).
An optional "t" field (seconds, relative or epoch) paces the replay; lines
without it go out 0.1 s apart. Run with the earpiece venv:
earpiece/.venv/bin/python scripts/replay.py ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "earpiece"))

from earpiece.client import EarpieceClient  # noqa: E402
from earpiece.config import _load_env  # noqa: E402

DEFAULT_GAP_S = 0.1


def load(path: str) -> list[dict]:
    events = []
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError as e:
            sys.exit(f"{path}:{n}: bad JSON ({e.msg})")
        if not isinstance(ev, dict) or "type" not in ev:
            sys.exit(f"{path}:{n}: each line must be an event object with a type")
        events.append(ev)
    return events


def schedule(events: list[dict]) -> list[tuple[float, dict]]:
    """(seconds from start, event) with any absolute timestamps made relative."""
    ts = [e.get("t") for e in events]
    t0 = next((t for t in ts if isinstance(t, (int, float))), 0.0)
    out, last = [], 0.0
    for ev, t in zip(events, ts):
        last = t - t0 if isinstance(t, (int, float)) else last + DEFAULT_GAP_S
        out.append((last, {k: v for k, v in ev.items() if k != "t"}))
    return out


def main(argv=None) -> int:
    _load_env()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("jsonl")
    p.add_argument("--host", default=os.environ.get("KITE_EARPIECE_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("KITE_EARPIECE_PORT", "5010")))
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--loop", action="store_true")
    p.add_argument("--spacetime", action="store_true", help="also push every event to SpacetimeDB")
    a = p.parse_args(argv)

    plan = schedule(load(a.jsonl))
    if not plan:
        sys.exit(f"{a.jsonl}: no events")
    ear = EarpieceClient(a.host, a.port, source="replay")
    st = None
    if a.spacetime:
        from earpiece.spacetime import SpacetimeBridge
        st = SpacetimeBridge()
        print(f"[replay] mirroring to {st.host}/{st.db}")
    print(f"[replay] {len(plan)} events, {plan[-1][0] / a.speed:.1f}s, to udp://{a.host}:{a.port}")
    try:
        while True:
            start = time.monotonic()
            for at, ev in plan:
                delay = start + at / a.speed - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                ear.send(dict(ev))
                if st is not None:
                    st.publish({**ev, "source": ev.get("source", "replay")})
            if not a.loop:
                break
    except KeyboardInterrupt:
        pass
    if st is not None:
        st.flush()
        print(f"[replay] spacetime {st.stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
