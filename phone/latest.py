"""The newest phone position from SpacetimeDB, readable from any computer with internet (no Bluetooth, no LAN).

    python -m phone.latest            # print each phone's newest position once
    python -m phone.latest --watch    # print it every second

    from phone.latest import latest   # in a script: latest() -> [{"source", "lat", "lng", "accuracy", "age_s", ...}]

The receiver writes these when run with --spacetime (python -m phone.ble --spacetime). Standard library only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import ssl
import time
import urllib.request

HOST = os.environ.get("SPACETIME_HOST", "https://maincloud.spacetimedb.com")
DB = os.environ.get("SPACETIME_DB", "kite-sync-pl6rp")
QUERY = "SELECT * FROM latest WHERE type = 'position'"


def _ssl() -> ssl.SSLContext:
    try:                                   # python.org builds on macOS ship without CA certificates
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def latest(timeout: float = 5.0) -> list[dict]:
    """Every phone's newest position, newest first. Raises OSError when SpacetimeDB can't be reached."""
    req = urllib.request.Request(f"{HOST}/v1/database/{DB}/sql", data=QUERY.encode(), method="POST")
    with urllib.request.urlopen(req, timeout=timeout, context=_ssl()) as r:   # nosec B310: fixed https host
        reply = json.load(r)
    return fixes(reply)


def _num(v, lo: float, hi: float) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not lo <= v <= hi:
        return None
    return float(v)


def fixes(reply, now: float | None = None) -> list[dict]:
    """The phone positions in an SQL reply, newest first. The table is public and anyone can push to it, so a
    row whose payload isn't an object with a finite lat and lng on the globe is skipped, not trusted."""
    now = time.time() if now is None else now
    try:
        names = [e["name"]["some"] for e in reply[0]["schema"]["elements"]]
        rows = list(reply[0]["rows"])
    except (KeyError, IndexError, TypeError):
        return []
    out = []
    for row in rows:
        d = dict(zip(names, row)) if isinstance(row, list) else {}
        source = d.get("source")
        if not isinstance(source, str) or not source.startswith("phone"):
            continue
        try:
            p = json.loads(d["payload"])
            t = float(d["t"][0]) / 1e6
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        if not isinstance(p, dict) or not math.isfinite(t):
            continue
        lat, lng = _num(p.get("lat"), -90, 90), _num(p.get("lng"), -180, 180)
        if lat is None or lng is None:
            continue
        fix = {**p, "lat": lat, "lng": lng, "source": source, "t": t, "age_s": round(now - t, 1)}
        if p.get("accuracy") is not None:
            acc = _num(p.get("accuracy"), 0, 1e6)
            if acc is None:
                fix.pop("accuracy")
            else:
                fix["accuracy"] = acc
        out.append(fix)
    return sorted(out, key=lambda p: -p["t"])


def line(p: dict) -> str:
    acc = f"  +/-{p['accuracy']:.0f} m" if p.get("accuracy") is not None else ""
    return f"{p['source']:<12} {p['lat']:.6f}, {p['lng']:.6f}{acc}  ({p['age_s']:.0f} s ago)"


def main() -> None:
    a = argparse.ArgumentParser(prog="python -m phone.latest", description=__doc__.splitlines()[0])
    a.add_argument("--watch", action="store_true", help="print every second until Ctrl+C")
    args = a.parse_args()
    while True:
        try:
            rows = latest()
            print("\n".join(line(p) for p in rows) if rows else "no phone position yet")
        except (OSError, ValueError) as e:
            print(f"can't reach SpacetimeDB: {e}")
        if not args.watch:
            return
        time.sleep(1)
        print()


if __name__ == "__main__":
    main()
