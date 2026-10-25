"""The newest phone position from SpacetimeDB, readable from any computer with internet (no Bluetooth, no LAN).

    python -m phone.latest            # print each phone's newest position once
    python -m phone.latest --watch    # print it every second

    from phone.latest import latest   # in a script: latest() -> [{"source", "lat", "lng", "accuracy", "age_s", ...}]

The receiver writes these when run with --spacetime (python -m phone.ble --spacetime). Standard library only.
"""

from __future__ import annotations

import argparse
import json
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
    names = [e["name"]["some"] for e in reply[0]["schema"]["elements"]]
    out = []
    for row in reply[0]["rows"]:
        d = dict(zip(names, row))
        if not str(d.get("source", "")).startswith("phone"):
            continue
        p = json.loads(d["payload"])
        t = d["t"][0] / 1e6
        out.append({**p, "source": d["source"], "t": t, "age_s": round(time.time() - t, 1)})
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
        except OSError as e:
            print(f"can't reach SpacetimeDB: {e}")
        if not args.watch:
            return
        time.sleep(1)
        print()


if __name__ == "__main__":
    main()
