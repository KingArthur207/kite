"""Phone GPS from SpacetimeDB to the earpiece. The Kite app (phone-app/) pushes each fix to SpacetimeDB over the
phone's own internet; this reads them and sends `position` events to the earpiece over UDP, with the same "Kite has
lost the phone" announcements as the Bluetooth receiver (phone/link.py does both).

    python -m phone.cloud                           # earpiece on this computer (KITE_EARPIECE_HOST, default 127.0.0.1)
    python -m phone.cloud --ear-host 192.168.137.2  # earpiece on another computer: start it with --bind 0.0.0.0
    python -m phone.cloud --no-send                 # print the walker's position only

It reads the append-only `event` table, not `latest`: the earpiece mirrors every position it is given back to
SpacetimeDB under the same source, which overwrites the app's row in `latest` and would echo forever. Only rows that
carry the app's `fix_t` are forwarded, and the earpiece's copy never has one. Standard library plus phone/ and
earpiece/ (no Bluetooth).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import urllib.request
from collections.abc import Callable

from .ble import LinePrinter
from .fix import format_line
from .latest import DB, HOST, _ssl
from .link import LINK_STALE_S, PhoneLink

POLL_S = 0.5
REBASE_S = 10.0               # nothing new for this long: check whether the database was published afresh
IDENT = re.compile(r"^[\w.:-]{1,64}$")   # the module's source rule (dashboard/spacetimedb/src/index.ts)


def sql(query: str, host: str = HOST, db: str = DB, timeout: float = 5.0):
    req = urllib.request.Request(f"{host}/v1/database/{db}/sql", data=query.encode(), method="POST")
    with urllib.request.urlopen(req, timeout=timeout, context=_ssl()) as r:   # nosec B310: fixed http(s) host
        return json.load(r)


def rows(reply) -> list[tuple[int, dict | None]]:
    """(id, the app's fix or None) for every row of an `event` SQL reply, oldest first. The table is public, so a
    row is untrusted: anything that isn't a JSON object with a finite fix_t is None (still counted for the id)."""
    try:
        names = [e["name"]["some"] for e in reply[0]["schema"]["elements"]]
        raw = list(reply[0]["rows"])
    except (KeyError, IndexError, TypeError):
        return []
    out = []
    for row in raw:
        d = dict(zip(names, row)) if isinstance(row, list) else {}
        rid = d.get("id")
        if isinstance(rid, bool) or not isinstance(rid, int):
            continue
        try:
            p = json.loads(d["payload"])
        except (KeyError, TypeError, ValueError):
            p = None
        t = p.get("fix_t") if isinstance(p, dict) else None
        ok = isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t)
        out.append((rid, p if ok else None))
    return sorted(out, key=lambda r: r[0])


def to_fix(p: dict) -> dict:
    """The app's position payload -> the phone fix format phone/fix.py parses (and validates)."""
    return {"t": p["fix_t"], "lat": p.get("lat"), "lng": p.get("lng"), "acc": p.get("accuracy"),
            "speed": p.get("speed"), "course": p.get("course")}


class CloudFeed:
    """Feeds new app rows into a PhoneLink. The first poll only notes where the log ends, so a fix from an earlier
    walk is never replayed as the walker's position now."""

    def __init__(self, link: PhoneLink, query: Callable[[str], object] = sql, source: str = "phone",
                 clock: Callable[[], float] = time.monotonic):
        if not IDENT.match(source):
            raise ValueError(f"bad source {source!r}")
        self.link, self.query, self.source, self.clock = link, query, source, clock
        self.after: int | None = None
        self._last_new = clock()
        self.stats = {"rows": 0, "fed": 0, "skipped": 0}

    def _select(self, after: int):
        # source matches IDENT (no quotes) and after is an int, so neither can change the query
        q = f"SELECT * FROM event WHERE type = 'position' AND source = '{self.source}' AND id > {int(after)}"  # nosec B608
        return rows(self.query(q))

    def poll(self) -> int:
        """One round trip. Returns how many fixes went into the link. Raises OSError / ValueError when unreachable."""
        now = self.clock()
        if self.after is None:
            got = self._select(0)
            self.after = got[-1][0] if got else 0
            self._last_new = now
            return 0
        got = self._select(self.after)
        if not got:
            if now - self._last_new >= REBASE_S:
                self._last_new = now
                every = self._select(0)
                newest = every[-1][0] if every else 0
                if newest < self.after:              # ids went backwards: the database is a fresh one
                    self.after = newest
            return 0
        self._last_new = now
        fed = 0
        for rid, p in got:
            self.after = max(self.after, rid)
            self.stats["rows"] += 1
            if p is None:
                self.stats["skipped"] += 1           # the earpiece's own copy, or junk
                continue
            if self.link.on_message(json.dumps(to_fix(p))) is not None:
                fed += 1
        self.stats["fed"] += fed
        return fed


def run(args, query: Callable[[str], object] | None = None, stream=None, stop: Callable[[], bool] | None = None) -> int:
    printer = LinePrinter(stream)
    send = None
    if not args.no_send:
        from earpiece.client import EarpieceClient
        host = args.ear_host or os.environ.get("KITE_EARPIECE_HOST", "127.0.0.1")
        send = EarpieceClient(host=host, port=args.ear_port, source=args.source).send
    st_host = os.environ.get("SPACETIME_HOST", HOST)
    st_db = os.environ.get("SPACETIME_DB", DB)
    link = PhoneLink(send=send, log=printer.log, source=args.source, link_stale_s=args.stale,
                     announce=not args.no_announce)
    feed = CloudFeed(link, query or (lambda q: sql(q, st_host, st_db)), args.source)
    printer.log(f"[phone] reading {args.source!r} positions from {st_host}/v1/database/{st_db}")
    printer.log("        on the phone: open the Kite app (phone-app/) and tap the screen")
    if send is None:
        printer.log("        --no-send: positions are printed only, not sent to the earpiece")
    t_end = time.monotonic() + args.seconds if args.seconds else None
    down = False
    try:
        while not (stop and stop()):
            try:
                feed.poll()
                if down:
                    printer.log("[phone] SpacetimeDB reachable again")
                down = False
            except (OSError, ValueError) as e:
                if not down:
                    printer.log(f"[phone] can't reach SpacetimeDB: {e}")
                down = True
            now = time.monotonic()
            link.tick(now)
            printer.line(format_line(link.fix, link.fix_age(now), link.link_age(now), link.link_ok,
                                     link.stats["forwarded"]), now)
            if t_end is not None and now >= t_end:
                break
            time.sleep(POLL_S)
    except KeyboardInterrupt:
        pass
    printer.log(f"[phone] stopped: {feed.stats['fed']} fixes from the app, {link.stats['forwarded']} positions sent")
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m phone.cloud", description=__doc__.splitlines()[0])
    p.add_argument("--no-send", action="store_true", help="print only, nothing to the earpiece")
    p.add_argument("--ear-host", help="earpiece address (default KITE_EARPIECE_HOST or 127.0.0.1)")
    p.add_argument("--ear-port", type=int, default=5010)
    p.add_argument("--source", default="phone", help="the app's source in SpacetimeDB (default phone)")
    p.add_argument("--stale", type=float, default=LINK_STALE_S, help="seconds of silence before 'lost the phone'")
    p.add_argument("--no-announce", action="store_true", help="don't speak 'lost the phone' / 'the phone is back'")
    p.add_argument("--seconds", type=float, default=0, help="stop after this many seconds (0: run until Ctrl+C)")
    return p


def main(argv: list[str] | None = None) -> int:
    return run(parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
