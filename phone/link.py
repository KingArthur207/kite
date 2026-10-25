"""Transport-independent phone link logic: newest fix, staleness, forwarding to the earpiece, loss announcements.

PhoneLink is pure logic with an injectable clock and sender, so the Bluetooth server, the USB server and the tests
all drive the same object:

    link = PhoneLink(send=EarpieceClient(source="phone").send)
    link.on_message(raw_bytes)        # from a transport, any thread
    link.tick()                       # a few times a second: forwards positions, announces loss / recovery

Loss is announced with a `say` event (priority "system", a stable key so it is never queued twice), not a `status`:
the earpiece treats every `status` as the drone's (app.py _status), so a phone status would overwrite the drone's
state on the caregiver dashboard.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Callable

from .fix import Fix, LineAssembler, parse_fix, to_position

LINK_STALE_S = 5.0          # no message from the phone for this long: "Kite has lost the phone."
GPS_STALE_S = 15.0          # link up but no new GPS fix for this long: "The phone has no GPS fix."
FORWARD_HZ = 2.0            # positions to the earpiece (each is also mirrored to SpacetimeDB)

LOST_TEXT = "Kite has lost the phone. Directions are paused."
BACK_TEXT = "The phone is back."
NO_GPS_TEXT = "The phone has no GPS fix."


class PhoneLink:
    def __init__(self, send: Callable[[dict], None] | None = None, clock: Callable[[], float] = time.monotonic,
                 link_stale_s: float = LINK_STALE_S, gps_stale_s: float = GPS_STALE_S,
                 forward_hz: float = FORWARD_HZ, announce: bool = True, log: Callable[[str], None] | None = None,
                 source: str = "phone", walker: bool | None = None):
        self.source = source
        self.walker = walker                       # None: the only phone; False: mirrored, no guidance, silent
        self.send = send or (lambda msg: None)
        self.clock = clock
        self.link_stale_s, self.gps_stale_s = link_stale_s, gps_stale_s
        self.forward_every = 1.0 / forward_hz if forward_hz > 0 else 0.0
        self.announce = announce
        self.log = log or (lambda s: None)
        self.fix: Fix | None = None                # newest message (may have no coordinates)
        self.last_msg: float | None = None         # laptop clock of the newest valid message
        self.last_gps: float | None = None         # laptop clock when the newest NEW GPS fix arrived
        self._last_t: float | None = None          # phone timestamp of that fix
        self._last_forward = -1e9
        self._pending = False                      # a new GPS fix not yet forwarded
        self.link_ok = False
        self._ever = False
        self._gps_warned = False
        self.stats = {"messages": 0, "bad": 0, "forwarded": 0, "lost": 0}
        self._lock = threading.Lock()
        self._asm = LineAssembler()

    # ------------------------------------------------------------ input (any thread)
    def on_chunk(self, chunk: bytes) -> int:
        """Bluetooth: raw bytes of one GATT write. Returns how many messages it completed."""
        with self._lock:
            lines = self._asm.feed(bytes(chunk))
        for line in lines:
            self.on_message(line)
        return len(lines)

    def on_message(self, raw) -> Fix | None:
        """One whole JSON message. Bad ones are counted and dropped."""
        try:
            fix = parse_fix(raw, now=self.clock())
        except ValueError:                       # BadMessage
            self.stats["bad"] += 1
            return None
        with self._lock:
            self.stats["messages"] += 1
            self.fix = fix
            self.last_msg = fix.received
            if fix.has_position and (fix.t is None or fix.t != self._last_t):
                self._last_t = fix.t
                self.last_gps = fix.received
                self._pending = True
        return fix

    # ------------------------------------------------------------ ages
    def link_age(self, now: float | None = None) -> float | None:
        now = self.clock() if now is None else now
        return None if self.last_msg is None else max(0.0, now - self.last_msg)

    def fix_age(self, now: float | None = None) -> float | None:
        now = self.clock() if now is None else now
        return None if self.last_gps is None else max(0.0, now - self.last_gps)

    # ------------------------------------------------------------ periodic
    def tick(self, now: float | None = None) -> list[dict]:
        """Forward the newest fix (rate-limited) and announce loss / recovery. Returns what was sent."""
        now = self.clock() if now is None else now
        out: list[dict] = []
        speak = self.announce and self.walker is not False     # only the walker's phone talks to the walker
        with self._lock:
            age = None if self.last_msg is None else now - self.last_msg
            up = age is not None and age <= self.link_stale_s
            if up and not self.link_ok:
                self.link_ok = True
                if self._ever:
                    self.log(f"[{self.source}] back")
                    if speak:
                        out.append(self._say(BACK_TEXT, "phone-link"))
                else:
                    self.log(f"[{self.source}] connected")
                self._ever = True
            elif not up and self.link_ok:
                self.link_ok = False
                self.stats["lost"] += 1
                self.log(f"[{self.source}] lost: nothing for {age:.1f} s")
                if speak:
                    out.append(self._say(LOST_TEXT, "phone-link"))
            gps_age = None if self.last_gps is None else now - self.last_gps
            if up and self._ever and gps_age is not None and gps_age > self.gps_stale_s and not self._gps_warned:
                self._gps_warned = True
                if speak:
                    out.append(self._say(NO_GPS_TEXT, "phone-gps"))
            if gps_age is not None and gps_age <= self.gps_stale_s:
                self._gps_warned = False
            if up and self._pending and now - self._last_forward >= self.forward_every and self.fix is not None:
                pos = to_position(self.fix, self.source, self.walker)
                if pos is not None:
                    out.append(pos)
                    self.stats["forwarded"] += 1
                self._pending = False
                self._last_forward = now
        for msg in out:
            try:
                self.send(msg)
            except Exception as e:               # the earpiece being down must never stop the link
                self.log(f"[phone] send failed: {type(e).__name__}: {e}")
        return out

    def alive(self, now: float) -> bool:
        return self.last_msg is not None and now - self.last_msg <= self.link_stale_s

    def _say(self, text: str, key: str, priority: str = "system") -> dict:
        return {"type": "say", "source": self.source, "text": text, "priority": priority, "key": key, "ttl": 10}

    def status(self, now: float | None = None) -> dict:
        """Small JSON for the phone's status characteristic / WebSocket: what the laptop has."""
        now = self.clock() if now is None else now
        fa = self.fix_age(now)
        return {"ok": self.link_ok, "n": self.stats["messages"], "fwd": self.stats["forwarded"],
                "fix_age": None if fa is None else round(fa, 1)}


def short_id(central) -> str:
    """A stable 4-hex id for a central (CoreBluetooth's per-central UUID); "" when the backend gives none."""
    return hashlib.sha1(str(central).encode()).hexdigest()[:4] if central else ""


class PhoneHub:
    """Several phones at once. Each central gets its own PhoneLink (own reassembly buffer, own staleness) and the
    source "phone-<id>"; every phone's position goes to the earpiece (and so to the caregiver map), but only the
    walker's phone has walker=true and may speak. The walker is the first phone to connect (or `prefer`), and the
    next live phone once it goes quiet.

    ponytail: a phone that leaves keeps its PhoneLink forever; fine for a handful of phones in a demo."""

    def __init__(self, send: Callable[[dict], None] | None = None, clock: Callable[[], float] = time.monotonic,
                 walker: str | None = None, log: Callable[[str], None] | None = None, **link_kw):
        self.send = send or (lambda msg: None)
        self.clock = clock
        self.log = log or (lambda s: None)
        self.prefer = walker.lower().removeprefix("phone-") if walker else None
        self.link_kw = link_kw
        self.links: dict[str, PhoneLink] = {}      # insertion order = the order the phones connected
        self.walker: str | None = None             # a key of self.links
        self._lock = threading.Lock()

    def link_for(self, central=None) -> PhoneLink:
        pid = short_id(central)
        with self._lock:
            link = self.links.get(pid)
            if link is None:
                link = self.links[pid] = PhoneLink(
                    send=self.send, clock=self.clock, log=self.log, source=f"phone-{pid}" if pid else "phone",
                    walker=False, **self.link_kw)
        return link

    def on_chunk(self, chunk: bytes, central=None) -> int:
        return self.link_for(central).on_chunk(chunk)

    def walker_link(self) -> PhoneLink | None:
        return self.links.get(self.walker) if self.walker is not None else None

    def tick(self, now: float | None = None) -> list[dict]:
        now = self.clock() if now is None else now
        with self._lock:
            links = dict(self.links)
        out = self._pick_walker(links, now)
        for pid, link in links.items():
            link.walker = pid == self.walker
            out += link.tick(now)
        return out

    def _pick_walker(self, links: dict[str, PhoneLink], now: float) -> list[dict]:
        live = [pid for pid, link in links.items() if link.alive(now)]
        cur = links.get(self.walker) if self.walker is not None else None
        if self.prefer in live:
            new = self.prefer
        elif cur is not None and (cur.alive(now) or not live):
            new = self.walker                      # keep the walker, even while lost (it says so itself)
        else:
            new = live[0] if live else None
        if new == self.walker:
            return []
        self.walker = new
        link = links[new]
        self.log(f"[phone] walker is now {link.source}" + (f" ({cur.source} went quiet)" if cur else ""))
        if cur is None or cur.link_ok or not link.announce:
            return []                              # a silent handover: directions never stopped
        if not link.link_ok and link._ever:
            return []                              # its own tick says "The phone is back"
        msg = link._say(BACK_TEXT, "phone-link")   # the walker heard "lost"; another phone took over
        try:
            self.send(msg)
        except Exception as e:
            self.log(f"[phone] send failed: {type(e).__name__}: {e}")
        return [msg]
