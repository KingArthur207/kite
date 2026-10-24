"""Tiny client for teammates: send events to a running earpiece in one line each.

    from earpiece.client import EarpieceClient
    ear = EarpieceClient()                                   # localhost:5010 (UDP)
    ear.obstacles([{"id": 7, "kind": "person", "bearing": -20, "distance": 2.4}])  # every frame
    ear.guidance("In 20 meters, turn left onto Lyon Street", key="step-3")
    ear.beacon(bearing=-35, distance=42, label="Palace of Fine Arts")
    ear.hazard("crosswalk", state="dont_walk", bearing=5, distance=9)
    ear.scene("Wide sidewalk, light crowd near the corner, crosswalk ahead")

UDP is fire-and-forget: it never blocks the caller's perception loop, and a
missing earpiece costs nothing. Voice requests ("take me to ...") come back over
the WebSocket; see `requests()`.
"""

from __future__ import annotations

import itertools
import json
import math
import socket

# macOS refuses UDP datagrams over net.inet.udp.maxdgram (9216 bytes by default) and Kite only
# ever uses the 64 nearest objects, so a crowded frame is trimmed to fit instead of vanishing.
MAX_DATAGRAM = 9000
MAX_OBJECTS = 64


def _distance(o) -> float:
    d = o.get("distance", o.get("distance_m")) if isinstance(o, dict) else None
    return float(d) if isinstance(d, (int, float)) and not isinstance(d, bool) and math.isfinite(d) else math.inf


class EarpieceClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 5010, source: str = "client"):
        self.addr = (host, port)
        self.source = source
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._seq = itertools.count(1)
        self.trimmed = 0               # obstacle frames cut down to fit one datagram
        self.dropped = 0               # datagrams the OS refused (earpiece not running, too big)

    def send(self, msg: dict) -> None:
        msg.setdefault("source", self.source)
        data = json.dumps(msg, separators=(",", ":")).encode()
        if len(data) > MAX_DATAGRAM and isinstance(msg.get("objects"), list):
            data = self._fit(msg)
        try:
            self._sock.sendto(data, self.addr)
        except OSError:
            self.dropped += 1          # earpiece not running: never crash the sender

    def _fit(self, msg: dict) -> bytes:
        """The nearest objects that fit in one datagram (the earpiece keeps the nearest 64 anyway)."""
        self.trimmed += 1
        objs = sorted(msg["objects"], key=_distance)[:MAX_OBJECTS]
        while True:
            data = json.dumps({**msg, "objects": objs}, separators=(",", ":")).encode()
            if len(data) <= MAX_DATAGRAM or not objs:
                return data
            objs = objs[: len(objs) * 3 // 4]

    def obstacles(self, objects: list[dict], source: str | None = None) -> None:
        """The FULL set of objects in this frame (ids not listed are treated as gone)."""
        self.send({"type": "obstacles", "source": source or self.source, "seq": next(self._seq),
                   "objects": objects})

    def heartbeat(self, source: str | None = None) -> None:
        self.send({"type": "heartbeat", "source": source or self.source})

    def guidance(self, text: str, key: str | None = None, ttl: float | None = None,
                 bearing: float | None = None) -> None:
        self.send({"type": "guidance", "text": text, "key": key, "ttl": ttl, "bearing": bearing})

    def beacon(self, bearing: float | None = None, distance: float | None = None,
               label: str | None = None, active: bool = True) -> None:
        self.send({"type": "beacon", "bearing": bearing, "distance": distance, "label": label,
                   "active": active})

    def hazard(self, kind: str, state: str | None = None, text: str | None = None,
               bearing: float | None = None, distance: float | None = None, ttl: float | None = None) -> None:
        self.send({"type": "hazard", "kind": kind, "state": state, "text": text, "bearing": bearing,
                   "distance": distance, "ttl": ttl})

    def say(self, text: str, priority: str = "info") -> None:
        self.send({"type": "say", "text": text, "priority": priority})

    def scene(self, text: str) -> None:
        self.send({"type": "scene", "text": text})

    def position(self, lat: float | None = None, lng: float | None = None, heading: float | None = None,
                 place: str | None = None) -> None:
        self.send({"type": "position", "lat": lat, "lng": lng, "heading": heading, "place": place})

    def status(self, battery: float | None = None, state: str | None = None, text: str | None = None) -> None:
        self.send({"type": "status", "battery": battery, "state": state, "text": text})

    def control(self, action: str) -> None:
        self.send({"type": "control", "action": action})


def requests(url: str = "ws://127.0.0.1:5011", kinds=("voice_request", "caregiver_alert")):
    """Blocking generator of outbound events (e.g. {'type':'voice_request','intent':'navigate',
    'destination':'Crissy Field'}). Reconnects if the earpiece restarts."""
    import time

    from websockets.sync.client import connect
    while True:
        try:
            with connect(url) as ws:
                for raw in ws:
                    m = json.loads(raw)
                    if m.get("type") in kinds:
                        yield m
        except OSError:
            time.sleep(1.0)
