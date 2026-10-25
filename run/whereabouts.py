"""Phone GPS fixes from SpacetimeDB, shown in the kite window.

The phone PWA writes a ``position`` row while its GPS watch is on. This module
reads that row. A signed fix that still arrives through the UNO Q is kept as a
second source. It only listens. It does not talk to the drone.

``parse_fix`` and ``format_fix`` are pure. The listener threads never touch Tk.
"""

from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
from datetime import datetime, timezone

_LINK_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hardware", "arduino", "unoq_link")
_MAX_TEXT = 2000


def parse_fix(text) -> dict | None:
    """A ``location_fix`` object, or None when ``text`` is not one."""
    if not isinstance(text, str) or not text or len(text) > _MAX_TEXT:
        return None
    try:
        raw = json.loads(text)
    except ValueError:
        return None
    if not isinstance(raw, dict) or raw.get("kind") != "location_fix":
        return None
    lat, lon = _num(raw.get("lat")), _num(raw.get("lon"))
    if lat is None or lon is None or not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
        return None
    fix = {
        "lat": lat,
        "lon": lon,
        "accuracy_m": _num(raw.get("accuracy_m")),
        "altitude_m": _num(raw.get("altitude_m")),
        "speed_mps": _num(raw.get("speed_mps")),
        "heading_deg": _num(raw.get("heading_deg")),
        "timestamp": raw.get("timestamp") if isinstance(raw.get("timestamp"), str) else None,
    }
    stamp = fix["timestamp"] or ""
    fix["key"] = f"{stamp}|{lat:.5f}|{lon:.5f}"
    return fix


def format_fix(fix: dict) -> str:
    """One console line. ``accepted`` means the board has already logged the fix."""
    bits = [f"gps: {fix['lat']:.5f}, {fix['lon']:.5f}"]
    if fix.get("accuracy_m") is not None:
        bits.append(f"±{fix['accuracy_m']:.0f}m")
    if fix.get("altitude_m") is not None:
        bits.append(f"alt {fix['altitude_m']:.0f}m")
    if fix.get("speed_mps") is not None:
        bits.append(f"{fix['speed_mps']:.1f}m/s")
    if fix.get("heading_deg") is not None:
        bits.append(f"hdg {fix['heading_deg']:.0f}")
    age = fix.get("age_s")
    if isinstance(age, (int, float)) and math.isfinite(age) and age >= 15:
        bits.append(f"{int(age)}s old")
    bits.append("live" if fix.get("accepted") else "on the way")
    return "  ".join(bits)


def _num(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _age_s(fix: dict, now: float) -> float:
    stamp = fix.get("timestamp")
    if isinstance(stamp, str):
        try:
            when = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            return max(0.0, now - when.timestamp())
        except ValueError:
            pass
    return max(0.0, now - float(fix.get("seen_at") or now))


def _stamp_key(fix: dict) -> tuple:
    """Newest timestamp wins. Missing timestamps sort last among otherwise equal keys."""
    stamp = fix.get("timestamp")
    if isinstance(stamp, str):
        try:
            when = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            return (1, when.timestamp())
        except ValueError:
            pass
    return (0, float(fix.get("seen_at") or 0.0))


def fix_from_position(row) -> dict | None:
    """A ``phone.latest`` row (``lat`` / ``lng``) as the fix the map draws, or None."""
    if not isinstance(row, dict):
        return None
    lat = _num(row.get("lat"))
    lon = _num(row.get("lng"))
    if lon is None:
        lon = _num(row.get("lon"))
    if lat is None or lon is None or not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
        return None
    when = row.get("t")
    stamp = None
    if isinstance(when, (int, float)) and not isinstance(when, bool) and math.isfinite(when) and when > 1_000_000_000:
        stamp = datetime.fromtimestamp(when, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    fix = {
        "lat": lat,
        "lon": lon,
        "accuracy_m": _num(row.get("accuracy") if row.get("accuracy") is not None else row.get("accuracy_m")),
        "altitude_m": _num(row.get("alt") if row.get("alt") is not None else row.get("altitude_m")),
        "speed_mps": _num(row.get("speed") if row.get("speed") is not None else row.get("speed_mps")),
        "heading_deg": _num(row.get("course") if row.get("course") is not None else row.get("heading")),
        "timestamp": stamp,
    }
    fix["key"] = f"{stamp or ''}|{lat:.5f}|{lon:.5f}"
    return fix


def _mods():
    if _LINK_DIR not in sys.path:
        sys.path.insert(0, _LINK_DIR)
    import kite_common
    import kite_link

    return kite_common, kite_link


def _payload(row) -> dict | None:
    raw = row.get("payload") if isinstance(row, dict) else None
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    try:
        body = json.loads(raw)
    except ValueError:
        return None
    return body if isinstance(body, dict) else None


class Whereabouts:
    """Background listener. ``latest()`` is the newest fix, or None."""

    def __init__(self, log) -> None:
        self.log = log
        self._lock = threading.Lock()
        self._fix: dict | None = None
        self._pending: dict[str, dict] = {}
        self._stop = threading.Event()
        self._stream = None
        self._cloud_warned = False
        self._sse_logged = False
        self._last_link: int | None = None
        self._last_reply: int | None = None
        self._cloud = None

    def start(self) -> None:
        threading.Thread(target=self._cloud_loop, name="whereabouts-cloud", daemon=True).start()
        threading.Thread(target=self._sse_loop, name="whereabouts-board", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()
        stream = self._stream
        if stream is not None:
            try:
                stream.close()
            except Exception:
                pass

    def latest(self) -> dict | None:
        with self._lock:
            if self._fix is None:
                return None
            fix = dict(self._fix)
        fix["age_s"] = _age_s(fix, time.time())
        return fix

    def _token(self) -> str:
        token = os.environ.get("KITE_TOKEN", "").strip()
        if token:
            return token
        try:
            _, kite_link = _mods()
            return kite_link.read_token()
        except Exception:
            return ""

    def _offer(self, fix: dict, accepted: bool) -> None:
        fresh = dict(fix)
        fresh["accepted"] = bool(accepted)
        fresh["seen_at"] = time.time()
        with self._lock:
            current = self._fix
            if current is not None and current.get("key") == fresh["key"]:
                if accepted:
                    current["accepted"] = True
                    current["seen_at"] = fresh["seen_at"]
                return
            if current is not None and _stamp_key(fresh) < _stamp_key(current):
                return
            self._fix = fresh

    def _remember(self, fix: dict) -> None:
        ident = fix.get("id") or ""
        if not ident:
            return
        self._pending[ident] = fix
        if len(self._pending) > 30:
            for old in list(self._pending)[:-30]:
                self._pending.pop(old, None)

    def _cloud_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._poll_phone()
                self._cloud_warned = False
            except Exception as exc:
                if not self._cloud_warned:
                    self._cloud_warned = True
                    self.log(f"gps: spacetime is not reachable yet ({type(exc).__name__})", "warn")
            try:
                self._poll_cloud()
            except Exception:
                pass
            self._stop.wait(2.0)

    def _poll_phone(self) -> None:
        from phone.latest import latest as phone_latest

        for row in phone_latest(timeout=5.0):
            fix = fix_from_position(row)
            if fix is not None:
                self._offer(fix, accepted=True)
                return

    def _poll_cloud(self) -> None:
        token = self._token()
        if not token:
            return
        kite_common, _kite_link = _mods()
        source = os.environ.get("KITE_SOURCE", kite_common.DEFAULT_SOURCE)
        if self._cloud is None:
            self._cloud = kite_common.Spacetime()
        cloud = self._cloud
        link = kite_common.link_source(source)
        reply = kite_common.reply_source(source)
        if self._last_link is None:
            self._last_link = cloud.last_id("control", link)
            self._last_reply = cloud.last_id("control", reply)
            return
        for row in cloud.rows("control", link, self._last_link):
            self._last_link = max(self._last_link, int(row["id"]))
            self._on_command(kite_common, token, source, row)
        for row in cloud.rows("control", reply, self._last_reply):
            self._last_reply = max(self._last_reply, int(row["id"]))
            self._on_reply(kite_common, token, row)

    def _on_command(self, kite_common, token: str, source: str, row: dict) -> None:
        body = _payload(row)
        if body is None or not kite_common.verify(token, body):
            return
        if body.get("cmd") != "send" or body.get("to") != source:
            return
        fix = parse_fix(body.get("text"))
        if fix is None:
            return
        fix["id"] = str(body.get("id") or "")
        self._remember(fix)
        self._offer(fix, accepted=False)

    def _on_reply(self, kite_common, token: str, row: dict) -> None:
        body = _payload(row)
        if body is None or not kite_common.verify(token, body) or not body.get("ok", True):
            return
        fix = self._pending.get(str(body.get("reply_to") or ""))
        if fix is not None:
            self._offer(fix, accepted=True)

    def _sse_loop(self) -> None:
        while not self._stop.is_set():
            try:
                direct = self._find_direct()
                if direct is None:
                    self._stop.wait(5.0)
                    continue
                kite_link = _mods()[1]
                with direct.open("/events", timeout=15) as stream:
                    self._stream = stream
                    if not self._sse_logged:
                        self._sse_logged = True
                        self.log(f"gps: board link is up at {direct}", "info")
                    for event in kite_link.iter_events(stream):
                        if self._stop.is_set():
                            return
                        if event.get("kind") != "got":
                            continue
                        fix = parse_fix(event.get("text"))
                        if fix is not None:
                            self._offer(fix, accepted=True)
            except Exception:
                self._stop.wait(3.0)
            finally:
                self._stream = None

    def _find_direct(self):
        token = self._token()
        if not token:
            return None
        kite_common, kite_link = _mods()
        chosen = os.environ.get("KITE_BOARD", "").strip()
        if chosen:
            host, port = kite_link.parse_board(chosen, kite_common.DEFAULT_PORT)
            ok, _why = kite_link.tcp_ok(host, port, 2.0)
            if ok:
                return kite_link.Direct(host, port, token, "KITE_BOARD")
        cloud = self._cloud or kite_common.Spacetime()
        self._cloud = cloud
        source = os.environ.get("KITE_SOURCE", kite_common.DEFAULT_SOURCE)
        info = kite_link.find_heartbeat(cloud, source, token)
        hb = info.get("hb") if info.get("valid") else None
        port = int((hb or {}).get("port") or kite_common.DEFAULT_PORT)
        candidates = []
        if hb:
            candidates += [("direct", ip, port) for ip in hb.get("ips") or [] if isinstance(ip, str)]
            if isinstance(hb.get("tailscale"), str):
                candidates.append(("tailscale", hb["tailscale"], port))
        candidates.append(("tailscale name", kite_common.TAILSCALE_NAME, port))
        for label, host, host_port, ok, _why in kite_link.reach(candidates, timeout=2.0):
            if ok:
                return kite_link.Direct(host, host_port, token, label)
        return None
