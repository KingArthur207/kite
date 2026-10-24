"""The laptop as a Bluetooth LE peripheral: the phone page (https://trykite.pages.dev/phone, in Bluefy on iPhone or
Chrome on Android) connects to it and writes GPS + compass fixes a few times a second.

    python -m phone.ble                       # advertise "Kite", print the walker's position live, feed the earpiece
    python -m phone.ble --seconds 20          # stop by itself (a quick advertise check)
    python -m phone.ble --no-send             # print only, nothing to the earpiece

GATT layout (UUIDs in phone/__init__.py):
    service  SERVICE_UUID
      FIX_CHAR_UUID     write / write without response: newline-framed JSON fixes (phone/fix.py), chunked by the
                        page to fit the ATT MTU, reassembled here (CoreBluetooth hands over each write as it comes)
      STATUS_CHAR_UUID  read / notify, ~40 bytes: {"ok":1,"n":<messages>,"e":<earpiece up>,"w":<Tello Wi-Fi>}

Uses `bless` (CoreBluetooth through PyObjC on macOS, BlueZ on Linux, WinRT on Windows). macOS asks once whether the
terminal app may use Bluetooth (System Settings > Privacy & Security > Bluetooth); without it advertising fails.
BLE runs next to the AirPods' Classic Bluetooth audio on the same controller: a few ~150-byte writes a second is a
tiny share of the radio's time (docs/phone.md has the evidence and the table check).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

from . import FIX_CHAR_UUID, SERVICE_UUID, STATUS_CHAR_UUID
from .fix import format_line
from .link import FORWARD_HZ, GPS_STALE_S, LINK_STALE_S, PhoneLink

ADVERTISED_NAME = "Kite"      # <= 10 characters, so macOS fits the 128-bit service UUID in the advertisement too
STATUS_EVERY_S = 1.0
TICK_S = 0.2


def _norm(uuid) -> str:
    return str(uuid).lower()


class GattHandlers:
    """The read / write callbacks bless calls (from CoreBluetooth's queue, not the asyncio loop)."""

    def __init__(self, link: PhoneLink, status=None):
        self.link = link
        self.status = status or (lambda: {})
        self.writes = 0
        self.ignored = 0

    def write(self, characteristic, value, **_kw) -> None:
        if _norm(getattr(characteristic, "uuid", "")) != FIX_CHAR_UUID:
            self.ignored += 1
            return
        self.writes += 1
        self.link.on_chunk(bytes(value or b""))

    def status_bytes(self) -> bytes:
        return json.dumps(self.status(), separators=(",", ":")).encode()

    def read(self, characteristic, **_kw) -> bytearray:
        if _norm(getattr(characteristic, "uuid", "")) == STATUS_CHAR_UUID:
            return bytearray(self.status_bytes())
        return bytearray(getattr(characteristic, "value", None) or b"")


class LinePrinter:
    """One live status line: rewritten in place on a terminal, one line every few seconds into a log file."""

    def __init__(self, stream=None, every_s: float = 5.0):
        self.stream = stream or sys.stdout
        self.tty = hasattr(self.stream, "isatty") and self.stream.isatty()
        self.every_s = every_s
        self._last = 0.0
        self._live = False

    def line(self, text: str, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        if self.tty:
            self.stream.write("\r\x1b[2K" + text)
            self._live = True
        elif now - self._last >= self.every_s:
            self.stream.write(text + "\n")
            self._last = now
        self.stream.flush()

    def log(self, text: str) -> None:
        if self.tty and self._live:
            self.stream.write("\r\x1b[2K")
        self.stream.write(time.strftime("%H:%M:%S ") + text + "\n")
        self.stream.flush()


async def build_server(handlers: GattHandlers, name: str = ADVERTISED_NAME, server_cls=None):
    """A bless server with Kite's service, not yet started. server_cls is injectable for tests."""
    if server_cls is None:
        from bless import BlessServer as server_cls
    from bless import GATTAttributePermissions, GATTCharacteristicProperties

    server = server_cls(name=name, loop=asyncio.get_running_loop())
    server.read_request_func = handlers.read
    server.write_request_func = handlers.write
    await server.add_new_service(SERVICE_UUID)
    await server.add_new_characteristic(
        SERVICE_UUID, FIX_CHAR_UUID,
        GATTCharacteristicProperties.write | GATTCharacteristicProperties.write_without_response,
        None, GATTAttributePermissions.writeable)
    await server.add_new_characteristic(
        SERVICE_UUID, STATUS_CHAR_UUID,
        GATTCharacteristicProperties.read | GATTCharacteristicProperties.notify,
        bytearray(b"{}"), GATTAttributePermissions.readable)
    return server


async def run(args, server_cls=None, stream=None, stop: asyncio.Event | None = None) -> int:
    printer = LinePrinter(stream)
    send = None
    watch = None
    if not args.no_send:
        from earpiece.client import EarpieceClient
        host = args.ear_host or os.environ.get("KITE_EARPIECE_HOST", "127.0.0.1")
        send = EarpieceClient(host=host, port=args.ear_port, source="phone").send
        if not args.no_watch:
            from .earpiece_watch import EarpieceWatch
            watch = EarpieceWatch(f"ws://{host}:{args.ear_ws_port}").start()
    link = PhoneLink(send=send, link_stale_s=args.stale, forward_hz=args.rate, announce=not args.no_announce,
                     log=printer.log)
    from .netinfo import on_tello_wifi
    tello = {"w": False, "t": -1e9}

    def status() -> dict:
        now = time.monotonic()
        if now - tello["t"] > 5:
            tello.update(w=on_tello_wifi(), t=now)
        return {"ok": int(link.link_ok), "n": link.stats["messages"], "e": int(bool(watch and watch.up)),
                "w": int(tello["w"])}

    handlers = GattHandlers(link, status)
    try:
        server = await build_server(handlers, args.name, server_cls)
        await server.start()
    except Exception as e:
        printer.log(f"[phone] could not start the Bluetooth peripheral: {type(e).__name__}: {e}")
        printer.log("        macOS: allow Bluetooth for this terminal app in System Settings > Privacy & Security > "
                    "Bluetooth, and check that Bluetooth is on.")
        return 1
    printer.log(f"[phone] advertising {args.name!r}, service {SERVICE_UUID}")
    printer.log("        on the phone: open https://trykite.pages.dev/phone (iPhone: in Bluefy) and tap Connect")
    if send is None:
        printer.log("        --no-send: positions are printed only, not sent to the earpiece")
    stop = stop or asyncio.Event()
    t_end = time.monotonic() + args.seconds if args.seconds else None
    last_status = 0.0
    try:
        while not stop.is_set():
            now = time.monotonic()
            link.tick(now)
            ear = "" if watch is None else ("earpiece up" if watch.up else "earpiece DOWN")
            printer.line(format_line(link.fix, link.fix_age(now), link.link_age(now), link.link_ok,
                                     link.stats["forwarded"], ear), now)
            if now - last_status >= STATUS_EVERY_S:
                last_status = now
                try:
                    server.get_characteristic(STATUS_CHAR_UUID).value = bytearray(handlers.status_bytes())
                    server.update_value(SERVICE_UUID, STATUS_CHAR_UUID)
                except Exception:
                    pass                          # nobody subscribed yet, or a backend without notify
            if t_end is not None and now >= t_end:
                break
            try:
                await asyncio.wait_for(stop.wait(), TICK_S)
            except asyncio.TimeoutError:
                pass
    finally:
        try:
            await server.stop()
        except Exception:
            pass
        if watch is not None:
            watch.stop()
        printer.log(f"[phone] stopped: {link.stats['messages']} messages, {link.stats['forwarded']} positions sent, "
                    f"{link.stats['bad']} bad")
    return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m phone.ble", description="Kite phone link over Bluetooth LE.")
    p.add_argument("--name", default=ADVERTISED_NAME, help="advertised name (<= 10 characters keeps the UUID in it)")
    p.add_argument("--ear-host", help="earpiece host (default KITE_EARPIECE_HOST or 127.0.0.1)")
    p.add_argument("--ear-port", type=int, default=5010, help="earpiece UDP port")
    p.add_argument("--ear-ws-port", type=int, default=5011, help="earpiece WebSocket port (status only)")
    p.add_argument("--rate", type=float, default=FORWARD_HZ, help="positions per second to the earpiece (default 2)")
    p.add_argument("--stale", type=float, default=LINK_STALE_S, help="seconds of silence before the phone is lost")
    p.add_argument("--no-send", action="store_true", help="print only, send nothing to the earpiece")
    p.add_argument("--no-announce", action="store_true", help="do not ask the earpiece to say the phone was lost")
    p.add_argument("--no-watch", action="store_true", help="do not watch the earpiece's WebSocket for status")
    p.add_argument("--seconds", type=float, default=0, help="stop after this long (0 = until Ctrl+C)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    _ = GPS_STALE_S
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        print()
        return 0


if __name__ == "__main__":
    sys.exit(main())
