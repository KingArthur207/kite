"""macOS audio-device facts via CoreAudio (ctypes, no extra deps), with safe fallbacks.

Verified on this Mac: the default-output listener fires 2-4 ms after a switch,
on a CoreAudio thread, with no run loop needed. AirPods are recognised by a
Bluetooth transport plus "airpods" in the name (the name is user-editable, the
transport is the reliable part). Battery comes from `system_profiler`
(~0.1 s), so it is polled rarely.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import struct
import subprocess
import sys
import threading
from typing import Callable

IS_MAC = sys.platform == "darwin"


def _fourcc(s: str) -> int:
    return struct.unpack(">I", s.encode())[0]


class _Addr(ctypes.Structure):
    _fields_ = [("mSelector", ctypes.c_uint32), ("mScope", ctypes.c_uint32), ("mElement", ctypes.c_uint32)]


_CA = _CF = None
_LISTENER = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_uint32, ctypes.c_uint32,
                             ctypes.POINTER(_Addr), ctypes.c_void_p)
_SYS = 1
_GLOB, _OUTP = _fourcc("glob"), _fourcc("outp")
_cb_refs: list = []

if IS_MAC:
    try:
        _CA = ctypes.CDLL(ctypes.util.find_library("CoreAudio"))
        _CF = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))
        _CA.AudioObjectGetPropertyData.argtypes = [ctypes.c_uint32, ctypes.POINTER(_Addr), ctypes.c_uint32,
                                                   ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
        _CA.AudioObjectGetPropertyDataSize.argtypes = [ctypes.c_uint32, ctypes.POINTER(_Addr), ctypes.c_uint32,
                                                       ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        _CA.AudioObjectAddPropertyListener.argtypes = [ctypes.c_uint32, ctypes.POINTER(_Addr), _LISTENER,
                                                       ctypes.c_void_p]
        _CF.CFStringGetCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32]
        _CF.CFRelease.argtypes = [ctypes.c_void_p]
    except (OSError, AttributeError):
        _CA = _CF = None


def _get(obj: int, sel: str, ctype, scope: int = _GLOB):
    a = _Addr(_fourcc(sel), scope, 0)
    v = ctype()
    n = ctypes.c_uint32(ctypes.sizeof(v))
    err = _CA.AudioObjectGetPropertyData(obj, ctypes.byref(a), 0, None, ctypes.byref(n), ctypes.byref(v))
    if err:
        raise OSError(f"{sel}: OSStatus {err}")
    return v.value


def _cfstr(obj: int, sel: str) -> str:
    ref = _get(obj, sel, ctypes.c_void_p)
    buf = ctypes.create_string_buffer(512)
    _CF.CFStringGetCString(ref, buf, 512, 0x08000100)
    _CF.CFRelease(ref)
    return buf.value.decode(errors="replace")


def _all() -> list[int]:
    a = _Addr(_fourcc("dev#"), _GLOB, 0)
    n = ctypes.c_uint32()
    _CA.AudioObjectGetPropertyDataSize(_SYS, ctypes.byref(a), 0, None, ctypes.byref(n))
    arr = (ctypes.c_uint32 * (n.value // 4))()
    _CA.AudioObjectGetPropertyData(_SYS, ctypes.byref(a), 0, None, ctypes.byref(n), arr)
    return list(arr)


def _has_output(dev: int) -> bool:
    a = _Addr(_fourcc("stm#"), _OUTP, 0)
    n = ctypes.c_uint32()
    return _CA.AudioObjectGetPropertyDataSize(dev, ctypes.byref(a), 0, None, ctypes.byref(n)) == 0 and n.value > 0


def _transport(dev: int) -> str:
    try:
        t = _get(dev, "tran", ctypes.c_uint32)
    except OSError:
        return "unknown"
    return {_fourcc("blue"): "bluetooth", _fourcc("blea"): "bluetooth-le",
            _fourcc("bltn"): "built-in"}.get(t, "other")


def outputs() -> list[dict]:
    """Every output device: name, transport, sample rate, is_default."""
    if _CA is None:
        try:
            import sounddevice as sd
            d = sd.default.device[1]
            return [{"name": x["name"], "transport": "unknown", "rate": x["default_samplerate"],
                     "default": i == d} for i, x in enumerate(sd.query_devices()) if x["max_output_channels"] > 0]
        except Exception:
            return []
    try:
        dflt = _get(_SYS, "dOut", ctypes.c_uint32)
        out = []
        for dev in _all():
            if not _has_output(dev):
                continue
            try:
                out.append({"name": _cfstr(dev, "lnam"), "transport": _transport(dev),
                            "rate": _get(dev, "nsrt", ctypes.c_double), "default": dev == dflt})
            except OSError:
                continue
        return out
    except Exception:
        return []


def default_output_name() -> str | None:
    for d in outputs():
        if d["default"]:
            return d["name"]
    return None


def headphones_output_name(prefer: str = "airpods") -> str | None:
    """The output to use for the walker: a device matching `prefer` (AirPods) first, else any
    Bluetooth output (spec: any stereo Bluetooth pair works). None if neither is connected."""
    outs = outputs()
    p = prefer.lower()
    for d in outs:
        if p and p in d["name"].lower():
            return d["name"]
    for d in outs:
        if d["transport"].startswith("bluetooth"):
            return d["name"]
    return None


def airpods_output_present(substr: str = "airpods") -> bool:
    s = substr.lower()
    for d in outputs():
        if s in d["name"].lower() and (d["transport"].startswith("bluetooth") or d["transport"] == "unknown"):
            return True
    return False


def airpods_output_rate(substr: str = "airpods") -> float | None:
    """< 44.1 kHz means the AirPods are in the low-quality two-way (call) profile:
    something is using their microphone."""
    for d in outputs():
        if substr.lower() in d["name"].lower():
            return d["rate"]
    return None


def on_device_change(fn: Callable[[], None]) -> bool:
    """fn() on a CoreAudio thread whenever the default output or the device list changes."""
    if _CA is None:
        return False

    def cb(obj, n, addrs, ctx):
        try:
            fn()
        except Exception:
            pass
        return 0

    c = _LISTENER(cb)
    _cb_refs.append(c)                    # keep alive or the callback is garbage-collected
    ok = True
    for sel in ("dOut", "dev#"):
        a = _Addr(_fourcc(sel), _GLOB, 0)
        ok &= _CA.AudioObjectAddPropertyListener(_SYS, ctypes.byref(a), c, None) == 0
    return ok


def airpods_battery() -> dict | None:
    """{'name':..., 'left': 85, 'right': 80, 'case': None} for connected Apple audio
    devices, or None. Fields only exist while connected; case is often missing."""
    if not IS_MAC:
        return None
    try:
        r = subprocess.run(["system_profiler", "SPBluetoothDataType", "-json"],
                           capture_output=True, text=True, timeout=5)
        root = json.loads(r.stdout)["SPBluetoothDataType"][0]
    except Exception:
        return None

    def pct(v):
        try:
            return int(str(v).strip().rstrip("%"))
        except (TypeError, ValueError):
            return None

    for dev in root.get("device_connected", []) or []:
        for name, p in dev.items():
            if not isinstance(p, dict):
                continue
            keys = ("device_batteryLevelLeft", "device_batteryLevelRight",
                    "device_batteryLevelCase", "device_batteryLevelMain")
            if "airpods" in name.lower() or any(k in p for k in keys[:2]):
                return {"name": name, "left": pct(p.get(keys[0])), "right": pct(p.get(keys[1])),
                        "case": pct(p.get(keys[2])), "main": pct(p.get(keys[3]))}
    return None


class BatteryWatch:
    """Polls AirPods battery once a minute and calls warn(text) at 20 % and 10 %."""

    def __init__(self, warn: Callable[[str], None], period_s: float = 60.0):
        self.warn = warn
        self.period = period_s
        self.last: dict | None = None
        self._warned: set[int] = set()
        self._stop = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self._run, name="battery", daemon=True).start()

    def check(self, info: dict | None) -> None:
        self.last = info
        if not info:
            return
        levels = [v for v in (info.get("left"), info.get("right"), info.get("main")) if v is not None]
        if not levels:
            return
        low = min(levels)
        for thr in (10, 20):
            if low <= thr and thr not in self._warned:
                self._warned.add(thr)
                self.warn(f"AirPods battery is at {low} percent.")
                return
        if low > 25:
            self._warned.clear()

    def _run(self) -> None:
        while not self._stop.is_set():
            self.check(airpods_battery())
            self._stop.wait(self.period)

    def stop(self) -> None:
        self._stop.set()
