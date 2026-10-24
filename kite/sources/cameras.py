"""Which cameras this Mac has, with the index cv2.VideoCapture(index) opens and their names (no new dependencies).

    opencv_cameras()   [(0, "MacBook Pro Camera", "<uniqueID>"), (1, "Kite's iPhone Camera", ...)]
    system_cameras()   names AVFoundation's device discovery reports (what ffmpeg -f avfoundation lists)
    find_camera("iphone") -> (index, name), or (None, name) for a camera only PyAV/ffmpeg can open

OpenCV's macOS backend numbers cameras as [AVCaptureDevice devicesWithMediaType:Video] + (Muxed), sorted by uniqueID.
opencv_cameras() makes that same call through the Objective-C runtime (ctypes), so its indices are the ones OpenCV
opens. Listing never opens a camera. ffmpeg's discovery list (also Desk View, Continuity Camera, screens) is
numbered differently: never use its numbers with cv2.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import sys
from ctypes import c_char_p, c_long, c_ulong, c_void_p

AVFOUNDATION = "/System/Library/Frameworks/AVFoundation.framework/AVFoundation"
AUTH_STATUS = {0: "not determined (the first open asks)", 1: "restricted", 2: "denied", 3: "authorized"}


class _ObjC:
    def __init__(self):
        self.lib = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
        self.avf = ctypes.cdll.LoadLibrary(AVFOUNDATION)
        self.lib.objc_getClass.restype = c_void_p
        self.lib.objc_getClass.argtypes = [c_char_p]
        self.lib.sel_registerName.restype = c_void_p
        self.lib.sel_registerName.argtypes = [c_char_p]
        self._send = ctypes.cast(self.lib.objc_msgSend, c_void_p).value

    def send(self, obj, sel: str, *args, restype=c_void_p, argtypes=()):
        # arm64: objc_msgSend must be called through a pointer cast to the exact signature
        fn = ctypes.CFUNCTYPE(restype, c_void_p, c_void_p, *argtypes)(self._send)
        return fn(obj, self.lib.sel_registerName(sel.encode()), *args)

    def cls(self, name: str):
        return self.lib.objc_getClass(name.encode())

    def const(self, name: str):
        return c_void_p.in_dll(self.avf, name).value

    def text(self, nsstring) -> str:
        if not nsstring:
            return ""
        raw = self.send(nsstring, "UTF8String", restype=c_char_p)
        return raw.decode("utf-8", errors="replace") if raw else ""


def opencv_cameras() -> list[tuple[int, str, str]]:
    """(cv2 index, name, uniqueID) in OpenCV's order. [] off macOS or if the runtime call fails."""
    if sys.platform != "darwin":
        return []
    try:
        oc = _ObjC()
        dev_cls = oc.cls("AVCaptureDevice")
        found = []
        for media in ("AVMediaTypeVideo", "AVMediaTypeMuxed"):
            arr = oc.send(dev_cls, "devicesWithMediaType:", oc.const(media), argtypes=[c_void_p])
            for i in range(oc.send(arr, "count", restype=c_ulong) if arr else 0):
                d = oc.send(arr, "objectAtIndex:", i, argtypes=[c_ulong])
                found.append((oc.text(oc.send(d, "localizedName")), oc.text(oc.send(d, "uniqueID"))))
    except (OSError, ValueError, AttributeError, TypeError):
        return []
    found.sort(key=lambda d: d[1])  # OpenCV: [d1.uniqueID compare:d2.uniqueID] (IDs are ASCII: same order)
    return [(i, name, uid) for i, (name, uid) in enumerate(found)]


def camera_permission() -> str | None:
    """This process's camera authorization (macOS), or None if unknown."""
    if sys.platform != "darwin":
        return None
    try:
        oc = _ObjC()
        st = oc.send(oc.cls("AVCaptureDevice"), "authorizationStatusForMediaType:", oc.const("AVMediaTypeVideo"),
                     restype=c_long, argtypes=[c_void_p])
    except (OSError, ValueError, AttributeError, TypeError):
        return None
    return AUTH_STATUS.get(st, f"status {st}")


def system_cameras() -> list[str]:
    """Video device names from FFmpeg's avfoundation device list (through PyAV, no ffmpeg binary). Screens excluded."""
    if sys.platform != "darwin":
        return []
    try:
        import av
        import av.logging
    except ImportError:
        return []
    names: list[str] = []
    prev = av.logging.get_level()
    av.logging.set_level(av.logging.INFO)  # the list is logged at INFO
    try:
        with av.logging.Capture(True) as logs:
            try:
                av.open("", format="avfoundation", options={"list_devices": "true"})
            except Exception:  # listing always ends with "Input/output error"
                pass
    finally:
        av.logging.set_level(prev)
    video = False
    for _level, _name, line in logs:
        line = line.strip()
        if "video devices" in line:
            video = True
        elif "audio devices" in line:
            video = False
        elif video and line.startswith("[") and "]" in line:
            name = line.split("]", 1)[1].strip()
            if name and not name.startswith("Capture screen"):
                names.append(name)
    return names


def _match(query: str, names: list[str]) -> str | None:
    q = query.lower()
    exact = [n for n in names if n.lower() == q]
    if exact:
        return exact[0]
    part = [n for n in names if q in n.lower()]
    return part[0] if part else None


def find_camera(query: str) -> tuple[int | None, str]:
    """Camera whose name contains `query` (case-insensitive; an exact name wins). Returns (cv2 index, name), or
    (None, name) when only AVFoundation's discovery list has it (OpenCV cannot open it: PyAV can, by name).
    LookupError lists the cameras when nothing matches."""
    cv = opencv_cameras()
    name = _match(query, [n for _, n, _ in cv])
    if name is not None:
        return next(i for i, n, _ in cv if n == name), name
    sysn = system_cameras()
    name = _match(query, sysn)
    if name is not None:
        return None, name
    have = sorted({n for _, n, _ in cv} | set(sysn))
    raise LookupError(f"no camera matching {query!r}; cameras: {', '.join(have) or 'none found'}"
                      " (Continuity Camera: iPhone near the Mac, same Apple ID, Wi-Fi and Bluetooth on on both)")
