"""Socket reader for the Tello's UDP H.264: our own UDP socket, PyAV's parser + decoder, no FFmpeg network code.

Same interface as pyav_stream.PyAVStream (start / read_seq / stop). Why it exists: on the team's Macs the FFmpeg
udp:// readers showed "no image". The Tello sends each frame as a burst of 1460-byte datagrams; if the receive
buffer overflows mid-burst, the keyframe (IDR) never arrives whole and nothing ever decodes. Here:

  * the socket asks for a big SO_RCVBUF (16 MB, stepping down until the OS accepts; macOS caps it near
    `sysctl kern.ipc.maxsockbuf`) and reports what it got,
  * one thread does nothing but drain the socket into a queue,
  * another feeds the bytes to av.CodecContext.parse() and decodes. Bytes before the first SPS are dropped; no frame
    is published before a keyframe, nor after a decode error (or a frame FFmpeg flags corrupt) until the next
    keyframe. Most lost datagrams raise nothing though: FFmpeg conceals them, so a smeared frame can get through
    until the next keyframe, as with the other backends. Only the newest frame is kept, converted to BGR at most
    `fps` times per second.

stats() and diagnose() tell "no packets at all" (permissions, firewall, the port held, no streamon) from "packets but
no keyframe" from "keyframes that don't decode" (loss).
"""

from __future__ import annotations

import logging
import queue
import socket
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit

import av

log = logging.getLogger("kite.video")

RCVBUF_ASK = 16 * 1024 * 1024
RCVBUF_MIN = 256 * 1024
NAL_SPS, NAL_IDR = 7, 5

MAC_CHECKLIST = ("macOS: System Settings > Privacy & Security > Local Network must allow your terminal app; "
                 "the firewall (System Settings > Network > Firewall) must allow incoming connections for python; "
                 "quit the Tello app; `lsof -nP -iUDP:{port}` must list only this program")


def _bind(host: str, port: int) -> tuple[socket.socket, int, int]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ask = RCVBUF_ASK
    while ask >= RCVBUF_MIN:  # macOS refuses (ENOBUFS) anything above its limit instead of clamping
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, ask)
            break
        except OSError:
            ask = ask * 3 // 4
    try:
        sock.bind((host or "0.0.0.0", port))
    except OSError as e:
        sock.close()
        raise RuntimeError(f"cannot bind UDP {port} ({e}): another program holds it (the Tello app, another kite "
                           f"script). Find it with: lsof -nP -iUDP:{port}") from None
    sock.settimeout(0.2)
    return sock, ask, sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)


def _maxsockbuf() -> int | None:
    if sys.platform != "darwin":
        return None
    try:
        return int(subprocess.run(["sysctl", "-n", "kern.ipc.maxsockbuf"], capture_output=True, text=True,
                                  timeout=2).stdout)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def nal_types(data: bytes) -> list[int]:
    """NAL unit types after every Annex B start code in data."""
    out, i = [], data.find(b"\x00\x00\x01")
    while 0 <= i < len(data) - 3:
        out.append(data[i + 3] & 0x1F)
        i = data.find(b"\x00\x00\x01", i + 3)
    return out


class SocketStream:
    def __init__(self, url: str, fps: int = 60, open_timeout_ms: int = 15000, read_timeout_ms: int = 15000):
        u = urlsplit(url)
        self.port = u.port or 11111
        self._sock, self.rcvbuf_asked, self.rcvbuf = _bind(u.hostname or "0.0.0.0", self.port)
        self.maxsockbuf = _maxsockbuf()
        log.info("video socket UDP %d: SO_RCVBUF %d bytes (asked %d)%s", self.port, self.rcvbuf, self.rcvbuf_asked,
                 f", macOS kern.ipc.maxsockbuf {self.maxsockbuf}" if self.maxsockbuf else "")
        self._interval = 1 / fps
        self._warn_after = max(open_timeout_ms, read_timeout_ms) / 1000.0
        # ponytail: unbounded; the decoder drains it whole every pass, so it only grows if decoding falls behind 30 fps
        self._q: queue.SimpleQueue[bytes] = queue.SimpleQueue()
        self._frame = None
        self._seq = 0
        self._ts = 0.0
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        self._t0 = time.monotonic()
        self.packets = self.bytes = self.max_queue = 0
        self.sps_seen = self.idr_seen = self.decode_errors = self.skipped_frames = self.frames_decoded = 0
        self.first_frame_s: float | None = None
        self._last_packet_t = 0.0
        self._rx = threading.Thread(target=self._receive, daemon=True, name="kite-video-rx")
        self._dec = threading.Thread(target=self._decode, daemon=True, name="kite-video-decode")

    def start(self) -> SocketStream:
        self._rx.start()
        self._dec.start()
        return self

    def _receive(self) -> None:
        recv, put = self._sock.recv, self._q.put
        while not self._stopped.is_set():
            try:
                data = recv(65536)
            except socket.timeout:
                continue
            except OSError:
                return
            put(data)
            self.packets += 1
            self.bytes += len(data)
            self._last_packet_t = time.monotonic()

    def _decode(self) -> None:
        codec = av.CodecContext.create("h264", "r")
        synced = False  # an SPS has been seen: bytes before it are useless to the decoder
        wait_key = True  # publish nothing until a keyframe decodes cleanly
        tail = b""  # last bytes of the previous chunk: a start code may straddle two datagrams
        pending, last_convert, last_warn = None, 0.0, time.monotonic()
        while not self._stopped.is_set():
            timeout = 0.2 if pending is None else max(0.0, last_convert + self._interval - time.monotonic())
            try:
                chunks = [self._q.get(timeout=timeout)]
            except queue.Empty:
                chunks = []
            while True:  # take everything already queued: one parse call per burst
                try:
                    chunks.append(self._q.get_nowait())
                except queue.Empty:
                    break
            self.max_queue = max(self.max_queue, len(chunks))
            if chunks:
                data = b"".join(chunks)
                scan = tail + data  # the scan sees the carry-over; the parser gets each byte once
                tail = scan[-3:]
                types = nal_types(scan)
                self.sps_seen += types.count(NAL_SPS)
                self.idr_seen += types.count(NAL_IDR)
                if not synced:
                    if NAL_SPS not in types:
                        continue
                    synced, data = True, scan[self._sps_offset(scan):]  # nothing fed yet: the tail is new too
                pending, wait_key = self._feed(codec, data, pending, wait_key)
            if pending is not None and time.monotonic() - last_convert >= self._interval:
                img = pending.to_ndarray(format="bgr24")
                now = time.time()
                with self._lock:
                    self._frame, self._seq, self._ts = img, self._seq + 1, now
                if self.first_frame_s is None:
                    self.first_frame_s = time.monotonic() - self._t0
                pending, last_convert = None, time.monotonic()
            if self._seq == 0 and time.monotonic() - last_warn > self._warn_after:
                last_warn = time.monotonic()
                log.warning("no video frame yet: %s", self.diagnose())

    @staticmethod
    def _sps_offset(data: bytes) -> int:
        i = data.find(b"\x00\x00\x01")
        while i >= 0 and data[i + 3] & 0x1F != NAL_SPS:
            i = data.find(b"\x00\x00\x01", i + 3)
        return i - 1 if i > 0 and data[i - 1] == 0 else i  # keep the 4-byte start code whole

    def _feed(self, codec, data: bytes, pending, wait_key: bool):
        try:
            packets = codec.parse(data)
        except av.error.FFmpegError:
            self.decode_errors += 1
            return pending, True
        for pkt in packets:
            try:
                frames = codec.decode(pkt)
            except av.error.FFmpegError:
                self.decode_errors += 1  # lost datagrams: wait for the next keyframe
                wait_key = True
                continue
            for f in frames:
                self.frames_decoded += 1
                if f.is_corrupt:
                    wait_key = True
                if wait_key and not (f.key_frame and not f.is_corrupt):
                    self.skipped_frames += 1
                    continue
                wait_key = False
                pending = f
        return pending, wait_key

    def stats(self) -> dict:
        now = time.monotonic()
        live_s = now - self._t0 - (self.first_frame_s or 0.0)
        return {"port": self.port, "rcvbuf_bytes": self.rcvbuf, "rcvbuf_asked": self.rcvbuf_asked,
                "maxsockbuf": self.maxsockbuf, "packets": self.packets, "bytes": self.bytes,
                "sps": self.sps_seen, "idr": self.idr_seen, "frames_decoded": self.frames_decoded,
                "frames_published": self._seq, "skipped_frames": self.skipped_frames,
                "decode_errors": self.decode_errors, "max_queue": self.max_queue,
                "first_frame_s": None if self.first_frame_s is None else round(self.first_frame_s, 3),
                "fps": round(self._seq / live_s, 1) if self.first_frame_s is not None and live_s > 0 else 0.0,
                "since_last_packet_s": round(now - self._last_packet_t, 2) if self.packets else None}

    def diagnose(self) -> str:
        s, waited = self.stats(), time.monotonic() - self._t0
        if not s["packets"]:
            return (f"no video packets reached UDP {self.port} in {waited:.0f} s: the drone is not streaming to this "
                    f"laptop (streamon not sent or ignored), or the OS drops them. " + MAC_CHECKLIST.format(port=self.port))
        if s["frames_published"]:
            return (f"ok: {s['frames_published']} frames, {s['decode_errors']} decode errors, "
                    f"SO_RCVBUF {self.rcvbuf} bytes")
        if not s["idr"]:
            return (f"{s['packets']} packets ({s['bytes'] // 1024} KB) arrived but no keyframe (SPS {s['sps']}, "
                    f"IDR 0): wait longer, or keyframes are being lost (SO_RCVBUF {self.rcvbuf} bytes)")
        return (f"{s['idr']} keyframes arrived but none decoded cleanly ({s['decode_errors']} decode errors, "
                f"{s['skipped_frames']} damaged frames): datagrams are being lost (SO_RCVBUF {self.rcvbuf} bytes, "
                "weak Wi-Fi?)")

    def read_seq(self, copy: bool = True):
        with self._lock:
            if self._frame is None:
                return -1, None, 0.0
            return self._seq, (self._frame.copy() if copy else self._frame), self._ts

    def stop(self) -> None:
        self._stopped.set()
        for t in (self._rx, self._dec):
            if t.is_alive():
                t.join(timeout=2)
        self._sock.close()
