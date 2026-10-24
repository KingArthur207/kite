"""Tello H.264 reader that owns its UDP socket: no FFmpeg udp:// protocol, no demuxer probing, no FFmpeg timeouts.

Same interface as video_stream.VideoStream and pyav_stream.PyAVStream (constructor(url_or_port, fps=,
open_timeout_ms=, read_timeout_ms=), start(), read_seq(copy=), stop()), plus stats(). Backend "raw", and
"auto" on macOS (opt-in until it has run against a real Tello; the default is pyav): on Arthur's Mac both FFmpeg-based readers received the Tello's
packets but never delivered a frame (h264 "non-existing PPS", "Stream timeout triggered after 5015 ms").

    receiver thread   recv() datagrams into a bounded queue, nothing else (never waits on decoding). The socket asks
                      for an 8 MB SO_RCVBUF; macOS grants up to kern.ipc.maxsockbuf (8 MB by default, ~7.4 MB
                      usable), Linux up to net.core.rmem_max, Windows normally all of it (stats()["rcvbuf"] and the
                      kite.tello_raw log line say what was granted). Works the same on Windows: plain UDP, and a
                      stray ICMP "port unreachable" (WSAECONNRESET) is skipped instead of ending the receiver.
    decoder thread    raw Annex B bytes -> av.CodecContext("h264").parse() -> decode(). Bytes before the first SPS
                      (joining mid-stream) are skipped, so the decoder only ever starts on SPS/PPS + IDR. Every frame
                      is decoded (P-frames need their predecessors), at most `fps` per second are converted to BGR,
                      and only the newest one is kept: latency never builds up.

Loss: the Tello's datagrams carry no sequence numbers, so a lost one shows up as a broken frame. FFmpeg conceals it
and the stream heals at the next IDR (frames keep flowing: the safety governor hovers after 1 s without one).
After a hard failure (decode errors with no frame at all for NO_FRAME_RESYNC_S while data keeps arriving, a queue
overflow, or a gap of GAP_RESYNC_S without data) the decoder is rebuilt and waits for the next SPS. Errors alone never
reset it: FFmpeg recovers by itself, and a reset throws away everything until the next SPS.

The constructor binds the socket but never waits for a frame (TelloVideoSource does the waiting and reconnects);
stop() frees the port within ~0.2 s. Tests pass `feed=`: an iterable of byte chunks (datagrams) instead of a socket.
If the 8 MB buffer is not granted and stats() shows queue drops, raise the limit: sudo sysctl -w kern.ipc.maxsockbuf=16777216
"""

from __future__ import annotations

import logging
import queue
import re
import socket
import threading
import time
from collections.abc import Iterable

import av

log = logging.getLogger("kite.tello_raw")
RCVBUF_BYTES = 8 * 1024 * 1024  # ~20 s of Tello video; macOS's default kern.ipc.maxsockbuf is 8 MB
QUEUE_DATAGRAMS = 8192  # ~12 MB of 1460-byte datagrams between the receiver and the decoder
GAP_RESYNC_S = 2.0  # no data this long: the decoder's references are stale, wait for a fresh SPS + IDR
MAX_ERRORS_IN_A_ROW = 50  # decode errors without a single frame ...
NO_FRAME_RESYNC_S = 3.0  # ... AND no frame for this long: only then rebuild the decoder and wait for an SPS
_EOF, _RESYNC = object(), object()  # queue markers
LINK_GAP_S = 0.2  # a pause between datagrams longer than this is a link stall (the Tello sends ~200+ per second)
LINK_STARVED_KB_S = 60  # under this the stream is starving (a healthy Tello sends well over 100 KB/s even at 1 Mbps)
def _flag(enum, name: str) -> int:
    """A codec flag by name across PyAV versions: lower case (low_delay) in some, upper case (LOW_DELAY) in others."""
    for n in (name, name.upper()):
        if hasattr(enum, n):
            return int(getattr(enum, n))
    raise AttributeError(f"PyAV {av.__version__}: {enum.__name__} has no {name} or {name.upper()}")


HOLD_MAX_S = 3.0  # drop_corrupt never freezes the picture longer than this
LINK_POOR_LOSS, LINK_FAIR_LOSS = 0.2, 0.05  # fraction of decoded frames FFmpeg flags as concealed after loss
LINK_HELP = "docs/internal/windows-drone.md, 'If the drone video is corrupted'"


def parse_port(url_or_port) -> tuple[str, int]:
    """'udp://@0.0.0.0:11111', 'udp://127.0.0.1:11111?x=y', '11111' or 11111 -> (bind host, port)."""
    if isinstance(url_or_port, int):
        return "0.0.0.0", url_or_port
    s = str(url_or_port).strip()
    if s.isdigit():
        return "0.0.0.0", int(s)
    m = re.fullmatch(r"udp://@?([^:/?]*):(\d+)(?:[/?].*)?", s)
    if not m:
        raise ValueError(f"not a UDP video address: {url_or_port!r} (use udp://@0.0.0.0:11111 or a port number)")
    return m.group(1) or "0.0.0.0", int(m.group(2))


def find_sps(data: bytes | bytearray, start: int = 0) -> int:
    """Offset of the first Annex B start code (00 00 01) that opens an SPS NAL (type 7), or -1."""
    i = start
    while True:
        i = data.find(b"\x00\x00\x01", i)
        if i < 0 or i + 3 >= len(data):
            return -1
        if data[i + 3] & 0x1F == 7:
            return i
        i += 3


class RawTelloStream:
    def __init__(self, url_or_port=11111, fps: float = 60, open_timeout_ms: int = 15000, read_timeout_ms: int = 15000,
                 rcvbuf: int = RCVBUF_BYTES, feed: Iterable[bytes] | None = None, queue_datagrams: int = QUEUE_DATAGRAMS,
                 drop_corrupt: bool = False):
        """Binds the video port (RuntimeError if another program holds it); does NOT wait for video.

        open_timeout_ms / read_timeout_ms are accepted for interface compatibility: nothing here gives up on a
        quiet socket (TelloVideoSource decides when to reconnect). fps caps the BGR conversions per second, with
        a burst of 2 so arrival jitter never drops frames of a stream running at the cap.
        feed: test input (byte chunks, decoded as fast as possible, nothing dropped) instead of a socket.
        drop_corrupt: after a frame FFmpeg flags as concealed (packet loss), hold the last clean frame until the next
            clean keyframe (at most HOLD_MAX_S), instead of delivering smeared frames.
        """
        self._interval = 1.0 / fps
        self._drop_corrupt = drop_corrupt
        self._held_since: float | None = None  # drop_corrupt: holding since this time, until a clean keyframe
        self._feed = feed
        self._sock = None
        self._q: queue.Queue = queue.Queue(maxsize=queue_datagrams)
        self._frame = None
        self._seq = 0
        self._ts = 0.0
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        self.finished = False  # feed mode: every chunk decoded
        self._t_start = None
        self._st = {"datagrams": 0, "bytes": 0, "frames_decoded": 0, "frames": 0, "corrupt_frames": 0,
                    "decode_errors": 0, "queue_drops": 0, "skipped_bytes": 0, "resyncs": 0, "keyframe_seen": False,
                    "time_to_first_datagram_s": None, "time_to_first_frame_s": None, "size": None,
                    "rcvbuf_requested": rcvbuf, "rcvbuf": None, "port": None, "last_error": None,
                    "max_gap_s": 0.0, "gaps_over_0.2s": 0}
        self._first_rx = 0.0
        self._last_rx = 0.0
        if feed is None:
            host, port = parse_port(url_or_port)
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                if rcvbuf:
                    try:
                        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf)
                    except OSError:  # above the OS maximum (macOS refuses instead of clamping): take what we get
                        for size in (rcvbuf // 2, rcvbuf // 4, rcvbuf // 8):
                            try:
                                sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, size)
                                break
                            except OSError:
                                continue
                sock.bind((host, port))
            except OSError as e:
                sock.close()
                raise RuntimeError(f"cannot bind UDP {host}:{port} for the Tello video ({e}): another program holds "
                                   f"it (macOS/Linux: lsof -nP -iUDP:{port}; Windows: netstat -ano -p udp | "
                                   f"findstr :{port})") from None
            sock.settimeout(0.1)  # the receiver checks the stop flag this often
            self._sock = sock
            self._st["rcvbuf"] = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
            self._st["port"] = sock.getsockname()[1]
            log.info("video socket UDP %d: SO_RCVBUF %d bytes granted (asked %d)", self._st["port"],
                     self._st["rcvbuf"], rcvbuf)
        self._rx = threading.Thread(target=self._receive if feed is None else self._feed_run, name="tello-raw-rx",
                                    daemon=True)
        self._dec = threading.Thread(target=self._decode_run, name="tello-raw-decode", daemon=True)

    # ------------------------------------------------------------------ public
    def start(self) -> RawTelloStream:
        if self._t_start is None:
            self._t_start = time.monotonic()
            self._dec.start()
            self._rx.start()
        return self

    def read_seq(self, copy: bool = True):
        """(seq, frame, timestamp) like VideoStream.read_seq; (-1, None, 0.0) before the first frame."""
        with self._lock:
            if self._frame is None:
                return -1, None, 0.0
            return self._seq, (self._frame.copy() if copy else self._frame), self._ts

    def stats(self) -> dict:
        """Counters since start(): datagrams / bytes received, frames_decoded (all), frames (converted to BGR),
        corrupt_frames (concealed after loss), decode_errors (+ last_error), queue_drops, skipped_bytes (waiting
        for an SPS), resyncs, time_to_first_datagram_s / time_to_first_frame_s (from start()), size (w, h), rcvbuf
        granted, last_datagram_age_s, queued."""
        with self._lock:
            out = dict(self._st)
        out["last_datagram_age_s"] = round(time.monotonic() - self._last_rx, 3) if self._last_rx else None
        out["queued"] = self._q.qsize()
        out["receiving_s"] = round(self._last_rx - self._first_rx, 3) if self._first_rx else 0.0
        return out

    def stop(self) -> None:
        self._stopped.set()
        if self._rx.is_alive():
            self._rx.join(timeout=1.0)  # recv() times out every 0.1 s
        if self._sock is not None:
            self._sock.close()  # the port is free again
        try:
            self._q.put_nowait(_EOF)  # wake the decoder at once
        except queue.Full:
            pass
        if self._dec.is_alive():
            self._dec.join(timeout=2.0)

    def __enter__(self) -> RawTelloStream:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # ------------------------------------------------------------------ receiver
    def _receive(self) -> None:
        buf = bytearray(65536)
        view = memoryview(buf)
        sock, q, st = self._sock, self._q, self._st
        while not self._stopped.is_set():
            try:
                n = sock.recv_into(buf)
            except socket.timeout:
                continue
            except ConnectionResetError:  # Windows: an ICMP "port unreachable" for this port; not fatal
                continue
            except OSError:
                return  # socket closed under us
            if not n:
                continue
            now = time.monotonic()
            data = bytes(view[:n])
            if st["time_to_first_datagram_s"] is None:
                st["time_to_first_datagram_s"] = round(now - self._t_start, 3)
            st["datagrams"] += 1
            st["bytes"] += n
            self._gap(now)
            try:
                q.put_nowait(data)
            except queue.Full:  # the decoder is far behind: drop the backlog, it restarts at the next SPS
                dropped = 0
                while True:
                    try:
                        q.get_nowait()
                        dropped += 1
                    except queue.Empty:
                        break
                st["queue_drops"] += dropped + 1
                q.put_nowait(_RESYNC)

    def _gap(self, now: float) -> None:
        """Note a datagram's arrival: the longest pause between two and how many exceeded LINK_GAP_S."""
        st = self._st
        if self._last_rx:
            gap = now - self._last_rx
            st["max_gap_s"] = round(max(st["max_gap_s"], gap), 3)
            if gap > LINK_GAP_S:
                st["gaps_over_0.2s"] += 1
        else:
            self._first_rx = now
        self._last_rx = now

    def _feed_run(self) -> None:
        st = self._st
        try:
            for chunk in self._feed:
                if self._stopped.is_set():
                    return
                if not chunk:
                    continue
                if st["time_to_first_datagram_s"] is None:
                    st["time_to_first_datagram_s"] = round(time.monotonic() - self._t_start, 3)
                st["datagrams"] += 1
                st["bytes"] += len(chunk)
                self._gap(time.monotonic())
                while not self._stopped.is_set():  # tests: never drop, wait for the decoder
                    try:
                        self._q.put(bytes(chunk), timeout=0.1)
                        break
                    except queue.Full:
                        continue
        finally:
            while not self._stopped.is_set():
                try:
                    self._q.put(_EOF, timeout=0.1)
                    break
                except queue.Full:
                    continue

    # ------------------------------------------------------------------ decoder
    def _new_codec(self):
        ctx = av.CodecContext.create("h264", "r")
        ctx.flags |= _flag(av.codec.context.Flags, "low_delay") | _flag(av.codec.context.Flags, "output_corrupt")
        # show_all: output frames from the first SPS/PPS on, before a full IDR has arrived (they fill in as intra
        # data comes). If the drone sends IDRs seconds apart, this gives a picture in ~0.6 s instead of one IDR
        # interval later (docs/mac-video-root-cause.md); stats()["keyframe_seen"] says when it is fully refreshed.
        ctx.flags2 |= _flag(av.codec.context.Flags2, "show_all")
        ctx.thread_type = "SLICE"  # frame threading would hold frames back (latency); slices do not
        return ctx

    def _publish(self, f) -> None:
        """BGR-convert one decoded frame and make it the newest."""
        st = self._st
        try:
            img = f.to_ndarray(format="bgr24")
        except Exception as e:
            st["decode_errors"] += 1
            st["last_error"] = f"to_ndarray: {e}"
            return
        with self._lock:
            self._frame, self._seq, self._ts = img, self._seq + 1, time.time()
            st["frames"] += 1
            if st["time_to_first_frame_s"] is None:
                st["time_to_first_frame_s"] = round(time.monotonic() - self._t_start, 3)
                st["size"] = (img.shape[1], img.shape[0])

    def _decode_run(self) -> None:
        st, q = self._st, self._q
        ctx = self._new_codec()
        synced = False
        pending = b""  # tail kept while looking for an SPS start code split across datagrams
        errors_in_a_row = 0
        last_frame_t = time.monotonic()  # last decoded frame (or sync), for the reset rule
        tokens, last_t = 2.0, time.monotonic()  # conversion budget: fps per second, burst of 2 (arrival jitter)

        def resync():
            nonlocal ctx, synced, pending, errors_in_a_row
            ctx = self._new_codec()
            synced, pending, errors_in_a_row = False, b"", 0
            st["resyncs"] += 1
            st["keyframe_seen"] = False

        eof = False
        while not eof and not self._stopped.is_set():
            try:
                item = q.get(timeout=0.1)
            except queue.Empty:
                if synced and self._last_rx and time.monotonic() - self._last_rx > GAP_RESYNC_S:
                    resync()  # long silence (Wi-Fi drop, streamoff): restart cleanly on the next SPS
                continue
            chunks = []
            while True:  # take everything queued: one parse() call per batch
                if item is _EOF:
                    eof = True
                    break
                if item is _RESYNC:
                    resync()
                    chunks = []
                else:
                    chunks.append(item)
                try:
                    item = q.get_nowait()
                except queue.Empty:
                    break
            data = b"".join(chunks)
            if not synced and data:
                data = pending + data
                i = find_sps(data)
                if i < 0:
                    keep = min(len(data), 3)  # a start code may straddle the next chunk
                    st["skipped_bytes"] += len(data) - keep
                    pending = data[-keep:] if keep else b""
                    data = b""
                else:
                    st["skipped_bytes"] += i
                    data, pending, synced = data[i:], b"", True
            packets = []
            try:
                if data:
                    packets = ctx.parse(data)
                if eof and synced:
                    packets += ctx.parse(None)  # flush the parser's last frame (feed mode)
            except Exception as e:  # FFmpegError on broken data; anything else must not kill the reader either
                st["decode_errors"] += 1
                st["last_error"] = f"parse: {e}"
                errors_in_a_row += 1
            newest = None
            for pkt in packets:
                try:
                    frames = ctx.decode(pkt)  # every frame is decoded: the next ones are predicted from it
                except Exception as e:
                    st["decode_errors"] += 1
                    st["last_error"] = f"decode: {e}"
                    errors_in_a_row += 1
                    continue
                for f in frames:
                    errors_in_a_row = 0
                    last_frame_t = time.monotonic()
                    st["frames_decoded"] += 1
                    if f.key_frame:
                        st["keyframe_seen"] = True
                    if f.is_corrupt:
                        st["corrupt_frames"] += 1
                        if self._drop_corrupt and self._held_since is None:
                            self._held_since = time.monotonic()
                    if self._held_since is not None:
                        # Later P-frames carry the damage forward without being flagged: hold the last clean picture
                        # until the next keyframe, but never longer than HOLD_MAX_S.
                        if f.key_frame and not f.is_corrupt or time.monotonic() - self._held_since > HOLD_MAX_S:
                            self._held_since = None
                        else:
                            st["held_frames"] = st.get("held_frames", 0) + 1
                            continue
                    if self._feed is not None:
                        self._publish(f)  # tests / offline: every frame
                    else:
                        newest = f  # live: only the newest of a batch is worth converting
            if newest is not None:
                now = time.monotonic()
                tokens = min(2.0, tokens + (now - last_t) / self._interval)
                last_t = now
                if tokens >= 1.0:  # under the fps cap
                    tokens -= 1.0
                    self._publish(newest)
            if errors_in_a_row >= MAX_ERRORS_IN_A_ROW and time.monotonic() - last_frame_t > NO_FRAME_RESYNC_S:
                resync()
                last_frame_t = time.monotonic()
        self.finished = eof


def rates(prev: dict, cur: dict, dt: float) -> dict:
    """Per-second rates between two stats() snapshots, for live displays (window HUD, terminal lines)."""
    dt = max(dt, 1e-6)

    def d(k):
        return (cur.get(k) or 0) - (prev.get(k) or 0)

    return {"packets_s": d("datagrams") / dt, "kb_s": d("bytes") / dt / 1024, "decoded_fps": d("frames_decoded") / dt,
            "shown_fps": d("frames") / dt, "corrupt_s": d("corrupt_frames") / dt, "errors_s": d("decode_errors") / dt,
            "resyncs": cur.get("resyncs", 0), "queue_drops": cur.get("queue_drops", 0),
            "keyframe_seen": cur.get("keyframe_seen"), "last_error": cur.get("last_error")}


def format_rates(r: dict) -> str:
    """One line, e.g. '280 pkt/s 190 KB/s | decoded 29.8 fps, shown 29.8 | corrupt 0.0/s errors 0.0/s | resyncs 0'."""
    line = (f"{r['packets_s']:.0f} pkt/s {r['kb_s']:.0f} KB/s | decoded {r['decoded_fps']:.1f} fps, shown "
            f"{r['shown_fps']:.1f} | corrupt {r['corrupt_s']:.1f}/s errors {r['errors_s']:.1f}/s | resyncs {r['resyncs']}"
            f" qdrops {r['queue_drops']}")
    if not r.get("keyframe_seen"):
        line += " | no full keyframe yet"
    return line


def link_verdict(st: dict) -> tuple[str, str]:
    """(grade, one line) on the radio link from a stats()-shaped dict: GOOD / FAIR / POOR / NONE. Packet rate and
    gaps come from the socket; loss is estimated from the H.264 stream itself (the Tello's datagrams carry no
    sequence numbers): the share of decoded frames FFmpeg had to conceal because slices were missing."""
    n = st.get("datagrams") or 0
    if not n:
        return "NONE", "link NONE: no video packet arrived"
    sec = max(st.get("receiving_s") or 0.0, 1e-3)
    kb_s = (st.get("bytes") or 0) / sec / 1024
    decoded, corrupt = st.get("frames_decoded") or 0, st.get("corrupt_frames") or 0
    loss = corrupt / decoded if decoded else None
    gaps = st.get("gaps_over_0.2s") or 0
    if kb_s < LINK_STARVED_KB_S or loss is None or loss > LINK_POOR_LOSS or gaps > 2:
        grade = "POOR"
    elif loss > LINK_FAIR_LOSS or gaps:
        grade = "FAIR"
    else:
        grade = "GOOD"
    line = (f"link {grade}: {n / sec:.0f} pkt/s, {kb_s:.0f} KB/s, max gap {st.get('max_gap_s') or 0:.2f} s "
            f"({gaps} over {LINK_GAP_S:g} s), "
            + (f"~{loss:.0%} of frames damaged by packet loss ({corrupt}/{decoded})" if decoded else "no frame decoded"))
    if grade != "GOOD":
        line += f"; see {LINK_HELP}"
    return grade, line
