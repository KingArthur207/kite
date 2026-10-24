"""The socket video reader on a synthetic Tello stream: PyAV-encoded H.264 sent as 1460-byte UDP datagrams to localhost.

The stream is built like the Tello's: 960x720, 30 fps, a keyframe (with SPS/PPS) every 30 frames, raw Annex B, every
frame sent as one back-to-back burst of datagrams. Frame i carries its number as 8 black/white bands across the top,
so a decoded frame can be checked against the frame that was sent.

    python tests/test_socket_stream.py     # burst comparison, socket vs pyav backend, prints frames decoded by each
"""

from __future__ import annotations

import random
from fractions import Fraction
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kite.sources.socket_stream import SocketStream, nal_types  # noqa: E402

W, H, FPS, GOP, N_FRAMES, DGRAM = 960, 720, 30, 30, 90, 1460
BANDS, BAND_H = 8, 64


def source_frame(i: int) -> np.ndarray:
    """Textured (so keyframes are Tello-sized, tens of KB) and numbered (top bands = i in binary)."""
    rng = np.random.default_rng(1234)
    base = getattr(source_frame, "_base", None)
    if base is None:
        small = rng.integers(0, 255, (H // 8, W // 8, 3), dtype=np.uint8)
        base = source_frame._base = np.kron(small, np.ones((8, 8, 1), np.uint8))
    img = np.roll(base, 4 * i, axis=1).copy()
    for b in range(BANDS):
        img[:BAND_H, b * W // BANDS:(b + 1) * W // BANDS] = 255 if (i >> b) & 1 else 0
    return img


def frame_number(img: np.ndarray) -> int:
    bits = [img[8:BAND_H - 8, b * W // BANDS + 16:(b + 1) * W // BANDS - 16].mean() > 128 for b in range(BANDS)]
    return sum(1 << b for b, on in enumerate(bits) if on)


def matches_source(img: np.ndarray) -> bool:
    """The decoded frame is clean: close to the source frame whose number it carries (no smeared loss damage)."""
    ref = source_frame(frame_number(img) % N_FRAMES)
    return float(np.abs(img.astype(np.int16) - ref.astype(np.int16)).mean()) < 12


def encode_stream(bit_rate: int = 3_000_000) -> list[bytes]:
    """N_FRAMES encoded frames, each the Annex B bytes of one access unit."""
    import av

    enc = av.CodecContext.create("libx264", "w")
    enc.width, enc.height, enc.pix_fmt = W, H, "yuv420p"
    enc.time_base, enc.framerate = Fraction(1, FPS), Fraction(FPS)
    enc.bit_rate = bit_rate
    enc.gop_size = GOP
    enc.options = {"preset": "ultrafast", "tune": "zerolatency", "keyint_min": str(GOP), "sc_threshold": "0"}
    out = []
    for i in range(N_FRAMES):
        f = av.VideoFrame.from_ndarray(source_frame(i)[:, :, ::-1].copy(), format="rgb24").reformat(format="yuv420p")
        f.pts = i
        out += [bytes(p) for p in enc.encode(f)]
    out += [bytes(p) for p in enc.encode(None)]
    assert len(out) == N_FRAMES
    return out


@pytest.fixture(scope="module")
def stream() -> list[bytes]:
    return encode_stream()


class Sender(threading.Thread):
    """Plays the encoded frames to 127.0.0.1:port in a loop: each frame one burst of 1460-byte datagrams.

    drop: probability of losing each datagram. burst_frames: frames sent back-to-back per burst (1 = the Tello's
    pacing; more = a stall-then-flush, like a Wi-Fi retry storm)."""

    def __init__(self, frames: list[bytes], port: int, drop: float = 0.0, burst_frames: int = 1, seed: int = 7):
        super().__init__(daemon=True)
        self.frames, self.port, self.drop, self.burst = frames, port, drop, burst_frames
        self.rng = random.Random(seed)
        self.stop_event = threading.Event()
        self.sent = self.dropped = self.frames_sent = 0

    def run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
        period, next_t, i = self.burst / FPS, time.monotonic(), 0
        while not self.stop_event.is_set():
            for _ in range(self.burst):
                au = self.frames[i % len(self.frames)]
                for k in range(0, len(au), DGRAM):
                    if self.drop and self.rng.random() < self.drop:
                        self.dropped += 1
                        continue
                    try:
                        sock.sendto(au[k:k + DGRAM], ("127.0.0.1", self.port))
                        self.sent += 1
                    except OSError:  # ENOBUFS when the burst outruns the loopback: a lost datagram, like Wi-Fi
                        self.dropped += 1
                i += 1
                self.frames_sent += 1
            next_t += period
            time.sleep(max(0.0, next_t - time.monotonic()))
        sock.close()

    def stop(self) -> None:
        self.stop_event.set()
        self.join(timeout=2)


def wait_frame(reader, timeout: float = 5.0):
    t_end = time.monotonic() + timeout
    while time.monotonic() < t_end:
        seq, img, _ = reader.read_seq(copy=False)
        if img is not None:
            return seq, img
        time.sleep(0.01)
    return -1, None


def test_synthetic_stream_is_tello_shaped(stream):
    assert stream[0].startswith(b"\x00\x00\x00\x01")
    for i in (0, GOP, 2 * GOP):  # SPS + PPS + IDR at every keyframe
        assert {7, 8, 5} <= set(nal_types(stream[i])), nal_types(stream[i])
    assert 5 not in nal_types(stream[1])
    assert len(stream[0]) > 20 * DGRAM  # a keyframe is a burst of tens of datagrams


def test_decodes_newest_frame_and_stops_cleanly(stream):
    port = 11191
    reader = SocketStream(f"udp://@0.0.0.0:{port}").start()
    tx = Sender(stream, port)
    tx.start()
    try:
        seq, img = wait_frame(reader)
        assert img is not None, reader.diagnose()
        assert img.shape == (H, W, 3) and matches_source(img)
        time.sleep(1.5)
        s1, img1, _ = reader.read_seq(copy=False)
        newest = (tx.frames_sent - 1) % N_FRAMES
        # newest-frame semantics: what we hold is (within a couple of frames) the last one sent, not a queued old one
        assert (newest - frame_number(img1)) % 256 <= 3, (newest, frame_number(img1))
        assert s1 > seq
        st = reader.stats()
        assert st["fps"] > 25 and st["decode_errors"] == 0, st
        assert st["first_frame_s"] is not None and st["rcvbuf_bytes"] >= 256 * 1024, st
        assert "ok" in reader.diagnose()
    finally:
        tx.stop()
        t0 = time.monotonic()
        reader.stop()
        assert time.monotonic() - t0 < 2.0
    assert not reader._rx.is_alive() and not reader._dec.is_alive()
    SocketStream(f"udp://@0.0.0.0:{port}").stop()  # the port is free again


def sample(reader, seconds: float) -> tuple[int, int]:
    """(new frames, smeared frames) published over `seconds`."""
    n = bad = 0
    last, t_end = reader.read_seq(copy=False)[0], time.monotonic() + seconds
    while time.monotonic() < t_end:
        seq, img, _ = reader.read_seq(copy=False)
        if img is not None and seq != last:
            last, n, bad = seq, n + 1, bad + (not matches_source(img))
        time.sleep(0.005)
    return n, bad


def test_recovers_at_the_next_keyframe_after_packet_loss(stream):
    port = 11192
    reader = SocketStream(f"udp://@0.0.0.0:{port}").start()
    tx = Sender(stream, port, drop=0.05)
    tx.start()
    try:
        assert wait_frame(reader)[1] is not None, reader.diagnose()
        n_lossy, bad_lossy = sample(reader, 2.0)
        assert tx.dropped > 0 and n_lossy > 0  # loss injected; the reader keeps going (no stall, no crash)
        # FFmpeg conceals lost datagrams without an error, so damaged frames get through until the next keyframe
        tx.drop = 0.0
        time.sleep(GOP / FPS + 0.3)  # the next keyframe has arrived
        n, bad = sample(reader, 1.0)
        assert n > 25 and bad == 0, (n, bad, bad_lossy, reader.stats())
    finally:
        tx.stop()
        reader.stop()


def test_diagnoses_silence_and_keyframe_less_streams(stream):
    port = 11193
    reader = SocketStream(f"udp://@0.0.0.0:{port}").start()
    try:
        time.sleep(0.3)
        msg = reader.diagnose()
        assert "no video packets" in msg and "Local Network" in msg and "lsof -nP -iUDP:11193" in msg
        tx = Sender(stream[1:GOP], port)  # P frames only: no SPS, no IDR
        tx.start()
        time.sleep(0.5)
        tx.stop()
        assert "no keyframe" in reader.diagnose(), reader.diagnose()
    finally:
        reader.stop()


def test_tello_source_socket_backend(stream):
    from kite.sources import TelloVideoSource

    port = 11189
    src = TelloVideoSource(url=f"udp://@0.0.0.0:{port}", backend="socket").start()
    tx = Sender(stream, port)
    tx.start()
    try:
        f = src.wait_first(5.0)
        assert f is not None and f.image.shape == (H, W, 3) and f.source == "tello" and matches_source(f.image)
        assert src.diagnosis().startswith("ok")
    finally:
        tx.stop()
        src.stop()
    assert src.read() is None
    with pytest.raises(ValueError, match="socket, pyav, opencv"):
        TelloVideoSource(backend="ffmpeg")


def test_a_held_port_names_lsof():
    blocker = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # another program on the port (no SO_REUSEADDR)
    blocker.bind(("0.0.0.0", 11190))
    try:
        with pytest.raises(RuntimeError, match="lsof -nP -iUDP:11190"):
            SocketStream("udp://@0.0.0.0:11190")
    finally:
        blocker.close()


# ------------------------------------------------------------------ socket vs pyav under bursts
def compare(backend: str, frames: list[bytes], port: int, burst_frames: int, seconds: float = 4.0) -> dict:
    """New frames one backend delivers over `seconds` after its first frame, and how many lost data smeared."""
    from kite.sources.pyav_stream import PyAVStream

    tx = Sender(frames, port, burst_frames=burst_frames)
    if backend == "socket":
        reader = SocketStream(f"udp://@0.0.0.0:{port}").start()
        tx.start()
    else:  # av.open probes the stream: it returns once packets flow, so the sender starts first
        tx.start()
        try:
            reader = PyAVStream(f"udp://@0.0.0.0:{port}", open_timeout_ms=8000, read_timeout_ms=8000).start()
        except RuntimeError as e:
            tx.stop()
            return {"backend": backend, "burst_frames": burst_frames, "error": str(e)}
    try:
        t0 = time.monotonic()
        if wait_frame(reader, 10.0)[1] is None:
            return {"backend": backend, "burst_frames": burst_frames, "first_frame_s": None, "delivered": 0}
        first, sent0 = time.monotonic() - t0, tx.frames_sent
        delivered, bad = sample(reader, seconds)
        out = {"backend": backend, "burst_frames": burst_frames, "first_frame_s": round(first, 2),
               "sent": tx.frames_sent - sent0, "delivered": delivered, "smeared": bad}
        if backend == "socket":
            out["frames_decoded"] = reader.stats()["frames_decoded"]
        return out
    finally:
        tx.stop()
        reader.stop()


@pytest.mark.slow
def test_socket_reader_keeps_up_with_bursts(stream):
    r = compare("socket", stream, 11194, burst_frames=15)
    assert r["smeared"] == 0 and r["delivered"] >= r["sent"] / 15, r  # newest-frame: >= 1 frame per burst, none damaged


def main() -> None:
    port = 11195
    for rate in (3_000_000, 12_000_000):
        frames = encode_stream(rate)
        print(f"stream {rate / 1e6:g} Mbit/s: keyframe {len(frames[0]) // 1024} KB ({-(-len(frames[0]) // DGRAM)} "
              f"datagrams), mean {sum(map(len, frames)) // len(frames) // 1024} KB/frame", flush=True)
        for burst in (1, 5, 30, 90):
            for backend in ("socket", "pyav"):
                print(compare(backend, frames, port, burst), flush=True)
                port += 1


if __name__ == "__main__":
    main()
