"""kite.sources.tello_raw.RawTelloStream (the macOS default Tello reader) and the video backend profile.

The stand-in Tello: ffmpeg/libx264 encodes 960x720 yuv420p H.264 (Annex B, SPS/PPS before every IDR, no B-frames,
one slice per frame), and a sender thread plays it over loopback UDP like the drone: each frame cut into 1460-byte
datagrams sent as one burst, 30 frames/s, looping, started BEFORE the reader (it joins mid-stream), optionally
dropping datagrams. The fast tests stay under ~2 s; the longer ones are marked slow.
"""

from __future__ import annotations

import glob
import importlib.util
import os
import random
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from kite.config import VIDEO_BACKENDS, Config, DroneCfg, resolve_video_backend
from kite.sources.tello_raw import RawTelloStream, find_sps, parse_port

ROOT = Path(__file__).resolve().parent.parent
FPS = 30
DATAGRAM = 1460
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


# ------------------------------------------------------------------ the stand-in Tello
_STREAMS: dict[int, bytes] = {}


def h264_stream(keyint: int, seconds: float = 4.0) -> bytes:
    """960x720 30 fps H.264 Annex B with an IDR (preceded by SPS + PPS) every `keyint` frames, ~3 Mbit/s."""
    if keyint not in _STREAMS:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", f"testsrc2=size=960x720:rate={FPS}",
             "-t", str(seconds), "-vf", "noise=alls=6:allf=t", "-c:v", "libx264", "-preset", "ultrafast",
             "-pix_fmt", "yuv420p", "-bf", "0", "-b:v", "3M", "-maxrate", "3M", "-bufsize", "1M",
             "-x264-params", f"keyint={keyint}:min-keyint={keyint}:scenecut=0:repeat-headers=1:sliced-threads=0",
             "-f", "h264", "-"], capture_output=True, check=True, timeout=60)
        _STREAMS[keyint] = out.stdout
    return _STREAMS[keyint]


def nals(data: bytes) -> list[tuple[int, int]]:
    """(offset of the start code, nal type) for every NAL unit."""
    out, i = [], 0
    while (i := data.find(b"\x00\x00\x01", i)) >= 0 and i + 3 < len(data):
        start = i - 1 if i > 0 and data[i - 1] == 0 else i
        out.append((start, data[i + 3] & 0x1F))
        i += 3
    return out


def access_units(data: bytes) -> list[tuple[bool, bytes]]:
    """Split into frames: (is_idr, bytes). A frame starts at an SPS / AUD, or at a slice right after a slice."""
    units, starts, prev = [], [], None
    for off, typ in nals(data):
        if typ in (7, 9) or (typ in (1, 5) and prev in (1, 5)):
            starts.append(off)
        prev = typ
    starts.append(len(data))
    for a, b in zip(starts, starts[1:]):
        au = data[a:b]
        units.append((any(t == 5 for _, t in nals(au)), au))
    return units


class TelloSender:
    """Plays access units over UDP like the Tello: one 1460-byte-datagram burst per frame at `fps`, looping."""

    def __init__(self, port: int, units, fps: float = FPS, loss: float = 0.0, start_frame: int = 0, seed: int = 1):
        self.port, self.units, self.fps, self.loss = port, units, fps, loss
        self.i = start_frame
        self.rng = random.Random(seed)
        self.sent = self.dropped = 0
        self.idr_times: list[float] = []  # monotonic time each IDR was sent
        self.paused = threading.Event()
        self._stop = threading.Event()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
        self.thread = threading.Thread(target=self._run, name="fake-tello", daemon=True)

    def start(self) -> TelloSender:
        self.thread.start()
        return self

    def _run(self) -> None:
        t_next = time.monotonic()
        while not self._stop.is_set():
            if not self.paused.is_set():
                idr, au = self.units[self.i % len(self.units)]
                if idr:
                    self.idr_times.append(time.monotonic())
                for k in range(0, len(au), DATAGRAM):
                    if self.loss and self.rng.random() < self.loss:
                        self.dropped += 1
                        continue
                    try:
                        self.sock.sendto(au[k:k + DATAGRAM], ("127.0.0.1", self.port))
                        self.sent += 1
                    except OSError:  # ENOBUFS on a full loopback queue: a lost datagram, like on Wi-Fi
                        self.dropped += 1
                self.i += 1
            t_next += 1.0 / self.fps
            delay = t_next - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                t_next = time.monotonic()

    def stop(self) -> None:
        self._stop.set()
        self.thread.join(timeout=2)
        self.sock.close()


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.bind(("0.0.0.0", port))
            return True
        except OSError:
            return False


def raw_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name.startswith("tello-raw") and t.is_alive()]


def wait_frame(stream, timeout: float):
    t_end = time.monotonic() + timeout
    while time.monotonic() < t_end:
        seq, img, ts = stream.read_seq(copy=False)
        if img is not None:
            return seq, img, ts
        time.sleep(0.005)
    return -1, None, 0.0


def count_new_frames(stream, seconds: float) -> tuple[int, list[float]]:
    """(new frames seen, their ages at read time) polling every 2 ms."""
    last, n, ages = stream.read_seq(copy=False)[0], 0, []
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        seq, _, ts = stream.read_seq(copy=False)
        if seq != last:
            n, last = n + 1, seq
            ages.append(time.time() - ts)
        time.sleep(0.002)
    return n, ages


# ------------------------------------------------------------------ pure helpers
def test_parse_port():
    assert parse_port("udp://@0.0.0.0:11111") == ("0.0.0.0", 11111)
    assert parse_port("udp://0.0.0.0:11111") == ("0.0.0.0", 11111)
    assert parse_port("udp://127.0.0.1:5000?overrun_nonfatal=1") == ("127.0.0.1", 5000)
    assert parse_port("udp://@:11111") == ("0.0.0.0", 11111)
    assert parse_port(11111) == ("0.0.0.0", 11111) and parse_port("11111") == ("0.0.0.0", 11111)
    with pytest.raises(ValueError):
        parse_port("rtsp://camera/stream")


def test_find_sps():
    sps, pps, idr, p = b"\x00\x00\x00\x01\x67AB", b"\x00\x00\x00\x01\x68C", b"\x00\x00\x01\x65D", b"\x00\x00\x01\x41E"
    assert find_sps(p + idr + sps + pps) == len(p + idr) + 1  # the 3-byte start code inside the 4-byte one
    assert find_sps(p + idr) == -1 and find_sps(b"\x00\x00\x01") == -1


# ------------------------------------------------------------------ bytes feed (no socket)
@needs_ffmpeg
def test_feed_joins_mid_stream_and_decodes_every_frame_from_the_first_sps():
    data = h264_stream(keyint=30)
    chunks = [data[i:i + DATAGRAM] for i in range(0, len(data), DATAGRAM)]
    start = len(chunks) // 5  # mid-GOP: everything before the next SPS is undecodable
    first_sps = find_sps(b"".join(chunks[start:]))
    expected = sum(1 for idr, au in access_units(b"".join(chunks[start:])[first_sps:]))
    s = RawTelloStream(feed=chunks[start:]).start()
    t0 = time.monotonic()
    while not s.finished and time.monotonic() - t0 < 10:
        time.sleep(0.01)
    seq, img, _ = s.read_seq()
    st = s.stats()
    s.stop()
    assert s.finished and img.shape == (720, 960, 3) and img.dtype == np.uint8
    assert st["frames"] == st["frames_decoded"] == expected == seq
    assert st["skipped_bytes"] == first_sps > 0 and st["decode_errors"] == 0 and st["corrupt_frames"] == 0
    assert st["size"] == (960, 720) and st["datagrams"] == len(chunks) - start and st["rcvbuf"] is None


@needs_ffmpeg
def test_feed_with_loss_keeps_delivering_and_heals_at_the_next_idr():
    data = h264_stream(keyint=30)
    chunks = [data[i:i + DATAGRAM] for i in range(0, len(data), DATAGRAM)]
    rng = random.Random(3)
    half = len(chunks) // 2
    lossy = [c for c in chunks[:half] if rng.random() >= 0.03] + chunks[half:]  # 3 % loss in the first half only

    def last_frame(feed):
        s = RawTelloStream(feed=feed).start()
        t0 = time.monotonic()
        while not s.finished and time.monotonic() - t0 < 10:
            time.sleep(0.01)
        out = s.read_seq()[1], s.stats()
        s.stop()
        return out

    clean_img, _ = last_frame(chunks)
    img, st = last_frame(lossy)
    assert st["frames"] > 0.75 * len(access_units(data))  # frames kept flowing (a lost datagram costs ~1 frame)
    assert st["corrupt_frames"] > 0  # the loss was seen (concealed frames)
    assert np.array_equal(img, clean_img)  # after the next IDR the picture is exact again


# ------------------------------------------------------------------ real UDP
@needs_ffmpeg
def test_udp_joins_mid_stream_fast_then_frees_the_port_without_leaking_threads():
    units = access_units(h264_stream(keyint=15))  # IDR every 0.5 s
    port = free_port()
    before = raw_threads()
    tx = TelloSender(port, units, start_frame=3).start()  # streaming before the reader exists
    time.sleep(0.15)
    t0 = time.monotonic()
    s = RawTelloStream(f"udp://@0.0.0.0:{port}", fps=60)
    assert time.monotonic() - t0 < 0.2  # "opening" never waits for video
    s.start()
    try:
        seq, img, ts = wait_frame(s, 0.5 + 1.0)  # IDR interval + 1 s
        st = s.stats()
        assert img is not None, st
        assert img.shape == (720, 960, 3) and img.dtype == np.uint8 and abs(time.time() - ts) < 1.0
        assert st["time_to_first_frame_s"] <= 1.5 and st["datagrams"] > 0 and st["size"] == (960, 720)
        assert st["rcvbuf"] >= 1 << 20, st  # asked for 8 MB; macOS grants up to kern.ipc.maxsockbuf
        assert st["queue_drops"] == 0 and st["decode_errors"] == 0
    finally:
        tx.stop()
        t1 = time.monotonic()
        s.stop()
        stop_s = time.monotonic() - t1
    assert stop_s < 0.5 and port_is_free(port)
    assert raw_threads() == before


def test_constructor_does_not_wait_and_a_held_port_raises():
    port = free_port()
    t0 = time.monotonic()
    s = RawTelloStream(port).start()  # nothing is sending
    assert time.monotonic() - t0 < 0.2 and s.read_seq() == (-1, None, 0.0)
    with pytest.raises(RuntimeError, match="lsof"):
        RawTelloStream(port)
    st = s.stats()
    assert st["datagrams"] == 0 and st["time_to_first_frame_s"] is None and st["port"] == port
    s.stop()
    assert port_is_free(port) and not raw_threads()


@pytest.mark.slow
@needs_ffmpeg
@pytest.mark.parametrize("loss", [0.0, 0.02])
def test_udp_rate_latency_and_loss(loss, capsys):
    units = access_units(h264_stream(keyint=30))
    port = free_port()
    tx = TelloSender(port, units, loss=loss, start_frame=7).start()
    time.sleep(0.2)
    s = RawTelloStream(port, fps=60).start()
    try:
        assert wait_frame(s, 1.0 + 1.0)[1] is not None, s.stats()
        n, ages = count_new_frames(s, 3.0)
        fps = n / 3.0
        if loss:  # stop the loss: the next IDR must bring clean frames back
            tx.loss = 0.0
            time.sleep(1.0 + 0.2)  # one IDR interval + slack
            c0 = s.stats()["corrupt_frames"]
            time.sleep(0.5)
            assert s.stats()["corrupt_frames"] == c0, "still concealing after a clean IDR"
        st = s.stats()
    finally:
        tx.stop()
        s.stop()
    age_ms = 1000 * sorted(ages)[len(ages) // 2]
    # end-to-end: the first IDR sent after start() -> its frame ready (the parser holds a frame until the next starts)
    t_first = s._t_start + st["time_to_first_frame_s"]
    idr_ms = 1000 * (t_first - max(t for t in tx.idr_times if t <= t_first))
    with capsys.disabled():
        print(f"\nraw loopback loss={loss:.0%}: first frame {st['time_to_first_frame_s']} s after start "
              f"({idr_ms:.0f} ms after its IDR was sent), {fps:.1f} fps, frame age at read {age_ms:.0f} ms (median), "
              f"SO_RCVBUF {st['rcvbuf']}, stats {st}")
    assert idr_ms < 100
    assert fps >= (27 if not loss else 20), st
    assert age_ms < 60
    assert st["queue_drops"] == 0 and st["datagrams"] <= tx.sent
    if loss:
        assert st["corrupt_frames"] > 0
    else:
        assert st["corrupt_frames"] == 0 and st["decode_errors"] == 0


@pytest.mark.slow
@needs_ffmpeg
def test_udp_fps_cap_converts_fewer_frames_but_decodes_all():
    units = access_units(h264_stream(keyint=15))
    port = free_port()
    tx = TelloSender(port, units, start_frame=2).start()
    s = RawTelloStream(port, fps=10).start()
    try:
        assert wait_frame(s, 1.5)[1] is not None
        d0 = s.stats()["frames_decoded"]
        n, _ = count_new_frames(s, 2.0)
        decoded = s.stats()["frames_decoded"] - d0
    finally:
        tx.stop()
        s.stop()
    assert 16 <= n <= 24 and decoded >= 55  # ~10 fps converted, all ~30 fps decoded


@pytest.mark.slow
@needs_ffmpeg
def test_udp_long_idr_interval_first_frame_within_one_interval():
    units = access_units(h264_stream(keyint=150, seconds=10))  # IDR every 5 s, like a lazy encoder
    port = free_port()
    tx = TelloSender(port, units, start_frame=20).start()  # joins 0.7 s after an IDR: ~4.3 s to the next
    time.sleep(0.1)
    s = RawTelloStream(port).start()
    try:
        seq, img, _ = wait_frame(s, 5.0 + 1.0)
        st = s.stats()
    finally:
        tx.stop()
        s.stop()
    assert img is not None and img.shape == (720, 960, 3), st
    assert st["skipped_bytes"] > 0 and st["time_to_first_frame_s"] <= 6.0


@pytest.mark.slow
@needs_ffmpeg
def test_tello_video_source_raw_backend_end_to_end_with_a_wifi_gap():
    from kite.sources import TelloVideoSource

    units = access_units(h264_stream(keyint=30))
    port = free_port()
    tx = TelloSender(port, units, start_frame=11).start()
    src = TelloVideoSource(url=f"udp://@0.0.0.0:{port}", backend="raw").start()
    try:
        f = src.wait_first(2.0)
        assert f is not None and f.image.shape == (720, 960, 3) and f.source == "tello"
        assert not f.image.flags.writeable
        tx.paused.set()  # 3 s Wi-Fi gap: the decoder resyncs, frames come back at the next SPS + IDR
        time.sleep(3.0)
        s1 = src.read().seq
        tx.paused.clear()
        time.sleep(1.0 + 0.5)
        s2 = src.read().seq
        st = src.stats()
        assert s2 - s1 > 10 and st["backend"] == "raw" and st["resyncs"] >= 1, st
    finally:
        tx.stop()
        t = time.monotonic()
        src.stop()
        stop_s = time.monotonic() - t
    assert src.read() is None and stop_s < 1.0 and port_is_free(port)


# ------------------------------------------------------------------ a real capture (kite.video_doctor), if present
def _doctor_captures() -> list[Path]:
    env = os.environ.get("KITE_VIDEO_DOCTOR_CAPTURE")
    roots = [Path(env)] if env else []
    roots += [Path(p) for p in sorted(glob.glob(str(ROOT / "recordings" / "video_doctor_*")))]
    roots += [Path(p) for p in sorted(glob.glob(os.path.expanduser("~/Desktop/kite/recordings/video_doctor_*")))]
    return [r for r in dict.fromkeys(roots) if (r / "video.h264").is_file() and (r / "packets.csv").is_file()]


@pytest.mark.slow
@pytest.mark.skipif(not _doctor_captures(), reason="no recordings/video_doctor_*/ capture (python -m kite.video_doctor)")
def test_replay_real_tello_capture_through_the_raw_reader(capsys):
    """The drone's own datagrams, re-sent over loopback with their recorded sizes and timing."""
    cap = _doctor_captures()[-1]
    data = (cap / "video.h264").read_bytes()
    rows = (cap / "packets.csv").read_text().splitlines()[1:]
    times, sizes = zip(*[(float(r.split(",")[0]), int(r.split(",")[1])) for r in rows]) if rows else ((), ())
    assert sum(sizes) == len(data)
    port = free_port()
    s = RawTelloStream(port).start()
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    t0, off, ttff = time.monotonic(), 0, None
    try:
        for t, n in zip(times, sizes):
            delay = t - (time.monotonic() - t0)
            if delay > 0:
                time.sleep(delay)
            tx.sendto(data[off:off + n], ("127.0.0.1", port))
            off += n
            if ttff is None and s.read_seq(copy=False)[1] is not None:
                ttff = time.monotonic() - t0
        time.sleep(0.3)
        st = s.stats()
    finally:
        tx.close()
        s.stop()
    dur = max(times[-1], 1e-3) if times else 1e-3
    with capsys.disabled():
        print(f"\nreplayed {cap.name}: {len(sizes)} datagrams over {dur:.1f} s -> {st['frames_decoded']} frames decoded "
              f"({st['frames_decoded'] / dur:.1f} fps), first frame {st['time_to_first_frame_s']} s, stats {st}")
    assert st["frames"] > 0, f"the real Tello stream did not decode: {st}"
    assert st["size"] == (960, 720)


# ------------------------------------------------------------------ the backend profile
def test_auto_backend_is_raw_on_macos_and_pyav_elsewhere():
    assert DroneCfg().video_backend == "pyav" and VIDEO_BACKENDS == ("auto", "raw", "socket", "pyav", "opencv")
    assert resolve_video_backend("auto", "darwin") == "raw"
    assert resolve_video_backend("auto", "linux") == "pyav" and resolve_video_backend("auto", "win32") == "pyav"
    assert resolve_video_backend(None, "darwin") == "pyav"  # the default: kite's own readers are opt-in
    for b in ("raw", "socket", "pyav", "opencv"):
        assert resolve_video_backend(b, "darwin") == b and resolve_video_backend(b.upper(), "linux") == b
    with pytest.raises(ValueError):
        resolve_video_backend("gstreamer")


@pytest.fixture(params=["darwin", "linux"])
def platform(request, monkeypatch):
    monkeypatch.setattr(sys, "platform", request.param)
    return request.param


def want(platform: str) -> str:
    return "raw" if platform == "darwin" else "pyav"


def test_tello_video_source_and_drone_resolve_auto(platform):
    from kite.drone import make_tello
    from kite.sources import TelloVideoSource
    from kite.sources.opencv_sources import stream_class
    from kite.sources.pyav_stream import PyAVStream
    from kite.sources.video_stream import VideoStream

    assert TelloVideoSource().backend == "pyav" and TelloVideoSource(backend="auto").backend == want(platform)
    assert TelloVideoSource(backend="opencv").backend == "opencv"
    with pytest.raises(ValueError):
        TelloVideoSource(backend="nope")
    assert make_tello(Config(), dry_run=True).video_backend == "pyav"
    assert make_tello(Config(drone=DroneCfg(video_backend="auto")), dry_run=True).video_backend == want(platform)
    assert make_tello(Config(drone=DroneCfg(video_backend="opencv")), dry_run=True).video_backend == "opencv"
    assert stream_class("raw") is RawTelloStream and stream_class("pyav") is PyAVStream
    assert stream_class("opencv") is VideoStream and stream_class("socket").__name__ == "SocketStream"


@pytest.fixture
def connect_spy(monkeypatch):
    """TelloDrone.connect records the drone's video backend and fails (no drone in tests)."""
    import kite.drone.tello as tello_mod

    seen = []

    def connect(self):
        seen.append((self.video_backend, self.dry_run))
        raise RuntimeError("no drone in tests")

    monkeypatch.setattr(tello_mod.TelloDrone, "connect", connect)
    return seen


@pytest.mark.parametrize("flag", [None, "raw", "pyav", "opencv", "auto"])
def test_record_picks_the_profile(platform, connect_spy, flag, tmp_path):
    from kite import record

    argv = ["--source", "tello", "--out", str(tmp_path)] + (["--video-backend", flag] if flag else [])
    assert record.main(argv) == 1  # connect failed
    assert connect_spy == [(want(platform) if flag == "auto" else flag or "pyav", True)]


@pytest.mark.parametrize("flag", [None, "raw", "opencv"])
def test_calibrate_picks_the_profile(platform, connect_spy, flag):
    from argparse import Namespace

    from kite import calibrate

    assert calibrate._open_tello(Namespace(video_backend=flag)) is None
    assert connect_spy == [(flag or "pyav", True)]
    with pytest.raises(SystemExit):  # the CLI rejects an unknown reader
        calibrate.main(["--source", "tello", "--video-backend", "gstreamer"])


@pytest.mark.parametrize("flag", [None, "raw", "pyav"])
def test_preflight_picks_the_profile(platform, connect_spy, flag):
    from kite import preflight

    checks = preflight.drone_checks(video_backend=flag)
    assert checks[0].name == "drone link" and checks[0].status == preflight.NO_GO
    assert connect_spy == [(flag or "pyav", True)]


def test_preflight_cli_accepts_the_flag(monkeypatch):
    from kite import preflight

    got = {}
    monkeypatch.setattr(preflight, "run", lambda **kw: got.update(kw) or [])
    assert preflight.main(["--quick", "--video-backend", "raw", "--json"]) == 0
    assert got["video_backend"] == "raw"


def test_preflight_says_why_there_is_no_frame():
    from kite import preflight

    none = preflight._no_frame_detail({"backend": "raw", "datagrams": 0}, 10)
    some = preflight._no_frame_detail({"backend": "raw", "datagrams": 900, "bytes": 1_300_000, "decode_errors": 4,
                                       "skipped_bytes": 1000}, 10)
    assert "no video packet" in none and "900 video packets" in some and "kite.video_doctor" in some


@pytest.mark.parametrize("spec,flag", [("tello", None), ("tello:raw", None), ("tello:opencv", None),
                                       ("tello", "pyav")])
def test_open_source_picks_the_profile(platform, connect_spy, spec, flag):
    from kite.sources import open as op

    cfg = Config()
    with pytest.raises(RuntimeError, match="no drone"):
        op.open_source(spec, cfg=cfg, **({"video_backend": flag} if flag else {}))
    explicit = flag or (spec.partition(":")[2] or None)
    assert connect_spy == [(explicit or "pyav", True)]
    assert cfg.drone.video_backend == "pyav"  # the caller's config is never edited
    assert op.parse_spec("tello:RAW") == ("tello", "raw")
    with pytest.raises(ValueError):
        op.parse_spec("tello:gstreamer")


def test_open_source_cli_accepts_the_flag(connect_spy):
    from kite.sources import open as op

    assert op.main(["tello", "--video-backend", "opencv"]) == 1  # connect failed, reported, exit 1
    assert connect_spy == [("opencv", True)]


def _drone_test():
    spec = importlib.util.spec_from_file_location("kite_drone_test", ROOT / "tests" / "drone_test.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_drone_test_picks_the_profile(platform):
    dt = _drone_test()
    _, args = dt.parse_args([])
    assert args.video_backend == "pyav"  # the default: kite's own readers are opt-in
    assert dt.parse_args(["--video-backend", "auto"])[1].video_backend == want(platform)
    for b in ("raw", "socket", "pyav", "opencv"):
        assert dt.parse_args(["--video-backend", b])[1].video_backend == b
    with pytest.raises(SystemExit):
        dt.parse_args(["--video-backend", "gstreamer"])


def test_drone_test_keeps_crlf():
    raw = (ROOT / "tests" / "drone_test.py").read_bytes()
    assert raw.count(b"\r\n") == raw.count(b"\n")


class _FakeTello:
    vs_udp_port = 0

    def __init__(self, port):
        self.port, self.streamons = port, 0

    def get_udp_video_address(self):
        return f"udp://@0.0.0.0:{self.port}"

    def streamon(self):
        self.streamons += 1


@needs_ffmpeg
def test_drone_test_opens_the_raw_reader_on_a_live_stream(capsys):
    dt = _drone_test()
    units = access_units(h264_stream(keyint=15))
    port = free_port()
    tx = TelloSender(port, units, start_frame=5).start()
    tello = _FakeTello(port)
    try:
        stream = dt.open_video(tello, 30, "raw", tries=1)
        assert stream is not None and stream.read_seq()[1].shape == (720, 960, 3)
        stream.stop()
    finally:
        tx.stop()
    assert tello.streamons == 1 and "raw reader, first frame" in capsys.readouterr().out


class _FakeDjiTello(_FakeTello):
    """The djitellopy surface TelloDrone.connect / close use."""

    ended = False

    def connect(self):
        pass

    def get_current_state(self):
        return {"bat": 90}

    def end(self):
        self.ended = True


@needs_ffmpeg
def test_tello_drone_connect_streams_through_the_raw_reader(monkeypatch):
    """TelloDrone.connect (streamon until packets arrive) -> frame_source -> frames, over loopback UDP."""
    monkeypatch.setattr(sys, "platform", "darwin")
    from kite.drone.tello import TelloDrone

    units = access_units(h264_stream(keyint=15))
    port = free_port()
    tx = TelloSender(port, units, start_frame=9).start()
    fake = _FakeDjiTello(port)
    drone = TelloDrone(tello=fake, dry_run=True, video_backend="auto")  # auto -> raw on darwin
    try:
        drone.connect()
        src = drone.frame_source()
        f = src.wait_first(1.5)
        assert src.backend == "raw" and f is not None and f.image.shape == (720, 960, 3)
        assert src.stats()["frames"] >= 1 and fake.streamons >= 1
    finally:
        drone.close()
        tx.stop()
    assert fake.ended and port_is_free(port) and not raw_threads()


def test_codec_flags_resolve_in_either_case():
    import enum
    from kite.sources.tello_raw import _flag

    class Upper(enum.IntFlag):
        LOW_DELAY = 1

    class Lower(enum.IntFlag):
        low_delay = 2

    assert _flag(Upper, "low_delay") == 1
    assert _flag(Lower, "low_delay") == 2


def test_drop_corrupt_holds_until_a_clean_keyframe():
    import time as _t
    from types import SimpleNamespace as F
    from kite.sources import tello_raw

    s = tello_raw.RawTelloStream.__new__(tello_raw.RawTelloStream)
    s._drop_corrupt, s._held_since = True, None
    frames = [F(key_frame=False, is_corrupt=True), F(key_frame=False, is_corrupt=False),
              F(key_frame=True, is_corrupt=False), F(key_frame=False, is_corrupt=False)]
    shown = []
    for f in frames:  # the same rule as the decode loop
        if f.is_corrupt and s._held_since is None:
            s._held_since = _t.monotonic()
        if s._held_since is not None:
            if f.key_frame and not f.is_corrupt or _t.monotonic() - s._held_since > tello_raw.HOLD_MAX_S:
                s._held_since = None
            else:
                continue
        shown.append(f)
    assert shown == frames[2:]
