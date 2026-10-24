"""Frame sources: file replay, read-only frames, and the Tello readers on a real H.264 stream over local UDP.

Ported from vigil's tests/test_sources.py and test_review_fixes_1_3.py. The slow tests use ffmpeg as a stand-in
Tello (raw H.264 at 960x720, 30 fps, to a UDP port) and run each reader with djitellopy imported first, as in
the real app: OpenCV and PyAV each bundle their own FFmpeg.
"""

import os
import shutil
import subprocess
import sys
import textwrap
import time

import cv2
import numpy as np
import pytest

from kite.sources import FrameSource, StaticImageSource, SteppedVideoSource, TelloVideoSource, VideoFileSource

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_FRAMES = 40
needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    """A 30 fps MJPG video whose frame i is a flat grey of level 5*i (so order is checkable)."""
    path = str(tmp_path_factory.mktemp("vid") / "ramp.avi")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (160, 120))
    assert w.isOpened()
    for i in range(N_FRAMES):
        w.write(np.full((120, 160, 3), 5 * i + 10, np.uint8))
    w.release()
    return path


def level(frame) -> int:
    return int(round((float(frame.image.mean()) - 10) / 5))


def test_stepped_source_is_ordered_and_finishes(video):
    src = SteppedVideoSource(video)
    got = []
    while (f := src.read()) is not None:
        got.append((f.seq, level(f)))
        assert f.image.shape == (120, 160, 3) and f.image.dtype == np.uint8
    assert [s for s, _ in got] == list(range(1, N_FRAMES + 1))
    assert [lv for _, lv in got] == list(range(N_FRAMES))
    assert src.finished
    src.stop()


@pytest.mark.skipif(bool(os.environ.get("CI")), reason="wall-clock timing; flaky on shared CI runners, runs locally")
def test_file_source_plays_in_real_time_and_serves_newest(video):
    with VideoFileSource(video) as src:
        first = src.wait_first(3.0)
        assert first is not None
        time.sleep(0.5)
        f = src.read()
        # ~15 frames in 0.5 s at 30 fps: newest frame, not the next queued one
        assert f.seq >= first.seq + 8
        assert abs(level(f) - (f.seq - 1)) <= 1
        t0 = time.time()
        while not src.finished and time.time() - t0 < 3:
            time.sleep(0.05)
        assert src.finished and src.read().seq == N_FRAMES


def test_file_source_loops_and_restarts_cleanly(video):
    with VideoFileSource(video, loop=True, fps=200) as src:
        src.wait_first(3.0)
        time.sleep(0.6)  # 120 frames at 200 fps > 40 frames in the file
        assert src.read().seq > N_FRAMES and not src.finished
    assert src.read() is None
    seq_before = src._seq
    src.start()
    f = src.wait_first(3)
    assert f is not None and f.seq > seq_before
    src.stop()


def test_frames_are_read_only():
    f = StaticImageSource(np.zeros((10, 10, 3), np.uint8)).read()
    with pytest.raises(ValueError):
        f.image[0, 0] = 1
    with pytest.raises(cv2.error):
        cv2.rectangle(f.image, (0, 0), (5, 5), (0, 0, 255), 1)


def test_team_videostream_reader_via_tello_source(video):
    # the Tello source wraps the team's VideoStream; point it at a file to exercise it without a drone
    import kite.sources.video_stream  # noqa: F401

    assert "nobuffer" in os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]
    src = TelloVideoSource(url=video, fps=60, backend="opencv").start()
    try:
        f = src.wait_first(5.0)
        assert f is not None and f.source == "tello" and f.seq >= 1
        seqs = [f.seq]
        t0 = time.time()
        while time.time() - t0 < 0.5:
            g = src.read()
            if g.seq != seqs[-1]:
                seqs.append(g.seq)
            time.sleep(0.002)
        assert seqs == sorted(seqs)  # never goes backwards
        assert abs(f.t - time.time()) < 5  # stamped with wall-clock time
    finally:
        src.stop()
    assert src.read() is None


def test_tello_source_before_start_and_default_cap():
    src = TelloVideoSource()
    assert src.read() is None and src.fps >= 60  # a cap equal to the 30 fps stream drops ~40 % of frames


def test_wait_first_timeout():
    class Never(FrameSource):
        def read(self):
            return None

    t0 = time.time()
    assert Never().wait_first(0.2) is None and time.time() - t0 >= 0.19


# ------------------------------------------------------------------ Tello readers on a live UDP H.264 stream
UDP_SCRIPT = textwrap.dedent("""
    import subprocess, sys, time
    sys.path.insert(0, {root!r})
    import djitellopy  # noqa: F401  (imports PyAV, like the real app)
    from kite.sources import TelloVideoSource
    port = {port}
    def sender():  # like the Tello: raw H.264 over UDP at 30 fps, 960x720
        return subprocess.Popen(["ffmpeg", "-loglevel", "quiet", "-re", "-f", "lavfi", "-i", "testsrc=size=960x720:rate=30",
                                 "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency", "-g", "15",
                                 "-f", "h264", f"udp://127.0.0.1:{{port}}"])
    ff = sender()
    src = TelloVideoSource(url=f"udp://@0.0.0.0:{{port}}", backend={backend!r}).start()
    first = src.wait_first(15)
    assert first is not None, "no frame"
    assert first.image.shape == (720, 960, 3), first.image.shape
    s0, t0 = src.read().seq, time.time()
    ages = []
    while time.time() - t0 < 3.0:
        f = src.read()
        ages.append(time.time() - f.t)
        time.sleep(0.01)
    s1, t1 = src.read().seq, time.time()
    print("RATE", (s1 - s0) / (t1 - t0), flush=True)
    print("AGE", sorted(ages)[len(ages) // 2], flush=True)
    ff.kill(); ff.wait()
    time.sleep(5.0)                       # 5 s Wi-Fi gap: frames must flow again afterwards
    ff = sender()
    time.sleep(8.0)
    s2 = src.read().seq
    print("RECOVERED", s2 - s1, "RECONNECTS", src.reconnects, flush=True)
    ff.kill(); ff.wait()
    time.sleep(1.5)                       # stream stalled: reader is inside grab()
    t = time.time(); src.stop(); print("STOP", time.time() - t, flush=True)
""")


@pytest.mark.slow
@needs_ffmpeg
@pytest.mark.parametrize("backend,port", [("raw", 11175), ("pyav", 11177), ("opencv", 11176)])
def test_tello_reader_full_rate_and_safe_stop_on_stalled_udp_stream(backend, port):
    code = UDP_SCRIPT.format(root=ROOT, port=port, backend=backend)
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=90)
    out = p.stdout
    assert p.returncode == 0, f"crashed (code {p.returncode}):\n{out}\n{p.stderr[-2000:]}"
    rate = float(out.split("RATE")[1].split()[0])
    recovered = int(out.split("RECOVERED")[1].split()[0])
    stop_s = float(out.split("STOP")[1].split()[0])
    assert rate > 25, f"delivered {rate:.1f} fps of a 30 fps stream"
    assert recovered > 30, out  # frames flow again after the gap (grab() waits it out, or the stream is reopened)
    assert stop_s < 6.0


@pytest.mark.slow
@needs_ffmpeg
@pytest.mark.parametrize("backend,port", [("raw", 11174), ("pyav", 11178), ("opencv", 11179)])
def test_tello_source_waits_for_a_stream_that_starts_after_the_open_timeout(backend, port):
    # the "can't find camera" bugs: video starting > 5 s after start() raised; keyframes 5 s apart never decoded
    src = TelloVideoSource(url=f"udp://@0.0.0.0:{port}", backend=backend).start()  # must not raise
    time.sleep(7.0)
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "quiet", "-re", "-f", "lavfi", "-i", "testsrc=size=960x720:rate=30",
                           "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency", "-g", "150",
                           "-f", "h264", f"udp://127.0.0.1:{port}?pkt_size=1460"])
    try:
        f = src.wait_first(15.0)
        assert f is not None and f.image.shape == (720, 960, 3)
    finally:
        ff.kill()
        ff.wait()
        src.stop()
    assert src.read() is None
