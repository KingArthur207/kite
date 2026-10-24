"""kite.sources.recording: replay a kite.record take (dry run on a synthetic video) with a synthetic telemetry.jsonl
of known values, and check frame order, pose alignment and interpolation, the empty-telemetry case and seek."""

import csv
import json
import shutil

import cv2
import numpy as np
import pytest

from kite.record import FileReplay, Recorder, RecordCfg
from kite.sources import FrameSource, RecordingFrameSource, RecordingSource
from kite.sources.recording import main

N, FPS = 30, 60.0


def level(img) -> int:
    return int(round((float(img.mean()) - 10) / 5))


@pytest.fixture(scope="module")
def take(tmp_path_factory):
    """A kite.record dry run on a 30-frame clip whose frame i is a flat grey of level 5*i + 10."""
    d = tmp_path_factory.mktemp("rec")
    path = str(d / "ramp.avi")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), FPS, (160, 120))
    for i in range(N):
        w.write(np.full((120, 160, 3), 5 * i + 10, np.uint8))
    w.release()
    src = FileReplay(path)
    try:
        run = Recorder(src, None, RecordCfg(out_dir=str(d / "out"), fps=FPS), {"source_kind": "file"}).run()
    finally:
        src.stop()
    return run


def frame_times(run):
    with open(run / "frames.csv", newline="") as f:
        return [float(r["t_mono"]) for r in csv.DictReader(f)]


def with_telemetry(take, tmp_path, packets, events=()):
    """A copy of the take with these state packets (and events) written as the recorder would."""
    run = tmp_path / "run"
    shutil.copytree(take, run)
    with open(run / "telemetry.jsonl", "w") as f:
        for p in packets:
            f.write(json.dumps({"t_wall": 0.0, **p}) + "\n")
        f.write('{"t_mono": 1')  # a torn last line (crash) is skipped
    with open(run / "events.jsonl", "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    return run


def state(t, yaw=0, pitch=0, h=100, tof=100, bat=80, roll=0):
    return {"t_mono": t, "pitch": pitch, "roll": roll, "yaw": yaw, "vgx": 0, "vgy": 0, "vgz": 0, "templ": 50,
            "temph": 52, "tof": tof, "h": h, "bat": bat, "baro": 12.3, "time": 0, "agx": 0, "agy": 0, "agz": -1000}


def read_all(src):
    out = []
    while True:
        ok, img = src.read()
        if not ok:
            return out
        out.append((src.index, level(img), src.t, src.pose))


def test_every_frame_in_order_without_telemetry(take):
    src = RecordingSource(take)
    assert len(src) == N and (take / "telemetry.jsonl").read_text() == ""
    got = read_all(src)
    assert [g[0] for g in got] == list(range(N)) and [g[1] for g in got] == list(range(N))
    assert [g[2] for g in got] == pytest.approx(frame_times(take))
    p = got[0][3]
    assert {"x", "y", "heading", "cam_height", "pitch", "t", "battery", "flying"} <= set(p)
    assert all(p[k] is None for k in ("x", "y", "heading", "cam_height", "pitch", "battery", "flying"))
    assert src.telemetry is None and src.read() == (False, None)
    # camera: the 960x720 intrinsics scaled to the 160x120 dry-run clip
    assert (src.camera.width, src.camera.height) == (160, 120)
    assert src.camera.fx == pytest.approx(load_fx() * 160 / 960)
    s = src.summary()
    assert s["frames"] == N and s["dropped_frames"] == 0 and s["telemetry_coverage"] == 0.0


def load_fx():
    from kite.camera import load_camera

    return load_camera().fx


def test_pose_is_interpolated_between_packets(take, tmp_path):
    ft = frame_times(take)
    t0 = ft[0] - 0.01
    ts = list(np.arange(t0, ft[-1] + 0.11, 0.1))  # a packet every 0.1 s across the take
    # h rises 1 cm per 10 ms, tof agrees with h (floor below), pitch -5 -> +5, yaw crosses +-180, battery steps down
    packets = [state(t, yaw=(170 + 40 * k + 180) % 360 - 180, pitch=-5 + k, h=100 + 10 * k, tof=100 + 10 * k,
                     bat=80 - k) for k, t in enumerate(ts)]
    events = [{"t_mono": ft[10], "t_wall": 0, "event": "takeoff -> ok", "flying": True},
              {"t_mono": ft[20], "t_wall": 0, "event": "land -> ok", "flying": False}]
    src = RecordingSource(with_telemetry(take, tmp_path, packets, events))
    got = read_all(src)
    assert [g[1] for g in got] == list(range(N))
    for i, _, t, p in got:
        k = (t - t0) / 0.1  # fractional packet number
        assert p["cam_height"] == pytest.approx((100 + 10 * k) / 100, abs=1e-6)
        assert p["pitch"] == pytest.approx(-5 + k, abs=1e-6)
        assert p["heading"] == pytest.approx((170 + 40 * k) % 360, abs=1e-6)  # unwrapped across 180
        assert p["battery"] == 80 - round(k)  # nearest, not interpolated
        assert abs(p["tel_dt"]) <= 0.05 + 1e-9
        assert p["flying"] is (10 <= i < 20)
        assert p["x"] is None and p["y"] is None and p["t"] == t
    assert src.telemetry["bat"] == 80 - round((got[-1][2] - t0) / 0.1)  # the raw nearest packet
    s = src.summary()
    assert s["telemetry_coverage"] == 1.0 and s["flying_frames"] == 10
    assert s["battery_start"] == 80 and s["battery_end"] == 80 - (len(ts) - 1)


def test_cam_height_uses_h_when_tof_disagrees_and_none_beyond_the_gap(take, tmp_path):
    ft = frame_times(take)
    # one packet right on frame 5 only: over a 30 cm step ToF says 0.9 m, h says 1.2 m
    src = RecordingSource(with_telemetry(take, tmp_path, [state(ft[5], h=120, tof=90, pitch=3)]))
    poses = [src.pose_at(i) for i in range(N)]
    assert poses[5]["cam_height"] == pytest.approx(1.2) and poses[5]["pitch"] == 3
    for i, p in enumerate(poses):
        near = abs(ft[i] - ft[5]) <= 0.2
        assert (p["cam_height"] is not None) == near, i
        assert p["flying"] is False  # telemetry but no flight events (record only): on the ground
    # ToF out of range (6553) -> 'h'
    src = RecordingSource(with_telemetry(take, tmp_path / "b", [state(ft[5], h=150, tof=6553)]))
    assert src.pose_at(5)["cam_height"] == pytest.approx(1.5)


def test_seek_by_index_t_and_sec(take):
    src = RecordingSource(take)
    ft = frame_times(take)
    assert src.seek(20) == 20
    ok, img = src.read()
    assert ok and src.index == 20 and level(img) == 20
    assert src.seek(3) == 3  # backwards
    ok, img = src.read()
    assert level(img) == 3 and src.t == pytest.approx(ft[3])
    assert src.seek(t=ft[12] - 1e-4) == 12
    assert level(src.read()[1]) == 12
    assert src.seek(sec=ft[25] - ft[0]) == 25
    assert level(src.read()[1]) == 25
    assert src.seek(N + 5) == N and src.read() == (False, None)
    src.seek_grab_max = 0  # long jumps: the Matroska index of a finished take, both ways
    for i in (27, 4, 16, 0, 29):
        src.seek(i)
        ok, img = src.read()
        assert ok and level(img) == i


def test_realtime_paces_by_t_mono(take):
    import time

    src = RecordingSource(take, realtime=True, speed=2.0)
    t = time.monotonic()
    assert len(read_all(src)) == N
    span = (src.times[-1] - src.times[0]) / 2.0
    assert time.monotonic() - t >= 0.9 * span


def test_frame_source_adapter_and_cli(take, capsys):
    fs = RecordingSource(take).frame_source()
    assert isinstance(fs, FrameSource) and isinstance(fs, RecordingFrameSource)
    seqs = []
    while True:
        f = fs.read()
        if fs.finished:
            break
        seqs.append(f.seq)
        assert level(f.image) == len(seqs) - 1 and not f.image.flags.writeable and fs.pose["t"] is not None
    assert seqs == list(range(1, N + 1))
    fs.stop()
    assert main([str(take)]) == 0
    out = capsys.readouterr().out
    assert f"frames      {N}" in out and "dropped     0" in out and "0 packets" in out
