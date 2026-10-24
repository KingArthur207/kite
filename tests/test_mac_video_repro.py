"""Tello-like UDP H.264 against the live readers (docs/mac-video-root-cause.md). Slow: each case streams ~15 s.

The sender (scripts/tello_video_sim.py) imitates the drone: 960x720 30 fps raw Annex B, 1460-byte datagrams sent as
one burst per frame, already streaming when the reader opens. Each reader runs in its own process
(scripts/mac_video_repro.py) like the real app. These pin down the behaviours the report relies on.
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = [pytest.mark.slow, pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")]


def _repro():
    spec = importlib.util.spec_from_file_location("mac_video_repro", ROOT / "scripts" / "mac_video_repro.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(reader: str, cond: str, port: int, seconds: float = 10.0) -> dict:
    return _repro().one(reader, cond, port, seconds, None, [])


@pytest.mark.parametrize("reader", ["pyav_kite", "parser"])
def test_ffmpeg_h264_survives_heavy_random_loss(reader):
    # 20 % of datagrams dropped at random: FFmpeg conceals and keeps delivering. Random Wi-Fi loss alone does not
    # explain "packets arrive but zero frames".
    r = run(reader, "loss20", 11260)
    assert r["frames"] > 100, r


def test_parser_reader_first_frame_within_one_keyframe_interval():
    # raw socket + CodecContext.parse(): no format probe, no find_stream_info -> the first keyframe after opening
    # is shown (keyframes 5 s apart: < 5 s + margin)
    r = run("parser", "keyint5s", 11262, seconds=12)
    assert r["frames"] > 100 and r["first_frame_s"] < 6.5, r


def test_nobuffer_probe_throws_away_the_first_keyframe():
    # kite's PyAVStream (fflags nobuffer): avformat_find_stream_info decodes the first keyframe and, with nobuffer,
    # drops those packets, so the first delivered frame is the SECOND keyframe (~5 s later with 5 s keyframes)
    kite = run("pyav_kite", "keyint5s", 11264, seconds=14)
    noprobe = run("pyav_noprobe", "keyint5s", 11266, seconds=14)
    assert kite["first_frame_s"] is not None and noprobe["first_frame_s"] is not None, (kite, noprobe)
    assert kite["first_frame_s"] - noprobe["first_frame_s"] > 3.0, (kite, noprobe)


def test_opencv_cannot_open_without_a_keyframe_in_the_first_mebibyte():
    # OpenCV (no input format forced) detects raw H.264 only if SPS+PPS+a slice show up in the first 1 MiB
    # (FFmpeg's format_probesize default). Keyframes 5 s apart at 3 Mbit/s: open fails in ~3 s, any timeout.
    r = run("opencv_kite_15s", "keyint5s", 11268, seconds=4)
    assert r["frames"] == 0 and "can't capture by name" in r["signatures"], r


def test_showall_decoder_shows_frames_before_the_first_keyframe():
    # parameter sets every 1 s, IDR every 8 s (the real log's pattern): the default decoder waits for the IDR,
    # flags2=+showall / flags=+output_corrupt shows (initially grey) frames as soon as SPS/PPS are known
    plain = run("parser", "rare_idr_params1s", 11270, seconds=10)
    showall = run("parser_showall", "rare_idr_params1s", 11272, seconds=10)
    assert showall["first_frame_s"] is not None and showall["first_frame_s"] < 2.0, showall
    assert plain["first_frame_s"] is None or plain["first_frame_s"] > 4.0, plain
