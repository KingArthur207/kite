"""python -m kite.calibrate on synthetic chessboard views: a board projected by a known camera (fx 1000, cx 480, cy 360,
small distortion) at varied poses, through the image-folder, kite.record-folder (video.mkv) and live-capture paths; and
the known-object fallback's maths."""

import json
import math

import cv2
import numpy as np
import pytest

from kite.calibrate import (
    ViewSelector,
    board_image,
    calibrate,
    capture_live,
    find_board,
    fx_from_known_object,
    main,
    verdict,
)
from kite.camera import Camera, load_camera
from kite.sources import FrameSource
from kite.types import Frame

W, H = 960, 720
K = np.array([[1000.0, 0, 480.0], [0, 1000.0, 360.0], [0, 0, 1]])
DIST = np.array([-0.05, 0.02, 0.0005, -0.0003, 0.0])
PATTERN, SQ, S = (9, 6), 0.025, 40  # inner corners, square (m), square in the board image (px)


def _poses(n, seed=0):
    """Board poses that keep the whole board (with its white margin) in the frame, tilted up to ~40 deg."""
    rng = np.random.default_rng(seed)
    c = np.array([(PATTERN[0] - 1) * SQ / 2, (PATTERN[1] - 1) * SQ / 2, 0])
    outer = np.array([[-2, -2, 0], [PATTERN[0] + 1, -2, 0], [PATTERN[0] + 1, PATTERN[1] + 1, 0],
                      [-2, PATTERN[1] + 1, 0]], float) * SQ - c
    out = []
    while len(out) < n:
        rvec = np.array([rng.uniform(-0.7, 0.7), rng.uniform(-0.7, 0.7), rng.uniform(-0.3, 0.3)])
        tvec = np.array([rng.uniform(-0.15, 0.15), rng.uniform(-0.1, 0.1), rng.uniform(0.45, 0.8)])
        proj, _ = cv2.projectPoints(outer, rvec, tvec, K, DIST)
        p = proj.reshape(-1, 2)
        if p[:, 0].min() > 10 and p[:, 0].max() < W - 10 and p[:, 1].min() > 10 and p[:, 1].max() < H - 10:
            out.append((rvec, tvec))
    return out


@pytest.fixture(scope="module")
def views():
    """BGR 960x720 renders of the board at 16 poses, with lens distortion, blur and noise."""
    board = board_image(PATTERN, square_px=S)
    # board-image pixel centre (u, v) -> board metres, centred on the inner-corner grid (corners sit on pixel edges)
    c = np.array([(PATTERN[0] - 1) * SQ / 2, (PATTERN[1] - 1) * SQ / 2])
    A = np.array([[SQ / S, 0, (0.5 / S - 2) * SQ - c[0]], [0, SQ / S, (0.5 / S - 2) * SQ - c[1]], [0, 0, 1]])
    # distorted output pixel -> ideal pixel (where to sample the undistorted render)
    grid = np.stack(np.meshgrid(np.arange(W), np.arange(H)), -1).reshape(-1, 1, 2).astype(np.float64)
    und = cv2.undistortPoints(grid, K, DIST, P=K).reshape(H, W, 2).astype(np.float32)
    rng = np.random.default_rng(1)
    imgs = []
    for rvec, tvec in _poses(16):
        R = cv2.Rodrigues(rvec)[0]
        Hm = K @ np.column_stack([R[:, 0], R[:, 1], tvec]) @ A
        ideal = cv2.warpPerspective(board, Hm, (W, H), flags=cv2.INTER_LINEAR, borderValue=110)
        img = cv2.remap(ideal, und[..., 0], und[..., 1], cv2.INTER_LINEAR, borderValue=110)
        img = cv2.GaussianBlur(img, (3, 3), 0.6).astype(np.float32) + rng.normal(0, 2.0, img.shape)
        imgs.append(cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
    return imgs


def _assert_close(cam: Camera, tol=0.02):
    assert abs(cam.fx - 1000) < tol * 1000, cam
    assert abs(cam.fy - 1000) < tol * 1000, cam
    assert abs(cam.cx - 480) < 15 and abs(cam.cy - 360) < 15, cam


def test_every_synthetic_view_is_detected(views):
    found = [find_board(cv2.cvtColor(v, cv2.COLOR_BGR2GRAY), PATTERN) for v in views]
    assert all(f is not None for f in found)


def test_folder_of_images(views, tmp_path):
    folder = tmp_path / "shots"
    folder.mkdir()
    for i, v in enumerate(views):
        cv2.imwrite(str(folder / f"{i:02d}.png"), v)
    out = tmp_path / "cam.json"
    assert main(["--source", str(folder), "--out", str(out)]) == 0
    cam = load_camera(out)  # load_camera ignores the extra keys
    _assert_close(cam)
    assert (cam.width, cam.height) == (W, H)
    d = json.loads(out.read_text())
    assert d["method"] == "chessboard" and d["n_views"] >= 8 and len(d["dist"]) == 5 and d["rms_px"] < 1.0
    assert d["dist"][0] < 0  # barrel distortion recovered with the right sign


def test_record_folder_video(views, tmp_path):
    """A kite.record-style folder with video.mkv (MJPG): each view held for 3 frames, so the stillness check passes."""
    rec = tmp_path / "20261024_010000_tello"
    rec.mkdir()
    vw = cv2.VideoWriter(str(rec / "video.mkv"), cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (W, H))
    assert vw.isOpened()
    for v in views:
        for _ in range(3):
            vw.write(v)
    vw.release()
    out = tmp_path / "cam.json"
    assert main(["--source", str(rec), "--out", str(out), "--no-preview"]) == 0
    _assert_close(load_camera(out))


def test_too_few_views_is_not_saved(views, tmp_path):
    folder = tmp_path / "few"
    folder.mkdir()
    for i, v in enumerate(views[:4]):
        cv2.imwrite(str(folder / f"{i}.png"), v)
    out = tmp_path / "cam.json"
    assert main(["--source", str(folder), "--out", str(out)]) == 2
    assert not out.exists()


class _FakeLive(FrameSource):
    """A live camera: every read() is a new frame; each view lasts `hold` frames (moving board in between)."""

    name = "fake"

    def __init__(self, imgs, hold=4):
        self.imgs, self.hold, self.seq = imgs, hold, 0

    def read(self):
        self.seq += 1
        return Frame(self.imgs[(self.seq // self.hold) % len(self.imgs)], self.seq / 30, self.seq, self.name)


def test_live_auto_capture(views, tmp_path):
    frames_dir = tmp_path / "calib"
    got, size = capture_live(_FakeLive(views), PATTERN, n=10, max_s=10, preview=False, save_dir=frames_dir)
    assert len(got) == 10 and size == (W, H)
    assert len(list(frames_dir.glob("view_*.png"))) == 10  # kept for a re-run without the drone
    cal = calibrate(got, size, PATTERN)
    _assert_close(cal.camera)
    assert cal.rms < 1.0


def test_selector_needs_still_and_new_views():
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 900, (54, 2)).astype(np.float32)
    sel = ViewSelector(min_move_px=70, still_px=3)
    assert sel.offer(a) == "hold still"  # no previous frame to compare with
    assert sel.offer(a + 1) == "captured"
    assert sel.offer(a + 1.5) == "move / tilt to a new view"  # same view again
    assert sel.offer(a + 100) == "hold still"  # moving
    assert sel.offer(a + 101) == "captured"
    assert sel.offer(None) == "no board"
    assert sel.offer((a + 1)[::-1].copy()) == "hold still"  # after a miss, a frame never counts as still


def test_known_object_math():
    assert fx_from_known_object(500, 0.5, 1.0) == pytest.approx(1000)
    assert fx_from_known_object(240, 0.6, 2.4) == pytest.approx(960)
    with pytest.raises(ValueError):
        fx_from_known_object(0, 0.5, 1.0)
    # hfov follows: 960 px wide at fx 1000 -> 2 atan(0.48) = 51.3 deg
    assert Camera(fx=1000).hfov_deg == pytest.approx(math.degrees(2 * math.atan(0.48)))


def test_known_object_cli(views, tmp_path):
    img = tmp_path / "one.png"
    cv2.imwrite(str(img), views[0])
    out = tmp_path / "cam.json"
    assert main(["--source", str(img), "--known-object", "0.60", "--distance-m", "2.0", "--px-width", "300",
                 "--out", str(out), "--no-preview"]) == 0
    d = json.loads(out.read_text())
    assert d["fx"] == pytest.approx(1000) and d["fy"] == pytest.approx(1000) and d["method"] == "known-object"
    assert (d["cx"], d["cy"]) == (480, 360)
    # 0.60 m at 2.0 m spanning 600 px -> fx 2000: refused without --force
    out2 = tmp_path / "bad.json"
    assert main(["--source", str(img), "--known-object", "0.60", "--distance-m", "2.0", "--px-width", "600",
                 "--out", str(out2), "--no-preview"]) == 2
    assert not out2.exists()


def test_verdict_bands():
    assert verdict(Camera(fx=1000, fy=1000), 0.4) == ([], [])
    warn, err = verdict(Camera(fx=1200, fy=1200), 1.5)
    assert len(warn) == 3 and not err  # fx, fy outside 850-1150, RMS > 1
    warn, err = verdict(Camera(fx=1500, fy=1500), 3.0)
    assert len(err) == 3
