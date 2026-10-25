"""Calibrate the chest webcam (fx, fy, cx, cy and lens distortion) into a profile perception.near loads.

    python -m perception.calibrate_camera --print-board board.png                 # ChArUco 7x5, print it (A4)
    python -m perception.calibrate_camera --images calib/ --square-mm 30 --marker-mm 22    # photos of the board
    python -m perception.calibrate_camera --video calib.mp4                        # a clip waving the board
    python -m perception.calibrate_camera --live webcam:Logitech                   # live, on YOUR machine only
    python -m perception.calibrate_camera --images calib/ --pattern chessboard --board 9x6

Writes data/cameras/<camera>.json (e.g. data/cameras/logitech_c270.json), which `python -m perception.near
--profile auto` prefers over the spec numbers whenever the USB name matches that camera. --out to write
elsewhere.

How to take good views (20 to 40 of them): the board 0.3 to 1.5 m from the lens, filling a good part of the frame,
in every region of it (centre, all four corners, all edges: distortion lives at the edges), tilted 20 to 45 degrees
in different directions (flat-on views alone cannot separate fx from distance), held still (motion blur ruins
corners). Calibrate at the resolution you will run at: 1280x720 is what kite.sources asks a webcam for.

ChArUco (default) works with a partly visible or partly occluded board, so corner views are easy. The chessboard
path reuses kite/calibrate.py's detector, view selection and calibration (read-only) for the Tello.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perception.near.intrinsics import BUILTIN, CameraProfile, calibrated_path, dfov_from_focal, match_profile  # noqa: E402

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
MIN_VIEWS = 8
MIN_CHARUCO_CORNERS = 8


def charuco_board(squares: tuple[int, int], square_mm: float, marker_mm: float, dictionary: str):
    import cv2
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary))
    return cv2.aruco.CharucoBoard(squares, square_mm / 1000.0, marker_mm / 1000.0, d)


def print_board(path: str, pattern: str, squares: tuple[int, int], board: tuple[int, int], square_mm: float,
                marker_mm: float, dictionary: str) -> Path:
    """An image to print at 100 % (A4 / Letter at 300 dpi) or show full screen on a second display."""
    import cv2
    px_per_mm = 300 / 25.4
    if pattern == "charuco":
        b = charuco_board(squares, square_mm, marker_mm, dictionary)
        size = (int(squares[0] * square_mm * px_per_mm) + 120, int(squares[1] * square_mm * px_per_mm) + 120)
        img = b.generateImage(size, marginSize=60)
    else:
        from kite.calibrate import board_image
        img = board_image(board, square_px=int(square_mm * px_per_mm))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), img)
    return p


def frames_from(images: str | None, video: str | None, max_frames: int = 400):
    import cv2
    if images:
        root = Path(images)
        files = sorted(f for f in (root.iterdir() if root.is_dir() else [root]) if f.suffix.lower() in IMAGE_EXT)
        for f in files:
            img = cv2.imread(str(f))
            if img is not None:
                yield img
        return
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or max_frames
    keep = set(np.linspace(0, max(0, n - 1), min(n, max_frames)).astype(int).tolist())
    i = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if i in keep:
            yield img
        i += 1
    cap.release()


def detect_charuco(gray, board):
    """(object points N x 3, image points N x 2) or None."""
    import cv2
    det = cv2.aruco.CharucoDetector(board)
    corners, ids, _, _ = det.detectBoard(gray)
    if ids is None or len(ids) < MIN_CHARUCO_CORNERS:
        return None
    obj, img = board.matchImagePoints(corners, ids)
    if obj is None or len(obj) < MIN_CHARUCO_CORNERS:
        return None
    return obj.reshape(-1, 3).astype(np.float32), img.reshape(-1, 2).astype(np.float32)


def solve(obj_pts: list, img_pts: list, size: tuple[int, int], fix_k3: bool = True):
    """cv2.calibrateCamera, then drop views whose error is over 3x the median once and refit."""
    import cv2
    flags = cv2.CALIB_FIX_K3 if fix_k3 else 0

    def fit(o, i):
        rms, K, dist, rv, tv = cv2.calibrateCamera(o, i, size, None, None, flags=flags)
        errs = []
        for oo, ii, r, t in zip(o, i, rv, tv):
            pr, _ = cv2.projectPoints(oo, r, t, K, dist)
            errs.append(float(np.sqrt(np.mean(np.sum((pr.reshape(-1, 2) - ii) ** 2, axis=1)))))
        return float(rms), K, dist.ravel(), errs

    rms, K, dist, errs = fit(obj_pts, img_pts)
    lim = max(3.0 * float(np.median(errs)), 1.0)
    keep = [j for j, e in enumerate(errs) if e <= lim]
    if MIN_VIEWS <= len(keep) < len(errs):
        obj_pts, img_pts = [obj_pts[j] for j in keep], [img_pts[j] for j in keep]
        rms, K, dist, errs = fit(obj_pts, img_pts)
    return rms, K, dist, len(obj_pts)


def to_profile(name: str, size: tuple[int, int], K, dist, rms: float, n: int, method: str) -> CameraProfile:
    w, h = size
    fx, fy = float(K[0, 0]), float(K[1, 1])
    key = match_profile(name)
    spec = BUILTIN[key] if key else None
    return CameraProfile(name, w, h, fx, fy, float(K[0, 2]), float(K[1, 2]),
                         [round(float(d), 6) for d in (list(dist) + [0.0] * 5)[:5]],
                         f"calibrated: {method}, {n} views, RMS {rms:.2f} px, {datetime.now():%Y-%m-%d %H:%M}",
                         round(dfov_from_focal(w, h, math.sqrt(fx * fy)), 2), spec.spec_url if spec else None,
                         round(rms, 4), spec.match if spec else [])


def verdict(prof: CameraProfile) -> list[str]:
    notes = []
    if prof.rms_px is not None and prof.rms_px > 1.0:
        notes.append(f"RMS {prof.rms_px:.2f} px is high: blurred or mis-detected views; retake sharper ones")
    if abs(prof.fx / prof.fy - 1) > 0.03:
        notes.append(f"fx / fy = {prof.fx / prof.fy:.3f}: pixels should be square; add tilted views")
    key = match_profile(prof.name)
    if key:
        spec = BUILTIN[key].scaled(prof.width, prof.height)
        off = prof.fx / spec.fx - 1
        notes.append(f"fx is {100 * off:+.1f} % from the {spec.name} spec ({spec.fx:.0f} px at "
                     f"{prof.width}x{prof.height}); every size-cue range moves by the same %")
    return notes


def collect(frames, pattern: str, board, chess: tuple[int, int], square_mm: float):
    import cv2
    obj_pts, img_pts, size = [], [], None
    for img in frames:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = (gray.shape[1], gray.shape[0])
        if pattern == "charuco":
            r = detect_charuco(gray, board)
            if r is not None:
                obj_pts.append(r[0])
                img_pts.append(r[1])
        else:
            from kite.calibrate import board_points, find_board
            c = find_board(gray, chess)
            if c is not None:
                obj_pts.append(board_points(chess, square_mm / 1000.0))
                img_pts.append(c)
    return obj_pts, img_pts, size


def live_frames(spec: str, seconds: float, every_s: float = 0.6, seen: dict | None = None):
    """Frames from a camera on the user's machine (prints progress; Ctrl-C ends early)."""
    from kite.sources import open_source
    src = open_source(spec)
    if seen is not None:
        seen["name"] = src.name.split(" ", 1)[-1]
    print(f"camera: {src.name}. Move the board around the frame and tilt it; {seconds:.0f} s.")
    t0, last = time.monotonic(), 0.0
    try:
        while time.monotonic() - t0 < seconds:
            ok, img = src.read()
            if ok and time.monotonic() - last >= every_s:
                last = time.monotonic()
                yield img
    except KeyboardInterrupt:
        pass
    finally:
        src.release()


def parse_wh(s: str) -> tuple[int, int]:
    a, b = (int(v) for v in s.lower().split("x"))
    return a, b


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m perception.calibrate_camera", description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--images", help="a folder (or one file) of board photos")
    src.add_argument("--video", help="a video of the board moving")
    src.add_argument("--live", help="a camera spec (webcam:Logitech): YOUR machine only")
    ap.add_argument("--print-board", help="write the board image to print or show, then exit")
    ap.add_argument("--pattern", choices=("charuco", "chessboard"), default="charuco")
    ap.add_argument("--squares", type=parse_wh, default=(7, 5), help="ChArUco squares across x down (default 7x5)")
    ap.add_argument("--board", type=parse_wh, default=(9, 6), help="chessboard INNER corners (default 9x6)")
    ap.add_argument("--square-mm", type=float, default=30.0, help="printed square side (only scales the poses)")
    ap.add_argument("--marker-mm", type=float, default=22.0, help="ChArUco marker side")
    ap.add_argument("--dict", default="DICT_5X5_100", help="ArUco dictionary")
    ap.add_argument("--seconds", type=float, default=45.0, help="--live: how long to collect")
    ap.add_argument("--name", help="camera name for the profile (default: the camera's own name, or 'Logitech C270')")
    ap.add_argument("--out", help="profile JSON path (default data/cameras/<name>.json)")
    ap.add_argument("--free-k3", action="store_true", help="also fit k3 (wide lenses; needs many edge views)")
    a = ap.parse_args(argv)
    if a.print_board:
        p = print_board(a.print_board, a.pattern, a.squares, a.board, a.square_mm, a.marker_mm, a.dict)
        print(f"wrote {p}: print at 100 % scale (300 dpi) and measure a square; pass --square-mm")
        return 0
    if not (a.images or a.video or a.live):
        ap.error("give --images, --video or --live (or --print-board)")
    board = charuco_board(a.squares, a.square_mm, a.marker_mm, a.dict) if a.pattern == "charuco" else None
    seen: dict = {}
    frames = live_frames(a.live, a.seconds, seen=seen) if a.live else frames_from(a.images, a.video)
    obj_pts, img_pts, size = collect(frames, a.pattern, board, a.board, a.square_mm)
    print(f"board found in {len(obj_pts)} views")
    if len(obj_pts) < MIN_VIEWS:
        print(f"need at least {MIN_VIEWS} views with the board found; take more (see --help)")
        return 1
    rms, K, dist, n = solve(obj_pts, img_pts, size, not a.free_k3)
    usb = seen.get("name")
    name = a.name or (BUILTIN[match_profile(usb)].name if match_profile(usb) else usb) or "Logitech C270"
    prof = to_profile(name, size, K, dist, rms, n, a.pattern)
    out = Path(a.out) if a.out else calibrated_path(prof.name)
    prof.save(out)
    print(f"wrote {out}\n  {prof.describe()}")
    for note in verdict(prof):
        print(f"  - {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
