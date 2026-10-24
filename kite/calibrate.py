"""Measure the Tello camera's intrinsics (fx, fy, cx, cy + distortion) and write them to data/tello_camera.json.

    python -m kite.calibrate --show-board                          # full-screen chessboard on a second screen
    python -m kite.calibrate --board-png board.png                 # ... or save it, to open on a phone / tablet
    python -m kite.calibrate --source tello                        # drone on the ground, ~30-45 s of video
    python -m kite.calibrate --source recordings/calib_20261024_0100   # re-run on the frames the Tello run saved
    python -m kite.calibrate --source recordings/20261024_070000_tello # a kite.record folder (video.mkv or frames/)
    python -m kite.calibrate --source clip.mp4                     # any video, or a folder of images
    python -m kite.calibrate --source tello --known-object 0.60 --distance-m 2.0   # one-frame fallback, no board

Chessboard method (OpenCV findChessboardCorners + cornerSubPix + calibrateCamera). --board is the number of INNER
corners (default 9x6, i.e. 10x7 squares). The square size only scales the board poses (extrinsics), never fx / fy /
cx / cy, so it defaults to 1.0 and a board on a laptop or phone screen needs no measuring.

With --source tello the drone is opened with dry_run=True (NO motion command can be sent), video only. A frame is
captured automatically when the whole board is found, the board is held still (sharp), and the view differs enough
from the views already captured. It stops at --n views or --max-s seconds after the first frame (battery), then
closes the drone link BEFORE computing, and keeps the captured frames on disk so a re-run costs no battery.
Good views: the board at 0.3-1.5 m, in every part of the frame (centre, corners, edges), tilted 20-45 deg in
different directions. Flat-on views alone cannot separate fx from the distance.

Fallback: --known-object WIDTH_M --distance-m D. One frame of an object of known width at a taped distance D
(camera lens to the object's plane, object facing the camera near the image centre); click its two edges (several
pairs are averaged) and fx = pixel_width * D / WIDTH_M. fy = fx, cx / cy = image centre, no distortion.

The file holds kite.camera.Camera's fields plus extra keys (dist, rms_px, n_views, method, ...), which
load_camera() ignores. Nothing is saved with fewer than --min-views good views or a clearly wrong result (--force).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from .camera import DEFAULT_PATH, Camera, save_camera
from .config import VIDEO_BACKENDS
from .sources.base import FrameSource

TELLO_SIZE = (960, 720)
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
VIDEO_EXT = (".mkv", ".mp4", ".avi", ".mov", ".m4v", ".h264")
FX_OK = (850.0, 1150.0)  # published Tello stream calibrations: 920 to 1077 px
FX_PLAUSIBLE = (700.0, 1400.0)  # outside this the result is wrong, not just unusual: refuse to save without --force
RMS_OK, RMS_PLAUSIBLE = 1.0, 2.0  # px
MIN_VIEWS = 8
MAX_FILE_VIEWS = 40  # views kept from a video / folder (calibrateCamera stays fast)
MAX_SAMPLED_FRAMES = 400  # a video is searched on at most this many frames (evenly spaced)
STILL_PX = 3.0  # live: mean corner shift between consecutive frames below this = board held still (no motion blur)
MIN_MOVE_FRAC = 0.04  # a new view's corners must move this fraction of the image diagonal from every captured view


@dataclass
class CalibratedCamera(Camera):
    """Camera plus what the calibration measured. load_camera() reads only the Camera fields back."""

    dist: list = field(default_factory=list)  # OpenCV order k1 k2 p1 p2 k3
    rms_px: float | None = None  # RMS reprojection error
    n_views: int = 0
    method: str = ""  # chessboard | known-object
    calibrated_at: str = ""


@dataclass
class Calibration:
    camera: CalibratedCamera
    rms: float
    per_view_px: list  # RMS reprojection error of each view kept
    tilts_deg: list  # angle between the board's normal and the optical axis, per view kept
    dropped: int = 0  # outlier views removed before the final fit


# ------------------------------------------------------------------ board
def parse_board(s: str) -> tuple[int, int]:
    try:
        c, r = (int(v) for v in s.lower().split("x"))
    except ValueError:
        raise argparse.ArgumentTypeError(f"--board {s!r}: use COLSxROWS inner corners, e.g. 9x6") from None
    if c < 3 or r < 3 or c == r:
        raise argparse.ArgumentTypeError("--board needs >= 3 inner corners each way and cols != rows (orientation)")
    return c, r


def board_image(pattern: tuple[int, int], square_px: int = 80, size: tuple[int, int] | None = None) -> np.ndarray:
    """White canvas with a chessboard of pattern = inner corners (cols, rows), at least one square of white margin
    (the detector needs the quiet zone). size = (w, h) centres it on a canvas of that size (e.g. 16:9)."""
    cols, rows = pattern[0] + 1, pattern[1] + 1
    bw, bh = cols * square_px, rows * square_px
    w, h = size or (bw + 2 * square_px, bh + 2 * square_px)
    w, h = max(w, bw + 2 * square_px), max(h, bh + 2 * square_px)
    img = np.full((h, w), 255, np.uint8)
    x0, y0 = (w - bw) // 2, (h - bh) // 2
    for j in range(rows):
        for i in range(cols):
            if (i + j) % 2 == 0:
                img[y0 + j * square_px:y0 + (j + 1) * square_px, x0 + i * square_px:x0 + (i + 1) * square_px] = 0
    return img


def show_board(pattern: tuple[int, int]) -> None:
    """Full-screen chessboard in an OpenCV window until a key is pressed. Point the camera at this screen."""
    import cv2

    img = board_image(pattern, square_px=120, size=(1920, 1080))
    win = "kite calibration board (any key closes)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(win, img)
    print(f"showing a {pattern[0]}x{pattern[1]} inner-corner board full screen: brightness up, any key closes")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def board_points(pattern: tuple[int, int], square: float = 1.0) -> np.ndarray:
    """The inner corners on the board plane (z = 0), in findChessboardCorners order (x fastest)."""
    obj = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    obj[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * square
    return obj


# ------------------------------------------------------------------ detection and view selection
def find_board(gray: np.ndarray, pattern: tuple[int, int]) -> np.ndarray | None:
    """Sub-pixel inner corners (N x 2 float32) if the whole board is visible, else None."""
    import cv2

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    ok, corners = cv2.findChessboardCorners(gray, pattern, flags=flags)
    if not ok:
        return None
    pts = corners.reshape(pattern[1], pattern[0], 2)
    # the refinement window must stay inside one square, or it snaps to a neighbouring corner
    sq = min(np.linalg.norm(np.diff(pts, axis=1), axis=2).min(), np.linalg.norm(np.diff(pts, axis=0), axis=2).min())
    half = int(np.clip(sq / 4, 2, 11))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01)
    corners = cv2.cornerSubPix(gray, corners, (half, half), (-1, -1), crit)
    return corners.reshape(-1, 2).astype(np.float32)


def _shift(a: np.ndarray, b: np.ndarray) -> float:
    """Mean corner distance between two detections (the board may be detected from either end)."""
    return float(min(np.linalg.norm(a - b, axis=1).mean(), np.linalg.norm(a - b[::-1], axis=1).mean()))


class ViewSelector:
    """Decides which detections become calibration views.

    still_px: the board must have moved less than this since the previous detection (sharp, no rolling-shutter
    smear); None skips the check (unrelated still images). min_move_px: a view is kept only if its corners moved at
    least this far (mean) from every view already kept.
    """

    def __init__(self, min_move_px: float, still_px: float | None = STILL_PX):
        self.min_move_px, self.still_px = min_move_px, still_px
        self.views: list[np.ndarray] = []
        self._prev: np.ndarray | None = None

    def offer(self, corners: np.ndarray | None) -> str:
        prev, self._prev = self._prev, corners
        if corners is None:
            return "no board"
        if self.still_px is not None and (prev is None or _shift(prev, corners) > self.still_px):
            return "hold still"
        if any(_shift(v, corners) < self.min_move_px for v in self.views):
            return "move / tilt to a new view"
        self.views.append(corners)
        return "captured"


# ------------------------------------------------------------------ calibration
def calibrate(views: list[np.ndarray], size: tuple[int, int], pattern: tuple[int, int], square: float = 1.0,
              min_views: int = MIN_VIEWS) -> Calibration:
    """cv2.calibrateCamera on the views (k3 fixed: too few views to pin it down). Views whose error is far above
    the others (a mis-ordered or smeared detection) are dropped once, if enough remain."""
    import cv2

    obj = board_points(pattern, square)

    def fit(vs):
        img = [v.reshape(-1, 1, 2).astype(np.float32) for v in vs]
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera([obj] * len(vs), img, size, None, None, flags=cv2.CALIB_FIX_K3)
        errs = []
        for v, r, t in zip(vs, rvecs, tvecs):
            proj, _ = cv2.projectPoints(obj, r, t, K, dist)
            errs.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - v) ** 2, axis=1)))))
        return float(rms), K, dist.ravel(), rvecs, errs

    if len(views) < 3:
        raise ValueError(f"{len(views)} views: calibration needs at least 3 (and {min_views} to save)")
    rms, K, dist, rvecs, errs = fit(views)
    dropped = 0
    limit = max(3.0 * float(np.median(errs)), 1.0)
    keep = [v for v, e in zip(views, errs) if e <= limit]
    if len(keep) < len(views) and len(keep) >= max(min_views, 3):
        dropped = len(views) - len(keep)
        views = keep
        rms, K, dist, rvecs, errs = fit(views)
    tilts = [math.degrees(math.acos(min(1.0, abs(cv2.Rodrigues(r)[0][2, 2])))) for r in rvecs]
    cam = CalibratedCamera(
        width=size[0], height=size[1], fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]),
        source=f"chessboard calibration, {len(views)} views, RMS {rms:.2f} px",
        dist=[round(float(d), 6) for d in dist], rms_px=round(rms, 4), n_views=len(views), method="chessboard",
        calibrated_at=datetime.now().isoformat(timespec="seconds"),
    )
    return Calibration(cam, rms, errs, tilts, dropped)


def fx_from_known_object(px_width: float, width_m: float, distance_m: float) -> float:
    """Pinhole: an object W m wide at depth D m spans fx * W / D px, so fx = px * D / W."""
    if px_width <= 0 or width_m <= 0 or distance_m <= 0:
        raise ValueError("pixel width, object width and distance must all be > 0")
    return px_width * distance_m / width_m


def known_object_camera(px_widths: list[float], width_m: float, distance_m: float, size: tuple[int, int]) -> CalibratedCamera:
    px = float(np.mean(px_widths))
    fx = fx_from_known_object(px, width_m, distance_m)
    return CalibratedCamera(
        width=size[0], height=size[1], fx=fx, fy=fx, cx=size[0] / 2, cy=size[1] / 2,
        source=f"known object: {width_m:g} m at {distance_m:g} m = {px:.1f} px (mean of {len(px_widths)} clicks)",
        n_views=1, method="known-object", calibrated_at=datetime.now().isoformat(timespec="seconds"),
    )


def verdict(cam: Camera, rms: float | None = None) -> tuple[list[str], list[str]]:
    """(warnings, errors). Errors mean the result is wrong rather than unusual: it is not saved without --force."""
    warn, err = [], []
    for name, v in (("fx", cam.fx), ("fy", cam.fy)):
        if not FX_PLAUSIBLE[0] <= v <= FX_PLAUSIBLE[1]:
            err.append(f"{name} {v:.0f} px is far outside the Tello's published 920-1077 px")
        elif not FX_OK[0] <= v <= FX_OK[1]:
            warn.append(f"{name} {v:.0f} px is outside {FX_OK[0]:.0f}-{FX_OK[1]:.0f} px (published 920-1077)")
    if abs(cam.fx - cam.fy) > 0.03 * cam.fx:
        warn.append(f"fx and fy differ by {abs(cam.fx - cam.fy) / cam.fx:.0%} (square pixels expected): add tilted views")
    if abs(cam.cx - cam.width / 2) > 0.1 * cam.width or abs(cam.cy - cam.height / 2) > 0.1 * cam.height:
        warn.append(f"principal point ({cam.cx:.0f}, {cam.cy:.0f}) is far from the image centre: cover the corners")
    if (cam.width, cam.height) != TELLO_SIZE:
        warn.append(f"frames are {cam.width}x{cam.height}, the Tello stream is {TELLO_SIZE[0]}x{TELLO_SIZE[1]}")
    if rms is not None:
        if rms > RMS_PLAUSIBLE:
            err.append(f"RMS reprojection error {rms:.2f} px > {RMS_PLAUSIBLE:g} px: bad detections or a moving board")
        elif rms > RMS_OK:
            warn.append(f"RMS reprojection error {rms:.2f} px > {RMS_OK:g} px (sharper / stiller views help)")
    return warn, err


def report(cal: Calibration) -> str:
    c = cal.camera
    lines = [f"calibrated from {c.n_views} views ({c.width}x{c.height})"
             + (f", {cal.dropped} outlier view(s) dropped" if cal.dropped else ""),
             f"  fx {c.fx:.1f} px   fy {c.fy:.1f} px   cx {c.cx:.1f}   cy {c.cy:.1f}",
             f"  RMS reprojection error {cal.rms:.3f} px (worst view {max(cal.per_view_px):.2f} px)",
             f"  hfov {c.hfov_deg:.1f} deg   vfov {c.vfov_deg:.1f} deg",
             "  distortion k1 k2 p1 p2 k3: " + " ".join(f"{d:+.4f}" for d in c.dist)]
    tilted = sum(t > 15 for t in cal.tilts_deg)
    lines.append(f"  board tilt {min(cal.tilts_deg):.0f}-{max(cal.tilts_deg):.0f} deg ({tilted} views over 15 deg)")
    if tilted < 3:
        lines.append("  WARNING: fewer than 3 tilted views: fx is poorly separated from the board distance")
    return "\n".join(lines)


# ------------------------------------------------------------------ inputs
def _images_in(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXT)


def resolve_source(source: str) -> tuple[str, Path]:
    """('video', file) or ('images', [folder]) for a file, an image folder, or a kite.record output folder."""
    p = Path(source)
    if p.is_dir():
        if (p / "video.mkv").is_file():
            return "video", p / "video.mkv"
        if (p / "frames").is_dir() and _images_in(p / "frames"):
            return "images", p / "frames"
        if _images_in(p):
            return "images", p
        raise ValueError(f"{p}: no video.mkv, frames/ or images ({', '.join(IMAGE_EXT)}) in this folder")
    if p.is_file():
        return ("image", p) if p.suffix.lower() in IMAGE_EXT else ("video", p)
    raise ValueError(f"no such file or folder: {source}")


def iter_frames(kind: str, path: Path, max_frames: int = MAX_SAMPLED_FRAMES):
    """Yields (step, BGR image). A video is decoded in full but only every step-th frame is yielded."""
    import cv2

    if kind == "image":
        img = cv2.imread(str(path))
        if img is not None:
            yield 1, img
        return
    if kind == "images":
        for f in _images_in(path):
            img = cv2.imread(str(f))
            if img is not None:
                yield 1, img
        return
    from .sources.opencv_sources import SteppedVideoSource

    cap = cv2.VideoCapture(str(path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    step = max(1, math.ceil(count / max_frames)) if count > 0 else 1
    src = SteppedVideoSource(path)
    try:
        i = 0
        while (f := src.read()) is not None:
            if i % step == 0:
                yield step, f.image
            i += 1
    finally:
        src.stop()


def collect_from_files(kind: str, path: Path, pattern: tuple[int, int], max_views: int = MAX_FILE_VIEWS):
    """Board views from a video / image folder. Returns (views, size, n_frames, n_found, n_wrong_size)."""
    import cv2

    sel: ViewSelector | None = None
    size = None
    n = found = wrong = 0
    for step, img in iter_frames(kind, path):
        n += 1
        if size is None:
            size = (img.shape[1], img.shape[0])
            # a video is sampled every `step` frames: allow the same board speed as the live check
            sel = ViewSelector(MIN_MOVE_FRAC * math.hypot(*size), STILL_PX * step if kind == "video" else None)
        if (img.shape[1], img.shape[0]) != size:
            wrong += 1  # calibrateCamera needs one image size
            continue
        corners = find_board(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), pattern)
        found += corners is not None
        sel.offer(corners)
    views = sel.views if sel else []
    if len(views) > max_views:  # keep an even spread over the recording
        views = [views[int(i)] for i in np.linspace(0, len(views) - 1, max_views)]
    return views, size, n, found, wrong


def capture_live(frames: FrameSource, pattern: tuple[int, int], n: int = 15, max_s: float = 45.0,
                 preview: bool = True, save_dir: Path | None = None, battery=None):
    """Auto-capture board views from a live source until n views or max_s seconds after the first frame (or q / Esc /
    Ctrl+C). Returns (views, size). Each captured frame is saved to save_dir (re-run with --source save_dir)."""
    import cv2

    first = frames.wait_first(15.0)
    if first is None:
        stats = frames.stats() if hasattr(frames, "stats") else None
        raise RuntimeError("no video frame within 15 s" + (f" (video reader: {stats})" if stats else ""))
    size = (first.width, first.height)
    if size != TELLO_SIZE:
        print(f"WARNING: frames are {size[0]}x{size[1]}, the Tello stream is {TELLO_SIZE[0]}x{TELLO_SIZE[1]}")
    sel = ViewSelector(MIN_MOVE_FRAC * math.hypot(*size), STILL_PX)
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
    win = "kite calibrate (q quits)"
    t0, last_seq, state, bat_s, bat_t = time.time(), None, "", "", 0.0
    try:
        while len(sel.views) < n and time.time() - t0 < max_s:
            f = frames.read()
            if f is None or f.seq == last_seq or (f.width, f.height) != size:
                if preview and cv2.waitKey(5) & 0xFF in (ord("q"), 27):
                    break
                elif not preview:
                    time.sleep(0.005)
                continue
            last_seq = f.seq
            corners = find_board(cv2.cvtColor(f.image, cv2.COLOR_BGR2GRAY), pattern)
            state = sel.offer(corners)
            if state == "captured":
                k = len(sel.views)
                if save_dir is not None:
                    cv2.imwrite(str(save_dir / f"view_{k:02d}.png"), f.image)
                print(f"\a  captured {k}/{n} at {time.time() - t0:.0f} s", flush=True)
            if battery is not None and time.time() - bat_t > 2.0:
                bat_t, b = time.time(), battery()
                bat_s = f" | bat {b:.0f}%" if b is not None else ""
            if preview:
                show = f.image.copy()  # frames are read-only
                for v in sel.views:  # outline of every captured view: aim for the empty parts of the frame
                    quad = v.reshape(pattern[1], pattern[0], 2)[[0, 0, -1, -1], [0, -1, -1, 0]]
                    cv2.polylines(show, [quad.astype(np.int32)], True, (0, 200, 0), 1)
                if corners is not None:
                    cv2.drawChessboardCorners(show, pattern, corners.reshape(-1, 1, 2), True)
                txt = f"{len(sel.views)}/{n} views | {max_s - (time.time() - t0):.0f} s left | {state}{bat_s}"
                cv2.rectangle(show, (0, 0), (show.shape[1], 34), (0, 0, 0), -1)
                cv2.putText(show, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow(win, show)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        print("\nCtrl+C: stopping the capture")
    finally:
        if preview:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
    print(f"capture done: {len(sel.views)} views in {time.time() - t0:.0f} s")
    return sel.views, size


# ------------------------------------------------------------------ known-object fallback
def click_widths(image: np.ndarray, width_m: float, distance_m: float) -> list[float]:
    """Click the object's left and right edges (repeat for more pairs, averaged). Enter / space accepts, u undoes,
    Esc cancels. Returns the pixel width of each pair."""
    import cv2

    pts: list[tuple[int, int]] = []
    win = "click the two edges (Enter accepts, u undo, Esc cancel)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, lambda ev, x, y, *_: pts.append((x, y)) if ev == cv2.EVENT_LBUTTONDOWN else None)
    widths: list[float] = []
    try:
        while True:
            show = image.copy()
            widths = [math.dist(pts[i], pts[i + 1]) for i in range(0, len(pts) - 1, 2)]
            for i, w in enumerate(widths):
                a, b = pts[2 * i], pts[2 * i + 1]
                cv2.line(show, a, b, (0, 255, 0), 1)
                cv2.putText(show, f"{w:.0f} px", (a[0], a[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            for p in pts:
                cv2.circle(show, p, 3, (0, 0, 255), -1)
            txt = "click left edge, then right edge"
            if widths:
                txt = (f"{len(widths)} pair(s), mean {np.mean(widths):.1f} px -> fx "
                       f"{fx_from_known_object(float(np.mean(widths)), width_m, distance_m):.0f}")
            cv2.rectangle(show, (0, 0), (show.shape[1], 34), (0, 0, 0), -1)
            cv2.putText(show, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(win, show)
            k = cv2.waitKey(20) & 0xFF
            if k in (13, 10, 32) and widths:
                return widths
            if k == ord("u") and pts:
                pts.pop()
            if k == 27:
                return []
    finally:
        cv2.destroyAllWindows()
        cv2.waitKey(1)


def freeze_live_frame(frames: FrameSource, preview: bool = True) -> np.ndarray | None:
    """Live preview until space (keeps that frame) or Esc. Without a preview: the first frame."""
    import cv2

    first = frames.wait_first(15.0)
    if first is None or not preview:
        return None if first is None else first.image.copy()
    win = "aim at the object: space freezes the frame, Esc cancels"
    try:
        while True:
            f = frames.read()
            cv2.imshow(win, f.image)
            k = cv2.waitKey(15) & 0xFF
            if k == 32:
                return f.image.copy()
            if k == 27:
                return None
    finally:
        cv2.destroyAllWindows()
        cv2.waitKey(1)


# ------------------------------------------------------------------ CLI
def _save(cam: CalibratedCamera, rms: float | None, args) -> int:
    warn, err = verdict(cam, rms)
    for w in warn:
        print(f"  WARNING: {w}")
    for e in err:
        print(f"  BAD: {e}")
    print("verdict: " + ("BAD" if err else "CHECK the warnings" if warn else "OK")
          + f" (expected fx {FX_OK[0]:.0f}-{FX_OK[1]:.0f} px" + (f", RMS < {RMS_OK:g} px)" if rms is not None else ")"))
    if args.no_save:
        print("not saved (--no-save)")
        return 0
    if err and not args.force:
        print(f"NOT saved to {args.out}: fix the views and re-run (or --force)")
        return 2
    print(f"saved {save_camera(cam, args.out)}")
    return 0


def _open_tello(args):
    from .config import Config
    from .drone import make_tello

    cfg = Config()
    if args.video_backend:
        cfg.drone.video_backend = args.video_backend
    drone = make_tello(cfg, dry_run=True)  # dry run: no motion command can ever be sent
    print("connecting to the Tello (DRY RUN: video only, no motion) ...", flush=True)
    try:
        drone.connect()
    except Exception as e:
        drone.close()
        print(f"cannot connect: {e}\nJoin the TELLO-xxxxxx Wi-Fi, close the phone app, and check that "
              f"`lsof -nP -iUDP:8889 -iUDP:8890 -iUDP:11111` prints nothing.")
        return None
    return drone


def _battery(drone):
    return lambda: drone.telemetry().battery_pct


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m kite.calibrate", description="Measure the Tello camera's fx/fy/cx/cy.")
    p.add_argument("--source", help="'tello', a video, a kite.record folder, a folder of images, or one image")
    p.add_argument("--board", type=parse_board, default=(9, 6), help="inner corners COLSxROWS (default 9x6)")
    p.add_argument("--square", type=float, default=1.0, help="square size, any unit; only scales the extrinsics, "
                   "never fx (default 1.0)")
    p.add_argument("--n", type=int, default=15, help="tello: stop after this many views (default 15)")
    p.add_argument("--max-s", type=float, default=45.0, help="tello: stop this many seconds after the first frame "
                   "(default 45)")
    p.add_argument("--min-views", type=int, default=MIN_VIEWS, help=f"refuse to save below this (default {MIN_VIEWS})")
    p.add_argument("--out", default=str(DEFAULT_PATH), help="output json (default data/tello_camera.json)")
    p.add_argument("--no-save", action="store_true", help="report only")
    p.add_argument("--force", action="store_true", help="save even a result that fails the sanity check")
    p.add_argument("--no-preview", action="store_true", help="no OpenCV windows")
    p.add_argument("--frames-dir", default=None, help="tello: where the captured frames go "
                   "(default recordings/calib_<stamp>)")
    p.add_argument("--video-backend", choices=VIDEO_BACKENDS, default=None,
                   help="Tello H.264 reader (default pyav; opt-in: raw, socket, auto = raw on macOS)")
    p.add_argument("--show-board", action="store_true", help="show the chessboard full screen and exit")
    p.add_argument("--board-png", default=None, help="write the chessboard to this PNG and exit")
    p.add_argument("--known-object", type=float, metavar="WIDTH_M", help="fallback: width of an object (m)")
    p.add_argument("--distance-m", type=float, help="fallback: camera to object distance (m)")
    p.add_argument("--px-width", type=float, help="fallback: the object's width in px (skips clicking)")
    p.add_argument("--frame", type=int, default=0, help="fallback on a video: frame number to measure (default 0)")
    args = p.parse_args(argv)

    if args.board_png or args.show_board:
        if args.board_png:
            import cv2

            cv2.imwrite(args.board_png, board_image(args.board, square_px=120, size=(1920, 1080)))
            print(f"wrote {args.board_png}: open it full screen, brightness up")
        if args.show_board:
            show_board(args.board)
        return 0
    if not args.source:
        p.error("--source is required (or --show-board / --board-png)")
    if (args.known_object is None) != (args.distance_m is None):
        p.error("--known-object WIDTH_M needs --distance-m D (and vice versa)")
    tello = args.source.lower() == "tello"

    if args.known_object is not None:
        return _known_object(args, tello)

    if tello:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        frames_dir = Path(args.frames_dir or f"recordings/calib_{stamp}")
        drone = _open_tello(args)
        if drone is None:
            return 1
        try:
            print(f"battery {drone.telemetry().battery_pct}%  |  show the board: 0.3-1.5 m, all over the frame, tilted "
                  f"20-45 deg, hold still a moment (beep = captured)", flush=True)
            views, size = capture_live(drone.frame_source(), args.board, args.n, args.max_s, not args.no_preview,
                                       frames_dir, _battery(drone))
        except RuntimeError as e:
            print(f"capture failed: {e}")
            return 1
        finally:
            print("closing the drone link (stream off) ...", flush=True)
            drone.close()  # before the (slow-ish) maths: the stream costs battery
        print(f"captured frames kept in {frames_dir} (re-run: python -m kite.calibrate --source {frames_dir})")
    else:
        try:
            kind, path = resolve_source(args.source)
        except ValueError as e:
            p.error(str(e))
        views, size, n, found, wrong = collect_from_files(kind, path, args.board)
        print(f"{path}: {n} frames searched, board in {found}, {len(views)} distinct sharp views"
              + (f", {wrong} frames of another size skipped" if wrong else ""))

    if len(views) < args.min_views:
        print(f"only {len(views)} good views (need {args.min_views}): NOT saved. More distinct views: move the board "
              f"to new parts of the frame and tilt it.")
        return 2
    cal = calibrate(views, size, args.board, args.square, args.min_views)
    print(report(cal))
    return _save(cal.camera, cal.rms, args)


def _known_object(args, tello: bool) -> int:
    import cv2

    if args.known_object <= 0 or args.distance_m <= 0:
        print("--known-object and --distance-m must be > 0")
        return 2
    if tello:
        drone = _open_tello(args)
        if drone is None:
            return 1
        try:
            img = freeze_live_frame(drone.frame_source(), not args.no_preview)
        finally:
            drone.close()  # one frame is all we need
    else:
        try:
            kind, path = resolve_source(args.source)
        except ValueError as e:
            print(e)
            return 2
        img = None
        for i, (_, im) in enumerate(iter_frames(kind, path, max_frames=10 ** 9)):
            if i == args.frame:
                img = im
                break
    if img is None:
        print("no frame to measure")
        return 1
    size = (img.shape[1], img.shape[0])
    if args.px_width is not None:
        widths = [args.px_width]
    elif args.no_preview:
        print("--no-preview needs --px-width (nothing to click on)")
        return 2
    else:
        widths = click_widths(img, args.known_object, args.distance_m)
    if not widths:
        print("cancelled")
        return 1
    cam = known_object_camera(widths, args.known_object, args.distance_m, size)
    print(f"known object {args.known_object:g} m at {args.distance_m:g} m: {np.mean(widths):.1f} px "
          f"(mean of {len(widths)}) -> fx {cam.fx:.1f} px, hfov {cam.hfov_deg:.1f} deg ({size[0]}x{size[1]})")
    return _save(cam, None, args)


if __name__ == "__main__":
    sys.exit(main())
