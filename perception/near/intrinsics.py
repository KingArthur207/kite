"""Camera intrinsics for the chest webcam: spec profiles, calibrated profiles, auto-selection, undistortion.

A pinhole camera maps a point (X, Y, Z) in its own frame (x right, y down, z forward) to the pixel
u = fx X/Z + cx, v = fy Y/Z + cy. Every range cue in perception/near depends on fx and fy: the size cue
is linear in them (Z = fy H / h_px), the ground cue through the angle below the horizon. So the first
thing the module does is find out which camera it is looking through.

Built-in profiles come from the manufacturer's diagonal field of view (dFOV) at the native video mode:

    f = (diagonal in px / 2) / tan(dFOV / 2)        square pixels, principal point at the centre

| Camera | Native mode | dFOV | f at native | Spec page |
|---|---|---|---|---|
| Logitech C270 | 1280x720 | 55 deg | 1410.5 px | https://www.logitech.com/en-us/products/webcams/c270-hd-webcam.960-000694.html |
| Logitech C310 | 1280x720 | 60 deg | 1271.8 px | https://www.logitech.com/en-us/products/webcams/c310-hd-webcam.html |
| Logitech C920 | 1920x1080 | 78 deg | 1360.1 px | https://www.logitech.com/en-us/products/webcams/c920-pro-hd-webcam.960-001055.html |

They are "from spec, calibrate for accuracy": a spec dFOV is rounded to the degree (1 degree is about 2 %
of f, so 2 % of every size-cue range), the lens distortion is unknown (zeros), and a 4:3 mode such as
640x480 may crop the sensor differently from the 16:9 mode the spec describes. `python -m
perception.calibrate_camera` measures all of that and writes a profile JSON that replaces the spec one.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROFILE_DIR = ROOT / "data" / "cameras"          # calibrated profiles land here (one JSON per camera)

SPEC_NOTE = "from spec, calibrate for accuracy"


@dataclass
class CameraProfile:
    """Intrinsics at one resolution. `dist` is OpenCV's order (k1, k2, p1, p2, k3)."""
    name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist: list[float] = field(default_factory=lambda: [0.0] * 5)
    source: str = SPEC_NOTE                 # "from spec, ..." or "calibrated ..."
    dfov_deg: float | None = None
    spec_url: str | None = None
    rms_px: float | None = None             # calibration reprojection error, if calibrated
    match: list[str] = field(default_factory=list)   # USB-name patterns that select this profile

    # ------------------------------------------------------------------ derived
    @property
    def calibrated(self) -> bool:
        return self.source.startswith("calibrated")

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], float)

    @property
    def hfov_deg(self) -> float:
        return math.degrees(2 * math.atan(self.width / (2 * self.fx)))

    @property
    def vfov_deg(self) -> float:
        return math.degrees(2 * math.atan(self.height / (2 * self.fy)))

    @property
    def has_distortion(self) -> bool:
        return any(abs(float(d)) > 1e-9 for d in self.dist)

    def scaled(self, width: int, height: int) -> CameraProfile:
        """The same camera at another frame size. Same aspect ratio: every term scales (a downscaled mode).
        Another aspect ratio (640x480 from a 16:9 spec): ASSUMED to keep the horizontal field of view with
        square pixels, the principal point at the centre; only a calibration at that mode can confirm it."""
        if (width, height) == (self.width, self.height):
            return self
        sx, sy = width / self.width, height / self.height
        if abs(sx - sy) < 1e-3:
            return replace(self, width=int(width), height=int(height), fx=self.fx * sx, fy=self.fy * sy,
                           cx=self.cx * sx, cy=self.cy * sy)
        f = self.fx * sx
        return replace(self, width=int(width), height=int(height), fx=f, fy=f, cx=width / 2, cy=height / 2,
                       source=f"{self.source}; {width}x{height} assumed to keep the horizontal FOV")

    def resized(self, width: int, height: int) -> CameraProfile:
        """The same pixels resampled to width x height (each axis independently: a depth model's input)."""
        sx, sy = width / self.width, height / self.height
        return replace(self, width=int(width), height=int(height), fx=self.fx * sx, fy=self.fy * sy,
                       cx=self.cx * sx, cy=self.cy * sy)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hfov_deg"], d["vfov_deg"] = round(self.hfov_deg, 2), round(self.vfov_deg, 2)
        return d

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return p

    @classmethod
    def load(cls, path: str | Path) -> CameraProfile:
        d = json.loads(Path(path).read_text())
        keys = cls.__dataclass_fields__
        prof = cls(**{k: d[k] for k in keys if k in d})
        prof.dist = [float(x) for x in (prof.dist or [])][:5] + [0.0] * max(0, 5 - len(prof.dist or []))
        return prof

    # ------------------------------------------------------------------ undistortion
    def undistort_points(self, pts: np.ndarray, iters: int = 8) -> np.ndarray:
        """Distorted pixels (N x 2) -> ideal pinhole pixels (N x 2) under the same K. Brown-Conrady inverse by
        fixed-point iteration (OpenCV's undistortPoints does the same); no OpenCV needed."""
        pts = np.asarray(pts, float).reshape(-1, 2)
        if not self.has_distortion:
            return pts.copy()
        k1, k2, p1, p2, k3 = (list(self.dist) + [0.0] * 5)[:5]
        xd = (pts[:, 0] - self.cx) / self.fx
        yd = (pts[:, 1] - self.cy) / self.fy
        x, y = xd.copy(), yd.copy()
        for _ in range(iters):
            r2 = x * x + y * y
            radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
            dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
            dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
            x = (xd - dx) / radial
            y = (yd - dy) / radial
        return np.stack([x * self.fx + self.cx, y * self.fy + self.cy], axis=1)

    def distort_points(self, pts: np.ndarray) -> np.ndarray:
        """Ideal pinhole pixels -> distorted pixels (the forward model; synthetic renders and tests)."""
        pts = np.asarray(pts, float).reshape(-1, 2)
        k1, k2, p1, p2, k3 = (list(self.dist) + [0.0] * 5)[:5]
        x = (pts[:, 0] - self.cx) / self.fx
        y = (pts[:, 1] - self.cy) / self.fy
        r2 = x * x + y * y
        radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
        xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        return np.stack([xd * self.fx + self.cx, yd * self.fy + self.cy], axis=1)

    def undistort_box(self, x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float]:
        """Axis-aligned box in distorted pixels -> the box around its undistorted corners and edge midpoints."""
        if not self.has_distortion:
            return x1, y1, x2, y2
        xm, ym = (x1 + x2) / 2, (y1 + y2) / 2
        p = self.undistort_points([(x1, y1), (xm, y1), (x2, y1), (x1, ym), (x2, ym), (x1, y2), (xm, y2), (x2, y2)])
        # the extreme edge points, not the hull: a straight edge bends under distortion; use its midpoint row
        return float(min(p[0, 0], p[3, 0], p[5, 0])), float(p[1, 1]), float(max(p[2, 0], p[4, 0], p[7, 0])), \
            float(p[6, 1])

    def undistort_image(self, img: np.ndarray) -> np.ndarray:
        """Whole frame through cv2.undistort (the HUD only; the estimator undistorts the few points it needs)."""
        if not self.has_distortion:
            return img
        import cv2
        return cv2.undistort(img, self.K, np.asarray(self.dist, float))

    # ------------------------------------------------------------------ the startup line
    def describe(self, person_h: float = 1.7, at_m: float = 2.0) -> str:
        """One line naming what was loaded, plus one fact that makes the numbers tangible."""
        px = self.fy * person_h / at_m
        tag = "calibrated" if self.calibrated else SPEC_NOTE
        fact = (f"a {person_h:.2f} m person at {at_m:g} m stands {px:.0f} px tall"
                + (" - taller than the frame, so close people are ranged by their shoulders"
                   if px > self.height else f", {100 * px / self.height:.0f} % of the frame"))
        dist = " with lens distortion" if self.has_distortion else ""
        return (f"{self.name} ({tag}): fx {self.fx:.0f} / fy {self.fy:.0f} px at {self.width}x{self.height}, "
                f"{self.hfov_deg:.1f} x {self.vfov_deg:.1f} deg{dist}; {fact}.")


def focal_from_dfov(width: int, height: int, dfov_deg: float) -> float:
    """Diagonal field of view -> focal length in px (square pixels, centred principal point)."""
    if not 1.0 < dfov_deg < 179.0:
        raise ValueError(f"diagonal FOV {dfov_deg} deg is not a lens")
    return math.hypot(width, height) / 2 / math.tan(math.radians(dfov_deg) / 2)


def dfov_from_focal(width: int, height: int, f: float) -> float:
    return math.degrees(2 * math.atan(math.hypot(width, height) / 2 / f))


def spec_profile(name: str, width: int, height: int, dfov_deg: float, url: str, match: list[str]) -> CameraProfile:
    f = focal_from_dfov(width, height, dfov_deg)
    return CameraProfile(name, width, height, f, f, width / 2, height / 2, [0.0] * 5, SPEC_NOTE, dfov_deg, url,
                         None, match)


BUILTIN: dict[str, CameraProfile] = {
    "c270": spec_profile("Logitech C270", 1280, 720, 55.0,
                         "https://www.logitech.com/en-us/products/webcams/c270-hd-webcam.960-000694.html",
                         [r"\bc270\b", r"hd webcam c270"]),
    "c310": spec_profile("Logitech C310", 1280, 720, 60.0,
                         "https://www.logitech.com/en-us/products/webcams/c310-hd-webcam.html",
                         [r"\bc310\b", r"hd webcam c310"]),
    "c920": spec_profile("Logitech C920", 1920, 1080, 78.0,
                         "https://www.logitech.com/en-us/products/webcams/c920-pro-hd-webcam.960-001055.html",
                         [r"\bc92[02][a-z]?\b", r"hd pro webcam c92"]),       # C920, C920s, C920e, C922 share 78 deg
    # not a Logitech: the default the rest of Kite uses for an unknown phone-like camera (fx 700 at 960 wide)
    "generic": CameraProfile("generic webcam", 960, 540, 700.0, 700.0, 480.0, 270.0, [0.0] * 5,
                             "assumed: fx 700 px at 960 wide (perception/ default), calibrate for accuracy"),
}
DEFAULT_LOGITECH = "c270"      # the team's camera: black folding clip, round lens


def match_profile(usb_name: str | None) -> str | None:
    """USB / OS camera name ("Logitech HD Webcam C270", "HD Pro Webcam C920") -> built-in key, or None."""
    if not usb_name:
        return None
    low = usb_name.lower()
    for key, prof in BUILTIN.items():
        if any(re.search(p, low) for p in prof.match):
            return key
    return None


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def calibrated_path(name: str) -> Path:
    return PROFILE_DIR / f"{_slug(name)}.json"


def resolve(spec: str | None, usb_name: str | None = None, frame_size: tuple[int, int] | None = None
            ) -> tuple[CameraProfile, str]:
    """--profile value -> (profile at the frame size, why it was chosen).

    spec: a JSON path, a built-in key (c270, c310, c920, generic), or "auto"/None: the camera's USB name picks a
    built-in model, and a calibrated JSON for that model in data/cameras/ wins over the spec numbers. A name
    containing "logitech" with no model number falls back to the C270 (the team's camera)."""
    why = ""
    prof: CameraProfile | None = None
    if spec and spec not in ("auto",):
        p = Path(spec).expanduser()
        if p.suffix.lower() == ".json" or p.is_file():
            prof, why = CameraProfile.load(p), f"profile file {p}"
        elif spec.lower() in BUILTIN:
            prof, why = BUILTIN[spec.lower()], f"--profile {spec.lower()}"
        else:
            raise ValueError(f"--profile {spec!r}: a JSON file or one of {', '.join(BUILTIN)}")
    if prof is None:
        key = match_profile(usb_name)
        if key is None and usb_name and "logitech" in usb_name.lower():
            key, why = DEFAULT_LOGITECH, f"'{usb_name}' has no model number: assuming the team's C270"
        elif key is not None:
            why = f"USB name '{usb_name}' matched {key.upper()}"
        else:
            key, why = ("generic", "no camera name to match: generic webcam default") if not usb_name else \
                ("generic", f"'{usb_name}' is not a known model: generic webcam default")
        prof = BUILTIN[key]
        cal = calibrated_path(prof.name)
        if cal.is_file():
            shown = cal.relative_to(ROOT) if cal.is_relative_to(ROOT) else cal
            prof, why = CameraProfile.load(cal), f"{why}; calibrated profile {shown}"
    if frame_size is not None:
        prof = prof.scaled(*frame_size)
    return prof, why
