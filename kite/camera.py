"""Tello camera intrinsics: one place for fx / fy / cx / cy, so perception, the recorder and the sim agree.

load_camera() reads data/tello_camera.json (written by the calibration) and falls back to the published estimate.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "data" / "tello_camera.json"


@dataclass
class Camera:
    width: int = 960  # the Tello video stream (cropped from the photo sensor)
    height: int = 720
    fx: float = 920.0  # px; published estimate, 920 to 1077 px depending on the source (spec section 3)
    fy: float = 920.0
    cx: float = 480.0
    cy: float = 360.0
    source: str = "published estimate (uncalibrated)"

    @property
    def hfov_deg(self) -> float:
        return math.degrees(2 * math.atan(self.width / (2 * self.fx)))

    @property
    def vfov_deg(self) -> float:
        return math.degrees(2 * math.atan(self.height / (2 * self.fy)))

    def scaled(self, width: int, height: int) -> Camera:
        """The same field of view on a width x height image (itself if that is already its size)."""
        if (width, height) == (self.width, self.height):
            return self
        sx, sy = width / self.width, height / self.height
        return Camera(int(width), int(height), self.fx * sx, self.fy * sy, self.cx * sx, self.cy * sy,
                      f"{self.source}, scaled to {width}x{height}")

    @classmethod
    def from_hfov(cls, width: int, height: int, hfov_deg: float, source: str) -> Camera:
        """Square pixels, principal point at the centre, focal length from a horizontal field of view."""
        f = width / (2 * math.tan(math.radians(hfov_deg) / 2))
        return cls(int(width), int(height), f, f, width / 2, height / 2, source)

    def to_dict(self) -> dict:
        return asdict(self)


def load_camera(path: str | Path | None = None) -> Camera:
    p = Path(path) if path else DEFAULT_PATH
    if not p.is_file():
        return Camera()
    d = json.loads(p.read_text())
    return Camera(**{k: d[k] for k in Camera.__dataclass_fields__ if k in d})


def save_camera(cam: Camera, path: str | Path | None = None) -> Path:
    p = Path(path) if path else DEFAULT_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cam.to_dict(), indent=1) + "\n")
    return p
