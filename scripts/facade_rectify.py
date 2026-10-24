"""Rectify a street-level photo of a building wall into a facade texture (Mapillary phase 2).

    python scripts/facade_rectify.py            # self-check on a synthetic wall, no network

Not wired to Mapillary yet: there was no MAPILLARY_TOKEN when this was written. With a token the
plan is, per street-facing wall in sim/public/osm/moss.json:
  1. GET https://graph.mapillary.com/images?bbox=<wall bbox + 25 m>&fields=id,computed_geometry,
     computed_compass_angle,camera_parameters,camera_type,captured_at,width,height,thumb_2048_url
     (camera_type "perspective" only).
  2. Keep images 5 to 25 m in front of the wall whose view axis is within 35 degrees of the wall's
     inward normal and that see all four corners (project() below); take the most frontal, then newest.
  3. rectify() the wall quad into a texture at 40 px/m and write it to sim/public/textures/mapillary/
     with the image id in CREDITS.md ("Mapillary contributors", CC BY-SA 4.0). Walls without a good
     image keep the Poly Haven material.

Camera model (Mapillary conventions): local metres x east, y north, z up; heading = compass angle,
clockwise from north; focal = camera_parameters[0] as a fraction of max(width, height); pitch optional
(Street View Static, scripts/streetview_facades.py), no roll. ponytail: ignores radial distortion;
undistort first if rectified walls come out bowed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Camera:
    x: float          # metres east of the sim origin
    y: float          # metres north
    z: float          # metres above the street
    heading: float    # degrees clockwise from north
    focal: float      # fraction of max(width, height)
    width: int
    height: int
    pitch: float = 0.0  # degrees, + = up (Street View Static's pitch); no roll

    def axes(self):
        h, p = math.radians(self.heading), math.radians(self.pitch)
        level = np.array([math.sin(h), math.cos(h), 0.0])
        up = np.array([0.0, 0.0, 1.0])
        fwd = math.cos(p) * level + math.sin(p) * up
        right = np.array([math.cos(h), -math.sin(h), 0.0])
        down = math.sin(p) * level - math.cos(p) * up
        return fwd, right, down

    @property
    def f_px(self) -> float:
        return self.focal * max(self.width, self.height)


def project(cam: Camera, pts: np.ndarray) -> np.ndarray:
    """World points (N, 3) to pixels (N, 2); NaN for points behind the camera."""
    fwd, right, down = cam.axes()
    d = pts - np.array([cam.x, cam.y, cam.z])
    depth = d @ fwd
    with np.errstate(divide="ignore", invalid="ignore"):
        u = cam.width / 2 + cam.f_px * (d @ right) / depth
        v = cam.height / 2 + cam.f_px * (d @ down) / depth
    out = np.stack([u, v], axis=1)
    out[depth <= 0.1] = np.nan
    return out


def homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """3x3 H with dst ~ H @ src for 4 point pairs (DLT, SVD)."""
    rows = []
    for (x, y), (u, v) in zip(src, dst):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(rows, float))
    H = vt[-1].reshape(3, 3)
    return H / H[2, 2]


def sample(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear sample img (H, W, C) at float pixel coords (pixel centres at +0.5); returns (values, valid)."""
    h, w = img.shape[:2]
    x, y = u - 0.5, v - 0.5
    valid = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0) & (x <= w - 1) & (y <= h - 1)
    x, y = np.where(valid, x, 0), np.where(valid, y, 0)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    top = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
    bot = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
    return top * (1 - fy) + bot * fy, valid


def rectify(img: np.ndarray, cam: Camera, a: tuple[float, float], b: tuple[float, float], wall_h: float,
            px_per_m: float = 40.0) -> tuple[np.ndarray, np.ndarray]:
    """Wall from ground point a to b (metres east/north), wall_h tall, as seen from the street, into a
    texture: column 0 is a, row 0 is the roofline. Returns (texture, valid mask)."""
    L = math.dist(a, b)
    tw, th = max(1, round(L * px_per_m)), max(1, round(wall_h * px_per_m))
    corners = np.array([[a[0], a[1], wall_h], [b[0], b[1], wall_h], [b[0], b[1], 0.0], [a[0], a[1], 0.0]])
    img_pts = project(cam, corners)
    if np.isnan(img_pts).any():
        raise ValueError("wall corner behind the camera")
    H = homography(np.array([[0, 0], [tw, 0], [tw, th], [0, th]], float), img_pts)
    ys, xs = np.mgrid[0:th, 0:tw] + 0.5
    p = H @ np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)])
    u, v = (p[0] / p[2]).reshape(th, tw), (p[1] / p[2]).reshape(th, tw)
    return sample(img.astype(float), u, v)


# --- self-check: render a known wall texture through a camera, rectify it back, compare.
def _render_wall(tex: np.ndarray, cam: Camera, a, b, wall_h) -> np.ndarray:
    """Ray-cast the camera image of a textured vertical wall (grey elsewhere)."""
    fwd, right, down = cam.axes()
    vs, us = np.mgrid[0:cam.height, 0:cam.width] + 0.5
    rays = (fwd[:, None] * cam.f_px + right[:, None] * (us.ravel() - cam.width / 2)
            + down[:, None] * (vs.ravel() - cam.height / 2))
    a3, b3 = np.array([*a, 0.0]), np.array([*b, 0.0])
    along = (b3 - a3) / np.linalg.norm(b3 - a3)
    normal = np.array([-along[1], along[0], 0.0])
    o = np.array([cam.x, cam.y, cam.z])
    t = ((a3 - o) @ normal) / (normal @ rays)
    hit = o[:, None] + rays * t
    s = (hit - a3[:, None]).T @ along
    z = hit[2]
    L = np.linalg.norm(b3 - a3)
    th, tw = tex.shape[:2]
    vals, ok = sample(tex.astype(float), s / L * tw, (wall_h - z) / wall_h * th)
    ok &= (t > 0) & (s >= 0) & (s <= L) & (z >= 0) & (z <= wall_h)
    out = np.where(ok[:, None], vals, 128.0)
    return out.reshape(cam.height, cam.width, -1)


def selftest():
    # Pure projection: a point straight ahead lands on the image centre, one to the right lands right.
    cam = Camera(x=0, y=0, z=1.5, heading=90, focal=0.8, width=1000, height=750)
    c = project(cam, np.array([[10.0, 0.0, 1.5], [10.0, -2.0, 1.5]]))
    assert np.allclose(c[0], [500, 375]), c
    assert c[1][0] > 500 and abs(c[1][1] - 375) < 1e-9, c
    # DLT is exact on its four points.
    src = np.array([[0, 0], [10, 0], [10, 5], [0, 5]], float)
    dst = np.array([[3, 4], [40, 2], [38, 30], [5, 29]], float)
    H = homography(src, dst)
    p = H @ np.c_[src, np.ones(4)].T
    assert np.allclose((p[:2] / p[2]).T, dst, atol=1e-6)

    # A 10 m x 6 m wall, 12 m north of a camera that looks at it 25 degrees off square.
    a, b, wall_h, ppm = (-4.0, 12.0), (6.0, 12.0), 6.0, 20
    yy, xx = np.mgrid[0:round(wall_h * ppm), 0:round(10 * ppm)]
    tex = np.stack([((xx // 20 + yy // 20) % 2) * 200 + 30, xx * 255 / xx.max(), yy * 255 / yy.max()], -1).astype(float)
    cam = Camera(x=-3.0, y=0.0, z=1.6, heading=25, focal=0.85, width=1600, height=1200)
    shot = _render_wall(tex, cam, a, b, wall_h)
    back, ok = rectify(shot, cam, a, b, wall_h, px_per_m=ppm)
    tilted = Camera(x=-3.0, y=0.0, z=1.6, heading=25, focal=0.85, width=1600, height=1200, pitch=15)
    back_t, ok_t = rectify(_render_wall(tex, tilted, a, b, wall_h), tilted, a, b, wall_h, px_per_m=ppm)
    inner_t = ok_t.copy()
    inner_t[:3], inner_t[-3:], inner_t[:, :3], inner_t[:, -3:] = False, False, False, False
    assert np.abs(back_t[..., 0] - tex[..., 0])[inner_t].mean() < 5, "pitched camera misregisters"
    assert back.shape[:2] == tex.shape[:2] and ok.mean() > 0.95, (back.shape, ok.mean())
    inner = ok.copy()
    inner[:3], inner[-3:], inner[:, :3], inner[:, -3:] = False, False, False, False
    err = np.abs(back[..., 0] - tex[..., 0])[inner]   # the checker: any misregistration shows on its edges
    # Edges blur a little through two resamplings (mean about 2.5/255). A 2% focal error, 0.2 m of
    # camera position or half a degree of heading gives a mean of 15 to 25.
    assert err.mean() < 5, err.mean()
    try:
        rectify(shot, Camera(x=-3, y=20, z=1.6, heading=25, focal=0.85, width=1600, height=1200), a, b, wall_h)
        raise AssertionError("a wall behind the camera must be refused")
    except ValueError:
        pass
    print(f"facade_rectify self-check ok: mean checker error {err.mean():.2f}/255, {ok.mean():.0%} of the wall seen")


if __name__ == "__main__":
    selftest()
