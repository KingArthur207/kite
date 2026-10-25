"""A tiny numpy ray caster: a chest camera walking down a sidewalk, with exact depth for every pixel.

Real datasets have metric ground truth for still frames, but motion stereo, temporal smoothing and ground
anchoring need video with known depth and a known camera height. So we render it: a ground plane, a sidewalk
slab with a 15 cm curb down to the road, boxes (bins, a parked car, cones, walls), vertical cylinders (poles,
a tree trunk, a standing person), procedural textures in world coordinates (so optical flow has something to
track, and white-ish walls have little, as in life), a sun, sky, and sensor noise. The camera bobs, sways and
rolls with the gait (vertical +-2 cm at the step frequency, pitch +-0.8 deg, roll +-0.7 deg, slow yaw).

World: X right, Y up, Z forward (metres). Camera: x right, y down, z forward; depth is camera z.

    from perception.bench_depth import synth
    seq = synth.sequence("sidewalk")        # frames, depth, ids, poses, boxes; cached under data/depth_bench/synth
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..depth_zoo import Intrinsics

FPS = 15.0
W, H = 640, 480
K_C270 = Intrinsics.from_fov(W, H, 60.0)          # Logitech C270: 60 degrees diagonal -> fx = fy = 692.8 px


# ------------------------------------------------------------------ procedural texture
def _hash(i: np.ndarray, j: np.ndarray, seed: int) -> np.ndarray:
    x = (i.astype(np.int64) * 374761393 + j.astype(np.int64) * 668265263 + seed * 2246822519) & 0xFFFFFFFF
    x = ((x ^ (x >> 13)) * 1274126177) & 0xFFFFFFFF
    x = x ^ (x >> 16)
    return (x & 0xFFFF).astype(np.float32) / 65535.0


def value_noise(u: np.ndarray, v: np.ndarray, seed: int = 0) -> np.ndarray:
    i, j = np.floor(u), np.floor(v)
    fu, fv = u - i, v - j
    fu, fv = fu * fu * (3 - 2 * fu), fv * fv * (3 - 2 * fv)
    a, b = _hash(i, j, seed), _hash(i + 1, j, seed)
    c, d = _hash(i, j + 1, seed), _hash(i + 1, j + 1, seed)
    return (a + (b - a) * fu) * (1 - fv) + (c + (d - c) * fu) * fv


def fbm(u: np.ndarray, v: np.ndarray, seed: int, base: float, octaves: int = 4) -> np.ndarray:
    out, amp, tot = np.zeros(u.shape, np.float32), 1.0, 0.0
    f = base
    for k in range(octaves):
        out += amp * value_noise(u * f, v * f, seed + 17 * k)
        tot += amp
        amp *= 0.55
        f *= 2.1
    return out / tot


# ------------------------------------------------------------------ scene primitives
@dataclass
class Box:
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]
    color: tuple[float, float, float]          # BGR 0..1
    tex: float = 4.0                           # texture frequency (1/m)
    contrast: float = 0.5
    name: str = "box"
    obstacle: bool = True                      # a discrete thing a detector could box (not walls or slabs)


@dataclass
class Cyl:
    x: float
    z: float
    r: float
    y0: float
    y1: float
    color: tuple[float, float, float]
    tex: float = 6.0
    contrast: float = 0.5
    name: str = "pole"
    obstacle: bool = True


@dataclass
class Scene:
    ground_y: float = 0.0
    ground_color: tuple[float, float, float] = (0.42, 0.44, 0.45)
    ground_tex: float = 1.5
    tiles: float = 0.0                         # joint spacing (m) on the ground plane, 0 = none
    sky: bool = True
    objects: list = field(default_factory=list)


def _sidewalk(curb_z: float = 60.0) -> list:
    """A sidewalk slab 15 cm above the road, x from -2.6 to 2.1 m, ending at curb_z."""
    return [Box((-2.6, -0.15, -5.0), (2.1, 0.0, curb_z), (0.62, 0.64, 0.66), tex=1.2, contrast=0.55,
                name="sidewalk", obstacle=False)]


def scene(name: str) -> Scene:
    if name in ("sidewalk", "standstill"):
        objs = _sidewalk() + [
            Box((-3.5, -0.15, -5.0), (-2.6, 9.0, 60.0), (0.55, 0.62, 0.78), tex=0.9, contrast=0.35, name="facade",
                obstacle=False),
            Box((2.4, -0.15, 6.0), (4.2, 1.45, 10.5), (0.25, 0.22, 0.65), tex=2.0, contrast=0.3, name="car"),
            Cyl(1.75, 5.0, 0.07, 0.0, 4.2, (0.30, 0.32, 0.30), name="pole"),
            Cyl(1.75, 17.0, 0.07, 0.0, 4.2, (0.30, 0.32, 0.30), name="pole"),
            Cyl(-2.0, 14.0, 0.18, 0.0, 3.2, (0.18, 0.28, 0.35), tex=8.0, name="tree"),
            Box((-2.45, 0.0, 3.0), (-1.95, 0.48, 4.6), (0.20, 0.35, 0.50), tex=5.0, name="bench"),
            Box((-0.10, 0.0, 11.0), (0.40, 0.95, 11.5), (0.15, 0.45, 0.20), tex=6.0, contrast=0.6, name="bin"),
        ]
        return Scene(ground_y=-0.15, ground_color=(0.30, 0.31, 0.33), ground_tex=2.5, objects=objs)
    if name == "curb":
        objs = _sidewalk(curb_z=9.0) + [
            Box((-3.5, -0.15, -5.0), (-2.6, 9.0, 9.0), (0.60, 0.58, 0.72), tex=0.9, contrast=0.35, name="facade",
                obstacle=False),
            Cyl(0.9, 6.0, 0.07, 0.0, 4.0, (0.30, 0.32, 0.30), name="pole"),
            Box((0.30, 0.0, 4.0), (0.62, 0.55, 4.3), (0.15, 0.40, 0.95), tex=8.0, contrast=0.4, name="cone"),
            Cyl(0.3, 15.0, 0.25, -0.15, 1.6, (0.35, 0.30, 0.45), tex=5.0, name="person"),
            Box((-6.0, -0.15, 24.0), (6.0, 6.0, 26.0), (0.50, 0.55, 0.60), tex=0.8, contrast=0.4, name="building",
                obstacle=False),
        ]
        return Scene(ground_y=-0.15, ground_color=(0.27, 0.28, 0.30), ground_tex=2.5, objects=objs)
    if name == "hallway":
        objs = [
            Box((-2.3, 0.0, -5.0), (-1.3, 2.7, 14.0), (0.80, 0.82, 0.84), tex=0.6, contrast=0.12, name="wall",
                obstacle=False),
            Box((1.3, 0.0, -5.0), (2.3, 2.7, 14.0), (0.78, 0.80, 0.83), tex=0.6, contrast=0.12, name="wall",
                obstacle=False),
            Box((-2.3, 2.7, -5.0), (2.3, 3.0, 14.0), (0.85, 0.85, 0.85), tex=0.5, contrast=0.08, name="ceiling",
                obstacle=False),
            Box((-2.3, 0.0, 14.0), (2.3, 2.7, 15.0), (0.55, 0.65, 0.75), tex=1.0, contrast=0.3, name="end wall",
                obstacle=False),
            Box((-1.3, 0.9, 4.0), (-1.28, 1.6, 5.2), (0.2, 0.3, 0.7), tex=3.0, contrast=0.6, name="poster",
                obstacle=False),
            Box((0.0, 0.0, 7.0), (0.5, 0.8, 7.5), (0.25, 0.30, 0.55), tex=5.0, contrast=0.5, name="box"),
            Cyl(-0.9, 10.5, 0.2, 0.0, 1.1, (0.20, 0.50, 0.30), tex=7.0, name="plant"),
        ]
        return Scene(ground_y=0.0, ground_color=(0.50, 0.47, 0.40), ground_tex=1.8, tiles=0.6, sky=False,
                     objects=objs)
    raise KeyError(name)


# ------------------------------------------------------------------ camera path
@dataclass
class Pose:
    pos: np.ndarray            # world position (m)
    yaw: float                 # degrees, + = right
    pitch: float               # degrees, + = down
    roll: float                # degrees

    def R_cw(self) -> np.ndarray:
        y, p, r = (math.radians(a) for a in (self.yaw, self.pitch, self.roll))
        Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
        Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
        Rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
        return Ry @ Rx @ Rz @ np.diag([1.0, -1.0, 1.0])

    def down(self) -> np.ndarray:
        """World down (0, -1, 0) in camera coordinates: the ground normal the anchoring estimates."""
        return self.R_cw().T @ np.array([0.0, -1.0, 0.0])


SEQS = {   # name: (scene, frames, speed m/s, start z, base pitch, ground height under the walker)
    "sidewalk": ("sidewalk", 112, 1.3, 0.0, 8.0, 0.0),
    "curb": ("curb", 96, 1.25, 0.0, 10.0, 0.0),
    "hallway": ("hallway", 90, 1.0, 0.0, 6.0, 0.0),
    "standstill": ("standstill", 45, 0.0, 7.0, 8.0, 0.0),
}


def path(seq: str, n: int | None = None, cam_h: float = 1.2) -> list[Pose]:
    _, frames, v, z0, pitch0, g = SEQS[seq]
    out = []
    for k in range(n or frames):
        t = k / FPS
        step = 1.9 * 2 * math.pi * t                       # 1.9 steps per second
        walk = 1.0 if v > 0 else 0.15                      # standing still: breathing sway only
        bob = 0.02 * walk * math.sin(step)
        sway = 0.02 * walk * math.sin(step / 2)
        pos = np.array([sway, g + cam_h + bob, z0 + v * t])
        out.append(Pose(pos, 1.0 * math.sin(2 * math.pi * 0.25 * t), pitch0 + 0.8 * walk * math.sin(step + 0.6),
                        0.7 * walk * math.sin(step / 2 + 0.3)))
    return out


# ------------------------------------------------------------------ ray casting
def _hit_plane(C, D, y0):
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (y0 - C[1]) / D[..., 1]
    return np.where(t > 1e-4, t, np.inf)


def _hit_box(C, D, b: Box):
    lo, hi = np.array(b.lo), np.array(b.hi)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / D
        t1, t2 = (lo - C) * inv, (hi - C) * inv
    tmin, tmax = np.minimum(t1, t2), np.maximum(t1, t2)
    tmin = np.nan_to_num(tmin, nan=-np.inf)
    tmax = np.nan_to_num(tmax, nan=np.inf)
    tn, tf = tmin.max(axis=-1), tmax.min(axis=-1)
    face = tmin.argmax(axis=-1)
    hit = (tf >= tn) & (tf > 1e-4)
    t = np.where(tn > 1e-4, tn, np.where(hit, tf, np.inf))
    return np.where(hit, t, np.inf), face


def _hit_cyl(C, D, c: Cyl):
    ox, oz = C[0] - c.x, C[2] - c.z
    a = D[..., 0] ** 2 + D[..., 2] ** 2
    b = 2 * (ox * D[..., 0] + oz * D[..., 2])
    cc = ox * ox + oz * oz - c.r * c.r
    disc = b * b - 4 * a * cc
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.sqrt(np.maximum(disc, 0))
        t = (-b - s) / (2 * a)
    y = C[1] + t * D[..., 1]
    ok = (disc > 0) & (t > 1e-4) & (y >= c.y0) & (y <= c.y1)
    return np.where(ok, t, np.inf)


SUN = np.array([0.45, 0.8, -0.4]) / np.linalg.norm([0.45, 0.8, -0.4])


def render(sc: Scene, pose: Pose, K: Intrinsics = K_C270, seed: int = 0, noise: float = 2.0):
    """-> (image BGR uint8, depth float32 (0 = sky / no hit), ids int16 (-1 ground, -2 sky, k = object k))."""
    rays = K.rays()
    R = pose.R_cw()
    D = rays @ R.T                                         # world directions with camera-z = 1, so t = depth
    C = pose.pos
    h, w = rays.shape[:2]
    best = _hit_plane(C, D, sc.ground_y)
    ids = np.where(np.isfinite(best), -1, -2).astype(np.int16)
    face = np.zeros((h, w), np.int8)
    for k, o in enumerate(sc.objects):
        if isinstance(o, Box):
            t, f = _hit_box(C, D, o)
        else:
            t, f = _hit_cyl(C, D, o), np.full((h, w), 3, np.int8)
        closer = t < best
        best = np.where(closer, t, best)
        ids = np.where(closer, k, ids).astype(np.int16)
        face = np.where(closer, f, face).astype(np.int8)
    P = C + D * np.where(np.isfinite(best), best, 0)[..., None]
    img = np.zeros((h, w, 3), np.float32)
    # sky
    sky = ids == -2
    if sky.any():
        el = np.clip(-(D[..., 1] / np.linalg.norm(D, axis=-1)), -1, 1)
        top = np.array([0.85, 0.62, 0.42], np.float32)
        hor = np.array([0.95, 0.88, 0.80], np.float32)
        g = np.clip(-el * 3, 0, 1)[..., None]
        sk = hor * (1 - g) + top * g
        cl = fbm(D[..., 0] / np.maximum(-D[..., 1], 0.05) * 0.3, D[..., 2] / np.maximum(-D[..., 1], 0.05) * 0.3,
                 seed + 99, 1.0, 3)
        sk = sk + (np.clip(cl - 0.55, 0, 1) * 0.6)[..., None]
        img[sky] = sk[sky]
    # ground
    gm = ids == -1
    if gm.any():
        u, v = P[..., 0], P[..., 2]
        n = fbm(u, v, seed + 1, sc.ground_tex)
        col = np.array(sc.ground_color, np.float32) * (0.65 + 0.7 * n)[..., None]
        if sc.tiles:
            line = (np.abs(((u / sc.tiles) % 1) - 0.5) > 0.47) | (np.abs(((v / sc.tiles) % 1) - 0.5) > 0.47)
            col = np.where(line[..., None], col * 0.6, col)
        shade = 0.55 + 0.45 * SUN[1]
        img[gm] = (col * shade)[gm]
    for k, o in enumerate(sc.objects):
        m = ids == k
        if not m.any():
            continue
        Pm = P[m]
        if isinstance(o, Box):
            f = face[m]
            nrm = np.zeros((f.size, 3))
            nrm[np.arange(f.size), f] = 1.0
            sgn = np.where(D[m][np.arange(f.size), f] > 0, -1.0, 1.0)
            nrm *= sgn[:, None]
            uu = np.where(f == 0, Pm[:, 2], Pm[:, 0])
            vv = np.where(f == 1, Pm[:, 2], Pm[:, 1])
        else:
            dx, dz = Pm[:, 0] - o.x, Pm[:, 2] - o.z
            nrm = np.stack([dx, np.zeros_like(dx), dz], 1) / o.r
            uu, vv = np.arctan2(dx, dz) * o.r, Pm[:, 1]
        n = fbm(uu, vv, seed + 7 + k, o.tex)
        col = np.array(o.color, np.float32)[None, :] * (1 - o.contrast + 2 * o.contrast * n)[:, None]
        if o.name == "sidewalk":          # paving joints every 1.5 m, the strongest texture a walker sees
            line = (np.abs(((Pm[:, 2] / 1.5) % 1) - 0.5) > 0.48) | (np.abs(((Pm[:, 0] / 1.5) % 1) - 0.5) > 0.485)
            col = np.where(line[:, None], col * 0.55, col)
        lam = np.clip(nrm @ SUN, 0, 1)
        img[m] = col * (0.5 + 0.5 * lam)[:, None]
    rng = np.random.default_rng(seed * 7919 + int(pose.pos[2] * 1000) % 100000)
    img = img * 255 + rng.normal(0, noise, img.shape)
    depth = np.where(np.isfinite(best) & (best <= 80.0), best, 0).astype(np.float32)   # past 80 m: like sky
    return np.clip(img, 0, 255).astype(np.uint8), depth, ids


def boxes(ids: np.ndarray, sc: Scene, min_px: int = 30) -> list[dict]:
    """The visible 2D box of every discrete object (what a perfect detector would report)."""
    out = []
    for k, o in enumerate(sc.objects):
        if not o.obstacle:
            continue
        ys, xs = np.nonzero(ids == k)
        if ys.size < min_px:
            continue
        out.append({"name": o.name, "x1": int(xs.min()), "y1": int(ys.min()), "x2": int(xs.max()) + 1,
                    "y2": int(ys.max()) + 1, "k": k})
    return out


@dataclass
class Sequence:
    name: str
    frames: np.ndarray         # (n, h, w, 3) uint8 BGR
    depth: np.ndarray          # (n, h, w) float32, 0 = sky
    ids: np.ndarray            # (n, h, w) int16
    down: np.ndarray           # (n, 3) ground normal in camera coordinates
    cam_h: np.ndarray          # (n,) true camera height above the walker's floor
    boxes: list                # per frame: list of dicts
    K: Intrinsics


def sequence(name: str, n: int | None = None, K: Intrinsics = K_C270, cache_dir=None) -> Sequence:
    """Render (or load from cache_dir/<name>-<w>x<h>-<n>.npz) one of SEQS."""
    import json
    from pathlib import Path
    scn, frames, *_ = SEQS[name]
    n = n or frames
    if cache_dir is not None:
        cache = Path(cache_dir) / f"{name}-{K.width}x{K.height}-{n}.npz"
        if cache.exists():
            z = np.load(cache, allow_pickle=False)
            return Sequence(name, z["frames"], z["depth"], z["ids"], z["down"], z["cam_h"],
                            json.loads(str(z["boxes"])), K)
    sc = scene(scn)
    poses = path(name, n)
    fr, dp, ids_, dn, ch, bx = [], [], [], [], [], []
    g0 = SEQS[name][5]
    for pose in poses:
        img, d, ids = render(sc, pose, K, seed=3)
        fr.append(img)
        dp.append(d)
        ids_.append(ids)
        dn.append(pose.down())
        ch.append(pose.pos[1] - g0)
        bx.append(boxes(ids, sc))
    seq = Sequence(name, np.stack(fr), np.stack(dp), np.stack(ids_), np.stack(dn), np.array(ch), bx, K)
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, frames=seq.frames, depth=seq.depth, ids=seq.ids, down=seq.down, cam_h=seq.cam_h,
                            boxes=json.dumps(seq.boxes))
    return seq
