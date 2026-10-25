"""The near system: frames (+ detections, + dense depth) -> per-object priorities in the walker's corridor.

    detections ──► undistort ──► track ──► cues (size_h, size_w, ground, depth, motion) ──► robust fusion
                                    │                                                          │
                                    └──► looming (box scale over ~0.9 s) ──────► Kalman [ln F, d ln F/dt]
    depth ──► ground anchor (RANSAC) ──► confidence + smoothing ──► corridor occupancy (no class)
                                                                         │
    corridor test (+-0.5 m at the filtered range) ◄──────────────────────┘ ──► priority with hysteresis

Detections are anything with x1, y1, x2, y2, kind, conf, label, tid (perception.fuse.Det fits), in the frame's
pixels. `use` switches cues off for the ablation in evaluate.py.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from . import corridor, cues, occupancy
from .depth import DepthFrame, DepthProcessor
from .fusion import LogKalman, fuse
from .intrinsics import CameraProfile
from .priority import PriorityState
from .priors import Prior, prior_for

ALL_CUES = ("size_h", "size_w", "ground", "depth", "motion", "loom")
EDGE_FRAC, EDGE_MIN_PX = 0.015, 8.0   # a box this close to the frame edge is cut by it (YOLO stops short of it)
TRACK_TTL_S = 0.7
ELEVATED_KINDS = {"overhead", "sign"}
OCC_KIND = {"obstacle", "wall", "curb", "stairs_down"}


@dataclass
class Track:
    id: str
    kind: str
    label: str
    prior: Prior
    state: PriorityState
    kf: LogKalman | None = None
    hist: deque = field(default_factory=lambda: deque(maxlen=24))
    lat_hist: deque = field(default_factory=lambda: deque(maxlen=24))
    last_t: float = 0.0
    box: tuple | None = None
    priority: int | None = None


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def lateral_rate(hist, window_s: float = 0.9, min_n: int = 4) -> float:
    """m/s sideways, least squares over the last window (0 with too little history)."""
    if not hist:
        return 0.0
    t_end = hist[-1][0]
    pts = [(t, x) for t, x in hist if t_end - t <= window_s]
    if len(pts) < min_n or pts[-1][0] - pts[0][0] < 0.25:
        return 0.0
    t = np.array([p[0] for p in pts])
    x = np.array([p[1] for p in pts])
    t = t - t.mean()
    return float(np.sum(t * (x - x.mean())) / np.sum(t * t))


def collision_course(lat_hist, hit, ttc: float | None, half: float) -> bool:
    """Will the object still overlap the corridor when it arrives? Its lateral offset extrapolated to the time
    of contact (the 'possible collision course' test of single-camera forward collision warning) with its
    footprint's half-width and 0.2 m of slack. Without a time or a lateral history: assume yes."""
    if ttc is None:
        return True
    lat_then = hit.lateral + lateral_rate(lat_hist) * min(ttc, 3.0)
    return abs(lat_then) <= half + 0.5 * abs(hit.right - hit.left) + 0.2


@dataclass
class NearFrame:
    t: float
    objects: list[dict]
    top: dict | None
    depth: DepthFrame | None
    pitch: float
    anchored: bool
    timings: dict = field(default_factory=dict)

    def event(self) -> dict:
        """The stream record: {t, priority, kind, confidence, ttc_s?, cues, objects}."""
        top = self.top or {}
        ev = {"type": "near", "t": round(self.t, 3), "priority": top.get("priority"), "kind": top.get("kind"),
              "id": top.get("id"), "confidence": top.get("confidence"), "distance_m": top.get("distance_m"),
              "cues": top.get("cues", {}), "objects": self.objects}
        if top.get("ttc_s") is not None:
            ev["ttc_s"] = top["ttc_s"]
        return ev


class NearSystem:
    def __init__(self, prof: CameraProfile, detector=None, depth: DepthProcessor | None = None, cam_h: float = 1.2,
                 pitch: float = 0.0, pitch_sigma: float = 3.0, v_ego: float = 1.2, still: bool = False,
                 size_thresholds: bool = True, use: tuple[str, ...] = ALL_CUES, occupancy_on: bool = True,
                 ignore_bottom: float = 0.0, half: float | None = None, n_up: int = 2, m_down: int = 5):
        self.prof, self.detector, self.depth = prof, detector, depth
        self.cam_h, self.pitch, self.pitch_sigma, self.v_ego = cam_h, pitch, pitch_sigma, v_ego
        self.still, self.size_thresholds, self.use = still, size_thresholds, set(use)
        self.occupancy_on, self.ignore_bottom, self.half = occupancy_on, ignore_bottom, half
        self.n_up, self.m_down = n_up, m_down
        self.tracks: dict[str, Track] = {}
        self._n = 0

    def _state(self) -> PriorityState:
        return PriorityState(self.n_up, self.m_down, self.still)

    # ------------------------------------------------------------------ tracks
    def _track_for(self, det, ubox, t: float, taken: set) -> Track:
        tid = getattr(det, "tid", None)
        if tid is not None:
            key = f"t{tid}"
        else:
            key, best = None, 0.3
            for k, tr in self.tracks.items():
                if k in taken or tr.box is None or k.startswith("occ:") or t - tr.last_t > TRACK_TTL_S:
                    continue
                o = iou(ubox, tr.box)
                if o > best:
                    key, best = k, o
            if key is None:
                self._n += 1
                key = f"n{self._n}"
        tr = self.tracks.get(key)
        if tr is None:
            label = getattr(det, "label", "") or det.kind
            tr = Track(key, det.kind, label, prior_for(det.kind, label), self._state())
            self.tracks[key] = tr
        return tr

    # ------------------------------------------------------------------ one frame
    def step(self, frame: np.ndarray | None, t: float, dets=None, frame_size: tuple[int, int] | None = None
             ) -> NearFrame:
        timings = {}
        if frame is not None:
            h, w = frame.shape[:2]
        else:
            w, h = frame_size or (self.prof.width, self.prof.height)
        prof = self.prof.scaled(w, h)
        if dets is None:
            ts = time.perf_counter()
            dets = self.detector(frame) if self.detector is not None else []
            timings["detect"] = (time.perf_counter() - ts) * 1000
        boxes_d = [(d.x1, d.y1, d.x2, d.y2) for d in dets]
        df = None
        pitch, psig, anchored = self.pitch, self.pitch_sigma, False
        if self.depth is not None and frame is not None and ({"depth"} & self.use or self.occupancy_on):
            ts = time.perf_counter()
            df = self.depth.process(frame, t, prof, boxes_d, self.v_ego)
            timings["depth"] = (time.perf_counter() - ts) * 1000
            if df.anchor.ok:
                pitch, psig, anchored = df.anchor.pitch, 1.0, True
        objs: list[dict] = []
        taken: set = set()
        for det in dets:
            ub = prof.undistort_box(det.x1, det.y1, det.x2, det.y2)
            e = max(EDGE_MIN_PX, EDGE_FRAC * min(w, h))
            box = cues.Box(*ub, cut_top=det.y1 <= e, cut_bottom=det.y2 >= h - e,
                           cut_left=det.x1 <= e, cut_right=det.x2 >= w - e)
            tr = self._track_for(det, ub, t, taken)
            taken.add(tr.id)
            o = self._object(tr, det, box, prof, df, pitch, psig, anchored, t)
            if o is not None:
                objs.append(o)
        if df is not None and self.occupancy_on and df.anchor.ok:     # heights need a fitted ground
            mask = occupancy.box_mask(boxes_d, df.z.shape, w, h, grow=0.08)
            hits = occupancy.occupancy(df, self.cam_h, pitch, corridor.half_width(self.v_ego), mask,
                                       self.ignore_bottom)
            named = [o for o in objs if o["distance_m"] is not None and o["corridor"] != "out"]
            for hit in hits:
                # one sound per thing: a rise right where a named object already is belongs to it
                if any(abs(o["distance_m"] - hit.forward) < 0.25 * o["distance_m"] + 0.3 and
                       abs(o["lateral_m"] - hit.lateral) < 0.6 for o in named) and hit.kind in ("obstacle", "wall"):
                    continue
                objs.append(self._occ_object(hit, df, t))
        for k in [k for k, tr in self.tracks.items() if t - tr.last_t > TRACK_TTL_S]:
            del self.tracks[k]
        ranked = sorted((o for o in objs if o["priority"] is not None),
                        key=lambda o: (o["priority"], o["distance_m"] if o["distance_m"] is not None else 99))
        return NearFrame(t, objs, ranked[0] if ranked else None, df, pitch, anchored, timings)

    def _object(self, tr: Track, det, box: cues.Box, prof, df, pitch, psig, anchored, t) -> dict | None:
        pr = tr.prior
        tr.kind = det.kind
        found: dict[str, cues.Cue | None] = {}
        if "size_h" in self.use:
            found["size_h"] = cues.size_height(box, prof, pr, pitch)
        if "size_w" in self.use:
            found["size_w"] = cues.size_width(box, prof, pr)
        if "ground" in self.use:
            found["ground"] = cues.ground(box, prof, self.cam_h, pitch, psig, det.kind in ELEVATED_KINDS)
        if "depth" in self.use and df is not None:
            found["depth"] = cues.depth(box, prof, df.z, df.conf, df.prof, pitch, df.anchor.sigma_log, anchored)
        sc = cues.box_scale(box)
        if sc is not None:
            which = "a" if not (box.cut_top or box.cut_bottom or box.cut_left or box.cut_right) else \
                ("w" if not (box.cut_left or box.cut_right) else "h")
            tr.hist.append((t, sc, which))
        loom = cues.looming(list(tr.hist)) if "loom" in self.use else cues.Loom(None, 0.0, math.inf, 0)
        if "motion" in self.use and not pr.moving:
            found["motion"] = cues.motion(loom, self.v_ego)
        upper = cues.size_bound(box, prof, pr)
        fz = fuse(found, upper)
        tr.last_t, tr.box = t, (box.x1, box.y1, box.x2, box.y2)
        if fz is None:
            hit = corridor.test_angular(prof, box.x1, box.x2, self.v_ego)
            p = tr.state.update(5.0 if hit.inside else None, 1.0) if hit.inside else tr.state.update(None)
            tr.priority = 3 if p is not None else None
            return self._record(tr, det, None, None, loom.tau, hit, found, [], tr.priority, "angular")
        if tr.kf is None:
            tr.kf = LogKalman(math.log(fz.d), fz.sigma ** 2, t)
        else:
            tr.kf.predict(t)
            tr.kf.update_distance(fz.d, fz.sigma)
        if loom.n >= 4 and math.isfinite(loom.rate_sigma):
            tr.kf.update_rate(loom.rate, loom.rate_sigma)
        d, dsig = tr.kf.d, tr.kf.sigma
        ttc, ttc_sig = tr.kf.ttc()
        # time to contact: the filter's (distance cues + looming), or looming alone when the filter lags. For the
        # P1 override use the cautious side of each: the filter's ttc + 1 sigma, looming's slow-side tau.
        cand = []
        if ttc is not None:
            cand.append((ttc, ttc + ttc_sig))
        if loom.tau is not None:
            cand.append((loom.tau, cues.tau_upper(loom)))
        ttc = min(c[0] for c in cand) if cand else None
        ttc_safe = min(c[1] for c in cand) if cand else None
        v_foot = min(box.y2, prof.height - 1.0)
        hit = corridor.test(prof, box.x1, box.x2, v_foot, d, pitch, self.v_ego, self.half)
        tr.lat_hist.append((t, hit.lateral))
        if ttc_safe is not None and not collision_course(tr.lat_hist, hit, ttc, corridor.half_width(self.v_ego)):
            ttc_safe = None          # closing, but it will be beside the walker by then (a car turning across)
        scale = pr.size_scale() if self.size_thresholds else 1.0
        if hit.inside or hit.edge:
            p = tr.state.update(d, scale, ttc_safe)
            if hit.edge and p is not None:
                p = max(p, 2)
        else:
            p = tr.state.update(None, scale)
        tr.priority = p
        return self._record(tr, det, d, dsig, ttc, hit, found, fz.rejected, p, "metric")

    def _record(self, tr, det, d, dsig, ttc, hit, found, rejected, p, mode) -> dict:
        conf = float(getattr(det, "conf", 1.0)) * (math.exp(-dsig) if dsig is not None else 0.5)
        conf *= 1.0 if hit.inside else 0.5
        return {"id": tr.id, "kind": tr.kind, "label": tr.label, "priority": p, "confidence": round(conf, 2),
                "distance_m": None if d is None else round(d, 2),
                "sigma_log": None if dsig is None else round(dsig, 3),
                "ttc_s": None if ttc is None else round(ttc, 2), "bearing": round(hit.bearing, 1),
                "lateral_m": round(hit.lateral, 2), "corridor": "inside" if hit.inside else "edge" if hit.edge
                else "out", "range_mode": mode,
                "cues": {k: round(c.d, 2) for k, c in found.items() if c is not None},
                "rejected": list(rejected)}

    def _occ_object(self, hit: occupancy.OccHit, df: DepthFrame, t: float) -> dict:
        key = f"occ:{hit.kind}"
        tr = self.tracks.get(key)
        if tr is None:
            tr = Track(key, hit.kind, hit.kind, prior_for(hit.kind), self._state())
            self.tracks[key] = tr
        sig = math.hypot(df.anchor.sigma_log, 0.15)
        if tr.kf is None:
            tr.kf = LogKalman(math.log(max(hit.forward, 0.3)), sig ** 2, t)
        else:
            tr.kf.predict(t)
            tr.kf.update_distance(max(hit.forward, 0.3), sig)
        tr.last_t = t
        d = tr.kf.d
        ttc, _ = tr.kf.ttc()
        p = tr.state.update(d, 1.0, ttc)
        tr.priority = p
        return {"id": key, "kind": hit.kind, "label": f"depth {hit.kind}", "priority": p,
                "confidence": hit.confidence, "distance_m": round(d, 2), "sigma_log": round(tr.kf.sigma, 3),
                "ttc_s": None if ttc is None else round(ttc, 2), "bearing": round(hit.bearing, 1),
                "lateral_m": round(hit.lateral, 2), "corridor": "inside", "range_mode": "occupancy",
                "cues": {"occupancy": round(hit.forward, 2)}, "rejected": []}
