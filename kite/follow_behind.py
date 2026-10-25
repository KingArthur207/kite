"""Follow behind: lock onto the person in front of the drone and hover behind and above their head, on rc sticks.

This is the follow run/app.py flies. kite/follow.py's FollowController stays for kite.record and the rehearsals.

Why a second controller. kite/follow.py reads the distance from one cue only, the head top's image row, assuming the
camera is already above_m over the head. Right after takeoff the camera is BELOW the head: the head top sits above the
image centre, that formula reads it as max_range_m (5 m), and the controller flies forward at full stick into and past
the walker while it is still climbing. Its tracker also acquires only a box cut off by the bottom edge, and it holds
the altitude only while it sees the walker. Here:

  PersonLock     locks the LARGEST person in view (the one right in front), confirmed by 3 hits in the detector's last
                 5 runs; sticks to that box (overlap, or a short jump between consecutive runs); after a loss it
                 prefers a person near where they were; after lost_s x 3 unseen it lets go and locks anyone.
  estimate()     range from up to three cues, fused by their expected errors (log space):
                   full_height  the whole body in frame (the box bottom where the floor is, when the altitude is
                                known): the walker's height over the box height
                   head_elev    head top in frame and the altitude known: (altitude - walker height) / tan(angle).
                                Works with the camera above OR below the head (the sign of the angle says which)
                   bbox_width   sides and feet in frame: a 0.48 m wide body
                 plus range_lo, the most conservative cue (and a 0.28 m wide body: the narrowest a person looks).
                 No cue (the box is cut at the top and the bottom, e.g. while climbing) -> no range -> no forward.
  FollowBehind   the control law, every tick:
                   ud       holds altitude_m ALWAYS: locking, following, lost
                   yaw      turns toward the person (bearing, deadband), or a kite.fly yaw source when given
                   fb       holds distance_m. Moves toward them only when even range_lo says it is safe and their box
                            does not fill the frame; too close -> backs off fast and climbs a little.
                            Feed-forward: they walk toward the drone -> back away at their speed at once
                   lr       0 (no pose keypoints, no orbit)
                 Not seen: keep the last command 0.35 s, hold 1 s, turn toward where they were for 20 s, hover.
                 Lost while close (they walked under the camera): back off to the side and climb for up to 2.5 s.
  BehindMode     the live glue for run/pilot.py: detector thread, start checks, tick, status. Same calls as
                 kite.follow.FollowMode.

rc signs as everywhere in kite: +lr right, +fb forward, +ud up, +yaw clockwise. Every rc goes through the governor.
"""

from __future__ import annotations

import math
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from .camera import Camera
from .follow import Detection, Detector, YawSource, _link_error, scaled_camera
from .types import Telemetry

BODY_WIDTH_M = 0.48  # shoulders incl. arms, seen from behind
BODY_DEPTH_M = 0.28  # chest to back: the narrowest a person can look
MIN_RANGE_M, MAX_RANGE_M = 0.3, 15.0


@dataclass
class BehindCfg:
    distance_m: float = 1.0  # horizontal distance behind the person
    altitude_m: float = 2.0  # hover height above the floor: >= person height + 0.2 m
    person_height_m: float = 1.7  # the walker's real height: 5 cm off is 10-20 % range error from above
    # lock
    min_conf: float = 0.25  # a head seen from above scores low
    confirm_hits: int = 3  # detector runs with the person in the last confirm_window runs
    confirm_window: int = 5
    lost_s: float = 1.5  # unseen this long: lost (lets go after 3x this)
    edge_px: float = 3.0  # a box edge within this of the frame edge is cut off
    # control
    yaw_gain: float = 1.2  # rc yaw per degree of bearing
    yaw_deadband_deg: float = 4.0
    dist_gain: float = 40.0  # rc forward per metre of distance error
    dist_deadband_m: float = 0.1
    alt_gain: float = 60.0  # rc up per metre of altitude error
    alt_deadband_m: float = 0.05
    max_rc_yaw: int = 40
    max_rc_forward: int = 50  # ~0.5 m/s at the governor's default max_rc 50: a slow walk
    max_rc_up: int = 30
    max_rc_backoff: int = 50  # backing away from a person who comes closer (the governor's max_rc also applies)
    approach_rate_mps: float = 0.2  # they come closer faster than this: back away at their speed right away
    min_range_m: float = 0.55  # the most conservative range below this: back off, never approach
    max_box_width_frac: float = 0.75  # a box wider than this fraction of the frame: back off
    search_yaw_rc: int = 20  # yaw while the person is lost


# ------------------------------------------------------------------ lock
@dataclass
class Lock:
    """The locked person as the UI and the controller read it (a copy)."""

    box: tuple[float, float, float, float]
    conf: float
    t: float  # when the box was last seen (frame time)
    size: tuple[int, int]
    confirmed: bool = False
    seen: bool = True  # matched on the newest detector run

    def age(self, now: float) -> float:
        return now - self.t


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def _area(d: Detection) -> float:
    return (d.x2 - d.x1) * (d.y2 - d.y1)


def _centre(b) -> tuple[float, float]:
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


class PersonLock:
    def __init__(self, cfg: BehindCfg | None = None):
        self.cfg = cfg or BehindCfg()
        self.release()

    def release(self) -> None:
        self.lock: Lock | None = None
        self._runs: deque = deque(maxlen=self.cfg.confirm_window)  # True: the locked person was seen on that run
        self._confirmed = False
        self._seen_last_run = False

    def _match(self, cands: list[Detection], width: int, height: int, now: float) -> Detection | None:
        lk = self.lock
        diag = math.hypot(width, height)
        lx, ly = _centre(lk.box)
        lw, lh = lk.box[2] - lk.box[0], lk.box[3] - lk.box[1]

        def dist(d):
            return math.hypot((d.x1 + d.x2) / 2 - lx, (d.y1 + d.y2) / 2 - ly)

        best = max(cands, key=lambda d: _iou(d.box, lk.box), default=None)
        if best is not None and _iou(best.box, lk.box) >= 0.2:
            return best
        if now - lk.t > self.cfg.lost_s:  # a real loss: anyone near where they were
            near = [d for d in cands if dist(d) < 0.25 * diag]
            return min(near, key=dist) if near else None
        if self._seen_last_run:  # a fast turn between consecutive runs moves the box more than its width
            ok = [d for d in cands if dist(d) < 0.15 * diag
                  and min(_area(d), lw * lh) / max(_area(d), lw * lh, 1e-9) > 0.35]
            if ok:
                return min(ok, key=dist)
        if not self._confirmed and cands:  # not committed yet: a one-frame blob must not block the real person
            return max(cands, key=_area)
        return None

    def update(self, dets: list[Detection], t: float, width: int, height: int) -> Lock | None:
        c = self.cfg
        cands = [d for d in dets if d.conf >= c.min_conf and d.x2 > d.x1 and d.y2 > d.y1]
        if self.lock is not None and t - self.lock.t > 3 * c.lost_s:
            self.release()  # gone for good: lock whoever is in front next
        if self.lock is None:
            if not cands:
                return None
            d = max(cands, key=_area)  # the person right in front is the largest
            self.lock = Lock(d.box, d.conf, t, (width, height))
            self._runs.append(True)
            self._seen_last_run = True
            return self.snapshot()
        d = self._match(cands, width, height, t)
        self._runs.append(d is not None)
        self._seen_last_run = d is not None
        lk = self.lock
        if d is not None:
            if t - lk.t > c.lost_s:
                self._confirmed = False  # back after a loss: confirm again with repeated hits
                self._runs.clear()
                self._runs.append(True)
            lk.box, lk.conf, lk.t, lk.size = d.box, d.conf, t, (width, height)
        self._confirmed = self._confirmed or sum(self._runs) >= c.confirm_hits
        lk.confirmed = self._confirmed and t - lk.t <= c.lost_s
        lk.seen = d is not None
        return self.snapshot()

    def snapshot(self) -> Lock | None:
        lk = self.lock
        return None if lk is None else Lock(lk.box, lk.conf, lk.t, lk.size, lk.confirmed, lk.seen)


# ------------------------------------------------------------------ range
@dataclass
class PersonObs:
    bearing_deg: float
    range_m: float | None
    range_lo_m: float | None
    src: str = ""
    cues: dict = field(default_factory=dict)


def _bearing(cam: Camera, x: float) -> float:
    return math.degrees(math.atan((x - cam.cx) / cam.fx))


def _range_to_height(cam: Camera, x: float, y: float, diff_m: float) -> float | None:
    """Horizontal distance to the point imaged at (x, y), knowing it is diff_m below the camera (negative: above).
    Level camera. None when the ray does not reach that height at a usable angle (> 1.5 deg)."""
    r, u = (x - cam.cx) / cam.fx, -(y - cam.cy) / cam.fy
    horiz = math.hypot(1.0, r)
    if diff_m == 0 or u * diff_m >= 0 or math.degrees(math.atan2(abs(u), horiz)) < 1.5:
        return None
    return abs(diff_m) / abs(u) * horiz


def _ok(r: float | None) -> bool:
    return r is not None and math.isfinite(r) and MIN_RANGE_M <= r <= MAX_RANGE_M


def estimate(box, cam: Camera, altitude_m: float | None, person_h: float, edge_px: float = 3.0) -> PersonObs:
    """`cam` must match the frame the box came from (scaled_camera)."""
    x1, y1, x2, y2 = box
    cx, w, h = (x1 + x2) / 2, x2 - x1, y2 - y1
    b = math.radians(_bearing(cam, cx))
    top_ok, bottom_ok = y1 > edge_px, y2 < cam.height - edge_px
    side_ok = x1 > edge_px and x2 < cam.width - edge_px
    cues: dict[str, float] = {}
    sig: dict[str, float] = {}
    # the box bottom is the feet only if it sits where the floor is: from above, a box ending inside the frame is
    # often the head and shoulders alone (read as a whole body, that is a person 5-10x too far away)
    feet = _range_to_height(cam, cx, y2, altitude_m) if (bottom_ok and altitude_m is not None) else None

    def on_floor(r: float) -> bool:
        return altitude_m is None or (feet is not None and 0.67 < r / feet < 1.5)

    if top_ok and bottom_ok and h > 1:
        cues["full_height"], sig["full_height"] = cam.fy * person_h / h / math.cos(b), 0.07
    if top_ok and altitude_m is not None and abs(altitude_m - person_h) >= 0.15:
        diff = altitude_m - person_h
        r = _range_to_height(cam, cx, y1, diff)
        if r is not None:
            # error: height prior + altimeter over the height difference, 3 deg of camera / body pitch over the angle
            e = math.atan2(abs(diff), r)
            dz = math.hypot(0.05, 0.03)
            cues["head_elev"] = r
            sig["head_elev"] = math.hypot(dz / abs(diff), math.radians(3.0) / max(math.sin(e) * math.cos(e), 1e-3))
    if side_ok and bottom_ok and w > 1:  # a box cut by the frame bottom may be narrower than the body
        cues["bbox_width"], sig["bbox_width"] = cam.fx * BODY_WIDTH_M / w / math.cos(b), 0.35
    cues = {k: v for k, v in cues.items() if _ok(v) and (k == "head_elev" or on_floor(v))}
    obs = PersonObs(math.degrees(b), None, None, cues=cues)
    if cues:
        wts = {k: 1.0 / sig[k] ** 2 for k in cues}
        obs.range_m = math.exp(sum(wts[k] * math.log(v) for k, v in cues.items()) / sum(wts.values()))
        obs.src = "+".join(sorted(cues, key=lambda k: -wts[k]))
    lows = list(cues.values())
    if side_ok and w > 1:
        lows.append(cam.fx * BODY_DEPTH_M / w / math.cos(b))
    obs.range_lo_m = min(lows) if lows else None
    return obs


# ------------------------------------------------------------------ control
def clamp(v: float, lim: float) -> float:
    return max(-lim, min(lim, v))


def deadband(v: float, db: float) -> float:
    return 0.0 if abs(v) < db else v - math.copysign(db, v)


class FollowBehind:
    def __init__(self, cfg: BehindCfg, camera: Camera, yaw_source: YawSource | None = None):
        self.c = cfg
        self.camera = camera
        self.yaw_source = yaw_source
        self.yaw_source_error: str | None = None
        self.reset()

    def reset(self) -> None:
        self.ranges: deque = deque(maxlen=5)  # (t, range): only recent ones count
        self.range_hist: deque = deque(maxlen=30)  # (t, filtered range): the approach rate
        self.fwd_hist: deque = deque(maxlen=60)  # (t, forward rc sent)
        self.last_close = False
        self.cmd = (0, 0, 0, 0)
        self.last_seen_t: float | None = None
        self.last_bearing = 0.0
        self._frame_t: float | None = None  # frame time of the last box steered on
        self.obs: PersonObs | None = None
        self.status = "starting"
        self.phase = "seen"  # seen | close (backing off, out of view) | between | holding | searching | hovering
        if self.yaw_source is not None:
            try:
                self.yaw_source.reset()
            except Exception as e:
                self._drop_yaw_source(e)

    def _drop_yaw_source(self, e: Exception) -> None:
        self.yaw_source_error = f"{type(e).__name__}: {e}"
        self.yaw_source = None

    def _range(self, now: float) -> float | None:
        recent = [r for t, r in self.ranges if now - t <= 0.7]
        return statistics.median(recent) if recent else None

    def _range_rate(self, now: float) -> float | None:
        """Slope of the filtered range over the last ~1 s (m/s, negative: they come closer); None across a gap."""
        h = [x for x in self.range_hist if now - x[0] <= 1.0]
        if len(h) < 5 or h[-1][0] - h[0][0] < 0.5 or now - h[-1][0] > 0.3:
            return None
        if any(b[0] - a[0] > 0.4 for a, b in zip(h, h[1:])):
            return None
        tm = sum(t for t, _ in h) / len(h)
        rm = sum(r for _, r in h) / len(h)
        den = sum((t - tm) ** 2 for t, _ in h)
        return None if den <= 0 else sum((t - tm) * (r - rm) for t, r in h) / den

    def altitude_rc(self, alt: float | None) -> int:
        if alt is None:
            return 0
        return int(clamp(deadband(self.c.altitude_m - alt, self.c.alt_deadband_m) * self.c.alt_gain, self.c.max_rc_up))

    def _climb_rc(self, alt: float | None) -> int:
        """Get clear of a person: climb, at most 0.25 m above the follow altitude (the governor also caps)."""
        return self.c.max_rc_up if (alt is None or alt < self.c.altitude_m + 0.25) else self.altitude_rc(alt)

    def _yaw(self, now: float, frame_t: float, bearing: float, rng: float | None, law: float) -> float:
        if self.yaw_source is None:
            return law
        try:
            return float(self.yaw_source.yaw(now, frame_t, bearing, rng if rng is not None else self.c.distance_m,
                                             law))
        except Exception as e:
            self._drop_yaw_source(e)
            return law

    def step(self, lock: Lock | None, tel: Telemetry | None, now: float) -> tuple[int, int, int, int]:
        c = self.c
        alt = tel.floor_altitude_m() if tel is not None else None
        ud = self.altitude_rc(alt)
        fresh = lock is not None and lock.confirmed and lock.seen and (self._frame_t is None or lock.t > self._frame_t)
        if fresh:
            self._frame_t = lock.t
            cam = scaled_camera(self.camera, *lock.size)
            p = self.obs = estimate(lock.box, cam, alt, c.person_height_m, c.edge_px)
            self.last_seen_t, self.last_bearing = now, p.bearing_deg
            if p.range_m is not None:
                self.ranges.append((now, p.range_m))
            r = self._range(now)
            if r is not None:
                self.range_hist.append((now, r))
            lows = [v for v in (p.range_lo_m, r) if v is not None]
            r_lo = min(lows) if lows else None
            yaw = clamp(deadband(p.bearing_deg, c.yaw_deadband_deg) * c.yaw_gain, c.max_rc_yaw)
            yaw = clamp(self._yaw(now, lock.t, p.bearing_deg, r, yaw), c.max_rc_yaw)
            width, height = lock.size
            too_big = (lock.box[2] - lock.box[0]) > c.max_box_width_frac * width
            fwd, approaching = 0.0, False
            if r is not None:
                e = deadband(r - c.distance_m, c.dist_deadband_m)
                if e > 0:  # too far: approach only if even the most conservative cue says it is safe
                    if r_lo is not None and r_lo > c.min_range_m and not too_big:
                        fwd = clamp(e * c.dist_gain, c.max_rc_forward)
                else:
                    fwd = clamp(e * c.dist_gain, c.max_rc_forward)
                # feed-forward: they walk toward the drone -> back away at their speed now (rc 100 ~ 1 m/s). Our
                # own forward motion also shrinks the range (video lags): subtract the speed we commanded
                rate = self._range_rate(now)
                own = [f for t, f in self.fwd_hist if 0.3 <= now - t <= 1.3]
                if rate is not None and own:
                    closing = rate + (sum(own) / len(own)) / 100.0
                    approaching = closing < -c.approach_rate_mps
                    if approaching:
                        fwd = min(fwd, clamp(100.0 * closing - 10, c.max_rc_backoff))
            if (r_lo is not None and r_lo < c.min_range_m) or too_big:
                fwd = -c.max_rc_backoff  # too close: back off fast and gain a little height
                ud = max(ud, self._climb_rc(alt) // 2)
            # coming closer, nearly too close, or the head top at the bottom of the frame: if they vanish now,
            # they are walking under the drone (a plain dropout at the follow distance is not this)
            self.last_close = approaching or (r is not None and r < c.min_range_m + 0.25) or lock.box[1] >= 0.88 * height
            self.cmd = (0, int(fwd), ud, int(yaw))
            self.fwd_hist.append((now, int(fwd)))
            rng = "?" if p.range_m is None else f"{p.range_m:.1f}"
            lo = "?" if r_lo is None else f"{r_lo:.1f}"
            self.status = f"person {rng} m (lo {lo}, {p.src or 'no range cue'}) bearing {p.bearing_deg:+.0f}"
            self.phase = "seen"
            return self.cmd
        unseen = math.inf if self.last_seen_t is None else now - self.last_seen_t
        if self.last_close and 0.2 < unseen < 2.5:
            # lost while close: they are walking under the camera. Back off, step out of their path (away from the
            # side they were on) and climb; never hover in their way
            side = -1 if self.last_bearing >= 0 else 1
            self.cmd = (side * c.max_rc_backoff, -c.max_rc_backoff, self._climb_rc(alt), 0)
            self.status = "person close and out of view: backing off to the side"
            self.phase = "close"
            return self.cmd
        if unseen < 0.35:  # between detector runs: keep the last command, fresh altitude
            self.phase = "between"
            return (self.cmd[0], self.cmd[1], ud, self.cmd[3])
        if unseen < 1.0:
            self.status = "person briefly lost: holding"
            self.phase = "holding"
            return (0, 0, ud, 0)
        if unseen < 20.0:
            d = 1 if self.last_bearing >= 0 else -1
            self.status = f"person lost {unseen:.0f} s: turning {'right' if d > 0 else 'left'}"
            self.phase = "searching"
            return (0, 0, ud, d * c.search_yaw_rc)
        self.status = "person lost: hovering"
        self.phase = "hovering"
        return (0, 0, ud, 0)


# ------------------------------------------------------------------ live glue for run/pilot.py
class BehindMode:
    """Detector thread + lock + FollowBehind. start(drone) refuses (returns why) unless flying, video is live and a
    confirmed person is locked; tick() returns the rc to send through the governor (0s once cancelled); hold_rc()
    is what to send while not following (climb to and hold the follow altitude, no other motion)."""

    def __init__(self, frames, detector_factory: Callable[[], Detector], cfg: BehindCfg, camera: Camera,
                 log: Callable | None = None, yaw_source: YawSource | None = None):
        self.frames = frames
        self.cfg = cfg
        self.lock = PersonLock(cfg)
        self.ctrl = FollowBehind(cfg, camera, yaw_source)
        self.yaw_name = "law" if yaw_source is None else str(getattr(yaw_source, "name", type(yaw_source).__name__))
        self.log = log or (lambda msg, quiet=False, **kw: None)
        self._factory = detector_factory
        self._mutex = threading.Lock()
        self._stop = threading.Event()
        self._lock_state: Lock | None = None
        self.last_frame_t: float | None = None
        self.ready = False
        self.error: str | None = None
        self.active = False
        self._said = "seen"
        self.det_fps = 0.0
        self._thread = threading.Thread(target=self._run, daemon=True, name="follow-detector")

    def start_tracker(self) -> BehindMode:
        self._thread.start()
        return self

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2)

    def _run(self) -> None:
        try:
            detect = self._factory()
        except Exception as e:
            self.error = f"detector failed to load: {e}"
            return
        self.ready = True
        last_seq, n, t_count = None, 0, time.time()
        while not self._stop.is_set():
            f = self.frames.read()
            if f is None or f.seq == last_seq:
                time.sleep(0.003)
                continue
            last_seq = f.seq
            try:
                dets = detect(f.image)
            except Exception as e:  # one bad frame must not kill tracking
                self.error = f"detector: {e!r}"
                dets = []
            with self._mutex:
                self._lock_state = self.lock.update(dets, f.t, f.width, f.height)
                self.last_frame_t = f.t
            n += 1
            if time.time() - t_count >= 2.0:
                self.det_fps, n, t_count = n / (time.time() - t_count), 0, time.time()

    def _snapshot(self) -> Lock | None:
        with self._mutex:
            lk = self._lock_state
            return None if lk is None else Lock(lk.box, lk.conf, lk.t, lk.size, lk.confirmed, lk.seen)

    def start(self, drone) -> str | None:
        now = time.time()
        lk = self._snapshot()
        why = None
        if not self.ready:
            why = self.error or "detector still loading"
        elif not drone.flying or drone.busy():
            why = "not flying (take off first)"
        elif getattr(drone, "land_requested", False):
            why = "landing"
        elif getattr(drone, "hovering_for_safety", False):
            why = "safety hover"
        elif _link_error(drone):
            why = f"Wi-Fi down ({_link_error(drone)})"
        elif self.last_frame_t is None or now - self.last_frame_t > 1.0:
            why = "no live video"
        elif lk is None or not lk.confirmed or lk.age(now) > 0.5:
            why = "no person in view"
        if why:
            self.log("follow refused", reason=why)
            return why
        self.ctrl.reset()
        self._said = "seen"
        self.active = True
        self.log("follow on", box=[round(v) for v in lk.box], conf=round(lk.conf, 2))
        return None

    def hold_rc(self, drone) -> tuple[int, int, int, int]:
        """Not following yet (locking): climb to and hold the follow altitude, nothing else."""
        return (0, 0, self.ctrl.altitude_rc(drone.telemetry().floor_altitude_m()), 0)

    def cancel(self, reason: str) -> None:
        if self.active:
            self.active = False
            self.log("follow off", reason=reason)

    def tick(self, drone, action: str = "ok", busy: bool = False) -> tuple[int, int, int, int]:
        if not self.active:
            return (0, 0, 0, 0)
        if action != "ok":
            self.cancel(f"safety {action}")
        elif not drone.flying or busy or getattr(drone, "land_requested", False):
            self.cancel("not flying / busy / landing")
        elif _link_error(drone):  # rc cannot reach it; never resume on its own when the link comes back
            self.cancel(f"Wi-Fi down ({_link_error(drone)})")
        if not self.active:
            return (0, 0, 0, 0)
        rc = self.ctrl.step(self._snapshot(), drone.telemetry(), time.time())
        phase = self.ctrl.phase
        if phase not in ("between", self._said):  # say each change once (between detector runs is not one)
            self._said = phase
            self.log("follow: person in view again" if phase == "seen" else f"follow: {self.ctrl.status}")
        self.log("follow", quiet=True, rc=list(rc), status=self.ctrl.status)
        return rc

    def status(self) -> str:
        if self.error and not self.ready:
            return f"follow ERROR {self.error}"
        if not self.ready:
            return "follow loading"
        lk = self._snapshot()
        now = time.time()
        who = "no person" if lk is None or lk.age(now) > self.cfg.lost_s else (
            f"person conf {lk.conf:.2f}{'' if lk.confirmed else ' (confirming)'}")
        yaw = "" if self.yaw_name == "law" else f", yaw {self.yaw_name if self.ctrl.yaw_source else 'law'}"
        if not self.active:
            return f"follow off ({who}, {self.det_fps:.0f} det/s{yaw})"
        return f"FOLLOW {self.ctrl.status}{yaw}"
