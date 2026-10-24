"""Fly the Tello from a live camera window: arrow keys move it, q lands, 3 cuts the motors.

    python tests/drone_fly.py                      # connect, open the camera window, fly with the keys
    python tests/drone_fly.py --dry-run            # on the table: video and keys work, no motion is ever sent
    python tests/drone_fly.py --video-backend raw  # another video reader (default pyav, as in tests/drone_test.py)

Keys (click the camera window first so it has the keyboard):
    t           take off (refused without live video: the safety governor would only hover)
    arrows      forward / back / left / right; hold to keep moving, let go and it stops within --pulse
    w / s       up / down
    a / d       turn left / right
    space       hover: all sticks to 0
    + / -       stick speed up / down (10 to 50, the safety cap)
    q           land; on the ground, quit
    3           EMERGENCY: motors off at once, the drone falls. Only if it is about to hit someone.
    Ctrl+C in the terminal lands, then quits.

Everything goes through kite's drone layer: TelloDrone (djitellopy commands, kite's video reader) inside
SafetyGovernor (stick cap, 0.5-2.3 m altitude band, lands on low battery or after 7 min, hovers when video or
telemetry stalls). The window, the keys and the 10 Hz control tick all run in one loop on the main thread: macOS only
lets the main thread draw windows, and no other thread can move the drone.
Safety: 3 x 3 m clear space, prop guards, a spotter. Works on macOS, Windows and Linux.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the kite package next to this file

# Arrow keys as cv2.waitKeyEx() reports them: Windows, macOS (Cocoa) and Linux (GTK / Qt) all differ.
ARROWS = {
    2490368: "up", 2621440: "down", 2424832: "left", 2555904: "right",  # Windows
    63232: "up", 63233: "down", 63234: "left", 63235: "right",          # macOS
    65362: "up", 65364: "down", 65361: "left", 65363: "right",          # Linux
}
# key -> (rc channel, sign); channels: 0 left/right, 1 forward/back, 2 up/down, 3 yaw
MOVES = {"up": (1, 1), "down": (1, -1), "left": (0, -1), "right": (0, 1),
         "w": (2, 1), "s": (2, -1), "a": (3, -1), "d": (3, 1)}
HELP = "t takeoff | arrows move | w/s up/down | a/d turn | space hover | +/- speed | q land/quit | 3 EMERGENCY"
WIN = "kite drone  -  click here for the keys  -  q land  -  3 EMERGENCY"


def key_name(code: int) -> str | None:
    """'up'/'down'/'left'/'right' for arrows, the lowercase character for printable keys, None for nothing."""
    if code is None or code < 0:
        return None
    if code in ARROWS:
        return ARROWS[code]
    ch = code & 0xFF
    if 32 <= ch < 127:
        return chr(ch).lower()
    return None


class Pilot:
    """Key presses -> drone commands, one control tick at a time. No threads, no sleeps: the caller's loop drives it."""

    def __init__(self, drone, speed: int = 30, pulse_s: float = 0.6, max_rc: int = 50, log=print):
        self.drone = drone
        self.speed = speed
        self.pulse_s = pulse_s
        self.max_rc = max_rc
        self.log = log
        self.hold = [0, 0, 0, 0]
        self.until = [0.0] * 4
        self.sent = (0, 0, 0, 0)
        self.action = "ok"
        self.quit = False
        self.note = ""  # last message, shown in the window
        self._n_events = 0

    def say(self, text: str) -> None:
        self.note = text
        self.log(f"  >> {text}")

    def state(self) -> str:
        d = self.drone
        if d.busy():
            return "busy"
        return "FLYING" if d.flying else "on ground"

    def on_key(self, key: str | None, now: float, video_age_s: float | None) -> None:
        if key is None:
            return
        d = self.drone
        if key == "3":
            if d.emergency() is False:  # the adapter says the packet could not be sent (Wi-Fi down)
                self.say("EMERGENCY NOT SENT (Wi-Fi down): press 3 again")
            else:
                self.hold, self.until = [0, 0, 0, 0], [0.0] * 4
                self.say("EMERGENCY: motors off")
        elif key == "q":
            if d.flying or d.busy():
                d.land()
                self.hold, self.until = [0, 0, 0, 0], [0.0] * 4
                self.say("landing")
            else:
                self.quit = True
        elif key == "t":
            if d.flying or d.busy():
                self.say("takeoff refused: already flying or busy")
            elif video_age_s is None or video_age_s > 1.0:
                self.say("takeoff refused: no live video (the safety governor would only hover)")
            else:
                d.takeoff()
                self.say("taking off")
        elif key == " ":
            self.hold, self.until = [0, 0, 0, 0], [0.0] * 4
        elif key in ("+", "="):
            self.speed = min(self.max_rc, self.speed + 10)
            self.say(f"speed {self.speed}")
        elif key in ("-", "_"):
            self.speed = max(10, self.speed - 10)
            self.say(f"speed {self.speed}")
        elif key in MOVES:
            i, sign = MOVES[key]
            self.hold[i] = sign * self.speed
            self.until[i] = now + self.pulse_s

    def tick(self, now: float, wall: float, last_frame_t: float | None) -> tuple[int, int, int, int]:
        """The 10 Hz control step: safety check, then the sticks (sent every tick: the rc keep-alive)."""
        d = self.drone
        self.action = d.check(wall, last_frame_t)
        busy = d.busy()  # also what completes takeoff / land: poll it every tick
        events = getattr(d, "events", [])
        for _, msg in events[self._n_events:]:
            self.say(f"safety: {msg}")
        self._n_events = len(events)
        if self.action != "ok":
            self.hold, self.until = [0, 0, 0, 0], [0.0] * 4  # never resume a stick after a safety hover or landing
        vals = tuple(h if now < u else 0 for h, u in zip(self.hold, self.until))
        if d.flying and not busy:
            d.rc(*vals)
            self.sent = vals
        else:
            self.sent = (0, 0, 0, 0)
        return self.sent


def land_and_wait(drone, timeout_s: float = 25.0, log=print) -> bool:
    """Land and wait until it is down; Ctrl+C is ignored meanwhile. True once confirmed on the ground."""
    if not (drone.flying or drone.busy()):
        return True
    log("LANDING - keep clear (Ctrl+C is ignored until it is down)")
    t_end, asked = time.time() + timeout_s, -1e9
    while time.time() < t_end:
        try:
            busy = drone.busy()
            if asked > 0 and not busy and not drone.flying:
                log("landed")
                return True
            if asked < 0 or (not busy and time.time() - asked > 2.0):
                drone.land()  # first ask, or the last one failed while still flying
                asked = time.time()
            drone.check(time.time(), time.time())  # keeps the governor's land re-send going
            time.sleep(0.05)
        except KeyboardInterrupt:
            log("  Ctrl+C ignored: still landing")
    log("!! LAND NOT CONFIRMED: the drone may still be airborne (it lands itself ~15 s after the last command)")
    return False


def hud(img, pilot: Pilot, tel, fps: float, link_error: str | None, dry_run: bool):
    import cv2

    h = tel.height_m if tel is not None else None
    bat = tel.battery_pct if tel is not None else None
    lines = [
        f"{'DRY RUN  ' if dry_run else ''}{pilot.state()}   bat {bat if bat is not None else '?'}%   "
        f"h {h:.2f} m   video {fps:4.1f} fps" if h is not None else
        f"{'DRY RUN  ' if dry_run else ''}{pilot.state()}   bat {bat if bat is not None else '?'}%   video {fps:4.1f} fps",
        f"sticks lr {pilot.sent[0]} fb {pilot.sent[1]} ud {pilot.sent[2]} yaw {pilot.sent[3]}   speed {pilot.speed}   "
        f"safety {pilot.action}" + (f"   WI-FI DOWN ({link_error})" if link_error else ""),
        pilot.note,
    ]
    for k, text in enumerate(lines):
        if text:
            y = 22 + 22 * k
            cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
    y = img.shape[0] - 12
    cv2.putText(img, HELP, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, HELP, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def fly(drone, src, args, cv2=None, clock=time.monotonic, wall=time.time) -> None:
    """The one loop: keys from the window, the 10 Hz control tick, the window. Lands on Ctrl+C or any error."""
    import numpy as np

    if cv2 is None:
        import cv2
    pilot = Pilot(drone, speed=args.speed, pulse_s=args.pulse, max_rc=args.max_rc)
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    blank = np.zeros((720, 960, 3), np.uint8)
    last_img, last_seq, last_frame_t = None, -1, None
    next_tick = 0.0
    fps, n_frames, fps_t = 0.0, 0, clock()
    print(HELP)
    try:
        while not pilot.quit:
            now = clock()
            f = src.read()
            if f is not None and f.seq != last_seq:
                last_seq, last_frame_t, last_img = f.seq, f.t, f.image
                n_frames += 1
            if now - fps_t >= 1.0:
                fps, n_frames, fps_t = n_frames / (now - fps_t), 0, now
            age = None if last_frame_t is None else wall() - last_frame_t
            pilot.on_key(key_name(cv2.waitKeyEx(1)), now, age)
            if now >= next_tick:
                next_tick = now + 0.1
                pilot.tick(now, wall(), last_frame_t)
            img = (last_img if last_img is not None else blank).copy()
            if last_img is None:
                cv2.putText(img, "waiting for video ...", (330, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            try:
                tel = drone.telemetry()
            except Exception:
                tel = None
            link = getattr(getattr(drone, "inner", drone), "link_error", None)
            cv2.imshow(WIN, hud(img, pilot, tel, fps, link, args.dry_run))
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:  # window closed: land and quit
                pilot.say("window closed")
                break
    except KeyboardInterrupt:
        print("\nCtrl+C")
    except Exception as e:  # a bug here must still land the drone
        print(f"\nerror: {e!r}")
        raise
    finally:
        land_and_wait(drone)
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except Exception:
            pass


def build_parser() -> argparse.ArgumentParser:
    from kite.config import VIDEO_BACKENDS

    p = argparse.ArgumentParser(description="Fly the Tello from a live camera window (arrows move, q lands, 3 cuts "
                                "the motors).")
    p.add_argument("--dry-run", action="store_true", help="video and keys work, no motion is ever sent")
    p.add_argument("--speed", type=int, default=30, help="stick value for the move keys, 10-50 (default 30)")
    p.add_argument("--pulse", type=float, default=0.6, help="seconds a key press drives its stick; holding the key "
                   "keeps it going (default 0.6)")
    p.add_argument("--video-backend", choices=VIDEO_BACKENDS, default=None,
                   help="video reader: pyav (default), opencv, raw, auto (raw on macOS)")
    p.add_argument("--min-battery", type=int, default=20, help="refuse to start below this battery %% (default 20)")
    return p


def main(argv: list[str] | None = None) -> int:
    from kite.config import Config, resolve_video_backend
    from kite.drone import SafetyGovernor, make_tello

    args = build_parser().parse_args(argv)
    if not 10 <= args.speed <= 50:
        print("--speed must be 10-50 (the safety governor caps sticks at 50)")
        return 2
    if not 0.2 <= args.pulse <= 2.0:
        print("--pulse must be 0.2-2.0 s")
        return 2
    os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")  # hide FFmpeg's decoder chatter before the first keyframe
    try:
        import av.logging

        av.logging.set_level(av.logging.PANIC)
    except Exception:
        pass
    from djitellopy import Tello

    Tello.LOGGER.setLevel(logging.CRITICAL)  # djitellopy's 0xCC binary-reply errors are harmless (it retries)

    cfg = Config()
    if args.video_backend:
        cfg.drone.video_backend = resolve_video_backend(args.video_backend)
    args.max_rc = int(cfg.safety.max_rc)
    drone = SafetyGovernor(make_tello(cfg, dry_run=args.dry_run), cfg.safety)
    print(f"connecting to the Tello ({'DRY RUN: no motion is sent' if args.dry_run else 'LIVE'}) ...")
    try:
        try:
            drone.connect()
        except Exception as e:
            print(f"cannot connect: {e}\nJoin the TELLO-xxxxxx Wi-Fi; close other Tello programs; allow Python through "
                  "the firewall (Windows: Private AND Public).")
            return 1
        bat = drone.telemetry().battery_pct
        print(f"connected, battery {bat}%")
        if not args.dry_run and bat is not None and bat < args.min_battery:
            print(f"battery {bat}% is under --min-battery {args.min_battery}%: not flying")
            return 1
        fly(drone, drone.frame_source(), args)
    finally:
        drone.close()  # lands if still flying (never in dry run), stops the video
    return 0


if __name__ == "__main__":
    sys.exit(main())
