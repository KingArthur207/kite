"""kite: the main app. Take off, lock onto the person in front, follow them; land on command; e cuts the motors.

    python run/app.py                 # live: drone on, laptop on its Wi-Fi
    python run/app.py --dry-run       # on the table: video, YOLO and commands, nothing reaches the motors
    python run/app.py --yaw law       # steer yaw by bearing instead of the fruit fly brain
    python run/app.py --yolo          # also show YOLO26n obstacle boxes on the camera (show only)

Each run also writes recordings/<time>_app/video.mkv and recordings/<time>_app/frames/NNNNNN.jpg (the camera
with the person boxes). Saving runs on its own thread and drops frames if the disk falls behind; it does not
change takeoff.

Console commands (type in the window, Enter):
    takeoff   take off, climb to --height, then follow the person in front of the drone
    land      stop following and land
    follow    look for the person again (while holding, or once they have been out of view for 2 s)
    hold      pause: stop and hover          resume   turn until the person is seen, then follow
    stop      while holding: look for the person again
    e         EMERGENCY: motors off at once (the drone drops if it is in the air), then the app quits
    status    battery, height, video and follow state
    quit      land, then close (closing the window does the same)

Stand about 2 m in front of the drone facing away from it before takeoff. It climbs to --height above the floor
(default --walker-height + 0.2), locks onto the largest person in view (you) and holds --behind (default 1.8 m) behind
you, reading the distance from the angle down to the top of your head, your height and your shoulders: give your real
--walker-height. Lost for 20 s: it drops to just below head height and turns 30 deg at a time to find you. Safety: an open space, prop guards, a
spotter. Everything flies through kite's SafetyGovernor (stick cap, 0.5-2.3 m altitude band, lands on low battery,
hovers when video or telemetry stall).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)  # the kite package next to this folder, installed or not
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def quiet_libraries() -> None:
    """Hide FFmpeg's decoder chatter before the first keyframe and djitellopy's harmless 0xCC reply errors."""
    os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
    try:
        import av.logging

        av.logging.set_level(av.logging.PANIC)
    except Exception:
        pass
    try:
        from djitellopy import Tello

        Tello.LOGGER.setLevel(logging.CRITICAL)
    except Exception:
        pass


def build_parser() -> argparse.ArgumentParser:
    from kite.config import VIDEO_BACKENDS

    p = argparse.ArgumentParser(prog="python run/app.py", description="kite: take off, lock onto the person in front, "
                                "follow them.")
    p.add_argument("--dry-run", action="store_true", help="video, YOLO and commands work, nothing reaches the motors")
    p.add_argument("--yaw", choices=("fly", "law", "retrained", "pd"), default="fly",
                   help="what steers yaw while following: fly (the fruit fly brain it flew with, FLY-YAW smooth2, "
                        "default), law (turn toward the person by bearing), retrained (kite.fly's retrained brain, "
                        "simulation-tested only) or pd")
    p.add_argument("--walker-height", type=float, default=1.7, help="the walker's real height WITH SHOES in m, to "
                   "+-2 cm (default 1.7): the follow distance is read from it, 5 cm off is 10-20 %% distance error")
    p.add_argument("--height", type=float, default=None, help="follow altitude above the floor, m (default: walker "
                   "height + 0.2, i.e. 1.9 m for 1.7 m)")
    p.add_argument("--behind", type=float, default=1.8, help="follow distance behind the person, m (default 1.8)")
    p.add_argument("--orbit", action="store_true", help="slide around behind the person when they turn (from the "
                   "pose model's shoulders); off by default, as flown")
    p.add_argument("--speak", action="store_true", help="also say the drone's announcements out loud (macOS say, "
                   "Windows speech)")
    p.add_argument("--video-backend", choices=VIDEO_BACKENDS, default=None,
                   help="video reader: pyav (default), opencv, raw, auto (raw on macOS)")
    p.add_argument("--yolo", action="store_true", help="also show YOLO26n (COCO) obstacle boxes on the camera; "
                   "show only, the drone ignores them")
    p.add_argument("--weights", default=None, help="YOLO pose weights (default data/models/yolo11n-pose.pt)")
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles that cannot print the console arrows must not crash
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    args = build_parser().parse_args(argv)
    min_height = round(args.walker_height + 0.2, 2)  # to the cm: 1.81 + 0.2 is 2.0100000000000002 in floats
    if args.height is None:
        args.height = min(min_height, 2.2)
    if not 0.8 <= args.height <= 2.2:
        print("--height must be 0.8-2.2 m (the safety governor holds 0.5-2.3 m)")
        return 2
    if args.height < min_height:
        print(f"--height must be at least --walker-height + 0.2 m ({min_height:.2f}): the follow flies "
              "above the head and reads the distance from the angle down to it")
        return 2
    quiet_libraries()

    from pilot import Pilot
    from ui import App

    app = App(lambda log: Pilot(args, log), title="kite" + ("  -  DRY RUN" if args.dry_run else ""))
    app.log(f"{'DRY RUN: nothing reaches the motors. ' if args.dry_run else ''}follow at {args.height:.2f} m, "
            f"{args.behind:.1f} m behind, steering: {args.yaw}", "info")
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
