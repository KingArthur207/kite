"""kite: the main app. Take off, lock onto the person in front, follow them; land on command; e cuts the motors.

    python run/app.py                 # live: drone on, laptop on its Wi-Fi
    python run/app.py --dry-run       # on the table: video, YOLO and commands, nothing reaches the motors
    python run/app.py --yaw fly       # steer yaw with the retrained kite.fly brain (simulation-tested only)

Console commands (type in the window, Enter):
    takeoff   take off, find the person right in front of the drone and lock on, then follow them
    land      stop following and land
    e         EMERGENCY: motors off at once (the drone drops if it is in the air), then the app quits
    status    battery, height, video and follow state
    quit      land, then close (closing the window does the same)

Stand about 1 m in front of the drone facing away from it before takeoff. It locks onto you (the largest person in
view), climbs to --height (default --walker-height + 0.3) and holds --behind behind you, reading the distance from the
angle to the top of your head: give your real --walker-height. Safety: an open space, prop guards, a
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
    p.add_argument("--yaw", choices=("law", "fly", "pd"), default="law",
                   help="what steers yaw while following: law (turn toward the person, default), fly (the retrained "
                        "kite.fly brain, simulation-tested only) or pd")
    p.add_argument("--walker-height", type=float, default=1.7, help="the walker's real height in m (default 1.7): "
                   "the follow distance is read from it, 5 cm off is 10-20 %% distance error")
    p.add_argument("--height", type=float, default=None, help="follow altitude above the takeoff point, m "
                   "(default: walker height + 0.3, within the 2.3 m safety ceiling)")
    p.add_argument("--behind", type=float, default=1.0, help="follow distance behind the person, m (default 1.0)")
    p.add_argument("--video-backend", choices=VIDEO_BACKENDS, default=None,
                   help="video reader: pyav (default), opencv, raw, auto (raw on macOS)")
    p.add_argument("--weights", default=None, help="YOLO weights (default data/models/yolo11n.pt)")
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles that cannot print the console arrows must not crash
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    args = build_parser().parse_args(argv)
    if args.height is None:
        args.height = min(round(args.walker_height + 0.3, 2), 2.2)
    if not 0.8 <= args.height <= 2.2:
        print("--height must be 0.8-2.2 m (the safety governor holds 0.5-2.3 m)")
        return 2
    if abs(args.height - args.walker_height) < 0.2:
        print("--height must be at least 0.2 m above (or below) --walker-height: the distance is read from the "
              "angle down (or up) to the top of the head")
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
