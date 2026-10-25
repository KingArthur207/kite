"""perception.near: is something directly in front of me, and how urgent is it? Camera only.

Several independent distance cues per object (known size, ground contact, ground-anchored dense depth, motion
stereo from the walker's own speed) fused by inverse variance in log-distance, looming time-to-contact as a
state of a per-track Kalman filter, a +-0.5 m walking corridor, and priorities P1 (< 2 m), P2 (2 to 5 m),
P3 (>= 5 m) with hysteresis. See perception/near/README.md.

    python -m perception.near --source webcam:Logitech        # live (your machine)
    python -m perception.near --source data/imagined/doorway.mp4 --no-send --radar
"""

from .. import __name__ as _parent  # noqa: F401  (perception/__init__ puts earpiece/, sim/ on sys.path)
