"""Street perception for Kite.

One call, every frame source: ``process(frame, camera, pose)`` returns
earpiece events. The drone or sim view is segmented. The chest camera
reports close obstacles. A direction is a say with priority nav, or a hint.
"""

from .camera import (
    CHEST_CAMERA,
    CHEST_POSE,
    DEMO_POSE,
    DRONE_POSE,
    TELLO_CAMERA,
    Camera,
    Pose,
)
from .events import Event
from .pipeline import Perception

__all__ = [
    "CHEST_CAMERA",
    "CHEST_POSE",
    "Camera",
    "DEMO_POSE",
    "DRONE_POSE",
    "Event",
    "Perception",
    "Pose",
    "TELLO_CAMERA",
]
