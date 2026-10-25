"""Play a frame source through perception and the segmentation window."""

from __future__ import annotations

from .camera import Camera, Pose


def freshen(source, camera: Camera, pose: Pose) -> tuple[Camera, Pose]:
    """Take fx, fy, height, and pitch from a sim message when it sent them."""
    fx = getattr(source, "fx", None)
    fy = getattr(source, "fy", None)
    if fx and fy:
        camera = Camera(float(fx), float(fy), camera.role)
    height = getattr(source, "height_m", None)
    pitch = getattr(source, "pitch", None)
    if height is not None and pitch is not None:
        pose = Pose(float(height), float(pitch))
    return camera, pose


def play(
    perception,
    source,
    camera: Camera,
    pose: Pose,
    display,
    *,
    chest=None,
    chest_camera: Camera | None = None,
    chest_pose: Pose | None = None,
    max_frames: int = 0,
    on_event=None,
    on_image=None,
    camera_fn=None,
) -> int:
    seen = 0
    chest_open = chest
    try:
        while True:
            if max_frames and seen >= max_frames:
                break
            frame = source.read()
            if frame is None:
                if getattr(source, "ended", True):
                    break
                if display.poll_quit():
                    break
                waiter = getattr(display, "wait", None)
                if waiter and waiter(1) in (ord("q"), 27):
                    break
                continue
            if camera_fn is not None:
                camera = camera_fn(frame, camera)
            camera, pose = freshen(source, camera, pose)
            events = perception.process(frame, camera, pose)
            _emit(events, on_event)
            title = "Kite chest" if camera.role == "chest" else "Kite street"
            if perception.image is not None:
                display.show(perception.image, title)
                if on_image is not None:
                    on_image(perception.image)
            if chest_open is not None and chest_camera is not None and chest_pose is not None:
                chest_open = _chest_once(
                    perception, chest_open, chest_camera, chest_pose, display, on_event
                )
            seen += 1
            if display.poll_quit():
                break
    finally:
        close = getattr(source, "close", None)
        if close:
            close()
        if chest_open is not None:
            chest_open.close()
        display.close()
    return seen


def _chest_once(perception, chest, camera, pose, display, on_event):
    frame = chest.read()
    if frame is None:
        if getattr(chest, "ended", False):
            chest.close()
            return None
        return chest
    events = perception.process(frame, camera, pose)
    _emit(events, on_event)
    if perception.image is not None:
        display.show(perception.image, "Kite chest")
    return chest


def _emit(events, on_event) -> None:
    if on_event is None:
        return
    for event in events:
        on_event(event)
