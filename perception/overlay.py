"""Overlay frames for review and the README: boxes, masks, kind / bearing / distance, how each range
was measured, the fitted pitch, and any hazard or stop this frame. Never used on live cameras."""

from __future__ import annotations

import cv2
import numpy as np

# BGR
RED, ORANGE, YELLOW, GREEN, CYAN, WHITE = (60, 60, 230), (0, 140, 255), (0, 215, 255), (90, 200, 90), \
    (230, 200, 0), (255, 255, 255)
COLOR = {"person": GREEN, "bike": CYAN, "vehicle": RED, "construction": ORANGE, "overhead": ORANGE,
         "curb": YELLOW, "stairs_down": YELLOW, "stairs_up": YELLOW}


def _label(img, text, x, y, col, scale=0.5):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    y = max(th + 4, y)
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y + 2), (25, 25, 25), -1)
    cv2.putText(img, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)


def draw(frame: np.ndarray, r, camera: str, title: str | None = None) -> np.ndarray:
    """r: pipeline.FrameResult (or anything with dets, objects, calib, timings, events)."""
    img = frame.copy()
    h, w = img.shape[:2]
    tint = img.copy()
    by_id = {o["id"]: o for o in r.objects}
    for i, d in enumerate(r.dets):
        o = by_id.get(str(d.tid if d.tid is not None else i))
        if o is None:
            continue
        col = RED if o.get("stop") else COLOR.get(o["kind"], WHITE)
        if d.mask is not None:
            m = cv2.resize(d.mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
            tint[m] = col
        cv2.rectangle(img, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), col, 2)
        rng = {"ground": "g", "depth": "d", "size": "s"}.get(o.get("range"), "?")
        _label(img, f"{o['kind']} {o['distance']:.1f}m {o['bearing']:+.0f}d [{rng}]",
               int(d.x1), int(d.y1) - 3, col)
    img = cv2.addWeighted(tint, 0.25, img, 0.75, 0)
    for k, o in enumerate(getattr(r, "depth_objects", None) or []):
        _label(img, f"depth: {o['kind']} {o['bearing']:+.0f}d {o['distance']:.1f}m", 10, 50 + 24 * k,
               RED if o.get("stop") else YELLOW, 0.55)
    c = r.calib
    ms = " ".join(f"{k} {v:.0f}ms" for k, v in r.timings.items())
    _label(img, title or f"{camera}  pitch {c.pitch:+.1f}d ({c.source})  {ms}", 6, 22, WHITE, 0.55)
    y = h - 12
    for ev in r.events:
        if ev["type"] == "hazard":
            _label(img, ev["text"], 10, y, ORANGE, 0.6)
            y -= 26
    stop = [o for o in r.objects if o.get("stop")]
    if stop:
        o = min(stop, key=lambda o: o["distance"])
        _label(img, f"STOP - {o['kind']} {o['distance']:.1f} m", 10, y, RED, 0.8)
    return img
