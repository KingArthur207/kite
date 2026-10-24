"""Make the sim's clean renders look like Tello frames. Parameters: sim/public/tello_camera_model.json.

    from tello_cam import TelloCam, load_model
    cam = TelloCam(load_model())
    bgr = cam.process(bgr, t=time.monotonic(), yaw_rate_dps=30.0)   # one frame, in order

Per frame, in this order: lens distortion (+ optional rolling-shutter skew), motion blur from the turn rate,
lens softness, vignetting, auto-exposure gain (adapts slowly), shot + read noise, colour response, then one JPEG
pass at low quality for the H.264 blockiness. stream() decides latency / drop / repeat for the frame link; the
caller (sim/frames.py) applies it. scripts/sim_real_gap.py --fit tunes the JSON against real footage.
"""

from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "public" / "tello_camera_model.json"
CALIBRATION_PATH = HERE.parent / "data" / "tello_camera.json"   # kite.calibrate output, if the Tello was calibrated


def load_model(path: str | Path | None = None, calibration: str | Path | None = CALIBRATION_PATH) -> dict:
    m = json.loads(Path(path or MODEL_PATH).read_text())
    cal = Path(calibration) if calibration else None
    if cal and cal.is_file():
        dist = json.loads(cal.read_text()).get("dist")
        if dist:
            m["lens"]["dist"] = list(dist)
            m["lens"]["dist_source"] = str(cal.name)
    return m


def luma(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


class TelloCam:
    def __init__(self, model: dict, seed: int | None = None):
        self.m = copy.deepcopy(model)
        self.rng = np.random.default_rng(seed)
        self.pyrng = random.Random(seed)
        self.gain: float | None = None   # auto-exposure state
        self.t: float | None = None
        self._maps: dict = {}            # (w, h) -> (map_x, map_y, vignette)

    # ------------------------------------------------------------------ geometry
    def _geometry(self, w: int, h: int):
        if (w, h) not in self._maps:
            f = self.m["lens"]["render_fx"]
            K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)
            u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            pts = np.stack([u.ravel(), v.ravel()], 1).reshape(-1, 1, 2)
            # Output pixel (distorted, like the real lens) <- pinhole render at the undistorted point.
            und = cv2.undistortPoints(pts, K, np.array(self.m["lens"]["dist"], np.float64)).reshape(h, w, 2)
            mx = (und[..., 0] * f + w / 2).astype(np.float32)
            my = (und[..., 1] * f + h / 2).astype(np.float32)
            r2 = ((u - w / 2) ** 2 + (v - h / 2) ** 2) / ((w / 2) ** 2 + (h / 2) ** 2)
            vig = (1 - self.m["vignette"]["strength"] * r2).astype(np.float32)[..., None]
            self._maps[(w, h)] = (mx, my, vig)
        return self._maps[(w, h)]

    # ------------------------------------------------------------------ one frame
    def process(self, bgr: np.ndarray, t: float | None = None, yaw_rate_dps: float = 0.0) -> np.ndarray:
        m = self.m
        h, w = bgr.shape[:2]
        mx, my, vig = self._geometry(w, h)
        f = m["lens"]["render_fx"]
        omega = math.radians(yaw_rate_dps)
        if m["rolling_shutter"]["enabled"] and omega:
            # Row r is read (r/h - 0.5) * readout later than the middle row; the scene has slid by f * omega * dt px.
            rows = (np.arange(h, dtype=np.float32) / h - 0.5) * m["rolling_shutter"]["readout_ms"] / 1000
            mx = mx + (f * omega * rows)[:, None]
        img = cv2.remap(bgr, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        blur_px = abs(f * omega * m["exposure"]["exposure_ms"] / 1000) if m["motion_blur"]["enabled"] else 0
        if blur_px >= 1.5:
            k = int(round(blur_px)) | 1
            img = cv2.blur(img, (k, 1))
        sigma = m["lens"]["softness_sigma_px"]
        if sigma > 0.05:
            img = cv2.GaussianBlur(img, (0, 0), sigma)

        x = img.astype(np.float32) * vig
        x *= self._exposure(float(luma(img).mean()), t)
        n = m["noise"]
        sd = np.sqrt(n["shot"] * np.clip(x, 0, 255) + n["read"] ** 2)
        x += self.rng.standard_normal(x.shape, np.float32) * sd
        x = self._colour(np.clip(x, 0, 255))
        out = np.clip(x + 0.5, 0, 255).astype(np.uint8)
        ok, jpg = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(m["compression"]["jpeg_quality"])])
        return cv2.imdecode(jpg, cv2.IMREAD_COLOR) if ok else out

    def _exposure(self, mean_luma: float, t: float | None) -> float:
        e = self.m["exposure"]
        want = float(np.clip(e["target_luma"] / max(mean_luma, 1.0), e["min_gain"], e["max_gain"]))
        if self.gain is None or t is None or self.t is None:
            self.gain = want                      # first frame (or stateless use): already settled
        else:
            a = 1 - math.exp(-max(t - self.t, 0.0) / max(e["adapt_s"], 1e-3))
            self.gain += (want - self.gain) * a
        self.t = t
        return self.gain

    def _colour(self, x: np.ndarray) -> np.ndarray:
        c = self.m["color"]
        r, g, b = c["gains_rgb"]
        x = x * np.array([b, g, r], np.float32)                 # BGR
        y = x @ np.array([0.114, 0.587, 0.299], np.float32)
        x = y[..., None] + (x - y[..., None]) * c["saturation"]
        x = 128 + (x - 128) * c["contrast"]
        x = np.clip(x, 0, 255) / 255
        if c["gamma"] != 1.0:
            x = x ** (1 / c["gamma"])
        return c["black_lift"] + (255 - c["black_lift"]) * x

    # ------------------------------------------------------------------ the video link
    def stream(self) -> tuple[str, float]:
        """("ok" | "drop" | "dup", latency in s) for the next frame."""
        s = self.m["stream"]
        lo, hi = s["latency_ms"]
        p = self.pyrng.random()
        kind = "drop" if p < s["drop_p"] else "dup" if p < s["drop_p"] + s["dup_p"] else "ok"
        return kind, self.pyrng.uniform(lo, hi) / 1000


if __name__ == "__main__":      # self-check: python sim/tello_cam.py
    cam = TelloCam(load_model(), seed=1)
    flat = np.full((720, 960, 3), 200, np.uint8)
    out = cam.process(flat)
    assert out.shape == flat.shape
    centre, corner = luma(out)[340:380, 460:500].mean(), luma(out)[:40, :40].mean()
    assert corner < centre - 10, (centre, corner)           # vignette
    assert abs(float(luma(out).mean()) - cam.m["exposure"]["target_luma"]) < 40   # auto-exposure pulled it to target
    assert luma(out)[340:380, 460:500].std() > 0.5          # noise
    bars = np.zeros((720, 960, 3), np.uint8); bars[:, ::16] = 255
    still, turning = TelloCam(cam.m, 1).process(bars), TelloCam(cam.m, 1).process(bars, yaw_rate_dps=90)
    assert np.abs(np.diff(luma(turning).astype(float), axis=1)).mean() < np.abs(np.diff(luma(still).astype(float), axis=1)).mean()
    kinds = [TelloCam(cam.m, s).stream()[0] for s in range(400)]
    assert 0 < kinds.count("drop") < 40 and all(0.1 <= TelloCam(cam.m, s).stream()[1] <= 0.2 for s in range(20))
    print("tello_cam OK")
