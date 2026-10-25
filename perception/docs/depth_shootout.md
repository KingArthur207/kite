# Depth shoot-out: which depth source should Kite run live?

## Recommendation

**Run Depth Anything V2 Metric Indoor Small (Apache-2.0), anchored to the floor, fused with motion stereo.** Not one
model, a combination, and the numbers below are why.

- **No model's own metric scale can be trusted.** Raw, every Depth Anything V2 variant is far off on the chest-camera
  walks (AbsRel 0.50 to 1.23). Anchoring the scale to the floor, which sits a known 1.2 m below a chest camera,
  cuts the error by 3 to 8 times: Indoor Small goes from 0.496 to 0.145 AbsRel, and its corridor error goes from
  0.464 to 0.086.
- **Motion stereo is the most accurate source, where it can see.** Using only its confident pixels, it reaches
  0.046 corridor AbsRel and catches 100 % of the P1 frames. But it is blind when the walker stands still (0 %
  priority right on the standstill walk), because there is no parallax. That is the same reason ARCore's Depth
  API is described as needing camera motion.
- **The fusion keeps both strengths.** Indoor Small plus floor anchoring plus motion stereo gets the priority band
  right 84 % of the time, catches 100 % of P1 frames and raises a false P1 on 3 % of frames. Its corridor error is
  0.050, and the nearest obstacle in the corridor is placed within 0.15 to 0.30 m while walking.
- **The ScrollSafe rule** is box height over frame height in the middle half of the frame. Even with perfect boxes,
  it gets the band right 18 % of the time on the same walks and raises a false P1 56 % of the time.
- **Base is not worth it.** Outdoor Base fused scores 85 % against 84 % for Indoor Small fused, at about 3 times the
  latency (326 ms against 105 ms per frame on the M1 GPU). It is also CC-BY-NC.
- **Indoor beats Outdoor, even on outdoor scenes.** After anchoring, Outdoor Small and Outdoor Base never call P1 on
  the synthetic walks (0 % recall). Indoor Small, trained for 0 to 20 m, keeps near-field detail. Anchoring
  corrects its scale for streets anyway.
- **Temporal smoothing** adds 1 to 2 points to the priority score. It is cheap, so keep it.

**Latency on an 8 GB M1:**
- The model takes 105 ms per frame on the GPU, or 275 ms on the CPU.
- Motion stereo takes about 100 ms on the CPU.
- Run one after the other, the fused source takes 234 ms per frame, about 4 fps.
- The model runs on the GPU and motion stereo on the CPU, so pipelining them should approach the slower of the two
  (about 10 fps). That has not been measured yet.

**Honest limits:**
- On real laser-scanned photos (DIODE stills), every source is weak inside the corridor. The best priority score is
  50 to 70 %, and the floor fit is accepted on only 48 to 72 % of frames.
- Stills cannot use motion stereo at all.
- This is why `perception/near` does not rely on depth alone. It fuses this depth with per-object cues (known
  object size, ground contact) and looming time to contact.

**Not run on this machine:** Apple Depth Pro, UniDepth V2, Metric3D V2 and Video Depth Anything. Their adapters
are in `depth_zoo.py`; run them with `--models`.
- Depth Pro has 952 M parameters, too many for 8 GB next to the rest of Kite, and a research-only licence.
- UniDepth V2 is CC-BY-NC.

## Results

Ground truth: 100 DIODE validation frames (30 with enough floor in view to define the walking corridor) and four synthetic chest-camera walks (sidewalk 112, curb 96, hallway 90, standstill 45 frames at 15 fps, 640x480, 60 degree diagonal like a C270). Run time 11.4 min. Regenerate: `python -m perception.bench_depth` (see perception/README.md).

![GT and each depth source, turbo colormap (near = red)](depth_shootout.png)

### Synthetic walks (exact depth; motion, anchoring and fusion all apply)

| Depth source | Frames | AbsRel | RMSE (m) | delta<1.25 | Corridor AbsRel (0-5 m) | Priority right | P1 recall | False P1 | Pitch error (deg) | Floor fit ok | ms/frame |
|---|---|---|---|---|---|---|---|---|---|---|---|
| motion-stereo (confident px only) | 291 | 0.061 | 1.18 | 0.964 | 0.046 | 61 % | 100 % | 1 % | - | - | - |
| fused: da2-metric-indoor-s+anchor + motion stereo | 343 | 0.107 | 3.97 | 0.836 | 0.050 | 84 % | 100 % | 3 % | - | - | 234 |
| fused: da2-metric-outdoor-b+anchor + motion stereo | 343 | 0.117 | 4.42 | 0.853 | 0.050 | 85 % | 100 % | 1 % | - | - | 485 |
| fused: da2-metric-outdoor-s+anchor + motion stereo | 343 | 0.130 | 4.49 | 0.833 | 0.058 | 83 % | 100 % | 2 % | - | - | 246 |
| da2-relative-s+anchor (pitch prior) | 343 | 0.132 | 5.37 | 0.855 | 0.062 | 80 % | 74 % | 6 % | 0.5 | 100 % | 121 |
| da2-metric-indoor-s+anchor+temporal | 343 | 0.137 | 3.94 | 0.746 | 0.076 | 70 % | 91 % | 6 % | 1.2 | 100 % | 124 |
| da2-metric-indoor-s+anchor | 343 | 0.145 | 3.97 | 0.723 | 0.086 | 68 % | 91 % | 8 % | 1.2 | 100 % | 115 |
| da2-metric-outdoor-b+anchor | 343 | 0.166 | 4.29 | 0.767 | 0.100 | 73 % | 0 % | 0 % | 1.1 | 100 % | 363 |
| da2-metric-outdoor-b+anchor+temporal | 343 | 0.177 | 4.30 | 0.758 | 0.115 | 74 % | 0 % | 0 % | 1.1 | 100 % | 376 |
| da2-metric-outdoor-s+anchor | 343 | 0.207 | 4.25 | 0.698 | 0.144 | 68 % | 0 % | 3 % | 2.0 | 100 % | 124 |
| da2-metric-outdoor-s+anchor+temporal | 343 | 0.220 | 4.26 | 0.668 | 0.163 | 67 % | 0 % | 3 % | 2.0 | 100 % | 135 |
| da2-relative-s+anchor+temporal | 343 | 0.222 | 4.62 | 0.646 | 0.132 | 78 % | 54 % | 4 % | 2.0 | 100 % | 130 |
| da2-relative-s+anchor | 343 | 0.224 | 4.62 | 0.645 | 0.135 | 77 % | 54 % | 4 % | 2.0 | 100 % | 123 |
| da2-metric-indoor-s | 343 | 0.496 | 6.90 | 0.131 | 0.464 | 22 % | 100 % | 80 % | - | - | 108 |
| da2-metric-outdoor-b | 343 | 1.125 | 8.01 | 0.035 | 1.198 | 9 % | 0 % | 0 % | - | - | 354 |
| da2-metric-outdoor-s | 343 | 1.233 | 8.82 | 0.049 | 1.335 | 11 % | 0 % | 0 % | - | - | 118 |
| motion-stereo | 343 | 2.262 | 14.78 | 0.694 | 3.320 | 63 % | 100 % | 11 % | 0.8 | - | 102 |
| baseline: flat ground (dense, true pitch) | 343 | 2.623 | 33.61 | 0.608 | 0.233 | 9 % | 0 % | 0 % | - | - | 2 |
| baseline: ScrollSafe rule (boxes) | 343 | - | - | - | - | 18 % | 37 % | 56 % | - | - | - |
| baseline: flat ground (boxes) | 343 | - | - | - | - | 85 % | 0 % | 0 % | - | - | - |

### Priority band right, per synthetic walk

| Depth source | curb | hallway | sidewalk | standstill |
|---|---|---|---|---|
| baseline: ScrollSafe rule (boxes) | 1 % (- m) | 58 % (- m) | 7 % (- m) | 0 % (- m) |
| baseline: flat ground (boxes) | 70 % (- m) | 84 % (- m) | 93 % (- m) | 100 % (- m) |
| baseline: flat ground (dense, true pitch) | 21 % (- m) | 0 % (- m) | 11 % (- m) | 0 % (- m) |
| da2-metric-indoor-s | 14 % (1.60 m) | 62 % (0.77 m) | 7 % (2.14 m) | 0 % (2.83 m) |
| da2-metric-indoor-s+anchor | 41 % (1.05 m) | 73 % (0.66 m) | 76 % (0.74 m) | 100 % (0.92 m) |
| da2-metric-indoor-s+anchor+temporal | 43 % (0.98 m) | 76 % (0.62 m) | 77 % (0.68 m) | 100 % (0.92 m) |
| da2-metric-outdoor-b | 16 % (8.42 m) | 0 % (- m) | 14 % (3.75 m) | 0 % (5.58 m) |
| da2-metric-outdoor-b+anchor | 55 % (1.32 m) | 72 % (0.52 m) | 78 % (0.22 m) | 100 % (0.18 m) |
| da2-metric-outdoor-b+anchor+temporal | 57 % (1.43 m) | 71 % (0.87 m) | 79 % (0.30 m) | 100 % (0.19 m) |
| da2-metric-outdoor-s | 27 % (7.25 m) | 0 % (3.66 m) | 12 % (4.23 m) | 0 % (- m) |
| da2-metric-outdoor-s+anchor | 35 % (2.00 m) | 78 % (0.68 m) | 77 % (0.32 m) | 100 % (0.38 m) |
| da2-metric-outdoor-s+anchor+temporal | 31 % (2.18 m) | 76 % (0.76 m) | 78 % (0.43 m) | 100 % (0.39 m) |
| da2-relative-s+anchor | 75 % (0.50 m) | 80 % (0.35 m) | 68 % (0.80 m) | 100 % (1.06 m) |
| da2-relative-s+anchor (pitch prior) | 75 % (0.60 m) | 80 % (0.39 m) | 78 % (0.61 m) | 100 % (0.76 m) |
| da2-relative-s+anchor+temporal | 76 % (0.66 m) | 80 % (0.37 m) | 69 % (0.74 m) | 100 % (1.06 m) |
| fused: da2-metric-indoor-s+anchor + motion stereo | 69 % (0.15 m) | 88 % (0.30 m) | 87 % (0.17 m) | 100 % (0.92 m) |
| fused: da2-metric-outdoor-b+anchor + motion stereo | 69 % (0.21 m) | 93 % (0.26 m) | 88 % (0.04 m) | 100 % (0.18 m) |
| fused: da2-metric-outdoor-s+anchor + motion stereo | 59 % (0.38 m) | 91 % (0.27 m) | 90 % (0.04 m) | 100 % (0.38 m) |
| motion-stereo | 38 % (0.08 m) | 90 % (0.28 m) | 88 % (0.03 m) | 0 % (- m) |
| motion-stereo (confident px only) | 65 % (0.07 m) | 71 % (0.28 m) | 73 % (0.03 m) | 0 % (- m) |

In brackets: mean error of the nearest corridor obstacle's distance when it is within 5 m.

### DIODE outdoor (laser-scanned, stills)

| Depth source | Frames | AbsRel | RMSE (m) | delta<1.25 | Corridor AbsRel (0-5 m) | Priority right | P1 recall | False P1 | Pitch error (deg) | Floor fit ok | ms/frame |
|---|---|---|---|---|---|---|---|---|---|---|---|
| da2-relative-s (oracle affine fit to GT) | 60 | 0.338 | 7.81 | 0.579 | 1.640 | 70 % | 40 % | 0 % | - | - | 102 |
| da2-relative-s+anchor (pitch prior) | 20 | 0.453 | 8.36 | 0.643 | 2.619 | 50 % | 20 % | 0 % | - | 90 % | 102 |
| da2-metric-indoor-s | 60 | 0.524 | 10.63 | 0.171 | 1.500 | 45 % | 60 % | 33 % | - | - | 105 |
| da2-metric-indoor-s+anchor | 60 | 0.576 | 11.37 | 0.292 | 2.023 | 50 % | 20 % | 0 % | 3.6 | 67 % | 105 |
| da2-metric-outdoor-s+anchor | 60 | 0.638 | 9.84 | 0.275 | 2.421 | 20 % | 0 % | 0 % | 7.4 | 72 % | 106 |
| da2-metric-outdoor-b+anchor | 60 | 0.738 | 10.47 | 0.247 | 2.894 | 20 % | 0 % | 0 % | 6.0 | 72 % | 326 |
| da2-metric-outdoor-b | 60 | 0.826 | 8.06 | 0.301 | 3.914 | 20 % | 0 % | 0 % | - | - | 326 |
| da2-metric-outdoor-s | 60 | 0.839 | 7.96 | 0.321 | 3.865 | 15 % | 0 % | 0 % | - | - | 106 |
| da2-relative-s+anchor (shift 0) | 60 | 1.093 | 17.81 | 0.149 | 3.976 | 30 % | 0 % | 0 % | 6.1 | 43 % | 102 |
| baseline: flat ground (dense, true pitch) | 20 | 1.927 | 29.18 | 0.493 | 7.537 | 10 % | 0 % | 0 % | - | - | 3 |
| baseline: ScrollSafe rule (boxes) | 20 | - | - | - | - | 5 % | 0 % | 20 % | - | - | - |
| baseline: flat ground (boxes) | 20 | - | - | - | - | 20 % | 0 % | 0 % | - | - | - |

### DIODE indoor (laser-scanned, stills)

| Depth source | Frames | AbsRel | RMSE (m) | delta<1.25 | Corridor AbsRel (0-5 m) | Priority right | P1 recall | False P1 | Pitch error (deg) | Floor fit ok | ms/frame |
|---|---|---|---|---|---|---|---|---|---|---|---|
| da2-relative-s (oracle affine fit to GT) | 40 | 0.062 | 0.37 | 0.953 | 0.268 | 50 % | 56 % | 0 % | - | - | 106 |
| da2-metric-indoor-s | 40 | 0.279 | 1.27 | 0.527 | 0.379 | 30 % | 22 % | 0 % | - | - | 104 |
| da2-metric-indoor-s+anchor | 40 | 0.572 | 1.69 | 0.453 | 0.504 | 30 % | 33 % | 0 % | 7.6 | 48 % | 104 |
| da2-relative-s+anchor (shift 0) | 40 | 0.840 | 4.46 | 0.098 | 0.816 | 70 % | 78 % | 100 % | 15.1 | 22 % | 106 |
| da2-relative-s+anchor (pitch prior) | 10 | 0.968 | 8.28 | 0.310 | 0.514 | 70 % | 78 % | 100 % | - | 50 % | 98 |
| da2-metric-outdoor-b+anchor | 40 | 1.335 | 3.21 | 0.267 | 1.167 | 30 % | 33 % | 0 % | 9.2 | 62 % | 293 |
| da2-metric-outdoor-s+anchor | 40 | 1.405 | 3.11 | 0.222 | 0.874 | 40 % | 33 % | 0 % | 9.5 | 50 % | 119 |
| baseline: flat ground (dense, true pitch) | 10 | 2.037 | 15.95 | 0.404 | 1.687 | 0 % | 0 % | 0 % | - | - | 4 |
| da2-metric-outdoor-s | 40 | 2.286 | 5.03 | 0.111 | 2.567 | 0 % | 0 % | 0 % | - | - | 119 |
| da2-metric-outdoor-b | 40 | 2.448 | 5.42 | 0.106 | 2.872 | 0 % | 0 % | 0 % | - | - | 293 |
| baseline: ScrollSafe rule (boxes) | 10 | - | - | - | - | 30 % | 33 % | 0 % | - | - | - |
| baseline: flat ground (boxes) | 10 | - | - | - | - | 50 % | 44 % | 0 % | - | - | - |

### Latency per 640x480 frame

Measured on Apple M1 (mps for the models, numpy + OpenCV on the CPU for motion stereo and the wrappers), median over the run, first call excluded. CPU column: the same model forced onto the CPU.

| Depth source | Licence | ms/frame (GPU) | ms/frame (CPU) | fps alone (GPU) |
|---|---|---|---|---|
| da2-metric-indoor-s | Apache-2.0 | 105 | 275 | 9.5 |
| da2-metric-outdoor-b | CC-BY-NC-4.0 | 326 | 566 | 3.1 |
| da2-metric-outdoor-s | Apache-2.0 | 106 | 347 | 9.4 |
| da2-relative-s | Apache-2.0 | 102 | 263 | 9.8 |
| motion-stereo | ours (MIT, this repo) | 102 | 124 | 9.8 |

### How to read the columns

- **AbsRel**: mean |pred - true| / true over pixels with ground truth (sky and missing scans excluded). **RMSE** in metres. **delta<1.25**: share of pixels within 25 % of the truth.
- **Corridor AbsRel**: the same over the walking corridor only (0.5 m either side, up to 5 m ahead).
- **Priority right**: share of frames where the band of the nearest thing rising 0.25 to 2 m out of the floor in the corridor (P1 under 2 m, P2 2-5 m, P3 5-10 m, clear) matches the band the ground truth gives. Both use the true floor plane, so only the depth differs. **P1 recall**: true P1 frames called P1. **False P1**: P1 calls when the truth was P2 or farther.
- **Pitch error**: anchored fits only, against the floor plane of the ground truth. **Floor fit ok**: frames where the RANSAC floor was accepted (otherwise the raw output is scored).
- Box baselines use ground-truth boxes of every discrete object on the synthetic walks (a perfect detector) and YOLO11n COCO boxes on DIODE.

