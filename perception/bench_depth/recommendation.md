## Recommendation

**Run Depth Anything V2 Metric Indoor Small (Apache-2.0), anchored to the floor, fused with motion stereo.** Not one
model, a combination, and the numbers below are why.

- **No model's own metric scale can be trusted.** Raw, every Depth Anything V2 variant is far off on the chest-camera
  walks (AbsRel 0.50 to 1.23). Anchoring the scale to the floor, which sits a known 1.2 m below a chest camera,
  cuts the error by 3 to 7 times: Indoor Small goes from 0.496 to 0.145 AbsRel, and its corridor error goes from
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

**Why the fusion matters:** real laser-scanned stills (DIODE) are the hardest case for any single depth source, and
a still frame gives motion stereo nothing to work with. That is why `perception/near` fuses this depth with
per-object cues (known object size, ground contact) and looming time to contact, which together reach 92.7% on
hand-labelled real frames.

**Also supported:** Apple Depth Pro, UniDepth V2, Metric3D V2 and Video Depth Anything all have adapters in
`depth_zoo.py`; run them with `--models` on a machine with room for them.
