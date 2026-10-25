# kite.fly.behind: train the follow the app flies, and its fly brain, together

The yaw-only study in [`kite/fly`](../DESIGN.md) simulated `kite.follow.FollowController` 1.0 m behind the head. The
app (`run/app.py`) flies `kite.follow_behind.FollowMission` / `FollowBehind`: 1.8 m behind, 1.9 m above the floor, the
person found by YOLO pose (box + 17 keypoints), range from head elevation, shoulders and box size, and the fruit fly
brain smooth2 steering yaw. This package simulates **that** loop and tunes the follow and the brain together.

Nothing that exists changes. Old files are not edited: `kite/follow_behind.py`, `kite/flybrain/`, `kite/fly/` (sim,
train, data), `run/`, the smooth2 and retrained (`kite/fly/data/fly_yaw.json`) brains all stay as they are. Everything
here is new files. New models are new files under `kite/fly/data/behind/`.

## 1. Goals of the new models

1. Hold **1.8 m** behind the person (healthy band 1.5-2.1 m), never closer than 1.2 m in normal walking.
2. **Smooth** sticks on every axis (forward, yaw, up/down, lateral).
3. **Speed only when needed:** the forward speed may rise to **0.6 m/s** (stick 63 at the nominal 0.96 m/s per 100)
   **if and only if the person walks faster**. When the person is slow or stopped the drone matches them and never
   goes faster than needed to hold the distance. The flown follow is capped at stick 35 (about 0.34 m/s).
4. **No overrun:** when the person stops, slows, steps back or turns around, the drone stops short or backs off; it
   never passes over or beside them.
5. **No ceiling rush:** altitude stays near the follow height; back-off climbs are small.
6. Keep the person in view and the lock on them (bearing error, losses).

## 2. The controller: `kite/follow_tuned.py` (new)

```python
@dataclass
class TunedCfg(BehindCfg):          # every BehindCfg field keeps its meaning and default
    fwd_cap_fast: int = 35            # forward stick cap when the person walks fast (hard limit 63 ~ 0.6 m/s)
    fwd_cap_v0_mps: float = 0.35      # person speed at which the cap starts to rise above max_rc_forward
    fwd_cap_per_mps: float = 0.0      # cap rise (stick) per m/s of person speed above fwd_cap_v0_mps
    fwd_ff_gain: float = 0.0          # feed-forward: + fwd_ff_gain * person speed away (m/s) * RC_PER_MPS
    walker_speed_tau_s: float = 0.5   # low-pass of the person-speed estimate
    fwd_slew: float = 0.0             # max forward stick change per second, 0 = off
    backoff_climb_m: float = 0.25     # how far above altitude_m the back-off climbs may go (flown: 0.25)
    model_name: str = ""              # set by load_model
RC_PER_MPS = 104.0                  # stick per m/s at the nominal Tello forward gain (0.96 m/s per 100)
FWD_CAP_LIMIT = 63                  # never above this, whatever a model file says
```

- `TunedFollowBehind(FollowBehind)` overrides `step()` (and `_climb_rc`) with the same logic as `FollowBehind.step`,
  plus the knobs above. **With TunedCfg defaults it must send bit-identical rc to `FollowBehind` on any input**
  (a test feeds both the same scripted perception/telemetry for 60 s and compares every rc).
- Person speed estimate (m/s, + = walking away from the drone): `v_away = range_rate + own_speed`, where
  `range_rate` is `FollowBehind._range_rate` and `own_speed` is the mean commanded forward stick 0.3-1.3 s ago divided
  by `RC_PER_MPS` (the same terms the flown approach feed-forward uses). Low-pass it with `walker_speed_tau_s`; None
  when the rate is None (then the cap is `max_rc_forward` and there is no feed-forward).
- Forward when too far (`e > 0`, and the approach gates of the flown code still hold: `r_lo > min_range_m`, box not
  too big): `fwd = e * dist_gain + fwd_ff_gain * max(0, v_away) * RC_PER_MPS`, clamped to
  `cap = min(fwd_cap_fast, FWD_CAP_LIMIT, max_rc_forward + fwd_cap_per_mps * max(0, v_away - fwd_cap_v0_mps))`,
  never below `max_rc_forward`'s own clamp when `fwd_cap_per_mps == 0`. So the extra speed exists only while the
  person is measured walking away faster than `fwd_cap_v0_mps`.
- When at or inside the distance (`e <= 0`): exactly the flown law (no feed-forward pushes it closer).
- Approaching feed-forward, too-close back-off, lost-while-close, lost handling, altitude law: as flown, with the
  back-off climb ceiling `altitude_m + backoff_climb_m` instead of `+ 0.25`.
- `fwd_slew` limits the change of the forward stick per second (applied last, except that a too-close back-off is
  never slowed: safety first).
- `TunedFollowMission(FollowMission)` only overrides `_follow` to build `TunedFollowBehind`.
- `load_model(path) -> TunedCfg` reads a model file (section 6) and returns a TunedCfg with its `follow` overrides,
  `fly_params_path = path` when the model has a brain (`x`), and `steering` "fly" or "pid". `save_model(...)` writes one.

## 3. Simulator: `kite/fly/behind/` (new)

The loop is `run/pilot.py`'s FOLLOW loop, with the **real** `kite.follow_behind` classes:

```
every 0.05 s (+ U(0, 0.01) jitter):  newest frame (if any) -> mission.sense(frame, now)  [Perception: FakePose -> SimpleTracker
  -> TargetLock -> PersonEstimator] -> governor.check(now, last_frame_t) -> mission.step() -> SafetyGovernor.rc -> SimTello
physics: kite.fly.sim.drone.DroneModel in 0.01 s sub-steps (reused, not edited)
```

- **Arms** (`params.py`): `FLOWN` = `FollowMission` + `BehindCfg()` defaults + smooth2 (the app today, the baseline and
  normalizer); the tuned arms run `TunedFollowMission` with a TunedCfg built from x.
- **Start:** directly in FOLLOW (`mission._follow("sim start")` after setting state like `begin()` would after the
  climb), altitude = `altitude_m + U(-0.12, +0.08)`; the person 1.8 + U(-0.3, +0.6) m ahead, bearing ±10°, facing away
  ±20°, standing for the first 1.0-2.5 s. The lock needs its 3 confirming hits, as live.
- **Person:** height H ~ U(1.55, 1.90) m; the operator's `--walker-height` = H + U(-0.02, +0.02);
  `altitude_m = person_height_m + 0.2` (the app's default); shoulders 0.36-0.46 m.
- **Walk** (`walker.py`, new; may reuse kite.fly.sim.walker pieces by import): a unicycle with heading, per 0.01 s
  tick, drawn at reset from its own stream (common random numbers: the same walk for every arm). Segments:

  | kind | p | details |
  |---|---|---|
  | straight | 0.45 | 3-12 s, gentle curvature ±10-25°, new speed with p 0.6 |
  | corner | 0.20 | 60-100° at 45-90 °/s, then a straight |
  | stop | 0.17 | 1-8 s (a crossing) |
  | speed change | 0.10 | ramps to a new speed within the profile's range over 1-3 s |
  | turn around | 0.03 | 150-180°, then walks toward and past the drone's side for 2-5 s |
  | step back | per episode p 0.15 | one 0.3-0.8 m step toward the drone |

  Acceleration 0.8 m/s². Speeds per profile (below); no cap from the drone's limits: if the person outwalks the drone
  it falls behind, as it would live.
- **Fake YOLO pose** (`pose.py`): per frame, project a 3D body model (17 COCO keypoints: nose, eyes, ears at the head;
  shoulders at 0.82 H, elbows 0.63 H, wrists 0.48 H, hips 0.53 H, knees 0.29 H, ankles 0.04 H; widths from the
  shoulder draw; the face points 0.09 m forward of the head centre) with the drone's true pose (x, y, z, heading,
  pitch) and the true camera (fx = fy ~ U(1040, 1110), cx 480, cy 360, 960 x 720; mount pitch U(-1°, +1°)).
  - `PersonDet("person", conf, bbox, keypoints(17, 3) float32)` exactly as `PoseDetector` returns (most confident first).
  - bbox = the extent of the visible body (head top to the lowest visible point, shoulder/arm width) clipped to the
    frame, each edge + N(0, σ 2-6 px).
  - keypoint conf: in frame and facing the camera side: U(0.6, 0.95); face points seen from behind: U(0.05, 0.35);
    ears from behind U(0.4, 0.85); out of frame: U(0, 0.25) at an extrapolated position. Keypoint noise σ 2-5 px.
    Left/right shoulder label swap with p 0.01 per frame. Seen from behind the person's RIGHT shoulder is on the image
    right (follow_behind's facing() convention); check the sign against `PersonEstimator.facing`.
  - detected with p = base (U(0.92, 1.0)) × min(1, visible_body_height_px / 80)²; dropout bursts 0-0.15 Hz of
    0.2-1.0 s; false boxes 0-1 %; an optional **second person** (p 0.3 per episode) crossing 1-4 m from the drone,
    sometimes between the drone and the person, to exercise the lock.
- **Video timing:** frames captured at 30 fps; each arrives on the laptop `video_latency` later (U(0.15, 0.35) s
  train, 0.25 demo) and is handed to perception on the first control tick after it arrives (newest only; YOLO time
  U(0.02, 0.06) s is added to the latency). `Frame(image=np.zeros((720, 960, 3), np.uint8)` shared buffer, t = arrival,
  seq). The fake detector is the `detect` callable passed to `Perception`; it returns the detections of the frame it is
  given (keyed by seq), ignoring the pixels.
- **Telemetry** (`Telemetry` every tick): `height_m` = z + bias U(±0.05) + N(0, 0.01-0.03), rounded to 0.01;
  `tof_m` = distance to what is below (floor, or the person's head when within 0.3 m horizontally) + N(0, 0.01);
  `pitch_deg` = true pitch (+ nose up) + N(0, 0.5), rounded to whole degrees; `yaw_deg` whole degrees;
  `battery_pct` 80. `BehindCfg.pitch_sign = 1` already turns it into nose-up positive.
- **SimTello** implements `kite.drone.base.Drone`: rc as the existing sim's; `move(dir, cm)` / `rotate(deg)` run as
  timed motions (0.5 m/s, 60 °/s) with `busy()` True until done, so REACQUIRE works if the person is lost 20 s.
- **Governor:** the real `SafetyGovernor(SimTello, SafetyCfg(max_rc=65, max_altitude_m=2.3))` (65 so 0.6 m/s is
  possible; the flown follow never sends more than 50 on any axis, so FLOWN is unaffected).
- **Drone dynamics:** `DroneParams` drawn per profile (ranges below), indoor-calm hover drift, small wind.
- **Collision** ends the episode (charged as the rest of the episode lost): horizontal camera-to-head range < 0.3 m
  while above the head, or 3D distance camera-to-head < 0.35 m.

### Profiles (`profiles.py`)

| parameter | train | demo | fast | slow | stress |
|---|---|---|---|---|---|
| person speed (m/s) | 0.2-0.75 | 0.35, one 0.6 stretch | 0.5-0.8 | 0.1-0.35 | train |
| fwd_gain at stick 100 | 0.80-1.10 | 0.96 | train | train | 0.75-1.2 |
| tau_fwd / fwd dead (s) | 0.31-0.59 / 0.33-0.61 | 0.45 / 0.47 | train | train | train |
| yaw gain (°/s per 100) / tau / dead | 45-65 / 0.02-0.10 / 0.13-0.24 | 55 / 0.02 / 0.18 | train | train | train |
| vz gain / tau / dead | 0.25-0.40 / 0.02-0.10 / 0.29-0.53 | 0.32 / 0.02 / 0.41 | train | train | train |
| pitch_scale | 0.5-1.0 | 0.75 | train | train | train |
| video latency (s) | 0.15-0.35 | 0.25 | train | train | 0.35-0.5 |
| detection base p / dropout bursts (Hz) | 0.92-1.0 / 0-0.15 | 0.97 / 0.05 | train | train | 0.85-0.95 / 0.2 |
| drift σ / wind σ (m/s) | 0-0.04 / 0-0.05 | 0.02 / 0.02 | train | train | 0.05 / 0.1 |
| second person p | 0.3 | 0 | 0.3 | 0.3 | 0.5 |

Episodes are 60 s. Common random numbers: every random quantity comes from
`numpy.random.default_rng([seed, profile_id, stream_id])` drawn at reset; only the drone's position depends on the
arm. Seeds: training `BEHIND_TRAIN_BASE = 7_000_000 + 100_000 * cma_seed + gen * K + k`, selection 6000-6095,
test 8000-8199 (disjoint from each other and from kite.fly's).

## 4. Metrics (`metrics.py`), from simulator truth after the first 3 s

`run_behind_episode(...)` returns `kite.fly.interfaces.EpisodeResult` with `metrics` holding at least:

| key | meaning |
|---|---|
| `collided` | 1 if the episode ended in a collision |
| `rms_range_err_m` | RMS of (horizontal range − distance_m) |
| `in_band` | fraction of time with \|range − distance_m\| ≤ 0.3 |
| `too_close_s` / `very_close_s` | seconds with range < 1.2 m / < 0.8 m |
| `min_range_m` | smallest horizontal range |
| `overshoot_m` | over every person stop / slow-down / step back / turn around: max(0, 1.5 − smallest range in it and the 2 s after) |
| `far_s` | seconds with range > 2.6 m (fell behind) |
| `rms_bearing_deg`, `in_view`, `losses`, `time_lost_s` | as kite.fly's (lock on the true person; a lock on the second person counts as lost) |
| `wrong_lock_s` | seconds locked on the second person |
| `stick_fb`, `stick_yaw`, `stick_ud`, `stick_lr` | mean \|Δ sent stick\| per control tick while following |
| `accel_rms` | RMS of the drone's true horizontal acceleration (m/s²) |
| `alt_max_m`, `ceiling_s`, `rms_alt_err_m` | highest true altitude; seconds above altitude_m + 0.25; RMS altitude error |
| `speed_track_err` | RMS of (drone ground speed toward the person's heading − person speed) while the person walks |
| `excess_speed` | mean of max(0, drone speed − person speed − 0.15) while range < distance_m + 0.3 (going faster than needed) |
| `fast_unneeded_s` | seconds with drone ground speed > 0.40 m/s while the person walks < 0.30 m/s and range < distance_m + 0.3 |
| `max_speed_mps` | highest drone ground speed |
| `governor_events` | count of governor interventions (ceiling, floor, stale) |
| `start_failed` | 1 if the lock never confirmed in the first 5 s (excluded identically for every arm) |

## 5. Fitness (`fitness.py`), minimized

Per episode cost (each term dimensionless, about 1 for a mediocre follow):

```
J = 1.0 * (rms_range_err_m / 0.3)**2 capped 9       distance
  + 0.5 * (rms_bearing_deg / 15)**2 capped 9         keep them centred
  + 1.0 * time_lost_s / 10 + 0.5 * losses            keep the lock
  + 0.6 * (stick_fb / s_fb + stick_yaw / s_yaw + stick_ud / s_ud + stick_lr / s_lr) / 4    smooth sticks
        (s_* = FLOWN's mean stick change per axis on the demo profile, measured once and stored in FitnessCfg)
  + 0.3 * accel_rms / 0.3
  + 0.8 * speed_track_err / 0.15
  + 1.0 * excess_speed / 0.1 + 0.2 * fast_unneeded_s           speed only when needed
penalties:
  + 4.0 * collided + 1.0 * very_close_s + 0.3 * too_close_s + 3.0 * overshoot_m + 0.5 * ceiling_s + 0.5 * wrong_lock_s
```

Episode score = `J_arm / max(J_FLOWN(same seed), 0.5)`, clipped to [0, 6]; a collision scores 6; a failed start is
excluded. CMA minimizes the mean score over the K seeds. FLOWN scores 1.0 by construction. Weights and scales live in
one dataclass `FitnessCfg` so they can be changed and are recorded in every run.

As built (integration, 2026-10-24):
- **Distance tail.** Past the cap the distance term goes on as the tangent of the square, `9 + 6 (r - 3)` with
  `r = rms_range_err_m / 0.3` (`FitnessCfg.range_tail = "linear"`; `"cap"` is the flat cap above). In the simulator
  FLOWN trails 5-10 m behind on most seeds (stick 35 is ~0.34 m/s; demo's mean RMS error is 7.6 m), so a flat cap is a
  plateau there: dist_gain 15 + max_rc_forward 20 (9.6 m RMS) scored 1.001 against FLOWN with it, 1.32 with the tail.
- **Stick scales** measured on FLOWN, demo seeds 0-39, 60 s: `s_fb 4.76, s_yaw 0.170, s_ud 0.156, s_lr 0.090`.
- The score is computed as `1 + (J_arm - J_FLOWN) / max(J_FLOWN, 0.5)`, the same ratio whenever J_FLOWN >= 0.5, so
  FLOWN is exactly 1.0 on an easy seed too.

## 6. Model files (`kite/fly/data/behind/<name>.json`, new)

```json
{"name": "v3-joint", "arm": "FLY-YAW", "x": [34 brain numbers], "brain": "pursuit_core1.npz",
 "follow": {"dist_gain": 52.1, "...": "..."}, "steering": "fly",
 "trained": {"behind_arm": "V3-JOINT", "cma_seed": 2, "gen": 140, "fitness": 0.71, "time": "..."},
 "signoff": {...}}
```

`arm` / `x` / `brain` make the file a valid FlySteer params file (`FlySteer(params_path=file)`), so the app's existing
brain loader reads the brain straight from it. A follow-only model keeps smooth2's `x`. A law model has no `x` and
`"steering": "pid"`.

## 7. Arms and the parameter space (`params.py`)

Follow knobs (every tuned arm), bounds and the flown default (= TunedCfg default):

| name | lo | hi | scale | flown |
|---|---|---|---|---|
| dist_gain | 15 | 90 | log | 40 |
| dist_deadband_m | 0 | 0.3 | lin | 0.1 |
| max_rc_forward | 20 | 45 | lin | 35 |
| fwd_cap_fast | 35 | 63 | lin | 35 |
| fwd_cap_v0_mps | 0.2 | 0.5 | lin | 0.35 |
| fwd_cap_per_mps | 0 | 150 | lin | 0 |
| fwd_ff_gain | 0 | 1.2 | lin | 0 |
| walker_speed_tau_s | 0.2 | 1.5 | log | 0.5 |
| fwd_slew | 60 | 2000 | log | 2000 (= off) |
| max_rc_backoff | 30 | 63 | lin | 50 |
| approach_rate_mps | 0.05 | 0.5 | lin | 0.2 |
| min_range_m | 0.55 | 1.2 | lin | 0.55 |
| alt_gain | 20 | 100 | log | 60 |
| alt_deadband_m | 0.02 | 0.15 | lin | 0.05 |
| max_rc_up | 10 | 40 | lin | 30 |
| backoff_climb_m | 0 | 0.25 | lin | 0.25 |

Fly conditioning (fly arms): fly_deadband 0-8 (4), fly_hysteresis 0-6 (3), fly_slew 60-600 log (300),
fly_smoothing_ms 0-250 (0), fly_max_rc_yaw 20-60 (40; `max_rc_yaw` is set to the same value).
Law yaw (law arm): yaw_gain 0.5-3 log (1.2), yaw_deadband_deg 0-6 (4), max_rc_yaw 20-60 (40).
Integer knobs are rounded when the cfg is built.

| Arm | Knobs | Brain | Init |
|---|---|---|---|
| FLOWN | none (BehindCfg defaults) | smooth2 | — |
| V3-FOLLOW | follow + fly conditioning (21) | smooth2, frozen | flown values |
| V3-JOINT | follow + fly conditioning + the 34 FLY-YAW brain parameters (55) | trained | flown values + smooth2's x |
| V3-LAW | follow + law yaw (19) | none | flown values |

Brain dimensions get per-dimension CMA step multipliers like kite.fly's FLY-YAW (`dec_w_` 0.08, `dec_b_` 0.2 of a
brain base of 0.03 / sigma0), the follow dimensions a larger one, so one sigma0 fits both.

## 8. Training (`train.py`) and Modal (`modal_app.py`, a new Modal app `kite-behind`)

- CMA-ES (pycma, bounded [0, 1]^n) with common random numbers per generation; reuse `kite.fly.train.cma_train`'s
  backends, prices, metering and file helpers by import (never edit it). Every 5 generations score the mean on the
  selection seeds; keep the best as `best.json` (a model file, section 6). Resume from `state.pkl`.
- Plan: V3-FOLLOW pop 24, K 32, 60 gens; V3-JOINT pop 40, K 32, 100 gens, init from flown + smooth2; V3-LAW pop 20,
  K 32, 60 gens; 4 CMA seeds each for the fly arms, 2 for the law. Runs write to `results/behind/runs/<arm>_s<seed>_<tag>/`
  (Volume `kite-behind-runs` on Modal).
- The Modal image is built like kite.fly's (import probe refusing cv2 / ultralytics / torch / av / djitellopy) plus
  `kite/flybrain` (its .py, `data/brains/**`, `configs/*.yaml`) and `kite/follow_tuned.py`. A `probe` entrypoint times
  real episodes on Modal and checks how many containers the workspace runs at once.

## 9. Sign-off (`signoff.py`)

On the 200 test seeds × (train, demo, fast, slow, stress), every arm against FLOWN, paired per seed with 95 %
bootstrap CIs. A model is **recommended** only if, on train and demo:
1. mean score ≥ 10 % below FLOWN (CI below 0);
2. collisions ≤ FLOWN's, seed by seed (+1 margin), on every profile;
3. not worse (upper CI ≤ 10 % of FLOWN's mean or a small absolute margin) on too_close_s, very_close_s, overshoot_m,
   ceiling_s, losses, wrong_lock_s;
4. smoother: mean of the four stick changes ≤ FLOWN's;
5. on `slow`: excess_speed and fast_unneeded_s not above FLOWN's margin; on `fast`: rms_range_err_m and far_s below
   FLOWN's (it keeps up).
The report is written to `results/behind/report/` and copied to `docs/follow-v3-results.md`.

Simulation only. A recommended model stays opt-in until a real flight next to smooth2.

## 10. As built after the review (2026-10-25)

Three reviews (fidelity, objective, packaging) found the speed gate leaking, the objective blind to safety where FLOWN
trails, and three packaging holes. What changed, by section:

**Controller (2).** The person-speed estimate takes the drone's own speed over the same stretch of the past the range
rate describes (the video lag + the frame's YOLO age + the median's delay), as the drone really moved then: the
forward stick the governor passed (clamped to its `max_rc`, 0 during a safety hover) through a nominal Tello response
(`own_dead_s` 0.47, `own_tau_s` 0.45), weighted as a least-squares slope weights its window. DESIGN's "mean stick
0.3-1.3 s ago" led the range by ~1 s (the dead time), read slow walkers as fast and fed itself. The raised cap opens
only through a **gate**: `v_away > fwd_cap_v0_mps` with the range rate >= 0 (the drone at its present speed is not
gaining on them) for `fwd_cap_dwell_s` (1 s) on end; it shuts as soon as `v_away <= fwd_cap_v0_mps`. `forward_cap`
clamps `max_rc_forward` to 63 too (and `check()` refuses more). `fwd_slew` limits only speed-ups. Two new knobs, off
by default: `approach_model_w` (the approach feed-forward's closing speed from the aligned estimate instead of the
misaligned stick mean) and `approach_tau_s` (that closing speed low-passed). The defaults are still bit-identical to
`FollowBehind`. Model files carry `"governor": {"needs_max_rc", "trained_max_rc"}`: 0.6 m/s is stick 63, and
`kite.config.SafetyCfg.max_rc` is 50 (~0.48 m/s), so a model that raises its cap needs `SafetyCfg(max_rc >= 63)`.

**Simulator (3).** Telemetry pitch error: an OU process, train sd U(0, 0.5) deg and correlation U(0.5, 3) s, demo
0.25 / 1.5, stress DESIGN's white 0.5 (from none to DESIGN's level, so a model cannot lean on either; whether a real
Tello's integer attitude has per-packet noise is not measured). The ToF reads the highest part of the silhouette in
its 25 deg cone (shoulders, legs), else the floor. Every follow start sets the shared FlySteer's clamp to the
episode's `fly_max_rc_yaw` (FlyYaw.load's cache key leaves it out).

**Still open.** In this simulator FLOWN falls into a ~1 Hz fore-aft cycle even behind slow walkers: the camera tilts
with the drone, the app corrects the 0.3 s old frame with the current telemetry pitch, 1 deg is ~16 % of range at
0.2 m above the head, and the approach feed-forward backs off at -50 on the resulting range-rate swings (+-3 m/s
while the true range moves +-0.3 m). Switching the feed-forward off removes it; the new `approach_tau_s` damps it.
A flight log of telemetry pitch, the range estimate and the forward stick behind a standing or slowly walking person
would show whether the real app does this.

**Metrics (4).** `excess_speed` and `fast_unneeded_s` use the speed TOWARD the person (backing away and side steps
are never "too fast"). `fast_unneeded_s` counts at any range, while the person's speed over the last 1.5 s stayed
below 0.30 m/s, with the drone above max(0.40, that + 0.15). `speed_track_err` is time-weighted over the window.

**Fitness (5).** Only the distance term is scored relative to FLOWN on the same seed; every other term is an absolute
price, `score = 1 + (D_arm - D_FLOWN) / max(D_FLOWN, 0.5) + (Q_arm - Q_FLOWN) / 3`. Dividing all of J by FLOWN's
(~100 where it trails) had made a second under 0.8 m worth 0.01. A collision FLOWN did not have makes the fitness
`max(mean, 1) + 1` per such seed (a hard constraint). An altitude term (`0.3 x rms_alt_err_m / 0.05`); unneeded fast
flight costs 0.6 J per second (0.2 score). A failed start of the candidate alone is charged 6. Stick scales
re-measured: `s_fb 4.24, s_yaw 0.173, s_ud 0.154, s_lr 0.089`.

**Parameters (7).** `max_rc_forward` 20-35 (extra speed only through the gate), `fwd_cap_v0_mps` 0.3-0.5,
`approach_rate_mps` 0.05-0.3 (training cannot switch the back-away off), `alt_deadband_m` 0.02-0.08, and
`approach_model_w` 0-1, `approach_tau_s` 0-1 (flown 0), and the missed-frame `missing_s` 0.5-1.5 (1.5) and `creep_rc` 0-25 (10): V3-FOLLOW 25, V3-JOINT 59, V3-LAW 23 dimensions.

**Training (8).** Training and selection seeds alternate `train` and `slow` (`--profile`, default `train,slow`). A
run refuses to resume when the code changed (`kite.fly.behind.digest.run_digest`; `--allow-code-change` overrides),
and FLOWN's cache is keyed by the episode digest. The Modal backend meters a run by its own chunks
(`max(busy, containers x slowest chunk)`), not by the shared queue. train_all's re-spawn takes over the dead driver's
lease. `probe --determinism N` checks the containers compute bit-identical episodes. The CLI's default chunk is
modal_app's (`--chunk auto` for the cheapest projection).

**Sign-off (9).** The episode cache key is `digest.episode_digest` (now with `kite/config.py`, `kite/camera.py`,
`kite/sources/*`); a model trained under another governor `max_rc` is refused, and the report says what each needs.
