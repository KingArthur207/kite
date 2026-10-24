# kite.fly: fly-brain yaw steering for street following

A 1,446-neuron subgraph of the male fruit-fly connectome (MaleCNS v1.0) can steer the Tello's **yaw** while
`kite.follow` follows a walker. The wiring is fixed. Only the small adapters around it are tuned with CMA-ES
(an evolution strategy) in a simulator of kite's real follow pipeline:
- the encoder: walker bearing and size → spike rates into the LC10a/LC9/LC11 visual inputs;
- the readout: steering-neuron rates → yaw stick.

Everything else stays `kite.follow.FollowController`'s:
- forward with walker-speed feed-forward;
- lateral and altitude;
- clamps and slew;
- coasting, and the lost-walker hover.

The fly plugs in through `FollowController(yaw_source=...)` (`kite.follow.YawSource`), and only while
following. It ships only if it beats kite's own yaw law (section 7). Shared constants and types are in
`interfaces.py`.

## 1. Geometry (the kite spec)

- **Position:** `FollowCfg(behind_m=1.0, above_m=0.30)`: 1.0 m horizontally behind the head top, camera
  0.30 m above it.
  - `FollowController` holds the altitude the drone had when follow started (`reset(tel.height_m)`);
    the pilot sets it.
  - It turns the head top's image row into a range assuming the camera is exactly `above_m` above the head.
  - An altitude error is therefore a range bias. The simulator models that: start altitude = head +
    0.30 + U(-0.10, +0.15) m.
- **Ceiling:** outdoors there is none, so nothing in the follow path may cap altitude below 3 m.
- **What the camera sees:** it cannot tilt. With fy ≈ 920 px, the head top at 1.0 m / 0.30 m sits 16.7°
  below the axis, row ≈ 636 of 720.
  - The person box is cut by the bottom edge (the walker tracker requires it to touch it).
  - Braking pitches the nose up and can push the head out of the bottom of the frame.

## 2. Simulator (`kite/fly/sim/`): kite's real follow loop in a box-level world

- **The real code runs in the loop:**
  - the fake detector → `kite.follow.WalkerTracker` → `kite.follow.FollowController` (`behind_m 1.0`,
    `above_m 0.30`, the main defaults otherwise) at kite.record's **10 Hz**;
  - its rc goes through the `SafetyGovernor` rc clamp (`max_rc` from the profile) → the drone model at
    **20 Hz** (DT 0.05 s), with the rc held between control steps;
  - the yaw arm is the `yaw_source`.
- **Common random numbers:** every random quantity comes from its own stream
  `numpy.random.default_rng([seed, profile_id, stream_id])`, drawn at reset. The same (seed, profile) is
  the same world for every arm. The only arm-dependent part is where the drone is.
- **Truth vs. belief:** metrics and reward use truth. The controller sees only detector boxes (noisy,
  late), telemetry `height_m` (truth + bias + noise) and `pitch_deg` (truth + noise).
- **Speed:** pure Python floats per tick. Target < 0.2 ms per tick without the brain.

### 2.1 Drone (from the team's Tello lag test, 2026-09-26, stick 30)
- **Per axis** (forward, lateral, vertical, yaw): sent stick → a command dead time → a first-order lag →
  velocity.
- **Disturbances:** OU hover drift (tau 2 s) plus OU street wind (tau 3 s).
- **Pitch:** `pitch_scale · atan(a_fwd / g)`, nose-down when accelerating forward, capped at 25°.

### 2.2 Walker (street)
- **Body:** head top 1.50-1.95 m; head 0.20-0.26 m; shoulders 0.40-0.50 m wide. The person box spans
  shoulder width, head top to feet, clipped to the frame.
- **Segments:**

  | Kind | Probability | Duration / size | Notes |
  |---|---|---|---|
  | straight | 0.55 | 3-15 s | gentle curvature ±10-30° per segment; new speed 0.5-1.5 m/s with probability 0.6 |
  | corner turn | 0.25 | 60-100° | at 45-120 °/s |
  | stop (crossing) | 0.20 | 1-10 s | |

- **Speed cap:** `min(1.5, 0.9 · fwd_gain · max_rc / 100)`, so every episode can physically be followed.
  Report it: with `SafetyCfg.max_rc = 50` it is about 0.43 m/s.
- **Acceleration:** 1.0 m/s².
- **Start:** the walker stands facing away ± 30°. The drone starts behind at 1.0 + U(-0.2, +0.8) m, bearing
  offset ±15°.
- **Step back:** with probability 0.2 per episode, one 0.3-0.8 m step toward the drone (it stops at 0.8 m).
- **Episode:** 60 s.
- **Follow start:** a 1 s pre-roll, so the tracker confirms the walker (`min_hits`). Then
  `FollowController.reset(tel.height_m)`, as `FollowMode.start` does.

### 2.3 Camera and fake detector
- **Projection:** pinhole with the true fx. Project the head top, the box sides and the feet, with body
  pitch. Clip to 960 × 720.
- **Detectable:** head top inside the frame (`v_top < 716`) and visible height ≥ 6 px.
  Probability `min(1, (h_vis / 20)^2)`.
- **Confidence:** U(0.3, 0.9), a little lower for small visible boxes (the tracker drops < 0.25).
- **Detector stream:**
  - rate 6-25 Hz;
  - noise σ 1-5 px on each box edge;
  - iid dropout 0-15 %;
  - bursts at 0-0.2 Hz, each 0.2-1.5 s;
  - false boxes 0-2 % (random person-sized boxes, some touching the bottom edge).
  - A box carries its frame's arrival time = capture + video latency (0.15-0.45 s). It becomes available
    to the tracker after the detector time (0.03-0.10 s).

### 2.4 Profiles
- **`train`:**

  | Parameter | Range |
  |---|---|
  | fwd_gain_mps | 0.67-1.5 |
  | yaw_gain_dps | 39-72 |
  | vz_gain_mps | 0.23-0.42 |
  | tau_fwd_s | 0.31-0.59 |
  | tau_yaw_s | 0.02-0.10 |
  | tau_z_s | 0.02-0.10 |
  | fwd_dead_s | 0.33-0.61 |
  | yaw_dead_s | 0.13-0.24 |
  | vz_dead_s | 0.29-0.53 |
  | pitch_scale | 0.5-1.0 |
  | drift σ | 0-0.05 m/s |
  | wind σ | 0-0.15 m/s |
  | altimeter bias | ±0.05 m |
  | altimeter noise σ | 0.01-0.04 m |
  | pitch noise σ | 0.5° |
  | fx true | 850-1100 px |
  | camera fx believed | true × U(0.97, 1.03) |
  | max_rc | 60-100 |

- **`demo`:** the nominal values:

  | Parameter | Value |
  |---|---|
  | fwd_gain | 0.96 m/s |
  | yaw_gain | 55 °/s |
  | vz_gain | 0.32 m/s |
  | tau_fwd | 0.45 s |
  | tau_yaw | 0.02 s |
  | tau_z | 0.02 s |
  | fwd_dead | 0.47 s |
  | yaw_dead | 0.18 s |
  | vz_dead | 0.41 s |
  | fx | 920 px |
  | video latency | 0.25 s |
  | detector rate | 15 Hz |
  | wind σ | 0.05 m/s |
  | max_rc | 80 |

- **`stress`:** train, plus video latency 0.6-0.8 s, dropout 30 %, fx at the range edges, wind 0.15 m/s.
- **`fast`:** train, with walker speeds 1.1-1.5 m/s and max_rc 100.

## 3. The fly (`kite/fly/`)

- **`brain.py`:** event-driven LIF on the `.npz` CSR graph, Shiu et al. (Nature 2024) parameters.

  | v_rest | v_reset | v_th | tau_m | tau_syn | t_ref | delay | w_syn | Poisson factor | dt |
  |---|---|---|---|---|---|---|---|---|---|
  | -52 mV | -52 mV | -45 mV | 20 ms | 5 ms | 2.2 ms | 1.8 ms | 0.275 mV | 250 | 0.5 ms |

  - Inputs are Poisson spikes per input neuron at its channel rate.
  - `tick(rates[22], 50 ms)` returns mean rates of the 10 OUTPUT_GROUPS.
  - The weights load once per process; episodes get a cheap per-seed copy.
  - Target ≤ 3 ms per tick.
- **`encoder.py`:** TargetFeatures → 22 rates.
  - Per side, 8 LC10a bins with Gaussian tuning. The right side covers -overlap..+40°, the left mirrors it,
    bin 0 is frontal. Use the npz `meta["azimuth_bins"]`.
  - `rate = r_max · gain_bin · exp(-0.5((θ−c)/(σ s^a))²) · s^b · (1 + k_v |dθ|)`.
  - LC9/LC11 ≤ 10 Hz. Arousal mode "gain" (a multiplier on LC10a); P1 tonic at 0.
- **`decoder.py`:** low-pass the rates (tau ≤ 150 ms) → normalize by the calibration rates →
  `yaw = 60 · G · tanh(Σ_t w_t (n_tR − n_tL) + b)` → soft deadzone.
- **`params.py`:** 34 names, CMA in x ∈ [0,1]^34 (linear or log mapping).
  - Encoder: `enc_r_max_hz`, `enc_width_deg`, `enc_overlap_deg`, `enc_size_width_exp`, `enc_size_amp_exp`,
    `enc_vel_gain`, 16 × `enc_bin_gain_{L,R}{0..7}`, `enc_arousal`, `enc_lc9_hz`, `enc_lc11_hz`.
  - Readout: 5 × `dec_w_yaw_{DN}`, `dec_b_yaw`, `dec_g_yaw`, `dec_tau_yaw_ms`, `dec_deadzone`.

  | Name | lo | hi | scale | default |
  |---|---|---|---|---|
  | enc_r_max_hz | 20 | 400 | log | 160 |
  | enc_width_deg | 3 | 40 | log | 8 |
  | enc_overlap_deg | 0 | 20 | lin | 6 |
  | enc_size_width_exp | 0 | 2 | lin | 1 |
  | enc_size_amp_exp | -1 | 2 | lin | 0.5 |
  | enc_vel_gain | 0 | 3 | lin | 0.5 |
  | enc_bin_gain_* | 0.5 | 2 | log | 1 |
  | enc_arousal | 0.5 | 2 | log | 1 |
  | enc_lc9_hz | 0.1 | 10 | log | 2 |
  | enc_lc11_hz | 0.1 | 10 | log | 2 |
  | dec_w_yaw_* | -15 | 15 | lin | 0 |
  | dec_b_yaw | -5 | 5 | lin | 0 |
  | dec_g_yaw | 0 | 1 | lin | 0.8 |
  | dec_tau_yaw_ms | 10 | 150 | log | 80 |
  | dec_deadzone | 0 | 10 | lin | 0.5 |

  - The names and bounds match `data/reference/` (the previously trained values plus their readout
    normalization `norm`), so those load as the `FLY-YAW-PREV` arm and seed the fine-tune.
- **`calibrate.py` (FLY-YAW-HAND):**
  1. Show a spot at ±30, ±18, ±10, ±4, 0° and sizes 0.6 / 1.0 / 1.5.
  2. Record the DN rates (1 s warmup, 200 ms settle, 1 s measure, 3 repeats).
  3. Normalize each channel by its mean.
  4. Ridge-fit the yaw weights to a tanh argument of 0.04 per degree. Write `data/init/fly_yaw_hand.json`.
- **`steer.py` `FlySteer` (implements `kite.follow.YawSource`):**
  1. **Measure:** each new frame (a new `frame_t`) adds (bearing, range) to a latency-compensated constant-
     velocity filter. The capture time is `frame_t - video_lag_s`, and the filter predicts to each brain
     tick's time.
  2. **Tick the brain** at DT 0.05 s steps up to `t`: at most 3 ticks per call, resynchronizing after a
     long gap instead of catching up.
  3. **Encode** `s = behind_m / range` (or 1 when `size_blind`).
  4. **Decode, then condition the stick:** deadband 4, hysteresis 3, slew 300/s.
  5. **Fail safe:** any exception → `fallback` (the law's own value), for the rest of the episode.
     No measurement for > 0.5 s → no target input (the DNs relax).
  6. `FlySteer.from_file(path)` loads a params JSON.

## 4. Reward and fitness (follow only)

Per control step (dt 0.1 s):
```
r = -dt · (w_r e_z + w_x e_x² + w_j j + w_l l + w_c c) + events
```

| Term | Meaning |
|---|---|
| e_z | `min(3, max(0, |Z − 1.0| − 0.15) / 0.15)`, plus a far-side quadratic over 1 m |
| e_x | true bearing (rad) |
| j | `|Δ sent yaw| / 100` |
| l | no true detection for > 1 s, or the controller is in state lost |
| c | true range < `FollowCfg.min_range_m` |

- **Weights:** w_r 2.0, w_x 7.6, w_j 6.6, w_l 5.0, w_c 10.
- **Events:** walker lost -10, too close -20, collision (range < 0.3 m) -200 (the episode ends).
- **Fitness:** each episode return ÷ max(|KITE-LAW return on the same seed|, 20), clipped to [-4, 2],
  averaged. KITE-LAW scores about -1; higher is better.

## 5. Training (`kite/fly/train/`)

- **Algorithm:** pycma, bounded on [0,1]^n, with common random numbers per generation (every candidate
  sees the same K training seeds). Every 5 generations, score the distribution mean on SELECTION_SEEDS
  and keep the best as `best.json`.
- **Fine-tune plan:**

  | Arm | Init | sigma0 | Population | K follow episodes | Generations | CMA seeds |
  |---|---|---|---|---|---|---|
  | FLY-YAW | FLY-YAW-PREV (or HAND, whichever G2 favours) | 0.03 | 24 | 24 | 60 | 3 |
  | KITE-LAW-CMA | KITE-LAW (k_yaw, db_yaw_deg) | 0.1 | 12 | 24 | 30 | 1 |

- **Backends:** a local process pool for smoke tests, Modal (`evaluate_chunk.map`) for the real runs.
- **Budget:** projected and actual spend ≤ $25 for the whole fine-tune; the run stops itself past it.

## 6. Size mode
The encoder can see `s` from the head-top range, or `s = 1` (size-blind: the previous parameters were
trained size-blind in practice). G2 picks the mode with the lower `rms_bearing_deg` for FLY-YAW-PREV, and
training uses that mode.

## 7. Go / no-go gates (nothing ships on hope)

| Gate | When | Test | If it fails |
|---|---|---|---|
| G1 | before any Modal spend | KITE-LAW, `demo`, 20 seeds: 0 collisions; report in_view and following. Fly sign check: walker right → yaw > 0. FLY-YAW-PREV loads and steers. | If in_view < 0.7 for KITE-LAW, the 1.0 m / 0.30 m geometry itself is the problem: report it before training |
| G2 | baseline | 100 TEST seeds × (`train`, `demo`): KITE-LAW, FLY-YAW-HAND, FLY-YAW-PREV (both size modes) | If FLY-YAW-PREV already beats KITE-LAW clearly on every metric, report "already good" and keep it |
| G3 | after the fine-tune | 200 TEST seeds × (`train`, `demo`, `stress`, `fast`) | see below |

**G3 ships FLY-YAW only if, on `train` and `demo`, all of these hold:**
- its rms_bearing_deg is lower than KITE-LAW, KITE-LAW-CMA and FLY-YAW-PREV: the paired per-seed
  difference has a 95 % bootstrap CI below 0;
- losses are no higher than KITE-LAW;
- there are 0 collisions;
- stick_change ≤ 2 × KITE-LAW;
- `stress` and `fast` are not catastrophically worse.

**Fallbacks:**
- If KITE-LAW-CMA wins, kite gets the tuned classical gains, and that is reported honestly.
- If nothing beats KITE-LAW, nothing changes.

## 8. Revision 1, after the design check (supersedes the sections it names)

**R1. Fitness is yaw-attributable** (supersedes the fitness in §4).
- **Why:** the range terms are dominated by things the yaw arm cannot change: the altitude bias at follow
  start, and the fb law trailing a walking walker.
- **Episode terms:** `run_episode` still computes every term and returns each one in `terms`, under the keys
  `e_x`, `jerk`, `lost`, `e_z`, `too_close`, `ev_loss`, `ev_too_close`, `ev_collision`. `ret` is the full
  reward, used for reporting.
- **Training fitness:** `r_yaw = -(w_x Σe_x² + w_j Σj + w_l Σl)·dt + ev_loss`.
  - The score is `r_yaw / max(|r_yaw(KITE-LAW, same seed)|, 5)`, clipped to [-4, 0].
  - A collision scores -4.
  - Fitness = mean score; KITE-LAW ≈ -1.
- **Constraints:** range and safety become G3 constraints (R6), not fitness.

**R2. Latency** (supersedes the video latency in §2.3 and §2.4).
- The measured command dead times already include telemetry transport. Video lags telemetry only a little:
  the lag test logged 6-21 ms, while the onset analysis says up to about 0.2 s.
- **Video latency on top of the dynamics:**

  | Profile | Range |
  |---|---|
  | `train` | U(0.00, 0.25) s |
  | `demo` | 0.10 s |
  | `stress` | U(0.45, 0.65) s |

- **Detector time:** U(0.03, 0.10) s on top.
- **What every lag-aware arm believes:** always `FollowCfg.video_lag_s` (0.25 s, "to measure glass-to-glass").
  It is never the episode's truth.

**R3. Simulate what flies** (supersedes the parts of §2 it names).
- **Shared configuration:** every arm uses `FollowCfg(behind_m=1.0, above_m=0.30, max_yaw=70)`. The 70 is
  street corners; it is shared, so it is a fair comparison. FlySteer and PRED-PD take `behind_m` and
  `video_lag_s` from that same instance (assert it).
- **`max_rc`:**

  | Profile | max_rc |
  |---|---|
  | `train` | U(50, 100) |
  | `demo` | 50 (today's `SafetyCfg`) |
  | `fast` | 100 |
  | `stress` | train's range |

  It goes to `FollowController(max_rc=...)` as `FollowMode` does, and to the governor.
- **The governor is the real one:** the simulated drone implements `kite.drone.base.Drone` and is wrapped in
  the real `kite.drone.safety.SafetyGovernor(SafetyCfg(max_rc=..., max_altitude_m=3.0))`.
  `governor.check(now, last_frame_t)` runs every control step, as in kite.record. 3.0 m is the street
  ceiling; the indoor default of 2.3 m stays until the integration adds a street option.
- **Camera:** the believed camera is fixed at fx = fy = 920 (there is no calibration file today). The true fx
  is U(880, 1100).
- **Timing:** physics in 0.01 s sub-steps (exact dead times). The control period is 0.100 + U(0, 0.012) s,
  as kite.record's loop measures it.
- **The brain's Poisson noise** comes from the episode seed (common random numbers).

**R4. Starts and geometry.**
- **Starts are redrawn** (bounded attempts, from the start stream) until the head top is ≥ 20 px inside the
  frame for a level drone, identically for every arm.
- **`pitch_exits`:** the head leaving the frame while within 1.5 m.
- **`corner_losses`:** > 1 s gaps starting within 3 s of a corner turn.
- **G1 runs KITE-LAW on `demo`** at true fx 920, 1000 and 1077.

**R5. Person box** (supersedes §2.3's box). The box is the union of two parts, then clipped to the frame:
- **head:** 0.14-0.17 m wide, head top to chin;
- **torso:** 0.40-0.50 m wide, from the shoulder line (head top − 0.25 m) to the feet.

| View | Detection probability | Confidence |
|---|---|---|
| head only (no torso in frame) | `0.8 · min(1, (h_vis / 40)²)` | U(0.25, 0.6) |
| head and torso | `min(1, (h_vis / 20)²)` | U(0.4, 0.9) |

**R6. Gates** (supersede §7).
- **G1 and G2 use BASELINE_SEEDS** (never TEST_SEEDS). They inform the init and the size mode only; there is
  no "already good" shortcut.
- **G3, on TEST_SEEDS, with `train` and `demo`. FLY-YAW ships (opt-in) only if all of these hold:**
  1. Its mean rms_bearing_deg is ≥ 10 % lower than each of KITE-LAW, KITE-LAW-CMA, PRED-PD and FLY-YAW-PREV,
     and the paired per-seed bootstrap 95 % CI of each difference lies entirely below 0.
  2. Non-inferiority vs KITE-LAW on losses, time_lost_s, rms_range_err_m, too_close_s and pitch_exits:
     the paired difference's upper 95 % CI ≤ max(10 % of KITE-LAW's mean, a small absolute margin; losses
     0.1 per episode).
  3. On in_view and following, the paired difference's lower CI ≥ -0.02.
  4. 0 collisions.
  5. stick_change ≤ 2 × KITE-LAW.
  6. On `stress` and `fast`: rms_bearing ≤ 1.25 × KITE-LAW, with 0 collisions.
- **Outcomes:**
  - If PRED-PD beats KITE-LAW but FLY-YAW does not beat PRED-PD, the honest report is "prediction helps,
    the fly does not".
  - Even when the fly passes, the default stays KITE-LAW and the fly is opt-in until a real flight check:
    walker static at ±10° and ±20° (sign, no oscillation), then a straight walk plus one corner.

**R7. Arms to tune.**

| Arm | Parameters | Init / settings |
|---|---|---|
| FLY-YAW | 34 | 3 CMA seeds × 60 generations, population 24, K 24, sigma0 0.03 |
| KITE-LAW-CMA | k_yaw (log 0.5-4), db_yaw_deg (0-5) | 1 seed × 30 generations |
| PRED-PD | kp (log 0.5-4 stick/°), kd (0-1 stick per °/s), deadband (0-4°) | FlySteer's latency filter and stick conditioning around them; 1 seed × 30 generations |

**R8. Sign check (G1).** Steady walker at ±4, ±10 and ±18°: yaw signs must be right. Report the zero-crossing
offset; the previous readout's own fit is biased right by about 5-8°.

**Notes.**
- The previous parameters were trained with a stick-jerk weight of 26.4 (4× the frozen 6.6).
- The LC10a bins reach 40°, but in-frame bearings stay within ±27.6° at fx 920, so ~2 of 8 bins go unused.

## 9. Revision 2, after G2 (BASELINE_SEEDS 3000-3099, 100 seeds per profile)

**G2 results.** Mean rms_bearing_deg; Δ is paired against KITE-LAW, with the 95 % bootstrap CI.

| Arm | demo | Δ demo | train | Δ train | stick_change demo |
|---|---|---|---|---|---|
| KITE-LAW | 27.0 | | 40.1 | | 0.40 |
| FLY-YAW-PREV, size-blind | 27.3 | +1.2 % [-13, +16] | 39.4 | -1.8 % | 0.53 |
| FLY-YAW-PREV, s from range | 34.2 | +26 % | 43.8 | +9 % | 0.46 |
| FLY-YAW-HAND, size-blind | 23.9 | -11.6 % | 36.6 | -8.9 % | 2.26 |
| PRED-PD, untuned | 24.6 | -8.8 % | 39.1 | -2.5 % | 0.43 |

- KITE-LAW collides in 2/100 (demo) and 8/100 (train) episodes at 1.0 m behind. The cause is walker stops, not
  steering: the drone overshoots, the head drops under the frame, and it hovers blind.
- 1.2 m and 1.5 m behind: 0 collisions (the simulator's geometry sweep).

**R9. Training decisions.**
1. **Size mode:** size-blind (s = 1).
2. **Init:** FLY-YAW CMA seeds 1 and 2 start from FLY-YAW-PREV. Seed 3 starts from FLY-YAW-HAND, for diversity.
3. **Stick penalty:** w_j alone is ~1-2 % of r_yaw, so CMA would drift to a jittery stick. The episode score
   becomes:
   ```
   clip(r_yaw / max(|r_yaw(KITE-LAW)|, 5), -4, 0) - max(0, stick_change / max(stick_change(KITE-LAW), 0.1) - 1.8)
   ```
   The penalty keeps solutions inside G3's 2× stick gate.
4. **Training seeds:** offset by CMA seed (`TRAIN_SEED_BASE + 100000·cma_seed + gen·K + k`), so the three runs
   see different episodes.
5. **Collisions gate:** G3 criterion 4 becomes "collisions ≤ KITE-LAW's on each profile". At 1.0 m behind no
   arm can reach 0, because the geometry causes them. G3 also reports every arm at behind_m 1.2 and 1.5
   (`demo`), so the standoff can be chosen.

## 10. Revision 3, after the simulator check

**R10. Simulator fixes (before training).**
1. **Collisions no longer flatter an episode.**
   - After a collision, the remaining ticks up to `episode_s` count as lost, out of view, not following, with a
     true bearing of 90°. They are charged in the metrics and in `ret` (lost + e_x terms).
   - Rename the metric's meaning in docs to "overhead" (horizontal range < 0.3 m while above the head). Keep
     the `collision` key, and add `min_dist3d_m`.
2. **The hover is position-held.** While the sticks are at 0, the drone holds a position with a restoring time
   constant of U(1, 2) s, so drift and wind give a bounded wander (σ ≈ 0.05-0.15 m), not a random walk. With
   sticks off zero, drift and wind act as velocity disturbances, as now.
3. **Head-only detection is uncertain.** The head-only view's base probability becomes U(0.3, 0.8) in `train`
   and `stress` (the formula stays `p_head · min(1, (h_vis / 40)²)`), and 0.55 in `demo`. G1/G3 also report
   `demo` with p_head 0.3.
4. **Follow start must be possible.** Redraw the start, bounded and from the start stream, until the tracker
   confirms a walker touching the bottom edge within the pre-roll. If that fails after the attempts, the
   episode is flagged `start_failed` and excluded identically for every arm.
5. **`stick_change` counts following steps only.** It is the mean |Δ sent yaw| over control steps in state
   following, so losing the walker can't make an arm look calm.
6. **Detector timing:** detector rate ≤ 1 / detector time.
7. **Tests that can actually fail:**
   - the collision test asserts the rate over BASELINE_SEEDS stays ≤ 3/100, not 0 on a lucky slice;
   - the governor test gives the controller max_rc 100 against a governor at 50;
   - add tests for detector timing, detector CRN, the walker speed cap / acceleration / step-back stop, and
     losses / corner_losses / pitch_exits on constructed traces.

**R11. G3 statistics around collisions.**
1. **Paired metric comparisons** use only seeds where neither arm collided and the start did not fail. The
   number of seeds excluded is reported.
2. **Collisions are compared seed by seed:** `#(arm collides, KITE-LAW doesn't) ≤ #(KITE-LAW collides, arm
   doesn't) + 2` per profile.
3. **Context:** the simulator check found raising KITE-LAW's gain to k_yaw 2.5 already gives rms_bearing 16.9
   (from 27.0) and in_view 0.84 on `demo`. KITE-LAW-CMA is expected to be strong, and FLY-YAW must beat it.

## 11. Results (G3, TEST_SEEDS 2000-2199, 200 seeds per set, run once after training)

**Training** (Modal, $5.04, ~20 min):
- FLY-YAW: 3 CMA seeds × 60 generations. The best checkpoint is seed 1 at gen 50.
- KITE-LAW-CMA: k_yaw 2.34, db_yaw 1.25°.
- PRED-PD: kp 2.33, kd 0.09, deadband 1.52°.

**Mean rms_bearing_deg** (Δ is paired against KITE-LAW):

| Arm | train | demo | stress | fast | demo @1.2 m | demo @1.5 m | demo, p_head 0.3 |
|---|---|---|---|---|---|---|---|
| KITE-LAW | 46.0 | 28.8 | 47.5 | 48.0 | 19.3 | 12.8 | 33.6 |
| KITE-LAW-CMA | 35.8 (-24 %) | 18.9 (-35 %) | 44.0 | 41.1 | 12.4 | 9.0 | 25.0 |
| PRED-PD | 29.8 (-38 %) | 18.7 (-35 %) | 39.6 | 37.1 | 12.7 | 9.0 | 25.8 |
| FLY-YAW-PREV | 46.8 | 30.8 | 47.6 | 48.7 | 20.8 | 14.9 | 34.3 |
| FLY-YAW (retuned) | 43.4 (-6 %) | 25.8 (-10 %) | 45.7 | 46.3 | 17.1 | 11.0 | 31.5 |

**Verdict, computed mechanically: FLY-YAW fails G3** (criterion 1, and a too_close_s margin on train).
- The retuned fly beats kite's law (-6 % on train, -10 % on demo; CIs below 0) and the previous parameters
  (-8 % / -16 %).
- It is far behind both tuned classical arms (+24 % to +51 % worse on train, +37 % to +38 % on demo).
- Its safety metrics are level with KITE-LAW's on demo.

**The classical arms fail G3 too, on safety margins, not on steering.**
- Both have more separate losses and more too-close time.
- PRED-PD collides on more seeds in `stress` (9 seeds where KITE-LAW does not, vs 2 the other way) and in
  `fast` (8 vs 1).
- Holding the walker centred keeps the forward law engaged at short range.

**No flight behaviour changes.** The retuned fly (`data/fly_yaw.json`) and the tuned PRED-PD
(`data/pred_pd.json`) are available opt in only: `kite.record --follow-yaw fly|pd` and the rehearsal's
`--yaw fly|pd` (`steer.make_yaw_source`). The default stays KITE-LAW.

**Findings that matter more than the yaw law:**
- **The standoff:**

  | Standoff | KITE-LAW in_view | rms_bearing |
  |---|---|---|
  | 1.0 m | 0.72 | 28.8° |
  | 1.2 m | 0.85 | 19.3° |
  | 1.5 m | 0.94 | 12.8° |

- **A higher yaw gain** (k_yaw ≈ 2.3) is the cheapest steering win, but it needs the forward law's too-close
  behaviour fixed first.
- **Range runaway:** the uncompensated pitch makes the range estimate run away under acceleration.
- **Speed cap:** max_rc 50 caps following at ≈ 0.43 m/s of walking.
