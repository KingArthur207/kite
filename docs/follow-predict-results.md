# Follow prediction mode: simulation results

Follow mode has an opt-in prediction mode (`kite.record --follow-predict`, `FollowCfg.predict`, [`kite/predict.py`](../kite/predict.py)). An IMM Kalman filter predicts the walker 1.5 s ahead and a sampling MPC plans the forward, lateral and yaw sticks. Until now it had been checked only in the Moss-walk sim. This page runs it in the street follow simulator of [`kite/fly`](../kite/fly/DESIGN.md), the one that evaluated the yaw sources in [fly-yaw-results.md](fly-yaw-results.md). That simulator runs kite's real `WalkerTracker`, `FollowController` and `SafetyGovernor` at 10 Hz against measured Tello dynamics, video latency and detector noise. It used the same 200 held-out test seeds (60 s walks) and the same profiles, plus `train`, `stress` and `fast` at 1.2 m and 1.5 m behind. Every difference is paired per seed against the default reactive law (`KITE-LAW`), with 95 % bootstrap confidence intervals. This is simulation only; nothing here has flown, and no default changes.

## Answer

**Prediction mode is less safe than the reactive law at 1.0 m, 1.2 m and 1.5 m behind, and it is not better overall.** On every profile at every standoff it collides overhead on more seeds where the law doesn't than the other way round. It also spends more time too close (under 0.7 m), significantly on 10 of the 13 sets. It does steer better on the noisy `train` and `fast` worlds at 1.0 m and 1.2 m (3 to 5° less RMS bearing error). On `demo`, the profile closest to the real Tello, it steers worse at 1.2 m and 1.5 m and keeps the walker in view 8 to 10 points less. It fails the same G3 ship gate as the yaw sources on every criterion (1 to 6), so `--follow-predict` should stay off.

| behind | overhead collisions, seed by seed: predict alone / law alone (both) | RMS bearing (deg) | in view (points) | losses / walk | too close (s / walk) |
|---|---|---|---|---|---|
| 1.0 m | **48 / 9** (37), sign test p 2e-7 | −2.9 [−4.0, −1.7] | +1.4 [−0.2, +3.0] | +0.12 [+0.05, +0.18] | +0.33 [+0.22, +0.46] |
| 1.2 m | **45 / 13** (24), p 3e-5 | −1.2 [−2.4, −0.0] | −1.3 [−2.9, +0.4] | +0.16 [+0.10, +0.23] | +0.50 [+0.36, +0.65] |
| 1.5 m | **43 / 9** (15), p 2e-6 | +0.9 [−0.4, +2.2] | −3.4 [−5.2, −1.6] | +0.14 [+0.07, +0.21] | +0.44 [+0.31, +0.59] |
| 1.2 m, yaw clamp 40 (kite.record's defaults) | **43 / 11** (26), p 1e-5 | +0.9 [−0.2, +1.9] | −3.6 [−5.1, −2.1] | +0.11 [+0.05, +0.17] | +0.42 [+0.29, +0.56] |

Collisions are summed over `train`, `demo`, `stress` and `fast` (200 seeds each). The other columns are prediction minus law, averaged over those four profiles, with a 95 % CI from a bootstrap stratified by profile, over the seeds where neither arm collided. Lower is better except for "in view".

**Stick activity.** Prediction moves the yaw stick about twice as much as the law (+1.1 units per control step) and the lateral stick 4 to 10 times as much (+1.6). The forward stick is slightly calmer (−0.15 to −0.66). On `train` and `demo` the yaw stick changes more than twice as much as the law's, so it also fails gate criterion 5. The planner steered on 55 to 90 % of the following steps. The rest fell back to the reactive law, mostly in `stress` and `fast` (55 to 66 %), where the filter is unsure more often.

**`pitch_comp = -1` does not help the reactive law either** (details below). It adds forward-stick activity (+1.3 to +2.0 per step) and collides alone on 39 / 32 / 32 seeds against the law's 11 / 9 / 7 at 1.0 / 1.2 / 1.5 m. Keep `pitch_comp` at 0.

## Prediction against the law, per profile and standoff

Each cell is law → prediction, then the paired difference and its 95 % CI. A star means the CI excludes 0. Collisions are counted seed by seed. The stick columns are the mean |change| per control step while following, law → prediction.

| set | RMS bearing (deg) | in view (%) | losses / walk | too close (s / walk) | collisions: predict alone / law alone / both | stick change yaw / fb / lr | planned |
|---|---|---|---|---|---|---|---|
| train @ 1.0 m | 46.2 → 41.1: −5.1 [−7.0, −3.3] * | 51 → 54: +3.0 [+0.6, +5.6] * | 2.13 → 2.33: +0.19 [+0.09, +0.31] * | 0.33 → 0.67: +0.34 [+0.13, +0.59] * | 12 / 1 / 7 | 0.76→2.11 / 3.12→2.93 / 0.23→1.99 | 0.69 |
| demo @ 1.0 m | 28.3 → 28.2: −0.1 [−3.1, +2.8] | 73 → 70: −2.1 [−6.0, +1.9] | 2.31 → 2.42: +0.11 [−0.04, +0.26] | 0.99 → 1.59: +0.59 [+0.32, +0.89] * | 6 / 0 / 0 | 0.86→1.79 / 2.39→2.31 / 0.21→1.61 | 0.85 |
| stress @ 1.0 m | 46.0 → 44.0: −2.0 [−4.2, +0.0] | 50 → 51: +1.1 [−1.9, +4.2] | 2.27 → 2.35: +0.08 [−0.04, +0.20] | 0.28 → 0.36: +0.08 [−0.02, +0.18] | 15 / 5 / 8 | 1.75→2.79 / 4.94→4.90 / 0.66→2.48 | 0.55 |
| fast @ 1.0 m | 45.8 → 41.6: −4.2 [−6.3, −2.1] * | 50 → 53: +3.5 [+0.7, +6.6] * | 2.21 → 2.30: +0.09 [−0.04, +0.24] | 0.25 → 0.56: +0.31 [+0.05, +0.61] * | 15 / 3 / 22 | 0.66→1.71 / 3.49→3.18 / 0.26→1.88 | 0.61 |
| train @ 1.2 m | 39.9 → 35.9: −4.0 [−6.7, −1.6] * | 59 → 60: +1.6 [−1.5, +4.8] | 2.12 → 2.34: +0.22 [+0.11, +0.34] * | 0.17 → 0.78: +0.61 [+0.34, +0.93] * | 11 / 1 / 4 | 0.75→2.00 / 3.55→3.03 / 0.25→1.93 | 0.72 |
| demo @ 1.2 m | 18.7 → 21.9: +3.3 [+0.8, +5.9] * | 86 → 78: −7.9 [−11.4, −4.6] * | 2.27 → 2.40: +0.12 [−0.01, +0.25] | 0.41 → 1.17: +0.76 [+0.47, +1.07] * | 6 / 0 / 0 | 0.59→1.51 / 2.81→2.31 / 0.14→1.37 | 0.89 |
| stress @ 1.2 m | 42.6 → 41.7: −0.9 [−3.2, +1.4] | 56 → 55: −1.3 [−4.5, +2.0] | 2.26 → 2.44: +0.18 [+0.03, +0.35] * | 0.37 → 0.73: +0.36 [+0.10, +0.71] * | 14 / 5 / 5 | 1.59→2.89 / 5.47→5.46 / 0.56→2.62 | 0.61 |
| fast @ 1.2 m | 40.6 → 37.4: −3.2 [−5.5, −1.0] * | 56 → 59: +2.5 [−0.4, +5.7] | 2.12 → 2.24: +0.12 [+0.00, +0.24] | 0.17 → 0.43: +0.26 [−0.02, +0.56] | 14 / 7 / 15 | 0.66→1.79 / 3.95→3.62 / 0.26→1.98 | 0.66 |
| train @ 1.5 m | 32.5 → 31.3: −1.2 [−3.9, +1.5] | 67 → 65: −2.0 [−5.8, +1.8] | 2.06 → 2.33: +0.27 [+0.16, +0.39] * | 0.09 → 0.72: +0.62 [+0.33, +0.99] * | 10 / 1 / 4 | 0.61→1.77 / 4.22→3.35 / 0.29→1.82 | 0.75 |
| demo @ 1.5 m | 12.3 → 18.3: +6.1 [+3.5, +8.9] * | 94 → 84: −10.0 [−13.8, −6.4] * | 2.21 → 2.31: +0.10 [−0.04, +0.23] | 0.12 → 0.84: +0.72 [+0.41, +1.09] * | 2 / 0 / 0 | 0.49→1.45 / 3.40→2.48 / 0.17→1.31 | 0.90 |
| stress @ 1.5 m | 36.2 → 37.6: +1.4 [−1.1, +3.9] | 65 → 61: −3.7 [−7.2, −0.2] * | 2.50 → 2.60: +0.10 [−0.07, +0.27] | 0.08 → 0.35: +0.28 [+0.14, +0.43] * | 12 / 5 / 2 | 1.42→2.69 / 5.92→5.69 / 0.59→2.61 | 0.62 |
| fast @ 1.5 m | 35.7 → 32.9: −2.7 [−5.3, −0.2] * | 62 → 64: +2.1 [−1.7, +5.9] | 2.09 → 2.19: +0.10 [−0.02, +0.23] | 0.22 → 0.37: +0.15 [−0.08, +0.38] | 19 / 3 / 9 | 0.60→1.63 / 4.47→3.87 / 0.28→1.91 | 0.66 |
| demo, head-only p 0.3 | 33.4 → 31.7: −1.7 [−4.0, +0.7] | 67 → 66: −1.0 [−4.2, +2.1] | 2.47 → 2.50: +0.03 [−0.09, +0.14] | 0.68 → 1.25: +0.57 [+0.32, +0.86] * | 5 / 0 / 0 | 0.93→1.79 / 2.44→2.07 / 0.25→1.56 | 0.81 |

The law's numbers are those of the G3 report to the third decimal (for example `demo` 28.799° and 0.720 in view over all 200 seeds; the table's means are over the seeds where neither arm collided). At kite.record's own geometry (1.2 m, yaw clamp 40 instead of the simulator's shared 70), prediction against the law is: `train` 11 / 1 collisions alone, too close +0.49 s; `demo` 5 / 0, RMS bearing +5.2°, in view −10.7 points; `stress` 13 / 4; `fast` 14 / 6.

What the traces show (observations, not proven causes):

- Prediction holds a tighter standoff, then overshoots at stops as much as the law does. In seed 2000 on `demo` at 1.5 m, prediction follows at 1.67 to 1.70 m while the law trails at 1.8 to 2.0 m. When the walker stops at t = 8.4 s, both close in by about 0.85 m. The law ends at 0.89 m. Prediction ends at 0.66 m, where the head leaves through the bottom of the frame and follow never recovers. Over 40 `demo` seeds, the range drop after a stop is a little larger with prediction (0.78 against 0.69 m at 1.0 m behind, 0.72 against 0.65 m at 1.5 m). At 1.5 m it also meets stops closer (2.39 against 2.57 m). The law trailing well behind acts as a buffer. Head exits through the bottom of the frame fit this: +0.32 to +0.34 per walk on `train` and `demo`, against a 0.1 margin.
- The planner's model of the drone is simpler than the measured one the simulator flies, which may add to the overshoot. It has a 0.4 s first-order forward lag and no dead time (the Tello measurements: 0.33 to 0.61 s dead time plus 0.31 to 0.59 s lag). It assumes a yaw rate of 1.0 °/s per stick unit (measured: 0.39 to 0.72) and a fixed 0.25 s video lag (simulated: 0 to 0.65 s). None of this was retuned here; defaults stay as they are.
- The planner's standoff point, measured along the walker's trail, is only 2 to 7 % closer than `behind_m + gap_s × speed`, so the trail is not what brings it close.

## Pitch compensation on the reactive law

`pitch_comp = -1` (the sign the reviews found correct) against `pitch_comp = 0`, the law otherwise unchanged, averaged over the four profiles:

| behind | collisions: pitch_comp alone / law alone (both) | RMS bearing (deg) | in view (points) | too close (s / walk) | fb stick change |
|---|---|---|---|---|---|
| 1.0 m | 39 / 11 (35), p 9e-5 | +0.2 [−0.8, +1.1] | −0.7 [−1.9, +0.5] | +0.22 [+0.11, +0.33] | +1.26 [+1.12, +1.40] |
| 1.2 m | 32 / 9 (28), p 4e-4 | +0.3 [−0.7, +1.2] | −0.6 [−1.9, +0.8] | +0.30 [+0.19, +0.42] | +1.57 [+1.42, +1.71] |
| 1.5 m | 32 / 7 (17), p 7e-5 | +1.4 [+0.4, +2.4] | −1.3 [−2.7, +0.2] | +0.21 [+0.10, +0.32] | +1.95 [+1.79, +2.11] |

It steers no better, gets close more often and works the forward stick harder. Most of the extra collisions are on `fast` (19 / 4, 15 / 6 and 20 / 3 seeds). It fails G3 on criteria 1, 2, 3, 4 and 6.

A diagnostic run (law against `pitch_comp = -1` on `train`, `demo` and `fast` at 1.0 m and `demo` at 1.5 m, 200 seeds each; not committed) checked two suspected causes:

- **Whole-degree rounding is not the cause.** The Tello sends pitch as whole degrees and the simulator now does too. Unrounded pitch gives the same picture (`train`: 12 / 1 collisions alone, against 7 / 2 rounded).
- **The timing of the pitch matters.** `FollowController.measure` corrects a box from a frame captured 0.1 to 0.65 s earlier with the pitch at the control step. Using the pitch at the frame's capture time instead roughly halves the extra forward-stick activity on `train` (+0.57 against +1.39) and removes the extra too-close time on `demo` at 1.5 m (+0.00 against +0.16 s). Even then, compensation is at best neutral: no steering or in-view gain, and `fast` still collides alone 16 / 4.

Prediction with `pitch_comp = -1`, against the law with `pitch_comp = -1`, is roughly even on collisions at 1.0 m (32 / 30) and steers 5.6° better there. Against the actual default (the law with `pitch_comp = 0`) it still collides alone 48 / 18, 46 / 15 and 48 / 10 at 1.0, 1.2 and 1.5 m.

## How it was run

The simulator change ([`kite/fly`](../kite/fly/), on top of PR #98):

- **FollowCfg variants.** `python -m kite.fly.train.report` takes `ARM+field=value` arm specs, for example `KITE-LAW+predict` or `KITE-LAW+predict+pitch_comp=-1`. A variant runs the arm with those `FollowCfg` fields changed (`arms.check_follow` types and checks them; `predict` only on the law arms, since it and a yaw source are exclusive). G3 checks a variant against `KITE-LAW` alone. The report now also lists seed-by-seed collisions against `KITE-LAW` for every arm and set.
- **Inputs as kite.record gives them live.** The controller gets no route and `walker_xy = None`, and kite.record passes neither. Telemetry yaw and pitch come in whole degrees with yaw wrapped to ±180, as the Tello's state packet sends them; only `pitch_comp` and the prediction's dead reckoning read them, so the law's results are unchanged. As before: control steps every 0.100 + U(0, 0.012) s, frames carrying their arrival time, the believed camera (fx 920), `video_lag_s` 0.25 and the profile's `max_rc`.
- **New metrics** (reported, never trained on): `fb_change` and `lr_change` (like `stick_change`, for the forward and lateral sticks) and `planned` (the share of following steps the planner steered). The trace records which planner steered each step.

```bash
python -m kite.fly.train.report --arms KITE-LAW,KITE-LAW+predict,KITE-LAW+pitch_comp=-1,KITE-LAW+predict+pitch_comp=-1 \
    --seeds test --n 200 --profiles train,demo,stress,fast \
    --side-sets demo@1.2,demo@1.5,demo@p_head0.3,train@1.2,train@1.5,stress@1.2,stress@1.5,fast@1.2,fast@1.5 --name predict
python -m kite.fly.train.report --arms "KITE-LAW+max_yaw=40,KITE-LAW+predict+max_yaw=40" --vs "KITE-LAW+max_yaw=40" \
    --seeds test --n 200 --profiles "" --side-sets train@1.2,demo@1.2,stress@1.2,fast@1.2 --name predict_live
```

The 10,400 + 1,600 episodes took about 10 + 6 minutes on a laptop (12 + 3 worker processes, no Modal). The full tables and every episode's metrics are in `results/fly/report/predict.{md,json}` and `predict_live.{md,json}`, which are not committed.

Limits:

- These are simulated walks. In this simulator the drone's top speed at stick 50 barely covers the walker's, so every controller loses the walker about twice per walk.
- Paired cells leave out seeds where either arm collided (DESIGN R11.1). That flatters the arm that collides more, prediction here.
- The simulator has no GPS route, so the route prior that `kite.follow`'s rehearsal can use on a take with GPS was not tested. kite.record never passes one either.
