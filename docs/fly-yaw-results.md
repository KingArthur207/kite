# Fly-brain yaw: simulation results (G3)

This is the G3 report of [`kite/fly`](../kite/fly/DESIGN.md) (DESIGN sections 7 and 11), copied from `results/fly/report/g3.md`, which is not committed. Five yaw controllers ran in kite's follow simulator, each deciding only the yaw stick inside the real `kite.follow` loop: `KITE-LAW` (follow mode's own yaw law, what kite flies), `KITE-LAW-CMA` (the same law with CMA-ES-tuned gains), `PRED-PD` (the predictive PD, `--follow-yaw pd`), `FLY-YAW-PREV` (the fruit-fly circuit with its earlier parameters) and `FLY-YAW` (the fly retuned for kite, `--follow-yaw fly`). Each ran the same 200 held-out test seeds (60 s simulated walks) per profile, once, after training. Differences are paired per seed against `KITE-LAW` with 95 % bootstrap confidence intervals; `rms_bearing_deg`, the steering error, is lower-is-better. The verdict at the end is computed mechanically from the gates: no arm passed, so the law stays kite's default and `pd` / `fly` are opt-in only. All of this is simulation; none of these controllers has flown. Every arm, `KITE-LAW` included, ran at the kite spec's geometry: `behind_m` 1.0, `above_m` 0.30 and `max_yaw` 70 (`kite.record` defaults to 1.2 m and 40), so `KITE-LAW` here is the law at that geometry, not byte-for-byte what flies by default.

## Yaw controller evaluation: g3

200 test seeds x train, demo, stress, fast, demo@1.2, demo@1.5, demo@p_head0.3; episode 60.0 s; 2026-10-24T12:11:43+00:00. Fitness (DESIGN R1) = mean clip(r_yaw / max(|KITE-LAW r_yaw|, 5), -4, 0), collision -4 (failed starts 0), minus min(4, max(0, S / max(S_KITE-LAW, 0.1) - 1.8)) on the following-weighted mean stick_change (KITE-LAW about -1, higher is better). Diff = paired per-seed mean (arm - KITE-LAW) with 95 % bootstrap CI over the seeds where neither collided and the start did not fail (n, excluded). Side sets (not gated): demo@<m> = demo at that behind_m, demo@<field><v> = a sim override.

### train

| arm | n | err | start failed | fitness | coll | rms_bearing_deg | p95_bearing_deg | in_view | following | losses | time_lost_s | rms_range_err_m | in_band | too_close_s | pitch_exits | corner_losses | stick_change | min_dist3d_m | rms_bearing diff vs KITE-LAW (n, excluded) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| KITE-LAW | 200 | 0 | 0 | -1.120 | 8 | 45.989 | 66.825 | 0.508 | 0.415 | 2.135 | 27.972 | 7.983 | 0.019 | 0.753 | 0.440 | 0.675 | 0.735 | 1.363 |  |
| KITE-LAW-CMA | 200 | 0 | 0 | -0.951 | 9 | 35.845 | 53.550 | 0.618 | 0.508 | 2.310 | 21.877 | 6.157 | 0.022 | 1.003 | 0.495 | 0.585 | 0.971 | 1.263 | -10.93 [-13.55, -8.46] (190, 10) |
| PRED-PD | 200 | 0 | 0 | -0.867 | 9 | 29.830 | 45.486 | 0.693 | 0.582 | 2.395 | 17.744 | 4.944 | 0.024 | 1.002 | 0.565 | 0.525 | 1.319 | 1.204 | -17.06 [-20.63, -13.73] (190, 10) |
| FLY-YAW-PREV@blind | 200 | 0 | 0 | -1.203 | 7 | 46.768 | 67.563 | 0.494 | 0.404 | 2.125 | 28.710 | 8.145 | 0.018 | 0.827 | 0.415 | 0.685 | 0.825 | 1.368 | +1.05 [-0.60, +2.89] (192, 8) |
| FLY-YAW | 200 | 0 | 0 | -1.106 | 8 | 43.445 | 63.116 | 0.533 | 0.430 | 2.190 | 26.560 | 7.706 | 0.020 | 0.874 | 0.425 | 0.655 | 0.922 | 1.340 | -2.63 [-4.04, -1.28] (191, 9) |


### demo

| arm | n | err | start failed | fitness | coll | rms_bearing_deg | p95_bearing_deg | in_view | following | losses | time_lost_s | rms_range_err_m | in_band | too_close_s | pitch_exits | corner_losses | stick_change | min_dist3d_m | rms_bearing diff vs KITE-LAW (n, excluded) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| KITE-LAW | 200 | 0 | 0 | -1.000 | 0 | 28.799 | 45.166 | 0.720 | 0.603 | 2.345 | 16.302 | 3.220 | 0.031 | 1.220 | 0.445 | 0.555 | 0.926 | 1.177 |  |
| KITE-LAW-CMA | 200 | 0 | 0 | -0.837 | 0 | 18.857 | 30.599 | 0.820 | 0.696 | 2.460 | 10.829 | 2.282 | 0.037 | 1.457 | 0.510 | 0.435 | 0.969 | 1.109 | -9.94 [-12.75, -7.23] (200, 0) |
| PRED-PD | 200 | 0 | 0 | -0.852 | 1 | 18.675 | 30.273 | 0.828 | 0.706 | 2.495 | 10.326 | 2.207 | 0.038 | 1.443 | 0.505 | 0.430 | 1.214 | 1.119 | -10.15 [-13.21, -7.31] (199, 1) |
| FLY-YAW-PREV@blind | 200 | 0 | 0 | -1.229 | 0 | 30.758 | 47.519 | 0.675 | 0.563 | 2.310 | 18.720 | 3.587 | 0.030 | 1.183 | 0.415 | 0.570 | 0.940 | 1.179 | +1.96 [-0.05, +4.05] (200, 0) |
| FLY-YAW | 200 | 0 | 0 | -1.001 | 0 | 25.791 | 41.096 | 0.733 | 0.621 | 2.360 | 15.498 | 3.004 | 0.033 | 1.277 | 0.480 | 0.520 | 1.047 | 1.149 | -3.01 [-4.73, -1.21] (200, 0) |


### stress

| arm | n | err | start failed | fitness | coll | rms_bearing_deg | p95_bearing_deg | in_view | following | losses | time_lost_s | rms_range_err_m | in_band | too_close_s | pitch_exits | corner_losses | stick_change | min_dist3d_m | rms_bearing diff vs KITE-LAW (n, excluded) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| KITE-LAW | 200 | 0 | 0 | -1.195 | 13 | 47.500 | 68.757 | 0.486 | 0.323 | 2.275 | 29.052 | 8.669 | 0.012 | 0.673 | 0.320 | 0.650 | 1.723 | 1.593 |  |
| KITE-LAW-CMA | 200 | 0 | 0 | -1.493 | 14 | 43.952 | 65.275 | 0.527 | 0.348 | 2.345 | 26.721 | 8.360 | 0.013 | 0.832 | 0.390 | 0.605 | 2.912 | 1.564 | -3.42 [-6.21, -0.70] (179, 21) |
| PRED-PD | 200 | 0 | 0 | -1.160 | 20 | 39.606 | 58.964 | 0.589 | 0.395 | 2.395 | 23.390 | 7.163 | 0.015 | 0.916 | 0.455 | 0.550 | 2.459 | 1.442 | -9.25 [-12.10, -6.65] (178, 22) |
| FLY-YAW-PREV@blind | 200 | 0 | 0 | -1.260 | 13 | 47.571 | 69.488 | 0.497 | 0.328 | 2.305 | 28.448 | 8.523 | 0.013 | 0.782 | 0.320 | 0.635 | 1.462 | 1.556 | +0.35 [-1.10, +1.83] (186, 14) |
| FLY-YAW | 200 | 0 | 0 | -1.212 | 12 | 45.650 | 66.750 | 0.511 | 0.332 | 2.295 | 27.680 | 8.604 | 0.013 | 0.792 | 0.360 | 0.625 | 1.937 | 1.564 | -1.62 [-3.51, +0.17] (184, 16) |


### fast

| arm | n | err | start failed | fitness | coll | rms_bearing_deg | p95_bearing_deg | in_view | following | losses | time_lost_s | rms_range_err_m | in_band | too_close_s | pitch_exits | corner_losses | stick_change | min_dist3d_m | rms_bearing diff vs KITE-LAW (n, excluded) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| KITE-LAW | 200 | 0 | 0 | -1.375 | 25 | 47.991 | 69.673 | 0.485 | 0.377 | 2.165 | 29.236 | 11.215 | 0.013 | 0.485 | 0.335 | 0.660 | 0.619 | 1.577 |  |
| KITE-LAW-CMA | 200 | 0 | 0 | -1.291 | 29 | 41.067 | 59.746 | 0.558 | 0.436 | 2.220 | 25.286 | 9.933 | 0.014 | 0.464 | 0.360 | 0.570 | 1.009 | 1.493 | -8.55 [-11.00, -6.32] (171, 29) |
| PRED-PD | 200 | 0 | 0 | -1.268 | 32 | 37.129 | 55.203 | 0.619 | 0.502 | 2.210 | 21.885 | 8.065 | 0.016 | 0.614 | 0.455 | 0.460 | 1.075 | 1.363 | -13.48 [-16.94, -10.29] (167, 33) |
| FLY-YAW-PREV@blind | 200 | 0 | 0 | -1.434 | 26 | 48.683 | 70.097 | 0.475 | 0.364 | 2.120 | 29.793 | 11.494 | 0.012 | 0.441 | 0.300 | 0.670 | 0.752 | 1.590 | +0.25 [-0.62, +1.29] (173, 27) |
| FLY-YAW | 200 | 0 | 0 | -1.383 | 26 | 46.347 | 67.759 | 0.495 | 0.386 | 2.135 | 28.674 | 10.986 | 0.013 | 0.479 | 0.335 | 0.655 | 0.971 | 1.572 | -2.39 [-3.53, -1.41] (173, 27) |


### demo@1.2

| arm | n | err | start failed | fitness | coll | rms_bearing_deg | p95_bearing_deg | in_view | following | losses | time_lost_s | rms_range_err_m | in_band | too_close_s | pitch_exits | corner_losses | stick_change | min_dist3d_m | rms_bearing diff vs KITE-LAW (n, excluded) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| KITE-LAW | 200 | 0 | 0 | -0.997 | 0 | 19.323 | 31.858 | 0.852 | 0.734 | 2.275 | 8.934 | 2.276 | 0.050 | 0.460 | 0.175 | 0.460 | 0.605 | 1.348 |  |
| KITE-LAW-CMA | 200 | 0 | 0 | -0.861 | 0 | 12.368 | 21.273 | 0.911 | 0.789 | 2.295 | 5.798 | 1.720 | 0.057 | 0.516 | 0.220 | 0.395 | 0.799 | 1.312 | -6.95 [-9.44, -4.77] (200, 0) |
| PRED-PD | 200 | 0 | 0 | -0.875 | 0 | 12.674 | 20.971 | 0.916 | 0.794 | 2.285 | 5.518 | 1.686 | 0.056 | 0.523 | 0.170 | 0.370 | 0.909 | 1.316 | -6.65 [-9.23, -4.39] (200, 0) |
| FLY-YAW-PREV@blind | 200 | 0 | 0 | -1.154 | 0 | 20.816 | 34.240 | 0.823 | 0.706 | 2.200 | 10.441 | 2.518 | 0.048 | 0.496 | 0.175 | 0.485 | 0.828 | 1.351 | +1.49 [-0.31, +3.54] (200, 0) |
| FLY-YAW | 200 | 0 | 0 | -1.040 | 0 | 17.115 | 28.624 | 0.857 | 0.745 | 2.240 | 8.619 | 2.139 | 0.051 | 0.517 | 0.180 | 0.455 | 0.992 | 1.326 | -2.21 [-3.16, -1.27] (200, 0) |


### demo@1.5

| arm | n | err | start failed | fitness | coll | rms_bearing_deg | p95_bearing_deg | in_view | following | losses | time_lost_s | rms_range_err_m | in_band | too_close_s | pitch_exits | corner_losses | stick_change | min_dist3d_m | rms_bearing diff vs KITE-LAW (n, excluded) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| KITE-LAW | 200 | 0 | 0 | -0.993 | 0 | 12.799 | 22.289 | 0.936 | 0.827 | 2.225 | 4.384 | 1.683 | 0.066 | 0.122 | 0.045 | 0.375 | 0.502 | 1.652 |  |
| KITE-LAW-CMA | 200 | 0 | 0 | -0.903 | 0 | 9.000 | 15.854 | 0.954 | 0.842 | 2.235 | 3.428 | 1.548 | 0.072 | 0.162 | 0.035 | 0.355 | 0.658 | 1.637 | -3.80 [-5.39, -2.51] (200, 0) |
| PRED-PD | 200 | 0 | 0 | -0.925 | 0 | 8.977 | 15.293 | 0.965 | 0.853 | 2.225 | 2.787 | 1.479 | 0.075 | 0.145 | 0.040 | 0.325 | 0.763 | 1.639 | -3.82 [-5.99, -2.02] (200, 0) |
| FLY-YAW-PREV@blind | 200 | 0 | 0 | -1.157 | 0 | 14.911 | 24.873 | 0.895 | 0.790 | 2.175 | 6.596 | 1.988 | 0.065 | 0.144 | 0.035 | 0.395 | 0.741 | 1.652 | +2.11 [+0.71, +3.75] (200, 0) |
| FLY-YAW | 200 | 0 | 0 | -1.215 | 0 | 11.016 | 19.045 | 0.936 | 0.828 | 2.205 | 4.402 | 1.663 | 0.071 | 0.136 | 0.045 | 0.370 | 0.882 | 1.643 | -1.78 [-2.62, -1.07] (200, 0) |


### demo@p_head0.3

| arm | n | err | start failed | fitness | coll | rms_bearing_deg | p95_bearing_deg | in_view | following | losses | time_lost_s | rms_range_err_m | in_band | too_close_s | pitch_exits | corner_losses | stick_change | min_dist3d_m | rms_bearing diff vs KITE-LAW (n, excluded) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| KITE-LAW | 200 | 0 | 0 | -1.000 | 0 | 33.563 | 51.752 | 0.669 | 0.542 | 2.510 | 19.253 | 3.601 | 0.028 | 0.922 | 0.285 | 0.660 | 0.920 | 1.206 |  |
| KITE-LAW-CMA | 200 | 0 | 0 | -0.900 | 1 | 24.969 | 39.994 | 0.762 | 0.618 | 2.645 | 14.231 | 2.772 | 0.031 | 1.040 | 0.395 | 0.515 | 1.184 | 1.152 | -8.62 [-11.28, -6.11] (199, 1) |
| PRED-PD | 200 | 0 | 0 | -0.881 | 1 | 25.831 | 40.974 | 0.765 | 0.622 | 2.650 | 14.062 | 2.700 | 0.033 | 1.076 | 0.330 | 0.530 | 1.227 | 1.158 | -7.74 [-10.02, -5.62] (199, 1) |
| FLY-YAW-PREV@blind | 200 | 0 | 0 | -1.097 | 0 | 34.335 | 53.310 | 0.656 | 0.524 | 2.530 | 19.896 | 3.738 | 0.028 | 0.918 | 0.275 | 0.680 | 0.854 | 1.212 | +0.77 [-0.87, +2.58] (200, 0) |
| FLY-YAW | 200 | 0 | 0 | -1.014 | 0 | 31.487 | 48.971 | 0.681 | 0.550 | 2.550 | 18.657 | 3.471 | 0.029 | 0.991 | 0.310 | 0.615 | 1.086 | 1.191 | -2.08 [-3.69, -0.46] (200, 0) |


### G3 (test seeds)

**Verdict: FLY-YAW fails G3 on criteria [1, 2]; nothing beats KITE-LAW: no change**

- KITE-LAW-CMA: FAIL on criteria [2, 6]
  - [x] (1) train: rms_bearing >= 10% below KITE-LAW: 34.471 vs 45.406 deg (-24.1%); paired diff -10.935 deg, 95% CI [-13.545, -8.460], n 190 (10 seeds excluded: collision, failed start or failed episode)
  - [ ] (2) train: losses non-inferior to KITE-LAW: paired diff +0.174, 95% CI [+0.058, +0.300], n 190 (10 seeds excluded: collision, failed start or failed episode); upper CI must be <= 0.216
  - [x] (2) train: time_lost_s non-inferior to KITE-LAW: paired diff -6.582, 95% CI [-8.768, -4.519], n 190 (10 seeds excluded: collision, failed start or failed episode); upper CI must be <= 2.776
  - [x] (2) train: rms_range_err_m non-inferior to KITE-LAW: paired diff -1.812, 95% CI [-2.516, -1.166], n 190 (10 seeds excluded: collision, failed start or failed episode); upper CI must be <= 0.824
  - [ ] (2) train: too_close_s non-inferior to KITE-LAW: paired diff +0.259, 95% CI [+0.082, +0.475], n 190 (10 seeds excluded: collision, failed start or failed episode); upper CI must be <= 0.200
  - [ ] (2) train: pitch_exits non-inferior to KITE-LAW: paired diff +0.053, 95% CI [-0.026, +0.132], n 190 (10 seeds excluded: collision, failed start or failed episode); upper CI must be <= 0.100
  - [x] (3) train: in_view lower CI >= -0.02: paired diff +0.119, 95% CI [+0.081, +0.158], n 190 (10 seeds excluded: collision, failed start or failed episode)
  - [x] (3) train: following lower CI >= -0.02: paired diff +0.101, 95% CI [+0.069, +0.135], n 190 (10 seeds excluded: collision, failed start or failed episode)
  - [x] (4) train: collisions seed by seed vs KITE-LAW: collides alone on 2 seeds, KITE-LAW alone on 1 (both 7, n 200; allowed KITE-LAW's + 2); 0 failed episodes
  - [x] (5) train: stick_change <= 2 x KITE-LAW: 0.999 vs 2 x 0.746 over 190 kept seeds
  - [x] (1) demo: rms_bearing >= 10% below KITE-LAW: 18.857 vs 28.799 deg (-34.5%); paired diff -9.942 deg, 95% CI [-12.749, -7.227], n 200 (0 seeds excluded: collision, failed start or failed episode)
  - [ ] (2) demo: losses non-inferior to KITE-LAW: paired diff +0.115, 95% CI [+0.005, +0.240], n 200 (0 seeds excluded: collision, failed start or failed episode); upper CI must be <= 0.235
  - [x] (2) demo: time_lost_s non-inferior to KITE-LAW: paired diff -5.473, 95% CI [-7.461, -3.563], n 200 (0 seeds excluded: collision, failed start or failed episode); upper CI must be <= 1.630
  - [x] (2) demo: rms_range_err_m non-inferior to KITE-LAW: paired diff -0.938, 95% CI [-1.291, -0.605], n 200 (0 seeds excluded: collision, failed start or failed episode); upper CI must be <= 0.322
  - [ ] (2) demo: too_close_s non-inferior to KITE-LAW: paired diff +0.237, 95% CI [+0.082, +0.423], n 200 (0 seeds excluded: collision, failed start or failed episode); upper CI must be <= 0.200
  - [ ] (2) demo: pitch_exits non-inferior to KITE-LAW: paired diff +0.065, 95% CI [+0.005, +0.140], n 200 (0 seeds excluded: collision, failed start or failed episode); upper CI must be <= 0.100
  - [x] (3) demo: in_view lower CI >= -0.02: paired diff +0.100, 95% CI [+0.066, +0.137], n 200 (0 seeds excluded: collision, failed start or failed episode)
  - [x] (3) demo: following lower CI >= -0.02: paired diff +0.093, 95% CI [+0.062, +0.127], n 200 (0 seeds excluded: collision, failed start or failed episode)
  - [x] (4) demo: collisions seed by seed vs KITE-LAW: collides alone on 0 seeds, KITE-LAW alone on 0 (both 0, n 200; allowed KITE-LAW's + 2); 0 failed episodes
  - [x] (5) demo: stick_change <= 2 x KITE-LAW: 0.969 vs 2 x 0.926 over 200 kept seeds
  - [x] (6) stress: rms_bearing <= 1.25 x KITE-LAW, collisions seed by seed: rms bearing 42.525 vs 1.25 x 45.940 over 179 kept seeds; collides alone on 8 seeds, KITE-LAW alone on 7 (both 6, n 200; allowed KITE-LAW's + 2); 0 failed episodes
  - [ ] (6) fast: rms_bearing <= 1.25 x KITE-LAW, collisions seed by seed: rms bearing 36.165 vs 1.25 x 44.711 over 171 kept seeds; collides alone on 4 seeds, KITE-LAW alone on 0 (both 25, n 200; allowed KITE-LAW's + 2); 0 failed episodes
- PRED-PD: FAIL on criteria [2, 6]
  - [x] (1) train: rms_bearing >= 10% below KITE-LAW: 28.213 vs 45.271 deg (-37.7%); paired diff -17.058 deg, 95% CI [-20.630, -13.729], n 190 (10 seeds excluded: collision, failed start or failed episode)
  - [ ] (2) train: losses non-inferior to KITE-LAW: paired diff +0.263, 95% CI [+0.132, +0.405], n 190 (10 seeds excluded: collision, failed start or failed episode); upper CI must be <= 0.215
  - [x] (2) train: time_lost_s non-inferior to KITE-LAW: paired diff -10.702, 95% CI [-13.326, -8.201], n 190 (10 seeds excluded: collision, failed start or failed episode); upper CI must be <= 2.766
  - [x] (2) train: rms_range_err_m non-inferior to KITE-LAW: paired diff -3.027, 95% CI [-3.866, -2.213], n 190 (10 seeds excluded: collision, failed start or failed episode); upper CI must be <= 0.817
  - [ ] (2) train: too_close_s non-inferior to KITE-LAW: paired diff +0.252, 95% CI [+0.076, +0.458], n 190 (10 seeds excluded: collision, failed start or failed episode); upper CI must be <= 0.200
  - [ ] (2) train: pitch_exits non-inferior to KITE-LAW: paired diff +0.121, 95% CI [+0.037, +0.205], n 190 (10 seeds excluded: collision, failed start or failed episode); upper CI must be <= 0.100
  - [x] (3) train: in_view lower CI >= -0.02: paired diff +0.193, 95% CI [+0.148, +0.240], n 190 (10 seeds excluded: collision, failed start or failed episode)
  - [x] (3) train: following lower CI >= -0.02: paired diff +0.174, 95% CI [+0.132, +0.217], n 190 (10 seeds excluded: collision, failed start or failed episode)
  - [x] (4) train: collisions seed by seed vs KITE-LAW: collides alone on 2 seeds, KITE-LAW alone on 1 (both 7, n 200; allowed KITE-LAW's + 2); 0 failed episodes
  - [x] (5) train: stick_change <= 2 x KITE-LAW: 1.358 vs 2 x 0.745 over 190 kept seeds
  - [x] (1) demo: rms_bearing >= 10% below KITE-LAW: 18.554 vs 28.699 deg (-35.3%); paired diff -10.145 deg, 95% CI [-13.215, -7.307], n 199 (1 seeds excluded: collision, failed start or failed episode)
  - [ ] (2) demo: losses non-inferior to KITE-LAW: paired diff +0.151, 95% CI [+0.025, +0.291], n 199 (1 seeds excluded: collision, failed start or failed episode); upper CI must be <= 0.235
  - [x] (2) demo: time_lost_s non-inferior to KITE-LAW: paired diff -5.863, 95% CI [-7.948, -3.933], n 199 (1 seeds excluded: collision, failed start or failed episode); upper CI must be <= 1.615
  - [x] (2) demo: rms_range_err_m non-inferior to KITE-LAW: paired diff -0.978, 95% CI [-1.332, -0.656], n 199 (1 seeds excluded: collision, failed start or failed episode); upper CI must be <= 0.319
  - [ ] (2) demo: too_close_s non-inferior to KITE-LAW: paired diff +0.188, 95% CI [+0.052, +0.352], n 199 (1 seeds excluded: collision, failed start or failed episode); upper CI must be <= 0.200
  - [ ] (2) demo: pitch_exits non-inferior to KITE-LAW: paired diff +0.055, 95% CI [-0.005, +0.121], n 199 (1 seeds excluded: collision, failed start or failed episode); upper CI must be <= 0.100
  - [x] (3) demo: in_view lower CI >= -0.02: paired diff +0.107, 95% CI [+0.072, +0.144], n 199 (1 seeds excluded: collision, failed start or failed episode)
  - [x] (3) demo: following lower CI >= -0.02: paired diff +0.101, 95% CI [+0.070, +0.135], n 199 (1 seeds excluded: collision, failed start or failed episode)
  - [x] (4) demo: collisions seed by seed vs KITE-LAW: collides alone on 1 seeds, KITE-LAW alone on 0 (both 0, n 200; allowed KITE-LAW's + 2); 0 failed episodes
  - [x] (5) demo: stick_change <= 2 x KITE-LAW: 1.216 vs 2 x 0.926 over 199 kept seeds
  - [ ] (6) stress: rms_bearing <= 1.25 x KITE-LAW, collisions seed by seed: rms bearing 35.971 vs 1.25 x 45.220 over 178 kept seeds; collides alone on 9 seeds, KITE-LAW alone on 2 (both 11, n 200; allowed KITE-LAW's + 2); 0 failed episodes
  - [ ] (6) fast: rms_bearing <= 1.25 x KITE-LAW, collisions seed by seed: rms bearing 31.288 vs 1.25 x 44.766 over 167 kept seeds; collides alone on 8 seeds, KITE-LAW alone on 1 (both 24, n 200; allowed KITE-LAW's + 2); 0 failed episodes
- FLY-YAW: FAIL on criteria [1, 2]
  - [ ] (1) train: rms_bearing >= 10% below KITE-LAW: 42.842 vs 45.474 deg (-5.8%); paired diff -2.632 deg, 95% CI [-4.039, -1.284], n 191 (9 seeds excluded: collision, failed start or failed episode)
  - [ ] (1) train: rms_bearing >= 10% below KITE-LAW-CMA: 42.821 vs 34.430 deg (+24.4%); paired diff +8.391 deg, 95% CI [+6.154, +10.808], n 190 (10 seeds excluded: collision, failed start or failed episode)
  - [ ] (1) train: rms_bearing >= 10% below PRED-PD: 42.683 vs 28.185 deg (+51.4%); paired diff +14.499 deg, 95% CI [+11.346, +17.925], n 190 (10 seeds excluded: collision, failed start or failed episode)
  - [ ] (1) train: rms_bearing >= 10% below FLY-YAW-PREV@blind: 42.644 vs 46.293 deg (-7.9%); paired diff -3.648 deg, 95% CI [-5.737, -1.879], n 192 (8 seeds excluded: collision, failed start or failed episode)
  - [x] (2) train: losses non-inferior to KITE-LAW: paired diff +0.052, 95% CI [-0.031, +0.141], n 191 (9 seeds excluded: collision, failed start or failed episode); upper CI must be <= 0.215
  - [x] (2) train: time_lost_s non-inferior to KITE-LAW: paired diff -1.462, 95% CI [-2.840, -0.180], n 191 (9 seeds excluded: collision, failed start or failed episode); upper CI must be <= 2.787
  - [x] (2) train: rms_range_err_m non-inferior to KITE-LAW: paired diff -0.290, 95% CI [-0.706, +0.102], n 191 (9 seeds excluded: collision, failed start or failed episode); upper CI must be <= 0.831
  - [ ] (2) train: too_close_s non-inferior to KITE-LAW: paired diff +0.098, 95% CI [+0.007, +0.202], n 191 (9 seeds excluded: collision, failed start or failed episode); upper CI must be <= 0.200
  - [x] (2) train: pitch_exits non-inferior to KITE-LAW: paired diff -0.016, 95% CI [-0.073, +0.037], n 191 (9 seeds excluded: collision, failed start or failed episode); upper CI must be <= 0.100
  - [x] (3) train: in_view lower CI >= -0.02: paired diff +0.026, 95% CI [+0.002, +0.051], n 191 (9 seeds excluded: collision, failed start or failed episode)
  - [x] (3) train: following lower CI >= -0.02: paired diff +0.017, 95% CI [-0.005, +0.039], n 191 (9 seeds excluded: collision, failed start or failed episode)
  - [x] (4) train: collisions seed by seed vs KITE-LAW: collides alone on 1 seeds, KITE-LAW alone on 1 (both 7, n 200; allowed KITE-LAW's + 2); 0 failed episodes
  - [x] (5) train: stick_change <= 2 x KITE-LAW: 0.932 vs 2 x 0.752 over 191 kept seeds
  - [x] (1) demo: rms_bearing >= 10% below KITE-LAW: 25.791 vs 28.799 deg (-10.4%); paired diff -3.009 deg, 95% CI [-4.732, -1.206], n 200 (0 seeds excluded: collision, failed start or failed episode)
  - [ ] (1) demo: rms_bearing >= 10% below KITE-LAW-CMA: 25.791 vs 18.857 deg (+36.8%); paired diff +6.934 deg, 95% CI [+4.854, +9.239], n 200 (0 seeds excluded: collision, failed start or failed episode)
  - [ ] (1) demo: rms_bearing >= 10% below PRED-PD: 25.677 vs 18.554 deg (+38.4%); paired diff +7.123 deg, 95% CI [+4.819, +9.734], n 199 (1 seeds excluded: collision, failed start or failed episode)
  - [x] (1) demo: rms_bearing >= 10% below FLY-YAW-PREV@blind: 25.791 vs 30.758 deg (-16.2%); paired diff -4.968 deg, 95% CI [-6.842, -3.260], n 200 (0 seeds excluded: collision, failed start or failed episode)
  - [x] (2) demo: losses non-inferior to KITE-LAW: paired diff +0.015, 95% CI [-0.045, +0.075], n 200 (0 seeds excluded: collision, failed start or failed episode); upper CI must be <= 0.235
  - [x] (2) demo: time_lost_s non-inferior to KITE-LAW: paired diff -0.804, 95% CI [-2.208, +0.592], n 200 (0 seeds excluded: collision, failed start or failed episode); upper CI must be <= 1.630
  - [x] (2) demo: rms_range_err_m non-inferior to KITE-LAW: paired diff -0.217, 95% CI [-0.456, +0.007], n 200 (0 seeds excluded: collision, failed start or failed episode); upper CI must be <= 0.322
  - [x] (2) demo: too_close_s non-inferior to KITE-LAW: paired diff +0.057, 95% CI [-0.039, +0.173], n 200 (0 seeds excluded: collision, failed start or failed episode); upper CI must be <= 0.200
  - [x] (2) demo: pitch_exits non-inferior to KITE-LAW: paired diff +0.035, 95% CI [-0.025, +0.100], n 200 (0 seeds excluded: collision, failed start or failed episode); upper CI must be <= 0.100
  - [x] (3) demo: in_view lower CI >= -0.02: paired diff +0.014, 95% CI [-0.012, +0.039], n 200 (0 seeds excluded: collision, failed start or failed episode)
  - [x] (3) demo: following lower CI >= -0.02: paired diff +0.018, 95% CI [-0.005, +0.041], n 200 (0 seeds excluded: collision, failed start or failed episode)
  - [x] (4) demo: collisions seed by seed vs KITE-LAW: collides alone on 0 seeds, KITE-LAW alone on 0 (both 0, n 200; allowed KITE-LAW's + 2); 0 failed episodes
  - [x] (5) demo: stick_change <= 2 x KITE-LAW: 1.047 vs 2 x 0.926 over 200 kept seeds
  - [x] (6) stress: rms_bearing <= 1.25 x KITE-LAW, collisions seed by seed: rms bearing 44.096 vs 1.25 x 45.712 over 184 kept seeds; collides alone on 3 seeds, KITE-LAW alone on 4 (both 9, n 200; allowed KITE-LAW's + 2); 0 failed episodes
  - [x] (6) fast: rms_bearing <= 1.25 x KITE-LAW, collisions seed by seed: rms bearing 42.557 vs 1.25 x 44.945 over 173 kept seeds; collides alone on 2 seeds, KITE-LAW alone on 1 (both 24, n 200; allowed KITE-LAW's + 2); 0 failed episodes
