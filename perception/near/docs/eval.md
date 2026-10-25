### Synthetic stills (509 renders, Logitech C270 spec intrinsics)

| variant | accuracy | n | P1 missed (said P3/none) | false P1 |
|---|---|---|---|---|
| fused | 85.5 % | 509 | 3 | 10 |
| depth raw | 77.0 % | 509 | 3 | 27 |
| depth anchored | 84.1 % | 509 | 3 | 10 |
| size only | 67.4 % | 509 | 41 | 11 |
| ground only | 57.4 % | 509 | 54 | 0 |
| scrollsafe | 33.6 % | 509 | 63 | 153 |


Per-cue error against the true distance (median and 90th percentile of |error|, bias = median signed error, coverage = share of renders where the cue applied):

| cue | coverage | median | p90 | bias |
|---|---|---|---|---|
| depth | 100 % | 7.1 % | 19.3 % | -1.3 % |
| fused | 100 % | 5.2 % | 14.7 % | -0.8 % |
| ground | 52 % | 3.9 % | 8.8 % | -0.6 % |
| size_h | 37 % | 6.8 % | 23.7 % | +1.1 % |
| size_w | 66 % | 11.7 % | 37.8 % | -3.6 % |

Confusion, fused:

| truth \ predicted | P1 | P2 | P3 | none |
|---|---|---|---|---|
| P1 | 122 | 5 | 0 | 3 |
| P2 | 5 | 104 | 0 | 0 |
| P3 | 0 | 5 | 101 | 0 |
| none | 5 | 32 | 19 | 108 |

Confusion, ScrollSafe rule:

| truth \ predicted | P1 | P2 | P3 | none |
|---|---|---|---|---|
| P1 | 61 | 6 | 3 | 60 |
| P2 | 76 | 9 | 1 | 23 |
| P3 | 44 | 50 | 12 | 0 |
| none | 33 | 32 | 10 | 89 |

### Synthetic sequences (10 Hz, 5 seeds each)

| scenario | variant | frame accuracy | P1 onset vs truth | time to impact at P1 | priority changes (truth) | TTC error |
|---|---|---|---|---|---|---|
| walk to a pole | fused | 95 % | -0.2 s | 1.6 s | 2.0 (2) | 31.1 % |
| walk to a pole | no looming | 98 % | +0.0 s | 1.4 s | 2.0 (2) | 36.9 % |
| walk to a pole | no hysteresis | 97 % | -0.2 s | 1.6 s | 2.4 (2) | 31.1 % |
| walk to a pole | scrollsafe | 10 % | -6.9 s | 8.3 s | 1.0 (2) | - |
| oncoming person | fused | 91 % | +0.3 s | 1.2 s | 2.0 (2) | 10.6 % |
| oncoming person | no looming | 80 % | +0.8 s | 0.7 s | 2.0 (2) | 49.9 % |
| oncoming person | no hysteresis | 94 % | +0.3 s | 1.2 s | 2.0 (2) | 10.6 % |
| oncoming person | scrollsafe | 24 % | -1.8 s | 3.3 s | 1.8 (2) | - |
| fast cyclist head-on | fused | 91 % | +0.1 s | 1.4 s | 1.0 (1) | 11.3 % |
| fast cyclist head-on | no looming | 76 % | +0.7 s | 0.8 s | 1.0 (1) | 25.5 % |
| fast cyclist head-on | no hysteresis | 94 % | +0.1 s | 1.4 s | 1.0 (1) | 11.3 % |
| fast cyclist head-on | scrollsafe | 71 % | +0.8 s | 0.7 s | 2.0 (1) | - |
| person crossing at 3.5 m | fused | 84 % | - | - | 1.0 (2) | - |
| person crossing at 3.5 m | no looming | 84 % | - | - | 1.0 (2) | - |
| person crossing at 3.5 m | no hysteresis | 91 % | - | - | 1.0 (2) | - |
| person crossing at 3.5 m | scrollsafe | 73 % | - | - | 1.0 (2) | - |
| person standing at 4.6 m | fused | 99 % | - | - | 0.0 (0) | - |
| person standing at 4.6 m | no looming | 99 % | - | - | 0.0 (0) | - |
| person standing at 4.6 m | no hysteresis | 100 % | - | - | 0.0 (0) | - |
| person standing at 4.6 m | scrollsafe | 0 % | - | - | 0.0 (0) | - |
| pole standing at 1.8 m | fused | 98 % | +0.1 s | - | 0.2 (0) | - |
| pole standing at 1.8 m | no looming | 98 % | +0.1 s | - | 0.2 (0) | - |
| pole standing at 1.8 m | no hysteresis | 100 % | +0.0 s | - | 0.2 (0) | - |
| pole standing at 1.8 m | scrollsafe | 100 % | +0.0 s | - | 0.0 (0) | - |

### Lens distortion (k1 -0.12, k2 0.05; ground cue, whole object in view, 3 m and beyond)

| intrinsics used | median error | p90 |
|---|---|---|
| Logitech C270, calibrated (undistorts) | 2.8 % | 7.2 % |
| Logitech C270, spec (ignores distortion) | 2.8 % | 6.8 % |
| Logitech C920, calibrated (undistorts) | 2.8 % | 8.3 % |
| Logitech C920, spec (ignores distortion) | 2.8 % | 8.0 % |

### Real stills (hand labels, perception/near/near_labels.json)

| variant | accuracy | n | P1 missed (said P3/none) | false P1 |
|---|---|---|---|---|
| fused | 92.7 % | 41 | 0 | 0 |
| depth raw | 85.4 % | 41 | 0 | 0 |
| depth anchored | 87.8 % | 41 | 0 | 0 |
| size only | 90.2 % | 41 | 0 | 0 |
| ground only | 85.4 % | 41 | 0 | 0 |
| scrollsafe | 73.2 % | 41 | 0 | 5 |


Confusion, fused (labels with one acceptable answer):

| truth \ predicted | P1 | P2 | P3 | none |
|---|---|---|---|---|
| P2 | 0 | 4 | 0 | 1 |
| P3 | 0 | 0 | 4 | 1 |
| none | 0 | 0 | 0 | 20 |

| image | object | label | fused | distance | cues | ScrollSafe |
|---|---|---|---|---|---|---|
| scooter.jpg | e_scooter | P2 | none | - |  | none |
| scooter.jpg | pole | P3 | none | - |  | none |
| scooter.jpg | pole | none | none | - |  | none |
| scooter.jpg | vehicle | none | none | 38.46 m | size_h 42.16, size_w 31.25, depth 30.42 | P3 |
| doorway.jpg | person | P2/none | P2 | 3.37 m | size_h 3.0, size_w 2.09, ground 3.63, depth 4.17 | P1 |
| doorway.jpg | bike | none | none | - |  | none |
| oncoming.jpg | person | P2 | P2 | 2.88 m | size_w 2.71, depth 2.94 | P1 |
| oncoming.jpg | person | none | none | 3.31 m | size_h 3.69, ground 3.08, depth 3.11 | none |
| oncoming.jpg | person | none | none | 5.2 m | size_h 5.65, size_w 3.97, ground 4.74, depth 5.52 | none |
| oncoming.jpg | person | none | none | 5.34 m | size_h 5.76, size_w 5.21, ground 4.88, depth 5.29 | P2 |
| cyclist.jpg | bike | P2 | P2 | 4.67 m | size_h 4.67, size_w 2.27, ground 6.37, depth 7.47 | P1 |
| cyclist.jpg | person | P2/none | P2 | 4.67 m | size_h 4.67, size_w 2.27, ground 6.37, depth 7.47 | P1 |
| cyclist.jpg | pole | none | none | - |  | none |
| car_turning.jpg | vehicle | P2/P3 | P2 | 6.14 m | size_h 6.35, size_w 5.03, ground 5.87, depth 6.41 | P2 |
| car_turning.jpg | pole | none | none | - |  | none |
| street_dressing_poles.jpg | pole | none/P3 | none | - |  | none |
| street_dressing_poles.jpg | vehicle | P2/P3 | P3 | 6.84 m | size_h 9.54, size_w 11.61, ground 6.76, depth 6.77 | P2 |
| street_dressing_poles.jpg | person | P3 | P3 | 36.57 m | size_h 40.58, size_w 21.57, ground 22.41, depth 15.93 | P3 |
| street_dressing_sw_sidewalk.jpg | tree | P2/none | P2 | - |  | none |
| street_dressing_sw_sidewalk.jpg | fire_hydrant | P2 | P2 | - |  | none |
| street_dressing_sw_sidewalk.jpg | person | none | none | 16.46 m | size_h 16.99, size_w 12.14, ground 9.55, depth 9.11 | none |
| street_dressing_sw_sidewalk.jpg | vehicle | none | none | 11.6 m | size_h 18.69, size_w 18.27, ground 11.41, depth 11.47 | none |
| street_dressing_howard.jpg | trash_can | P2/none | P3 | - |  | none |
| street_dressing_howard.jpg | tree | none/P3 | P3 | - |  | none |
| street_dressing_howard.jpg | vehicle | none | none | 19.97 m | size_h 20.75, ground 14.89, depth 14.1 | none |
| street_dressing_corner_se.jpg | person | P2 | P2 | 3.57 m | size_w 3.35, depth 3.64 | P1 |
| street_dressing_corner_se.jpg | vehicle | P2/none | P2 | 4.45 m | size_h 8.03, size_w 8.82, ground 4.44, depth 4.34 | P2 |
| street_dressing_corner_se.jpg | person | none | none | 25.83 m | size_h 26.88, size_w 17.95, ground 14.93, depth 12.99 | none |
| osm_facades_howard.jpg | tree | none | none | - |  | none |
| osm_facades_howard.jpg | vehicle | none | none | 46.78 m | size_h 47.75, size_w 50.43, ground 33.76, depth 25.15 | P3 |
| osm_facades_corner_chest.jpg | person | none | none | 8.3 m | size_h 8.34, size_w 6.74, ground 8.73, depth 8.32 | none |
| osm_facades_corner_chest.jpg | curb | none/P3 | none | - |  | none |
| osm_facades_spawn.jpg | vehicle | none | none | 4.39 m | size_h 5.58, ground 4.05, depth 4.08 | none |
| osm_facades_spawn.jpg | person | P3 | P3 | 35.61 m | size_h 36.84, size_w 35.29, ground 22.64, depth 19.41 | P3 |
| osm_facades_spawn.jpg | person | none | none | - |  | none |
| street_dressing_spawn.jpg | vehicle | none | none | - |  | none |
| street_dressing_spawn.jpg | person | P3 | P3 | 36.39 m | size_h 37.87, size_w 26.85, ground 18.32, depth 57.3 | P3 |
| street_dressing_spawn.jpg | fire_hydrant | none | none | - |  | none |
| hybrid_capture_osm_chest.jpg | person | P3 | P3 | 11.39 m | size_h 11.84, size_w 9.76, ground 11.49, depth 10.66 | P3 |
| hybrid_capture_osm_chest.jpg | vehicle | none | none | 15.1 m | size_h 23.94, ground 14.88, depth 13.47 | none |
| hybrid_capture_osm_chest.jpg | curb | P3/none | none | - |  | none |

### Real clips (12 Hz; most urgent priority per frame vs the key object's label)

| variant | accuracy | n | P1 missed (said P3/none) | false P1 |
|---|---|---|---|---|
| fused | 77.0 % | 365 | 2 | 4 |
| depth raw | 36.4 % | 365 | 27 | 0 |
| depth anchored | 64.9 % | 365 | 27 | 0 |
| size only | 67.9 % | 365 | 27 | 37 |
| ground only | 49.3 % | 365 | 27 | 0 |
| scrollsafe | 35.9 % | 365 | 27 | 140 |


| clip | variant | frame accuracy | P1 onset (label) | changes | timeline (- = clear) |
|---|---|---|---|---|---|
| doorway | fused | 100 % | 3.75 s (3.8 s) | 2 | `-222222222222222222222222222222222222222222221111111111111111111111111111` |
| doorway | depth raw | 57 % | - (3.8 s) | 2 | `-----------33332222222222222222222222222222222222222222222222222222222222` |
| doorway | depth anchored | 78 % | 4.92 s (3.8 s) | 3 | `-222222222222222222222222222222222222222222222222222222222211111111111222` |
| doorway | size only | 68 % | 1.33 s (3.8 s) | 2 | `-222222222222222111111111111111111111111111111111111111111111111111111111` |
| doorway | ground only | 63 % | - (3.8 s) | 1 | `-222222222222222222222222222222222222222222222222222222222222222222222222` |
| doorway | scrollsafe | 47 % | 0.0 s (3.8 s) | 0 | `1111111111111111111111111111111111111111111111111111111111111111111111111` |
| oncoming | fused | 96 % | 2.5 s (3.4 s) | 2 | `-222222222222222222222222222221111111111111111111111111111111111111111111` |
| oncoming | depth raw | 55 % | - (3.4 s) | 1 | `-222222222222222222222222222222222222222222222222222222222222222222222222` |
| oncoming | depth anchored | 55 % | - (3.4 s) | 1 | `-222222222222222222222222222222222222222222222222222222222222222222222222` |
| oncoming | size only | 80 % | 1.5 s (3.4 s) | 2 | `-222222222222222221111111111111111111111111111111111111111111111111111111` |
| oncoming | ground only | 55 % | - (3.4 s) | 1 | `-222222222222222222222222222222222222222222222222222222222222222222222222` |
| oncoming | scrollsafe | 56 % | 0.0 s (3.4 s) | 0 | `1111111111111111111111111111111111111111111111111111111111111111111111111` |
| cyclist | fused | 99 % | - (-) | 1 | `-222222222222222222222222222222222222222222222222222222222222222222222222` |
| cyclist | depth raw | 0 % | - (-) | 1 | `-333333333333333333333333333333333333333333333333333333333333333333333333` |
| cyclist | depth anchored | 99 % | - (-) | 1 | `-222222222222222222222222222222222222222222222222222222222222222222222222` |
| cyclist | size only | 99 % | - (-) | 1 | `-222222222222222222222222222222222222222222222222222222222222222222222222` |
| cyclist | ground only | 60 % | - (-) | 2 | `-333333333333333333333333333322222222222222222222222222222222222222222222` |
| cyclist | scrollsafe | 0 % | 0.0 s (-) | 1 | `1111111111111111111111111111111111111111111111111111111111111111111113333` |
| scooter | fused | 0 % | - (3.8 s) | 2 | `-------------------------------333333333333333332222222222222222222222222` |
| scooter | depth raw | 0 % | - (3.8 s) | 2 | `-------------------------------------------------333333333---------------` |
| scooter | depth anchored | 0 % | - (3.8 s) | 3 | `-------------------------------------------------333333333333333-------33` |
| scooter | size only | 0 % | - (3.8 s) | 0 | `-------------------------------------------------------------------------` |
| scooter | ground only | 0 % | - (3.8 s) | 1 | `-------------------------------------------------333333333333333333333333` |
| scooter | scrollsafe | 0 % | - (3.8 s) | 0 | `3333333333333333333333333333333333333333333333333333333333333333333333333` |
| car_turning | fused | 90 % | 3.25 s (-) | 6 | `-22222222---33333333333333333333333333311111111111111222222222222--------` |
| car_turning | depth raw | 70 % | - (-) | 4 | `-33333333---33333333333333333333333333333333333333333333333333-----------` |
| car_turning | depth anchored | 93 % | - (-) | 5 | `-22222222---33333333333333333333333333332222222222222222222222222--------` |
| car_turning | size only | 93 % | - (-) | 5 | `-22222222---33333333333333333333333333332222222222222222222222222--------` |
| car_turning | ground only | 68 % | - (-) | 4 | `-33333333---333333333333333333333333333333333333333333333333333----------` |
| car_turning | scrollsafe | 77 % | 4.0 s (-) | 7 | `222222222-3333333333333333322222222222222222222211112222-----333333333333` |
