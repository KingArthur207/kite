# kite/flybrain: the fruit fly yaw controller run/app.py flies with

`kite.follow_behind.FlyYaw` loads `flyfollow.steer.FlySteer` from here: a 1,446-neuron leaky integrate-and-fire
pursuit circuit from the fruit fly connectome (MaleCNS) that turns the person's bearing into the Tello's yaw stick.
It decides only the yaw stick; distance, height, the lost-person logic and the SafetyGovernor stay in
`kite/follow_behind.py`.

| Path | What |
|---|---|
| `flyfollow/` | `steer.FlySteer` and what it imports: the latency-compensating box filter, the target encoder, the controller factory, the pursuit decoder |
| `flydrones/` | the LIF network and connectome loader (FlyDrones, MIT: `flydrones/LICENSE`) |
| `data/brains/pursuit_core1.npz` | the pursuit subgraph |
| `data/brains/trained/FLY-YAW_smooth2_best.json` | smooth2, the weights it flies with: trained, then fine-tuned twice for a smooth stick |
| `data/brains/init/FLY-CMA__pursuit_core1.json` | the readout's hand-calibration init the controller factory reads |
| `configs/controllers.yaml` | controller defaults and parameter bounds |

Imports are package-absolute (`kite.flybrain.flyfollow...`, `kite.flybrain.flydrones...`). `FLYFOLLOW_DATA` / `FLYFOLLOW_CONFIGS` override the data and config folders.
Output conditioning as flown: no low-pass, deadband 4, hysteresis 3, slew 300 per second, clamp 40; 50 ms brain ticks,
about 3 ms each on a laptop CPU.

Kite has two trained fly brains on the same circuit:

| Brain | File | `--yaw` | Tested |
|---|---|---|---|
| smooth2 | `kite/flybrain/data/brains/trained/FLY-YAW_smooth2_best.json` | `fly` (default) | flown |
| retrained (kite1) | `kite/fly/data/fly_yaw.json`, fine-tuned from smooth in kite's follow simulator | `retrained` | simulation |
