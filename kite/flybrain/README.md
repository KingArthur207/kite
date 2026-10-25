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
| `data/brains/trained/FLY-YAW_smooth2_best.json` | the trained weights it flew with: trained, then fine-tuned for a smooth stick |
| `data/brains/init/FLY-CMA__pursuit_core1.json` | the readout's hand-calibration init the controller factory reads |
| `configs/controllers.yaml` | controller defaults and parameter bounds |

The modules are vendored unchanged apart from their imports (`kite.flybrain.flyfollow...`,
`kite.flybrain.flydrones...`). `FLYFOLLOW_DATA` / `FLYFOLLOW_CONFIGS` override the data and config folders.
Output conditioning as flown: no low-pass, deadband 4, hysteresis 3, slew 300 per second, clamp 40; 50 ms brain ticks,
about 3 ms each on a laptop CPU.
