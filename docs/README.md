# Kite docs

Start with the [README](../README.md). It covers what Kite is, how it works, each prize track and how to run it. The pages here go deeper.

| Doc | What it covers |
|---|---|
| [operations.md](operations.md) | Two laptops over a Thunderbolt bridge, flying, recording and replaying the Tello, frame sources for detectors, and walker GPS |
| [network.md](network.md) | Two laptops over Thunderbolt, venue Wi-Fi, the offline SpacetimeDB fallback, and how `scripts/netcheck.py` checks a setup |
| [research/facts.md](research/facts.md) | The numbers behind the pitch (guide dog cost and wait, how many people are blind, what a cane can reach), with sources and how far to trust each one |
| [fly-yaw-results.md](fly-yaw-results.md) | The simulation evaluation of the opt-in `kite.fly` yaw sources (the fruit-fly circuit and the predictive PD) against follow mode's own yaw law, and why neither is the default |
| [follow-predict-results.md](follow-predict-results.md) | Follow mode's opt-in prediction mode (`--follow-predict`) and pitch compensation run in the same simulator against the reactive law at 1.0, 1.2 and 1.5 m behind, and why both stay off |
| [google-permission.md](google-permission.md) | The record of Google's permission to use 3D Tiles and Street View on one block of Moss St, and how the code keeps to it |
| [`../SECURITY.md`](../SECURITY.md) | How Kite handles keys, inputs and its dependency scans |

Each part of the code has its own README: [earpiece](../earpiece/README.md), [dashboard](../dashboard/README.md), [sim](../sim/README.md) and [site](../site/README.md).

[`internal/`](internal/) holds the team's working notes from the hackathon: the spec, status, pitch and Devpost drafts, and the shoot, demo and video runbooks. They are kept for the record and are not needed to understand or run Kite.
