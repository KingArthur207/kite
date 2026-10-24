// ?drone=sim: the drone camera is a body with Tello-like dynamics, flown by rc commands over the relay
// (kite/drone/sim.py). Without it the drone camera stays rigidly 0.3 m over the walker's head, 1 m behind.
//
// Model (numbers from spec section 3 and kite-notes-02 where measured, otherwise marked as guesses):
//   - rc sticks -100..100 set a body-frame velocity target: 0.96 m/s forward/sideways at full stick (measured
//     at stick 30, extrapolated), MAX_VZ_MS up/down (guess); velocity follows with a first-order lag TAU_S.
//   - yaw stick sets a yaw-rate target (MAX_YAW_DEG_S at 100, guess) with its own lag, clamped to that rate.
//   - altitude hold: ud 0 keeps the height. The body tilts with acceleration (pitch/roll = atan(a / g)) and the
//     fixed camera tilts with it, like the real one.
//   - wind: a slow random drift (Ornstein-Uhlenbeck, WIND_SIGMA_MS), seeded by ?seed.
//   - takeoff climbs to TAKEOFF_ALT_M, land descends; auto-land AUTOLAND_S after the last command (Tello).
//   - battery is fake: 100 % at takeoff, about 13 %/min of flight.
// Limits that are the governor's job (max_rc 50, altitude band, battery) live in kite/drone/safety.py, not here.
import * as THREE from "three";

export const MAX_V_MS = 0.96;
const MAX_VZ_MS = 0.7;            // guess
const MAX_YAW_DEG_S = 100;        // guess (djitellopy users report ~90-110 deg/s at full yaw stick)
const TAU_S = 0.4;                // velocity lag, first order (follow.py models 0.4 s too)
const YAW_TAU_S = 0.25;
const WIND_SIGMA_MS = 0.05, WIND_TAU_S = 4;
const TAKEOFF_ALT_M = 0.8, TAKEOFF_SPOOL_S = 1.5, CLIMB_MS = 0.5;
const AUTOLAND_S = 15;
const BATTERY_PCT_PER_S = 13 / 60;
const G = 9.81;

export type DroneState = "ground" | "takeoff" | "flying" | "landing";
export interface DroneCmd { cmd: string; lr?: number; fb?: number; ud?: number; yaw?: number }

function mulberry32(a: number) {
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const clampStick = (v: unknown) => Math.max(-100, Math.min(100, Number(v) || 0));

export function makeSimDrone(start: { x: number; n: number; heading: number }, seed = 1) {
  const rand = mulberry32(seed);
  const gauss = () => Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());
  const d = {
    x: start.x, n: start.n, alt: 0, heading: start.heading,
    ve: 0, vn: 0, vz: 0, yawRate: 0, pitch: 0, roll: 0,
    state: "ground" as DroneState, flightS: 0, battery: 100,
  };
  let rc = [0, 0, 0, 0];               // lr fb ud yaw, as last received
  let lastCmd = performance.now() / 1000, spool = 0;
  let windE = 0, windN = 0;
  let prevF = 0, prevR = 0;            // body-frame speed, for the tilt

  function command(c: DroneCmd) {
    lastCmd = performance.now() / 1000;
    if (c.cmd === "rc") rc = [c.lr, c.fb, c.ud, c.yaw].map(clampStick);
    else if (c.cmd === "stop") rc = [0, 0, 0, 0];
    else if (c.cmd === "takeoff" && d.state === "ground") { d.state = "takeoff"; spool = 0; d.battery = 100; d.flightS = 0; }
    else if (c.cmd === "land" && (d.state === "flying" || d.state === "takeoff")) d.state = "landing";
    else if (c.cmd === "emergency") { d.state = "ground"; d.alt = 0; }
    if (c.cmd !== "rc") rc = [0, 0, 0, 0];
  }

  function step(dt: number) {
    if (d.state === "ground") { d.ve = d.vn = d.vz = d.yawRate = d.pitch = d.roll = 0; return; }
    d.flightS += dt;
    d.battery = Math.max(0, 100 - BATTERY_PCT_PER_S * d.flightS);
    if (d.state === "flying" && performance.now() / 1000 - lastCmd > AUTOLAND_S) d.state = "landing";
    const h = THREE.MathUtils.degToRad(d.heading);
    const fwd = { e: Math.sin(h), n: Math.cos(h) }, right = { e: Math.cos(h), n: -Math.sin(h) };
    let [lr, fb, ud, yaw] = d.state === "flying" ? rc : [0, 0, 0, 0];
    if (d.state === "takeoff") {
      spool += dt;
      ud = spool < TAKEOFF_SPOOL_S ? 0 : (CLIMB_MS / MAX_VZ_MS) * 100;
      if (d.alt >= TAKEOFF_ALT_M) { d.state = "flying"; rc = [0, 0, 0, 0]; }
    } else if (d.state === "landing") {
      ud = -(CLIMB_MS / MAX_VZ_MS) * 100;
    }
    const tE = (fwd.e * fb + right.e * lr) * MAX_V_MS / 100, tN = (fwd.n * fb + right.n * lr) * MAX_V_MS / 100;
    const a = Math.min(1, dt / TAU_S);
    d.ve += (tE - d.ve) * a;
    d.vn += (tN - d.vn) * a;
    d.vz += ((ud * MAX_VZ_MS) / 100 - d.vz) * a;
    d.yawRate += ((yaw * MAX_YAW_DEG_S) / 100 - d.yawRate) * Math.min(1, dt / YAW_TAU_S);
    d.yawRate = Math.max(-MAX_YAW_DEG_S, Math.min(MAX_YAW_DEG_S, d.yawRate));
    // Wind drift: OU process per axis.
    const k = Math.min(1, dt / WIND_TAU_S), s = WIND_SIGMA_MS * Math.sqrt(2 * k);
    windE += -windE * k + s * gauss();
    windN += -windN * k + s * gauss();
    d.x += (d.ve + windE) * dt;
    d.n += (d.vn + windN) * dt;
    d.alt = Math.max(0, d.alt + d.vz * dt);
    d.heading = (((d.heading + d.yawRate * dt) % 360) + 360) % 360;
    if (d.state === "landing" && d.alt <= 0) { d.state = "ground"; rc = [0, 0, 0, 0]; }
    // Tilt from body-frame acceleration: nose down while speeding up forward, right side down speeding up right.
    const vF = d.ve * fwd.e + d.vn * fwd.n, vR = d.ve * right.e + d.vn * right.n;
    if (dt > 0) {
      d.pitch = -THREE.MathUtils.radToDeg(Math.atan((vF - prevF) / dt / G));
      d.roll = THREE.MathUtils.radToDeg(Math.atan((vR - prevR) / dt / G));
    }
    prevF = vF; prevR = vR;
  }

  /** The fixed forward camera, level with the body (so it tilts with it). */
  function placeCamera(cam: THREE.Camera) {
    cam.position.set(d.x, Math.max(0.1, d.alt), -d.n);
    cam.rotation.order = "YXZ";
    cam.rotation.set(THREE.MathUtils.degToRad(d.pitch), -THREE.MathUtils.degToRad(d.heading), -THREE.MathUtils.degToRad(d.roll));
  }

  /** Tello-like state packet (raw field names and units, see kite/record.py STATE_UNITS) plus sim truth. */
  function telemetry() {
    const h = THREE.MathUtils.degToRad(d.heading);
    const vF = d.ve * Math.sin(h) + d.vn * Math.cos(h), vR = d.ve * Math.cos(h) - d.vn * Math.sin(h);
    const yaw = d.heading > 180 ? d.heading - 360 : d.heading;
    return {
      h: Math.round(d.alt * 100), tof: Math.round(Math.max(0.1, d.alt) * 100), bat: Math.round(d.battery),
      yaw: Math.round(yaw), pitch: Math.round(d.pitch), roll: Math.round(d.roll),
      vgx: Math.round(vF * 10), vgy: Math.round(vR * 10), vgz: Math.round(-d.vz * 10), time: Math.round(d.flightS),
      state: d.state, x: +d.x.toFixed(3), y: +d.n.toFixed(3), alt: +d.alt.toFixed(3), heading: +d.heading.toFixed(2),
      rc: [...rc],
    };
  }

  return { d, command, step, placeCamera, telemetry };
}
