// Self-check for the splat alignment solve: node sim/splat_align_check.ts (Node 22.18+, from the repo root or sim/).
// A known similarity transform (COLMAP-style y-down, yawed, scaled, moved) must come back from 3 landmarks
// and from 2 landmarks plus the up axis.
import * as THREE from "three";
import { solveAlign, simToThree, type V3 } from "./src/splatAlign.ts";

const truth = new THREE.Matrix4().compose(
  new THREE.Vector3(12, 0.4, -30),
  new THREE.Quaternion().setFromEuler(new THREE.Euler(Math.PI, 0.7, 0, "YXZ")),
  new THREE.Vector3().setScalar(0.37),
);
const inv = truth.clone().invert();
// Sim points (east, north, up): two zebra corners and a garage door top edge.
const sims: V3[] = [[3.1, 2.9, 0.15], [-2.6, -2.8, 0.15], [8.4, -6.0, 2.4]];
const splatOf = (s: V3) => simToThree(s).applyMatrix4(inv).toArray() as V3;
const landmarks = sims.map((sim) => ({ sim, splat: splatOf(sim) }));
const test: V3 = [20, -15, 1];

for (const [label, align] of [
  ["3 landmarks", { landmarks }],
  ["2 landmarks + up", { landmarks: landmarks.slice(0, 2), up: new THREE.Vector3(0, 1, 0).transformDirection(inv).toArray() as V3 }],
] as const) {
  const err = new THREE.Vector3(...splatOf(test)).applyMatrix4(solveAlign(align)).distanceTo(simToThree(test));
  console.log(`${label}: ${err.toExponential(1)} m off at a point 25 m away`);
  if (!(err < 1e-6)) throw new Error(`${label}: alignment solve is wrong`);
}
console.log("ok");
