// Splat alignment: splat-local coordinates -> sim (three.js) world, from landmark pairs or an explicit
// transform. No Spark import, so `node sim/splat_align_check.ts` can run it.
import * as THREE from "three";

export type V3 = [number, number, number];
export interface Landmark { name?: string; splat: V3; sim: V3 }   // sim: metres east, north, up
export interface Align { landmarks?: Landmark[]; up?: V3; scale?: number; quaternion?: [number, number, number, number]; position?: V3 }

export const simToThree = ([e, n, u]: V3) => new THREE.Vector3(e, u, -n);
const centroid = (ps: THREE.Vector3[]) => ps.reduce((a, p) => a.add(p), new THREE.Vector3()).divideScalar(ps.length);

// Orthonormal frame from a direction and a second, non-parallel vector.
function basis(dir: THREE.Vector3, other: THREE.Vector3) {
  const e1 = dir.clone().normalize();
  const e3 = e1.clone().cross(other).normalize();
  const e2 = e3.clone().cross(e1);
  return new THREE.Matrix4().makeBasis(e1, e2, e3);
}

/** Splat-local -> three.js world (x east, y up, z south). */
export function solveAlign(a: Align): THREE.Matrix4 {
  const lm = a.landmarks ?? [];
  if (lm.length < 2) {
    const q = a.quaternion ? new THREE.Quaternion(...a.quaternion) : new THREE.Quaternion(1, 0, 0, 0);   // 180 deg about x
    return new THREE.Matrix4().compose(new THREE.Vector3(...(a.position ?? [0, 0, 0])), q, new THREE.Vector3().setScalar(a.scale ?? 1));
  }
  const src = lm.map((l) => new THREE.Vector3(...l.splat));
  const dst = lm.map((l) => simToThree(l.sim));
  // ponytail: rotation from the first 3 landmarks (or 2 plus up), not a least-squares (Horn) fit over all; residuals are logged
  // Two landmarks: the second axis is up (given in splat space, +y in sim), the first the horizontal part of 0 -> 1.
  const frame = (ps: THREE.Vector3[], up: THREE.Vector3) => {
    const d = ps[1].clone().sub(ps[0]);
    return ps.length >= 3 ? basis(d, ps[2].clone().sub(ps[0])) : basis(d.sub(up.clone().multiplyScalar(d.dot(up))), up);
  };
  const fs = frame(src, new THREE.Vector3(...(a.up ?? [0, -1, 0])).normalize());
  const fd = frame(dst, new THREE.Vector3(0, 1, 0));
  const rot = fd.multiply(fs.transpose());
  const cs = centroid(src), cd = centroid(dst);
  const spread = (ps: THREE.Vector3[], c: THREE.Vector3) => ps.reduce((s, p) => s + p.distanceTo(c), 0);
  const scale = spread(dst, cd) / spread(src, cs);
  const q = new THREE.Quaternion().setFromRotationMatrix(rot);
  const t = cd.clone().sub(cs.clone().applyQuaternion(q).multiplyScalar(scale));
  return new THREE.Matrix4().compose(t, q, new THREE.Vector3().setScalar(scale));
}
