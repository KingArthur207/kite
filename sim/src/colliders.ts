// ?colliders=1: draw what the walker collides with, on screen only. Red: wall segments (building faces,
// curbs, stoops, street props). Yellow: actor boxes (cars, people). Cyan: the walker's collision circle.
// Flat ribbons drawn on top of everything, so a trunk box shows through its crown.
import * as THREE from "three";

const Y = 0.3;   // m: line height, above the curb
const W = 0.05;  // m: half width of the drawn lines (1 px GL lines vanish in screenshots)
const mat = (color: number) => new THREE.MeshBasicMaterial({ color, depthTest: false, transparent: true, side: THREE.DoubleSide });
/** Flat ribbons along segments [x1, n1, x2, n2] as one geometry. */
function ribbons(segs: number[][]) {
  const pos: number[] = [];
  for (const [x1, n1, x2, n2] of segs) {
    const L = Math.hypot(x2 - x1, n2 - n1) || 1, ox = (-(n2 - n1) / L) * W, on = ((x2 - x1) / L) * W;
    const q = [[x1 + ox, n1 + on], [x2 + ox, n2 + on], [x2 - ox, n2 - on], [x1 - ox, n1 - on]];
    for (const i of [0, 1, 2, 0, 2, 3]) pos.push(q[i][0], Y, -q[i][1]);
  }
  return new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
}

export function colliderOverlay(scene: THREE.Scene, walls: number[][], radius: number) {
  const group = new THREE.Group();
  group.renderOrder = 999;
  group.add(new THREE.Mesh(ribbons(walls), mat(0xff2020)));
  const boxes = new THREE.Mesh(new THREE.BufferGeometry(), mat(0xffd000));
  const ring = new THREE.Mesh(new THREE.RingGeometry(radius - W, radius + W, 32).rotateX(-Math.PI / 2).translate(0, Y, 0), mat(0x00e5ff));
  group.add(boxes, ring);
  for (const o of group.children) o.renderOrder = 999;
  group.visible = false;
  scene.add(group);
  return {
    group,
    update(solids: THREE.Box3[], walker: { x: number; n: number }) {
      const segs: number[][] = [];
      for (const b of solids) {
        const c = [[b.min.x, -b.min.z], [b.max.x, -b.min.z], [b.max.x, -b.max.z], [b.min.x, -b.max.z]];
        c.forEach((q, i) => segs.push([...q, ...c[(i + 1) % 4]]));
      }
      boxes.geometry.dispose();
      boxes.geometry = ribbons(segs);
      ring.position.set(walker.x, 0, -walker.n);
    },
  };
}
