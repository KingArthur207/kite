// Placeholder world. The Gaussian splat replaces this file; keep the same exports:
// a scene, update(dt) for moving things, and solids() for collision boxes.
import * as THREE from "three";

const mat = (color: number) => new THREE.MeshLambertMaterial({ color });

interface Car { mesh: THREE.Mesh; axis: "x" | "z"; speed: number; min: number; max: number }
interface Ped { mesh: THREE.Mesh; z0: number; amp: number; phase: number }

export function buildWorld() {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x9ec9ff);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x666655, 1.6));
  const sun = new THREE.DirectionalLight(0xffffff, 1.5);
  sun.position.set(30, 60, 20);
  scene.add(sun);

  const flat = (w: number, d: number, x: number, z: number, y: number, color: number) => {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(w, d), mat(color));
    m.rotation.x = -Math.PI / 2;
    m.position.set(x, y, z);
    scene.add(m);
  };

  flat(400, 400, 0, 0, 0, 0x6b8f4e);                 // grass
  flat(4, 200, 0, -40, 0.01, 0xbdbdb5);               // sidewalk, north-south, x in [-2, 2]
  flat(7, 200, 5.7, -40, 0.01, 0x3a3a3c);             // main road, x in [2.2, 9.2]
  flat(200, 7, 0, -36.5, 0.012, 0x3a3a3c);            // cross street, z in [-40, -33]
  for (let i = 0; i < 7; i++) flat(4, 0.5, 0, -33.5 - i, 0.02, 0xf2f2f2); // crosswalk stripes

  const solidMeshes: THREE.Mesh[] = [];
  const box = (w: number, h: number, d: number, x: number, z: number, color: number) => {
    const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat(color));
    m.position.set(x, h / 2, z);
    scene.add(m);
    solidMeshes.push(m);
    return m;
  };

  // Buildings along the west side of the sidewalk, and past the cross street.
  box(8, 12, 10, -7.5, -5, 0xc9a27e);
  box(8, 9, 8, -7.5, -16, 0xa7b4c2);
  box(8, 15, 9, -7.5, -27, 0xd8cfc4);
  box(8, 10, 12, -7.5, -50, 0xb98b73);
  box(10, 18, 12, 16, -20, 0x8f9aa8);
  box(10, 11, 12, 16, -55, 0xcfc39b);
  box(0.6, 1.0, 0.6, -1.2, -12, 0x2e5e3a);           // bin on the sidewalk

  const cars: Car[] = [];
  const car = (color: number, axis: "x" | "z", lane: number, start: number, speed: number, min: number, max: number) => {
    const along = axis === "z";
    const m = box(along ? 1.8 : 4.4, 1.5, along ? 4.4 : 1.8, along ? lane : start, along ? start : lane, color);
    cars.push({ mesh: m, axis, speed, min, max });
  };
  car(0xb22222, "z", 4, 10, -8, -90, 30);            // northbound lane
  car(0x2255aa, "z", 4, -40, -8, -90, 30);
  car(0xeeeeee, "z", 7.5, -60, 7, -90, 30);           // southbound lane
  car(0x222222, "x", -35, -40, 6, -60, 60);           // cross street, eastbound
  car(0xd4a017, "x", -38.5, 30, -6, -60, 60);         // westbound

  const peds: Ped[] = [];
  const ped = (x: number, z0: number, amp: number, phase: number, color: number) => {
    const m = new THREE.Mesh(new THREE.CapsuleGeometry(0.25, 1.2, 4, 8), mat(color));
    m.position.set(x, 0.85, z0);
    scene.add(m);
    solidMeshes.push(m);
    peds.push({ mesh: m, z0, amp, phase });
  };
  ped(1.0, -20, 4, 0, 0x7a3e9d);
  ped(-1.0, -31, 1, 1, 0x3e7a5d);
  ped(0.6, -42, 2, 2, 0x9d6a3e);
  ped(-0.8, -43, 0, 0, 0x555555);

  let t = 0;
  function update(dt: number) {
    t += dt;
    for (const c of cars) {
      const p = c.mesh.position;
      let v = p[c.axis] + c.speed * dt;
      if (v < c.min) v = c.max;
      if (v > c.max) v = c.min;
      p[c.axis] = v;
    }
    for (const p of peds) p.mesh.position.z = p.z0 + p.amp * Math.sin(t * 0.3 + p.phase);
  }

  const solids = () => solidMeshes.map((m) => new THREE.Box3().setFromObject(m));
  return { scene, update, solids };
}
