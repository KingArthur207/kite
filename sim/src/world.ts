// Placeholder world. The Gaussian splat replaces this file; keep the same exports:
// a scene, update(dt) for moving things, and solids() for collision boxes.
import * as THREE from "three";
import { spawnActors } from "./actors";

const mat = (color: number) => new THREE.MeshLambertMaterial({ color });

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

  const actors = spawnActors(scene); // glTF cars and people (actors.ts)

  function update(dt: number) {
    actors.update(dt);
  }

  const solids = () => [...solidMeshes.map((m) => new THREE.Box3().setFromObject(m)), ...actors.solids()];
  return { scene, update, solids };
}
