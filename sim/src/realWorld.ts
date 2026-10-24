// ?world=real: Moss St built in Blender with facades rectified from Google Street View
// (scripts/streetview_facades.py, sim/blender/build_real_moss.py). The GLB is only the look; moss.json still gives
// collision, lanes, route, spawn and traffic through buildOsmWorld. Street View-derived pixels are never committed:
// the GLB lives in the gitignored public/local/, and without it the textured OSM street stays.
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { buildOsmWorld } from "./osmWorld";

export const REAL_GLB = "local/moss_real.glb";

export function buildRealWorld(opts: Parameters<typeof buildOsmWorld>[0] = {}) {
  const world = buildOsmWorld(opts);
  const flag = window as unknown as { __kiteReal: string };   // screenshot/YOLO scripts wait while "loading"
  flag.__kiteReal = "loading";
  new GLTFLoader().load(REAL_GLB, (gltf) => {
    gltf.scene.traverse((o) => { if ((o as THREE.Mesh).isMesh) o.castShadow = o.receiveShadow = true; });
    world.osmMeshes.removeFromParent();
    world.scene.add(gltf.scene);
    flag.__kiteReal = "ready";
    console.info(`[realWorld] ${REAL_GLB} loaded`);
  }, undefined, () => {
    // Vite answers a missing file with index.html, so a missing GLB lands here too.
    console.warn(`[realWorld] no ${REAL_GLB} (built locally, see sim/README.md); showing the textured OSM street`);
    flag.__kiteReal = "missing";
  });
  return world;
}
