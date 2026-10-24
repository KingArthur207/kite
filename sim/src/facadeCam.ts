// Dev page (facadecam.html?look=2, not in the build): the Moss world with no actors and a free camera, for
// comparing building fronts with Street View (scripts/facade_layouts.py compare). Pose in sim metres:
//   await kiteCam.ready; await kiteCam.shot({ pos: [x, n, y], look: [x, n, y], fov, w, h }) -> PNG data URL
import * as THREE from "three";
import { loadBuildingSheet, createPost, look2 } from "./look2";

const ready = (async () => {
  await loadBuildingSheet();
  const { buildOsmWorld } = await import("./osmWorld");
  const world = buildOsmWorld({ actors: false });
  const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  world.tune?.(renderer);
  renderer.setPixelRatio(1);
  document.body.appendChild(renderer.domElement);
  const post = look2 ? await createPost(renderer, world.scene) : null;
  const cam = new THREE.PerspectiveCamera(60, 4 / 3, 0.1, 2000);
  return { world, renderer, post, cam };
})();

(window as unknown as { kiteCam: unknown }).kiteCam = {
  ready: ready.then(() => true),
  async shot(o: { pos: number[]; look: number[]; fov?: number; w?: number; h?: number }) {
    const { world, renderer, post, cam } = await ready;
    const w = o.w ?? 1280, h = o.h ?? 960;
    renderer.setSize(w, h, false);
    renderer.setViewport(0, 0, w, h);
    cam.aspect = w / h; cam.fov = o.fov ?? 60; cam.updateProjectionMatrix();
    cam.position.set(o.pos[0], o.pos[2], -o.pos[1]);
    cam.lookAt(o.look[0], o.look[2], -o.look[1]);
    world.update(0);
    if (post) post(cam); else renderer.render(world.scene, cam);
    return renderer.domElement.toDataURL("image/png");
  },
};
