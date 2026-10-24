// ?world=splat&src=<url>: a Gaussian splat of Moss St (scripts/splat/) drawn with Spark, over the OSM
// Moss colliders, lanes, route, spawn and glTF traffic. The OSM meshes stay in the scene for picking
// but are hidden; O shows them, to check the alignment.
//
// Alignment: <src minus extension>.align.json, or ?align=<url>. Either landmark pairs, solved as a
// similarity transform (scale, rotation incl. the up-axis fix, translation), or an explicit transform:
//   {"landmarks": [{"name": "zebra NE corner", "splat": [x, y, z], "sim": [east, north, up]}, ...]}
//   {"up": [0, -1, 0], "landmarks": [two pairs]}      2 pairs: splat-space up given, the solve adds yaw
//   {"scale": 1, "quaternion": [x, y, z, w], "position": [x, y, z]}   three.js frame (x east, y up, z south)
// With no file the splat sits at the origin, flipped from COLMAP's y-down. L (with ?debug=1) collects
// landmark pairs under the cursor (see pick below and sim/README.md, Splat).
import * as THREE from "three";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import { buildOsmWorld } from "./osmWorld";
import { solveAlign, simToThree, type Align, type Landmark, type V3 } from "./splatAlign";

async function loadAlign(url: string): Promise<Align> {
  try {
    const r = await fetch(url);
    if (r.ok && !(r.headers.get("content-type") ?? "").includes("html")) return await r.json();
  } catch (e) { console.warn(`[splat] ${url}:`, e); }
  console.warn(`[splat] no alignment at ${url}: splat at the origin, y flipped (see sim/README.md, Splat)`);
  return {};
}

export function buildSplatWorld(opts: { src: string; align?: string; walkers: () => { x: number; n: number }[] }) {
  const osm = buildOsmWorld({ walkers: opts.walkers });
  osm.osmMeshes.visible = false;
  const splat = new SplatMesh({ url: opts.src });
  splat.matrixAutoUpdate = false;
  splat.visible = false;      // until it is aligned
  osm.scene.add(splat);
  splat.castShadow = false;   // addSkyAndSun turns shadows on for everything added; splats in the 4096 px shadow pass cost ~50 fps
  const t0 = performance.now();
  const alignUrl = opts.align ?? opts.src.replace(/\.[^./]+$/, "") + ".align.json";
  void Promise.all([splat.initialized, loadAlign(alignUrl)]).then(([, a]) => {
    splat.matrix.copy(solveAlign(a));
    splat.matrixWorldNeedsUpdate = true;
    splat.visible = true;
    for (const l of a.landmarks ?? []) {
      const err = new THREE.Vector3(...l.splat).applyMatrix4(splat.matrix).distanceTo(simToThree(l.sim));
      console.info(`[splat] landmark ${l.name ?? ""}: ${err.toFixed(2)} m off after the solve`);
    }
    console.info(`[splat] ${splat.packedSplats.numSplats} splats from ${opts.src} in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
    (window as unknown as { __kiteSplat: unknown }).__kiteSplat = { numSplats: splat.packedSplats.numSplats, ready: true };
  }).catch((e) => console.error(`[splat] failed to load ${opts.src}:`, e));

  // One SparkRenderer on the screen renderer (it owns its sort and textures, so main.ts draws the capture
  // frames on the same context). Spark sizes splats by the whole drawing buffer; we draw letterboxed
  // and inset viewports, so size them by the current viewport instead.
  function attach(renderer: THREE.WebGLRenderer) {
    const spark = new SparkRenderer({ renderer });
    const vp = new THREE.Vector4();
    const before = spark.onBeforeRender.bind(spark);
    spark.onBeforeRender = (r, s, c) => {
      before(r, s, c);
      if (!spark.viewpoint.target) spark.uniforms.renderSize.value.set(r.getCurrentViewport(vp).z, vp.w);
    };
    osm.scene.add(spark);
    spark.castShadow = false;
  }

  // L with the OSM meshes hidden picks a point on the splat; L with them shown (O) picks the matching
  // point on the OSM model and pairs it with the last splat point.
  const landmarks: Landmark[] = [];
  let splatPoint: V3 | null = null;
  const ray = new THREE.Raycaster();
  const r2 = (v: number) => Math.round(v * 100) / 100;
  function pick(ndc: THREE.Vector2, camera: THREE.Camera) {
    ray.setFromCamera(ndc, camera);
    if (!osm.osmMeshes.visible) {
      const hit = ray.intersectObject(splat, false)[0];
      splatPoint = hit ? (splat.worldToLocal(hit.point.clone()).toArray().map(r2) as V3) : null;
      console.info(`[splat] splat point ${JSON.stringify(splatPoint)}; now press O, aim at the same spot and press L`);
      return;
    }
    const hit = ray.intersectObject(osm.osmMeshes, true)[0];
    const sim = hit ? ([r2(hit.point.x), r2(-hit.point.z), r2(hit.point.y)] as V3) : null;
    console.info(`[splat] OSM point (E, N, up) ${JSON.stringify(sim)}`);
    if (!sim || !splatPoint) return;
    landmarks.push({ name: `landmark ${landmarks.length + 1}`, splat: splatPoint, sim });
    splatPoint = null;
    console.info(JSON.stringify({ landmarks }, null, 1));
  }
  window.addEventListener("keydown", (e) => {
    if (e.key.toLowerCase() === "o") osm.osmMeshes.visible = !osm.osmMeshes.visible;
  });

  return { ...osm, place: "Moss St at Howard St (splat)", attach, pick };
}
