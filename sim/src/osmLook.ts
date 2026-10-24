// Materials, sky and light for the Moss world. Textures and the HDRI are CC0 from Poly Haven
// (public/textures/CREDITS.md). Every surface has UVs in metres, so a texture's repeat is
// 1 / (its real-world size in metres).
import * as THREE from "three";
import { RGBELoader } from "three/examples/jsm/loaders/RGBELoader.js";

const BASE = "textures/";
const manager = new THREE.LoadingManager(() => {
  (window as unknown as { __kiteLook: string }).__kiteLook = "ready";   // screenshot/YOLO scripts wait on this
});
manager.onError = (url) => console.error(`[osmLook] failed to load ${url}`);
const loader = new THREE.TextureLoader(manager);
const cache = new Map<string, THREE.Texture>();

function tex(file: string, sizeM: number, srgb: boolean, rotate = false) {
  const key = `${file}@${sizeM}${rotate ? "r" : ""}`;
  let t = cache.get(key);
  if (!t) {
    t = loader.load(BASE + file);
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.repeat.set(1 / sizeM, 1 / sizeM);
    t.anisotropy = 8;
    if (rotate) t.rotation = Math.PI / 2;
    if (srgb) t.colorSpace = THREE.SRGBColorSpace;
    cache.set(key, t);
  }
  return t;
}

interface PbrOpts { rough?: boolean; roughness?: number; color?: number; vertexColors?: boolean; rotate?: boolean; normalScale?: number }

/** A Poly Haven material: `<name>_diff.jpg`, `<name>_nor.jpg` (OpenGL normals), optional `<name>_rough.jpg`. */
export function pbr(name: string, sizeM: number, o: PbrOpts = {}) {
  const ns = o.normalScale ?? 1;
  return new THREE.MeshStandardMaterial({
    map: tex(`${name}_diff.jpg`, sizeM, true, o.rotate),
    normalMap: name === "roof" ? null : tex(`${name}_nor.jpg`, sizeM, false, o.rotate),
    normalScale: new THREE.Vector2(ns, ns),
    roughnessMap: o.rough ? tex(`${name}_rough.jpg`, sizeM, false) : null,
    roughness: o.roughness ?? (o.rough ? 1 : 0.85),
    color: o.color ?? 0xffffff,
    vertexColors: o.vertexColors ?? false,
  });
}

/** Facade materials by key. Sizes are the Poly Haven scan sizes. Tinted per building via vertex colours. */
export const FACADE_SIZES = {
  stucco: 2.0, plaster: 3.2, brick: 3.0, concrete: 2.7, metal: 2.7, siding: 1.8,
} as const;
export type FacadeKind = keyof typeof FACADE_SIZES;

export function facadeMaterial(kind: FacadeKind) {
  return pbr(kind, FACADE_SIZES[kind], { vertexColors: true, rotate: kind === "siding", roughness: kind === "metal" ? 0.6 : 0.9 });
}

export const SUN_DIR = new THREE.Vector3(-0.45, 1.0, 0.55).normalize();   // afternoon sun from the SW

/** Sky (HDRI as background and image-based light) and a shadow-casting sun. */
export function addSkyAndSun(scene: THREE.Scene, focus: { x: number; n: number }, span: number) {
  scene.background = new THREE.Color(0xb9c4cc);   // until the HDRI arrives
  new RGBELoader(manager).load(BASE + "urban_street_01_1k.hdr", (hdr) => {
    hdr.mapping = THREE.EquirectangularReflectionMapping;
    scene.environment = hdr;      // each renderer builds its own PMREM from this
  });
  loader.load(BASE + "sky_puresky.jpg", (sky) => {   // open sky for the background, no foreign trees
    sky.mapping = THREE.EquirectangularReflectionMapping;
    sky.colorSpace = THREE.SRGBColorSpace;
    scene.background = sky;
    scene.backgroundIntensity = 1.6;    // LDR jpeg through ACES comes out dim
  });
  scene.environmentIntensity = 1.25;
  scene.fog = new THREE.Fog(0xc9d2da, 90, 480);   // haze so the edge of the modelled blocks fades out
  const sun = new THREE.DirectionalLight(0xfff1e0, 2.2);
  const target = new THREE.Vector3(focus.x, 0, -focus.n);
  sun.position.copy(target).addScaledVector(SUN_DIR, 120);
  sun.target.position.copy(target);
  sun.castShadow = true;
  const c = sun.shadow.camera;
  c.left = c.bottom = -span / 2; c.right = c.top = span / 2;
  c.near = 1; c.far = 300;
  sun.shadow.mapSize.set(4096, 4096);
  sun.shadow.bias = -0.0004;
  sun.shadow.normalBias = 0.03;
  scene.add(sun, sun.target);
  // Anything added later (cars, people) casts shadows too.
  scene.addEventListener("childadded", (e) => e.child.traverse((o) => { o.castShadow = true; }));
}

/** Renderer settings shared by the on-screen and capture renderers so their frames match. */
export function tuneRenderer(r: THREE.WebGLRenderer) {
  r.outputColorSpace = THREE.SRGBColorSpace;
  r.toneMapping = THREE.ACESFilmicToneMapping;
  r.toneMappingExposure = 1.0;
  r.shadowMap.enabled = true;
  r.shadowMap.type = THREE.PCFSoftShadowMap;
}

/** Seeded random in [0, 1) from an integer id (mulberry32). Same building, same look, every load. */
export function seeded(id: number) {
  let a = (id ^ 0x9e3779b9) >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Triangle soup builder with normals, UVs (metres) and colours. Quads get their winding fixed to face `n`. */
export class Geo {
  pos: number[] = []; nor: number[] = []; uv: number[] = []; col: number[] = [];
  quad(p: THREE.Vector3[], n: THREE.Vector3, uv: [number, number][], color = new THREE.Color(1, 1, 1)) {
    const e1 = p[1].clone().sub(p[0]), e2 = p[2].clone().sub(p[0]);
    const order = e1.cross(e2).dot(n) >= 0 ? [0, 1, 2, 0, 2, 3] : [0, 2, 1, 0, 3, 2];
    for (const i of order) {
      this.pos.push(p[i].x, p[i].y, p[i].z);
      this.nor.push(n.x, n.y, n.z);
      this.uv.push(uv[i][0], uv[i][1]);
      this.col.push(color.r, color.g, color.b);
    }
  }
  get empty() { return this.pos.length === 0; }
  build() {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(this.pos, 3));
    g.setAttribute("normal", new THREE.Float32BufferAttribute(this.nor, 3));
    g.setAttribute("uv", new THREE.Float32BufferAttribute(this.uv, 2));
    g.setAttribute("color", new THREE.Float32BufferAttribute(this.col, 3));
    return g;
  }
}
