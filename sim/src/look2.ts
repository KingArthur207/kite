// ?look=2: the Moss world closer to Street View. Opt-in until reviewed; to make it the default, change DEFAULT_LOOK
// to "2" (?look=1 then brings back the old look). Metrics mode (?metrics=1) never uses it.
//   - windows: glass that reflects the sky at grazing angles over a fake room behind it (interior mapping:
//     walls, floor, ceiling, some blinds), so they stop reading as flat dark rectangles
//   - siding: lap siding drawn at true scale (15 cm boards with a shadow line under each)
//   - bright midday sun, hard shadows; world-space grime on walls, trim and ground (breaks texture tiling)
//   - post: N8AO ambient occlusion, ACES via OutputPass, a light grade, SMAA
//   - the per-building sheet public/local/moss_buildings.json (optional, local only), see loadBuildingSheet
//   - osmFacades adds iron security gates and doorstep hedges; streetProps makes Moss's trees scrappier
import * as THREE from "three";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";
import { ShaderPass } from "three/examples/jsm/postprocessing/ShaderPass.js";
import { SMAAPass } from "three/examples/jsm/postprocessing/SMAAPass.js";

const DEFAULT_LOOK = "1";
const query = new URLSearchParams(location.search);
export const look2 = (query.get("look") ?? DEFAULT_LOOK) === "2" && query.get("metrics") !== "1";

/** One building's look, from public/local/moss_buildings.json (generated from Street View, gitignored). Any field may be missing. */
export interface BuildingSheet {
  main?: string; trim?: string; cladding?: "siding" | "stucco" | "brick" | "concrete" | "metal"; floors?: number;
  bays?: number; garages?: number; garage_colour?: string; gate?: boolean; hedge?: boolean;
  fire_escape?: boolean; mural?: boolean; confidence?: number;
}
export const buildingSheet = new Map<number, BuildingSheet>();

/** Load the sheet if it's there (look 2 only). entry.ts awaits this before building the world. */
export async function loadBuildingSheet() {
  if (!look2) return;
  const r = await fetch("local/moss_buildings.json").catch(() => null);
  if (!r?.ok || !r.headers.get("content-type")?.includes("json")) return;   // vite answers a missing file with index.html
  try {
    const j = (await r.json()) as { buildings?: Record<string, BuildingSheet> };
    for (const [id, b] of Object.entries(j.buildings ?? {})) buildingSheet.set(Number(id), b);
    console.info(`[look2] building sheet: ${buildingSheet.size} buildings`);
  } catch (e) { console.error("[look2] moss_buildings.json unreadable, using defaults", e); }
}

// Midday sun: SF in late October peaks about 40 deg up, due south. x east, y up, z south.
const SUN_AZ = THREE.MathUtils.degToRad(190), SUN_EL = THREE.MathUtils.degToRad(44);
const SUN_DIR = new THREE.Vector3(Math.sin(SUN_AZ) * Math.cos(SUN_EL), Math.sin(SUN_EL), -Math.cos(SUN_AZ) * Math.cos(SUN_EL));

// --- world-space grime. Value noise in the fragment shader, keyed off the world position.
const NOISE = /* glsl */ `
varying vec3 vKWorld;
float kHash(vec3 p) { p = fract(p * 0.3183099 + 0.1); p *= 17.0; return fract(p.x * p.y * p.z * (p.x + p.y + p.z)); }
float kNoise(vec3 x) {
  vec3 i = floor(x), f = fract(x); f = f * f * (3.0 - 2.0 * f);
  return mix(mix(mix(kHash(i), kHash(i + vec3(1,0,0)), f.x), mix(kHash(i + vec3(0,1,0)), kHash(i + vec3(1,1,0)), f.x), f.y),
             mix(mix(kHash(i + vec3(0,0,1)), kHash(i + vec3(1,0,1)), f.x), mix(kHash(i + vec3(0,1,1)), kHash(i + vec3(1,1,1)), f.x), f.y), f.z);
}
float kFbm(vec3 x) { return 0.55 * kNoise(x) + 0.3 * kNoise(x * 2.3) + 0.15 * kNoise(x * 5.1); }
`;
const GRIME = {
  // Walls: blotchy weathering at a few metres, rain streaks (noise stretched vertically), a dirty bottom metre.
  wall: /* glsl */ `{
    float blot = kFbm(vKWorld * 0.35);
    float streak = kNoise(vKWorld * vec3(3.0, 0.12, 3.0));
    float foot = 1.0 - smoothstep(0.0, 1.2, vKWorld.y);
    diffuseColor.rgb *= 0.92 + 0.14 * blot;
    diffuseColor.rgb *= 1.0 - 0.08 * smoothstep(0.55, 0.9, streak);
    diffuseColor.rgb = mix(diffuseColor.rgb, diffuseColor.rgb * vec3(0.74, 0.7, 0.64), foot * (0.5 + 0.5 * blot));
  }`,
  // Trim and frames: paint, so only faint variation and the dirty bottom.
  trim: /* glsl */ `{
    float blot = kFbm(vKWorld * 0.6);
    float foot = 1.0 - smoothstep(0.0, 1.0, vKWorld.y);
    diffuseColor.rgb *= 0.95 + 0.07 * blot;
    diffuseColor.rgb = mix(diffuseColor.rgb, diffuseColor.rgb * vec3(0.76, 0.73, 0.67), foot * 0.6);
  }`,
  // Ground: big soft stains so the tiled texture doesn't repeat visibly, plus smaller dark spots (gum, oil).
  ground: /* glsl */ `{
    float big = kFbm(vec3(vKWorld.xz * 0.12, 0.0));
    float spots = smoothstep(0.72, 0.85, kNoise(vec3(vKWorld.xz * 1.7, 3.0)));
    diffuseColor.rgb *= (0.88 + 0.22 * big) * (1.0 - 0.15 * spots);
  }`,
};
type Grime = keyof typeof GRIME;

const WORLD_POS = /* glsl */ `#include <worldpos_vertex>
  { vec4 kw = vec4(transformed, 1.0);
    #ifdef USE_INSTANCING
    kw = instanceMatrix * kw;
    #endif
    vKWorld = (modelMatrix * kw).xyz; }`;

function addGrime(m: THREE.Material, kind: Grime) {
  m.onBeforeCompile = (s) => {
    s.vertexShader = "varying vec3 vKWorld;\n" + s.vertexShader.replace("#include <worldpos_vertex>", WORLD_POS);
    s.fragmentShader = NOISE + s.fragmentShader.replace("#include <map_fragment>", "#include <map_fragment>\n" + GRIME[kind]);
  };
  m.customProgramCacheKey = () => `kite-grime-${kind}`;
  m.needsUpdate = true;
}

// Material names set where they're made (osmLook pbr/facadeMaterial, osmFacades).
const GRIME_BY_NAME: Record<string, Grime> = {
  stucco: "wall", plaster: "wall", brick: "wall", concrete: "wall", metal: "wall", siding: "wall",
  trim: "trim", frame: "trim", shutter: "trim", door: "trim",
  sidewalk: "ground", asphalt: "ground", curb: "ground",
};

// --- window glass with a room behind it. The pane is a 1x1-segment plane in instance space (x across, y up,
// z out of the wall), so |position.xy| is its half size at every vertex. Ray from the camera (in instance space)
// through the pane into a box room; the face it hits picks wall/floor/ceiling. Emissive, so the sun on the facade
// doesn't light the inside; the physical specular adds sky reflections on top, strong at grazing angles.
const ROOM_VERT = /* glsl */ `#include <begin_vertex>
  { mat4 im = mat4(1.0);
    #ifdef USE_INSTANCING
    im = instanceMatrix;
    #endif
    vKLocal = position;
    vKHalf = abs(position.xy);
    vKCam = (inverse(modelMatrix * im) * vec4(cameraPosition, 1.0)).xyz;
    vKSeed = fract(sin(dot(im[3].xyz, vec3(12.9898, 78.233, 37.719))) * 43758.5453); }`;
const ROOM_FRAG = /* glsl */ `#include <emissivemap_fragment>
  {
    vec3 d = normalize(vKLocal - vKCam);
    vec3 o = vec3(vKLocal.xy, 0.0);
    vec3 lo = vec3(-vKHalf.x - 1.2, -vKHalf.y - 0.9, -4.0), hi = vec3(vKHalf.x + 1.2, vKHalf.y + 0.45, 0.0);
    vec3 t = (mix(lo, hi, step(0.0, d)) - o) / d;
    float tm = min(min(t.x, t.y), t.z);
    vec3 p = o + d * tm;
    float hue = vKSeed;
    vec3 wall = mix(vec3(0.85, 0.8, 0.72), vec3(0.75, 0.8, 0.82), step(0.5, fract(hue * 3.7)));
    vec3 c = tm == t.z ? wall : tm == t.x ? wall * 0.7 : d.y < 0.0 ? vec3(0.32, 0.22, 0.15) : vec3(0.9);
    c *= mix(1.0, 0.45, -p.z / 4.0) * (0.75 + 0.5 * fract(hue * 11.3));
    // Blinds or a curtain on some windows, pulled down part way, just behind the glass.
    float drop = fract(hue * 7.31);
    if (hue > 0.45 && vKLocal.y > vKHalf.y - 2.0 * vKHalf.y * drop) {
      vec3 cloth = mix(vec3(0.85, 0.82, 0.75), vec3(0.55, 0.5, 0.45), step(0.8, hue));
      c = cloth * (0.85 + 0.15 * step(0.5, fract(vKLocal.y * 22.0)));
    }
    totalEmissiveRadiance += c * 0.13;
    diffuseColor.rgb *= 0.0;
  }`;

function roomGlass() {
  const m = new THREE.MeshPhysicalMaterial({ color: 0xffffff, metalness: 0, roughness: 0.02, ior: 1.6, envMapIntensity: 2.2 });
  m.onBeforeCompile = (s) => {
    const v = "varying vec3 vKLocal; varying vec2 vKHalf; varying vec3 vKCam; varying float vKSeed;\n";
    s.vertexShader = v + s.vertexShader.replace("#include <begin_vertex>", ROOM_VERT);
    s.fragmentShader = v + s.fragmentShader.replace("#include <emissivemap_fragment>", ROOM_FRAG);
  };
  m.customProgramCacheKey = () => "kite-room-glass";
  return m;
}

// --- lap siding at true scale: boards 15 cm high, each lapping over the one below, so its face tilts down a
// little and its bottom edge throws a thin shadow line. Drawn into canvases (colour and OpenGL normal map).
const SIDING_M = 3.0, BOARDS = 20;   // texture tile 3 m, 20 boards of 15 cm
function sidingTextures() {
  const S = 1000, b = S / BOARDS;
  const make = () => { const c = document.createElement("canvas"); c.width = c.height = S; return c; };
  const cc = make(), nc = make();
  const g = cc.getContext("2d")!, n = nc.getContext("2d")!;
  let seed = 9;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  for (let k = 0; k < BOARDS; k++) {
    const y0 = k * b;
    const shade = 228 + rnd() * 14;
    g.fillStyle = `rgb(${shade},${shade},${shade - 3})`; g.fillRect(0, y0, S, b);
    for (let i = 0; i < 6; i++) {   // faint grain under the paint
      g.fillStyle = `rgba(0,0,0,${0.015 + rnd() * 0.02})`;
      g.fillRect(0, y0 + rnd() * b, S, 1 + rnd() * 2);
    }
    g.fillStyle = "rgba(60,55,50,0.55)"; g.fillRect(0, y0 + b - 3, S, 3);        // shadow under the butt edge
    const joint = rnd() * S;                                                       // one butt joint per board
    g.fillStyle = "rgba(60,55,50,0.35)"; g.fillRect(joint, y0, 2, b);
    // Normal: canvas y runs down, the texture's v up. Face tilts down (ny < 0), the butt edge faces down hard.
    n.fillStyle = "rgb(128,112,252)"; n.fillRect(0, y0, S, b);
    n.fillStyle = "rgb(128,30,190)"; n.fillRect(0, y0 + b - 5, S, 5);
    n.fillStyle = "rgb(128,200,230)"; n.fillRect(0, y0, S, 2);                    // top of the next board, tucked under
  }
  const tex = (c: HTMLCanvasElement, srgb: boolean) => {
    const t = new THREE.CanvasTexture(c);
    t.wrapS = t.wrapT = THREE.RepeatWrapping;
    t.repeat.set(1 / SIDING_M, 1 / SIDING_M);
    t.anisotropy = 8;
    if (srgb) t.colorSpace = THREE.SRGBColorSpace;
    return t;
  };
  return { map: tex(cc, true), normalMap: tex(nc, false) };
}

/** Re-light and re-surface a built Moss world in place. Call once after the world is built. */
export function upgradeWorld(scene: THREE.Scene) {
  if (scene.fog instanceof THREE.Fog) { scene.fog.color.set(0xd6e0ea); scene.fog.near = 80; scene.fog.far = 520; }
  scene.environmentIntensity = 0.95;
  const glass = roomGlass();
  const siding = sidingTextures();
  const done = new Set<THREE.Material>();
  scene.traverse((o) => {
    if (o instanceof THREE.DirectionalLight && o.castShadow) {
      o.position.copy(o.target.position).addScaledVector(SUN_DIR, 120);
      o.color.set(0xfff6ea);
      o.intensity = 3.3;
      o.shadow.normalBias = 0.05;
    }
    if (!(o instanceof THREE.Mesh)) return;
    if ((o.material as THREE.Material).name === "glass") { o.material = glass; return; }
    for (const m of [o.material].flat() as THREE.Material[]) {
      if (done.has(m)) continue;
      done.add(m);
      if (m.name === "siding" && m instanceof THREE.MeshStandardMaterial) {
        Object.assign(m, siding);
        m.color.setScalar(0.92);
        m.normalScale.set(1, 1);
      }
      const kind = GRIME_BY_NAME[m.name];
      if (kind) addGrime(m, kind);
    }
  });
}

// --- post
const GradeShader = {
  uniforms: { tDiffuse: { value: null }, vignette: { value: 0.14 } },
  vertexShader: /* glsl */ `varying vec2 vUv; void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
  fragmentShader: /* glsl */ `
    uniform sampler2D tDiffuse; uniform float vignette; varying vec2 vUv;
    void main() {
      vec3 c = texture2D(tDiffuse, vUv).rgb;                       // display-referred (after OutputPass)
      float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
      c = mix(vec3(l), c, 1.08);                                    // a touch more saturation, like a phone
      c = mix(c, c * c * (3.0 - 2.0 * c), 0.2);                     // gentle S-curve: deeper shadows
      vec2 d = vUv - 0.5;
      c *= 1.0 - vignette * smoothstep(0.25, 0.85, dot(d, d) * 2.0);
      gl_FragColor = vec4(c, 1.0);
    }`,
};

/** A post chain per camera, sized to the renderer's current viewport. render(camera) draws into that viewport. */
export async function createPost(renderer: THREE.WebGLRenderer, scene: THREE.Scene) {
  const { N8AOPass } = await import("n8ao");   // only fetched with ?look=2
  const chains = new Map<THREE.Camera, { composer: EffectComposer; w: number; h: number }>();
  const vp = new THREE.Vector4();
  return (camera: THREE.Camera) => {
    renderer.getViewport(vp);
    let c = chains.get(camera);
    if (!c) {
      const composer = new EffectComposer(renderer);
      const ao = new N8AOPass(scene, camera, vp.z, vp.w);
      Object.assign(ao.configuration, {
        aoRadius: 1.6, distanceFalloff: 1.0, intensity: 4, gammaCorrection: false,
        halfRes: renderer.getPixelRatio() > 1,   // retina: AO at half res, depth-aware upsampled
      });
      if (query.get("ao") === "debug") ao.setDisplayMode("AO");
      composer.addPass(ao);
      composer.addPass(new OutputPass());
      composer.addPass(new ShaderPass(GradeShader));
      composer.addPass(new SMAAPass(vp.z * renderer.getPixelRatio(), vp.w * renderer.getPixelRatio()));
      c = { composer, w: 0, h: 0 };
      chains.set(camera, c);
    }
    if (c.w !== vp.z || c.h !== vp.w) { c.composer.setSize(vp.z, vp.w); c.w = vp.z; c.h = vp.w; }
    c.composer.render();
  };
}
