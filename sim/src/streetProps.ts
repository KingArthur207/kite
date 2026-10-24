// Geometry and canvas textures for the street dressing (streetDressing.ts places them): street trees
// with branching trunks and leaf-card canopies, wooden utility poles, sagging wires, an SF hydrant,
// curbside carts, a litter can, a kick scooter, sign faces and ground decals. No downloaded models:
// bark and pole wood are CC0 Poly Haven textures (public/textures/CREDITS.md), the rest is drawn here.
import * as THREE from "three";
import { seeded } from "./osmLook";

const UP = new THREE.Vector3(0, 1, 0);

/** A canvas texture drawn once. */
function canvasTex(size: number, draw: (g: CanvasRenderingContext2D, s: number) => void, srgb = true) {
  const c = document.createElement("canvas");
  c.width = c.height = size;
  draw(c.getContext("2d")!, size);
  const t = new THREE.CanvasTexture(c);
  if (srgb) t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 4;
  return t;
}

/** Leaf clusters on a transparent card: a few hundred small pointed leaves, denser in the middle. */
export const leafTexture = () => canvasTex(512, (g, S) => {
  const r = seeded(7);
  for (let i = 0; i < 900; i++) {
    const a = r() * Math.PI * 2, d = Math.sqrt(r()) * S * 0.44;
    const x = S / 2 + Math.cos(a) * d, y = S / 2 + Math.sin(a) * d;
    const len = 16 + r() * 18, wid = len * (0.42 + r() * 0.2);
    const light = 20 + r() * 26 + (1 - d / (S * 0.44)) * 4;
    g.fillStyle = `hsl(${84 + r() * 30}, ${32 + r() * 22}%, ${light}%)`;
    g.save();
    g.translate(x, y);
    g.rotate(r() * Math.PI * 2);
    g.beginPath();
    g.moveTo(-len / 2, 0);
    g.quadraticCurveTo(0, -wid, len / 2, 0);
    g.quadraticCurveTo(0, wid, -len / 2, 0);
    g.fill();
    g.restore();
  }
});

/** Soft irregular blob, white with alpha: stains, oil drips, contact shadows (tinted by vertex colour). */
export const blobTexture = () => canvasTex(128, (g, S) => {
  const r = seeded(11);
  for (let i = 0; i < 14; i++) {
    const x = S / 2 + (r() - 0.5) * S * 0.4, y = S / 2 + (r() - 0.5) * S * 0.4, rad = S * (0.12 + r() * 0.2);
    const grad = g.createRadialGradient(x, y, 0, x, y, rad);
    grad.addColorStop(0, "rgba(255,255,255,0.35)");
    grad.addColorStop(1, "rgba(255,255,255,0)");
    g.fillStyle = grad;
    g.fillRect(0, 0, S, S);
  }
}, false);

/** Cast-iron utility cover: diamond plate inside a rim. */
export const coverTexture = () => canvasTex(256, (g, S) => {
  g.fillStyle = "#3a3936"; g.beginPath(); g.arc(S / 2, S / 2, S / 2 - 2, 0, Math.PI * 2); g.fill();
  g.strokeStyle = "#555350"; g.lineWidth = 10; g.beginPath(); g.arc(S / 2, S / 2, S / 2 - 12, 0, Math.PI * 2); g.stroke();
  g.save(); g.beginPath(); g.arc(S / 2, S / 2, S / 2 - 20, 0, Math.PI * 2); g.clip();
  g.strokeStyle = "#4d4b47"; g.lineWidth = 5;
  for (let k = -S; k < 2 * S; k += 22) {
    g.beginPath(); g.moveTo(k, 0); g.lineTo(k + S, S); g.stroke();
    g.beginPath(); g.moveTo(k, S); g.lineTo(k + S, 0); g.stroke();
  }
  g.restore();
  g.fillStyle = "#5c5a55"; g.font = "bold 26px Helvetica, Arial"; g.textAlign = "center"; g.fillText("S.F.W.D.", S / 2, S / 2 + 9);
});

/** Sign faces. */
export const noParkingTexture = () => canvasTex(256, (g, S) => {
  g.fillStyle = "#f4f3ee"; g.fillRect(0, 0, S, S);
  g.strokeStyle = "#c01d22"; g.lineWidth = 12; g.strokeRect(10, 10, S - 20, S - 20);
  g.fillStyle = "#c01d22"; g.textAlign = "center"; g.font = "bold 46px Helvetica, Arial";
  g.fillText("NO", S / 2, 64); g.fillText("PARKING", S / 2, 112);
  g.font = "bold 34px Helvetica, Arial"; g.fillText("ANY TIME", S / 2, 160);
  g.font = "bold 22px Helvetica, Arial"; g.fillText("TOW-AWAY", S / 2, 200); g.fillText("ZONE", S / 2, 226);
});
export const streetNameTexture = (name: string, block: string) => canvasTex(512, (g, S) => {
  g.fillStyle = "#1d6b3a"; g.fillRect(0, 0, S, S / 4);
  g.strokeStyle = "#eef2ee"; g.lineWidth = 5; g.strokeRect(6, 6, S - 12, S / 4 - 12);
  g.fillStyle = "#eef2ee"; g.font = "bold 70px Helvetica, Arial"; g.textAlign = "left"; g.textBaseline = "middle";
  g.fillText(name, 26, S / 8 + 4);
  g.font = "bold 30px Helvetica, Arial"; g.textAlign = "right"; g.fillText(block, S - 24, S / 8 + 4);
});

/** Ground cover for tree basins: dark mulch speckle. */
export const mulchTexture = () => canvasTex(128, (g, S) => {
  g.fillStyle = "#3a2e24"; g.fillRect(0, 0, S, S);
  const r = seeded(5);
  for (let i = 0; i < 700; i++) {
    g.fillStyle = `hsl(${20 + r() * 20}, ${20 + r() * 25}%, ${10 + r() * 22}%)`;
    g.fillRect(r() * S, r() * S, 1 + r() * 4, 1 + r() * 2);
  }
});

// --- tree: a trunk forking into scaffold branches, leaf cards clustered at the twig ends.
export interface TreeParts { bark: THREE.BufferGeometry[]; leaves: LeafSoup }

/** Leaf cards as plain arrays, merged across trees into one mesh. */
export class LeafSoup {
  pos: number[] = []; nor: number[] = []; uv: number[] = []; col: number[] = [];
  /** A square card centred at c, facing n, with lighting normals pointing out of the crown. */
  card(c: THREE.Vector3, n: THREE.Vector3, size: number, spin: number, crown: THREE.Vector3, crownR: number, shade: number) {
    const t = new THREE.Vector3().crossVectors(n, Math.abs(n.y) > 0.9 ? new THREE.Vector3(1, 0, 0) : UP).normalize();
    const b = new THREE.Vector3().crossVectors(n, t);
    t.applyAxisAngle(n, spin); b.applyAxisAngle(n, spin);
    const corners = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([i, j]) => c.clone().addScaledVector(t, i * size / 2).addScaledVector(b, j * size / 2));
    const uvs = [[0, 0], [1, 0], [1, 1], [0, 1]];
    for (const k of [0, 1, 2, 0, 2, 3]) {
      const p = corners[k];
      this.pos.push(p.x, p.y, p.z);
      const out = p.clone().sub(crown);
      const reach = Math.min(1, out.length() / crownR);
      out.y += crownR * 0.35;                    // sky light from above
      out.normalize();
      this.nor.push(out.x, out.y, out.z);
      this.uv.push(uvs[k][0], uvs[k][1]);
      const ao = shade * (0.5 + 0.5 * reach) * (0.75 + 0.25 * Math.min(1, Math.max(0, (p.y - crown.y + crownR) / (2 * crownR))));
      this.col.push(ao, ao, ao);
    }
  }
  build() {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(this.pos, 3));
    g.setAttribute("normal", new THREE.Float32BufferAttribute(this.nor, 3));
    g.setAttribute("uv", new THREE.Float32BufferAttribute(this.uv, 2));
    g.setAttribute("color", new THREE.Float32BufferAttribute(this.col, 3));
    return g;
  }
}

/** Tapered open cylinder from a to b, UVs in metres (around, along). */
function limb(a: THREE.Vector3, b: THREE.Vector3, r0: number, r1: number, seg = 7) {
  const d = b.clone().sub(a), len = d.length();
  const g = new THREE.CylinderGeometry(r1, r0, len, seg, 1, true);
  const uv = g.getAttribute("uv");
  for (let i = 0; i < uv.count; i++) uv.setXY(i, uv.getX(i) * 2 * Math.PI * r0, uv.getY(i) * len);
  g.translate(0, len / 2, 0);
  g.applyMatrix4(new THREE.Matrix4().makeRotationFromQuaternion(new THREE.Quaternion().setFromUnitVectors(UP, d.normalize())));
  g.translate(a.x, a.y, a.z);
  return g;
}

/** A street tree at (x, y, z), about `height` m tall, crown clear of the sidewalk below 2.6 m.
 *  size: crown radius and trunk radius in m (defaults scale with height). */
export function streetTree(at: THREE.Vector3, height: number, seed: number, leaves: LeafSoup,
                           size: { crownR?: number; trunkR?: number } = {}): THREE.BufferGeometry[] {
  const r = seeded(seed);
  const bark: THREE.BufferGeometry[] = [];
  const crown = at.clone().add(new THREE.Vector3(0, height * 0.66, 0));
  const crownR = size.crownR ?? height * 0.32;
  const trunkTop = at.clone().add(new THREE.Vector3((r() - 0.5) * 0.25, 2.5 + r() * 0.6, (r() - 0.5) * 0.25));
  const trunkR = size.trunkR ?? 0.1 + height * 0.012;
  bark.push(limb(at.clone().add(new THREE.Vector3(0, -0.05, 0)), trunkTop, trunkR * 1.15, trunkR * 0.85, 9));
  const tips: THREE.Vector3[] = [];
  const grow = (from: THREE.Vector3, dir: THREE.Vector3, len: number, rad: number, depth: number) => {
    const to = from.clone().addScaledVector(dir, len);
    bark.push(limb(from, to, rad, rad * 0.7, depth < 2 ? 7 : 5));
    if (depth === 3) { tips.push(to); return; }
    if (depth >= 1) tips.push(from.clone().lerp(to, 0.6));
    const kids = depth === 0 ? 2 : 2 + (r() < 0.4 ? 1 : 0);
    for (let k = 0; k < kids; k++) {
      const axis = new THREE.Vector3(r() - 0.5, 0, r() - 0.5).cross(dir).normalize();
      const nd = dir.clone().applyAxisAngle(axis, 0.35 + r() * 0.35).applyAxisAngle(dir, (k / kids) * Math.PI * 2 + r());
      nd.y = Math.max(nd.y, 0.15); nd.normalize();
      grow(to, nd, len * (0.62 + r() * 0.15), rad * 0.62, depth + 1);
    }
  };
  const n = 3 + (r() < 0.5 ? 1 : 0);
  const spread = crownR * 0.75;
  for (let k = 0; k < n; k++) {
    const az = (k / n) * Math.PI * 2 + r() * 0.8, tilt = 0.55 + r() * 0.3;
    const dir = new THREE.Vector3(Math.cos(az) * Math.sin(tilt), Math.cos(tilt), Math.sin(az) * Math.sin(tilt));
    grow(trunkTop, dir, spread * (0.85 + r() * 0.3), trunkR * 0.7, 0);
  }
  grow(trunkTop, new THREE.Vector3((r() - 0.5) * 0.2, 1, (r() - 0.5) * 0.2).normalize(), height * 0.22, trunkR * 0.6, 1);   // leader
  for (const tip of tips) {
    const cards = 5 + Math.floor(r() * 3);
    for (let k = 0; k < cards; k++) {
      const c = tip.clone().add(new THREE.Vector3(r() - 0.5, (r() - 0.5) * 0.7, r() - 0.5).multiplyScalar(1.1));
      const nrm = new THREE.Vector3(r() - 0.5, r() - 0.3, r() - 0.5).normalize();
      leaves.card(c, nrm, 1.0 + r() * 0.7, r() * Math.PI, crown, crownR, 0.8 + r() * 0.3);
    }
  }
  return bark;
}

// --- small props as merged, vertex-coloured geometry: one material for the lot.
const col = (g: THREE.BufferGeometry, hex: number) => {
  const c = new THREE.Color(hex), n = g.getAttribute("position").count;
  g.setAttribute("color", new THREE.Float32BufferAttribute(Array.from({ length: n }, () => [c.r, c.g, c.b]).flat(), 3));
  g.deleteAttribute("uv");
  return g.index ? g.toNonIndexed() : g;
};
const boxG = (x: number, y: number, z: number, hex: number) => col(new THREE.BoxGeometry(x, y, z), hex);
const cylG = (r0: number, r1: number, h: number, hex: number, seg = 12) => col(new THREE.CylinderGeometry(r1, r0, h, seg).translate(0, h / 2, 0), hex);

/** SF low-pressure hydrant: white body, blue bonnet and caps. About 0.8 m tall. */
export function hydrantGeo() {
  const W = 0xe6e3da, B = 0x2d5aa3;
  const body = col(new THREE.LatheGeometry([
    [0.001, 0], [0.19, 0], [0.19, 0.05], [0.14, 0.07], [0.125, 0.1], [0.125, 0.58], [0.15, 0.6], [0.15, 0.64], [0.001, 0.64],
  ].map(([x, y]) => new THREE.Vector2(x, y)), 16), W);
  const bonnet = col(new THREE.SphereGeometry(0.14, 14, 7, 0, Math.PI * 2, 0, Math.PI / 2).translate(0, 0.64, 0), B);
  const nut = cylG(0.035, 0.03, 0.09, B, 5).translate(0, 0.76, 0);
  const side = (a: number) => [
    cylG(0.055, 0.055, 0.1, W).rotateZ(Math.PI / 2).translate(0.17, 0.42, 0).rotateY(a),
    cylG(0.062, 0.062, 0.04, B).rotateZ(Math.PI / 2).translate(0.23, 0.42, 0).rotateY(a),
  ];
  const pumper = [cylG(0.08, 0.08, 0.1, W).rotateX(Math.PI / 2).translate(0, 0.36, 0.14), cylG(0.09, 0.09, 0.04, B).rotateX(Math.PI / 2).translate(0, 0.36, 0.21)];
  return [body, bonnet, nut, ...side(0), ...side(Math.PI), ...pumper];
}

/** Recology curbside cart (landfill black, recycling blue, compost green), lid toward +z. */
export function cartGeo(hex: number) {
  const body = new THREE.CylinderGeometry(0.42, 0.36, 0.98, 4, 1).rotateY(Math.PI / 4).scale(1, 1, 1.12).translate(0, 0.53, 0);
  return [
    col(body, hex),
    boxG(0.64, 0.05, 0.74, hex).translate(0, 1.04, 0.02),
    boxG(0.5, 0.04, 0.04, 0x1a1a1a).translate(0, 0.98, -0.42),
    ...[-0.22, 0.22].map((x) => cylG(0.1, 0.1, 0.05, 0x111111, 10).rotateZ(Math.PI / 2).translate(x + (x > 0 ? 0.025 : -0.025), 0.1, -0.33)),
  ];
}

/** Dark green SF litter can with a domed lid. */
export function litterCanGeo() {
  return [cylG(0.3, 0.3, 0.9, 0x24382c, 16), col(new THREE.SphereGeometry(0.31, 16, 6, 0, Math.PI * 2, 0, Math.PI / 2).scale(1, 0.45, 1).translate(0, 0.9, 0), 0x24382c),
    boxG(0.22, 0.1, 0.05, 0x101010).translate(0, 0.86, 0.29)];
}

/** Rental kick scooter on its kickstand, deck along x. */
export function scooterGeo(hex: number) {
  const stem = cylG(0.022, 0.022, 1.05, 0x2a2a2a, 8).rotateZ(-0.12).translate(0.42, 0.12, 0);
  return [
    boxG(0.78, 0.06, 0.17, 0x1d1d1d).translate(0, 0.13, 0),
    boxG(0.5, 0.025, 0.15, hex).translate(-0.02, 0.17, 0),
    stem,
    boxG(0.05, 0.12, 0.08, hex).translate(0.5, 0.95, 0),
    cylG(0.016, 0.016, 0.5, 0x1d1d1d, 6).rotateX(Math.PI / 2).translate(0.54, 1.12, 0.25),
    ...[-0.38, 0.46].map((x) => col(new THREE.TorusGeometry(0.085, 0.03, 6, 14).translate(x, 0.115, 0), 0x111111)),
    boxG(0.24, 0.035, 0.14, hex).rotateZ(0.3).translate(-0.44, 0.2, 0),
  ];
}

/** Galvanized sign post, 2.7 m. */
export const signPostGeo = () => [cylG(0.03, 0.03, 2.7, 0x9a9d9b, 6)];

/** Wire as thin prisms along a sagging parabola from a to b. */
export function wireGeo(a: THREE.Vector3, b: THREE.Vector3, sag: number, radius: number, segs = 14) {
  const out: THREE.BufferGeometry[] = [];
  const at = (t: number) => a.clone().lerp(b, t).add(new THREE.Vector3(0, -sag * 4 * t * (1 - t), 0));
  for (let k = 0; k < segs; k++) {
    const p = at(k / segs), q = at((k + 1) / segs);
    const g = limb(p, q, radius, radius, 4);
    g.deleteAttribute("uv");
    out.push(g);
  }
  return { mesh: out, line: Array.from({ length: segs + 1 }, (_, k) => at(k / segs)) };
}

export { limb };
