// ?compare=1: the sim on the left, Google Street View on the right, synced to the drone camera's position and
// heading, for spotting where the 3D Moss St differs from reality. Street View is an interactive embed only: no
// imagery is downloaded or stored, and only Google's own panoramas inside SCOPE are shown (docs/google-permission.md).
//
// Keys (arrows walk as usual): X cycles which side is larger, P puts the sim camera on the panorama's spot,
// C saves the sim canvas to out/compare/<stamp>_sim.png and the view (pano id, POV, walker) plus an optional note to
// out/compare/notes.jsonl, through the dev server's /__compare endpoint (vite.config.ts). ?at=x,n,heading spawns
// the walker there, so a saved line's `reopen` URL brings the same view back.
import { CAM_W, DRONE_BEHIND, FX, toLatLng, toSim } from "./config";
import { inScope, loadStreetView, PANO_OPTS } from "./googleStreetView";

type Walker = { x: number; n: number; heading: number };
type Pano = { id: string; lat: number; lng: number; date: string };

const REQUERY_M = 3;            // look for a nearer panorama only once the camera moved this far
const RADIUS_M = 15;
const HFOV_DEG = (2 * Math.atan(CAM_W / 2 / FX) * 180) / Math.PI;   // the drone camera's, about 51°
// Street View's nominal FOV is 180 / 2^zoom, but measured (turn 8°, cross-correlate the pixel shift) it shows
// 59° at the nominal 1.81; +0.24 brings it to the drone's 51°. ?svzoom= overrides it if the framing looks off.
const ZOOM_FIX = 0.24;
const query = new URLSearchParams(location.search);
const ZOOM = Number(query.get("svzoom") ?? Math.log2(180 / HFOV_DEG) + ZOOM_FIX);
const SPLITS = [50, 65, 35];    // sim's share of the width, X cycles

export function startCompare(o: { walker: Walker; canvas: HTMLCanvasElement; view: HTMLElement }) {
  const { walker, canvas, view } = o;
  const at = query.get("at")?.split(",").map(Number);
  if (at?.length === 3 && at.every(Number.isFinite)) Object.assign(walker, { x: at[0], n: at[1], heading: at[2] });

  const side = document.createElement("div");
  side.style.cssText = "position:absolute;top:0;bottom:0;right:0;background:#000;display:flex;align-items:center;justify-content:center";
  const box = document.createElement("div");   // 4:3 like the sim's drone view, so both halves frame the same view
  box.style.cssText = "width:min(100%, calc(100vh * 4 / 3));aspect-ratio:4/3;position:relative";
  const label = document.createElement("div");
  label.style.cssText = "position:absolute;right:10px;top:10px;z-index:10;background:rgba(0,0,0,.6);padding:8px 10px;border-radius:6px;white-space:pre;pointer-events:none";
  side.append(box);
  document.body.append(side, label);

  let split = 0;
  const layout = () => {
    const s = SPLITS[split];
    view.style.right = `${100 - s}%`;
    side.style.left = `${s}%`;
    window.dispatchEvent(new Event("resize"));   // main.ts sizes its renderer to the view
  };
  layout();

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let svc: any = null, pano: any = null, sources: unknown[] = [];
  let here: Pano | null = null, note = "loading Street View", busy = false;
  let askedAt: { x: number; n: number } | null = null, povSet = "";
  let flash = "", flashAt = 0, saving = false;
  const say = (s: string) => { flash = s; flashAt = performance.now(); };

  const cam = () => {   // the drone camera: DRONE_BEHIND behind the walker, where the sim's main view is
    const h = (walker.heading * Math.PI) / 180;
    return { x: walker.x - Math.sin(h) * DRONE_BEHIND, n: walker.n - Math.cos(h) * DRONE_BEHIND };
  };

  loadStreetView(() => { note = "Google rejected the key (is the Maps JavaScript API enabled?)"; })
    .then((lib) => {
      svc = new lib.StreetViewService();
      sources = [lib.StreetViewSource.GOOGLE, lib.StreetViewSource.OUTDOOR];   // intersection: Google's outdoor imagery
      pano = new lib.StreetViewPanorama(box, { ...PANO_OPTS, visible: false, zoom: ZOOM });
      pano.addListener("pano_changed", () => { if (here && pano.getPano() !== here.id) pano.setPano(here.id); });  // never wander off
      note = "";
    })
    .catch((e) => { note = String(e); });

  async function lookup(c: { x: number; n: number }) {
    busy = true;
    askedAt = c;
    const ll = toLatLng(c.x, c.n);
    try {
      if (!inScope(ll.lat, ll.lng)) { here = null; note = "outside permitted area (Moss St, Folsom to Howard)"; return; }
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const d = await svc.getPanorama({ location: ll, radius: RADIUS_M, sources }).then((r: any) => r.data, () => null);
      const p = d?.location?.latLng;
      if (!d || !p) { here = null; note = `no Street View panorama within ${RADIUS_M} m`; return; }
      if (!/Google/.test(d.copyright ?? "")) { here = null; note = "nearest panorama is not Google's"; return; }
      if (!inScope(p.lat(), p.lng())) { here = null; note = "outside permitted area (nearest panorama is past the fence)"; return; }
      note = "";
      if (here?.id === d.location.pano) return;
      here = { id: d.location.pano, lat: p.lat(), lng: p.lng(), date: d.imageDate ?? "" };
      pano.setPano(here.id);
      povSet = "";
    } finally {
      busy = false;
      pano?.setVisible(!!here);
    }
  }

  function pov() { return { heading: walker.heading, pitch: 0, zoom: ZOOM }; }

  function offset() {   // where the panorama sits relative to the sim camera, in metres along and right of the heading
    if (!here) return null;
    const c = cam(), p = toSim(here), h = (walker.heading * Math.PI) / 180;
    const de = p.x - c.x, dn = p.y - c.n;
    return { dist: Math.hypot(de, dn), ahead: de * Math.sin(h) + dn * Math.cos(h), right: de * Math.cos(h) - dn * Math.sin(h) };
  }

  async function save(png: string) {   // png: the sim canvas only, never the panorama
    const stamp = new Date().toISOString().replace(/[:.]/g, "-");
    const text = window.prompt("Note for this view (optional):", "") ?? null;
    window.dispatchEvent(new Event("blur"));   // the prompt swallowed the keyup: main.ts clears held keys on blur
    if (text === null) return;                 // cancelled
    const c = cam(), ll = toLatLng(c.x, c.n), off = offset();
    const reopen = new URL(location.href);
    reopen.searchParams.set("at", `${walker.x.toFixed(2)},${walker.n.toFixed(2)},${walker.heading.toFixed(1)}`);
    const line = {
      time: new Date().toISOString(), note: text, sim_png: `${stamp}_sim.png`,
      walker: { x: +walker.x.toFixed(2), n: +walker.n.toFixed(2), heading: +walker.heading.toFixed(1) },
      camera: { lat: +ll.lat.toFixed(7), lng: +ll.lng.toFixed(7) },
      pano: here && { id: here.id, lat: here.lat, lng: here.lng, date: here.date, offset_m: off && +off.dist.toFixed(2) },
      pov: { heading: +walker.heading.toFixed(1), pitch: 0, zoom: +ZOOM.toFixed(3), hfov: +HFOV_DEG.toFixed(1) },
      reopen: reopen.toString(),
      maps: here && `https://www.google.com/maps/@?api=1&map_action=pano&pano=${here.id}&heading=${walker.heading.toFixed(0)}&pitch=0&fov=${HFOV_DEG.toFixed(0)}`,
    };
    const r = await fetch("/__compare", { method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ stamp, png, line }) }).catch(() => null);
    if (r?.ok) { say(`saved out/compare/${stamp}_sim.png`); return; }
    // No dev-server endpoint (vite preview / build): download both instead.
    for (const [href, name] of [[png, `${stamp}_sim.png`],
      [URL.createObjectURL(new Blob([JSON.stringify(line) + "\n"])), `${stamp}_note.jsonl`]]) {
      Object.assign(document.createElement("a"), { href, download: name }).click();
    }
    say("no /__compare endpoint: downloaded instead");
  }

  window.addEventListener("keydown", (e) => {
    const k = e.key.toLowerCase();
    if (e.repeat) return;
    if (k === "x") { split = (split + 1) % SPLITS.length; layout(); }
    if (k === "p" && here) {   // put the drone camera on the panorama
      const p = toSim(here), h = (walker.heading * Math.PI) / 180;
      Object.assign(walker, { x: p.x + Math.sin(h) * DRONE_BEHIND, n: p.y + Math.cos(h) * DRONE_BEHIND });
    }
    if (k === "c") saving = true;   // captured right after the next draw, while the canvas still has its pixels
  });

  return {
    /** Call right after the sim draws its screen. */
    frame() {
      if (saving) { saving = false; void save(canvas.toDataURL("image/png")).catch((e) => say(`save failed: ${e}`)); }
      if (svc && !busy) {
        const c = cam();
        if (!askedAt || Math.hypot(c.x - askedAt.x, c.n - askedAt.n) > REQUERY_M) void lookup(c);
      }
      const key = `${here?.id}|${walker.heading.toFixed(1)}`;
      if (pano && here && key !== povSet) { pano.setPov(pov()); povSet = key; }   // only on change, so a mouse drag holds
      const off = offset();
      label.textContent =
        `Street View compare\n` +
        (here ? `pano ${here.id}${here.date ? `  (${here.date})` : ""}\n` +
          `pano is ${off!.ahead.toFixed(1)} m ahead, ${off!.right.toFixed(1)} m right of the sim camera\n` : "") +
        `heading ${walker.heading.toFixed(0)}°   hfov ${HFOV_DEG.toFixed(0)}°   zoom ${ZOOM.toFixed(2)}\n` +
        (note ? `${note}\n` : "") +
        (flash && performance.now() - flashAt < 4000 ? `${flash}\n` : "") +
        `\nX resize sides   P sim camera to pano   C save sim shot + note`;
    },
  };
}
