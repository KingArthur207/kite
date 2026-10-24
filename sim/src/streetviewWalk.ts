// ?world=streetview: walk Moss St on Google Street View (Maps JavaScript API panorama), real imagery instead of the
// 3D worlds. Google's Cal Hacks reps allowed Street View, including our detector on the imagery, for this one street
// only (docs/google-permission.md), so only Google's own panoramas inside SCOPE are ever shown or stepped to.
//
// Up / Down step to the linked panorama nearest the view heading (or behind it), Left / Right turn TURN_STEP,
// B black screen, V reveal, H one debug crowd hazard (as in the 3D sim).
// Earpiece: `position` (the panorama's lat/lng, view heading, "Moss St") and `status`, like main.ts.
// Detector: the page sends {type: "sv_view", pano, lat, lng, heading} to the relay; sim/streetview_bridge.py fetches
// the Static API images for it and publishes them as the drone and chest frames. The page sends no pixels.
import { EARPIECE_URL, POSITION_HZ, RELAY_URL, toSim } from "./config";
import { reconnectingSocket } from "./net";
import route from "../public/osm/route.json";
import { inScope, loadStreetView, PANO_OPTS } from "./googleStreetView";

// ponytail: no @types/google.maps; the handful of calls below are checked by hand
// eslint-disable-next-line @typescript-eslint/no-explicit-any
declare const google: any;

const TURN_STEP = 20;           // degrees per Left / Right press
const STEP_MAX_OFF = 60;        // a link more than this off the wanted direction is not "ahead"
const WALKING_S = 1.5;          // status stays "walking" this long after a step
const VIEW_SETTLE_MS = 300;     // send sv_view once the view holds this long (turning fast costs no images)
const VIEW_RESEND_MS = 2000;    // and again every so often, for a bridge that started late

type LatLng = { lat: number; lng: number };
type Pano = LatLng & { id: string; date: string; links: { pano: string; heading: number }[] };

const query = new URLSearchParams(location.search);
const view = document.getElementById("view")!;
const hud = document.getElementById("hud")!;
hud.style.zIndex = "10";                         // above the panorama
const relay = reconnectingSocket(query.get("relay") ?? RELAY_URL);
const ear = reconnectingSocket(query.get("ear") ?? EARPIECE_URL);

const pts = (route as { points: LatLng[] }).points;
const counts = { serviceRequests: 0, panoramaLoads: 0, panosShown: 0, refused: 0, viewsSent: 0 };
const info = new Map<string, Pano | null>();     // panorama id -> its data, null if refused (not Google's or outside)
let svc: any, pano: any, here: Pano | null = null;
let heading = compass(pts[0], pts[1]);           // NW along Moss
let black = false, walkedAt = -1e9, lastDebug = -1e9, status = "", note = "loading Street View";

function compass(a: LatLng, b: LatLng): number {
  const dn = b.lat - a.lat, de = (b.lng - a.lng) * Math.cos((a.lat * Math.PI) / 180);
  return ((Math.atan2(de, dn) * 180) / Math.PI + 360) % 360;
}
const angleOff = (a: number, b: number) => Math.abs(((a - b + 540) % 360) - 180);
const AHEAD_M = [8, 15, 22];   // no linked panorama that way: look for one this far ahead (Howard is ~20 m)
const AHEAD_RADIUS_M = 12;
function ahead(p: LatLng, deg: number, m: number): LatLng {
  const r = (deg * Math.PI) / 180;
  return { lat: p.lat + (m * Math.cos(r)) / 111320, lng: p.lng + (m * Math.sin(r)) / (111320 * Math.cos((p.lat * Math.PI) / 180)) };
}

/** A panorama's data if it is Google's own and inside SCOPE; null otherwise. One request per panorama. */
async function check(req: object, id?: string): Promise<Pano | null> {
  if (id && info.has(id)) return info.get(id)!;
  counts.serviceRequests++;
  const d = await svc.getPanorama(req).then((r: any) => r.data, () => null);
  const ll = d?.location?.latLng;
  const ok = d && ll && /Google/.test(d.copyright ?? "") && inScope(ll.lat(), ll.lng());
  const p: Pano | null = ok ? { id: d.location.pano, lat: ll.lat(), lng: ll.lng(), date: d.imageDate ?? "",
    links: (d.links ?? []).map((l: any) => ({ pano: l.pano, heading: l.heading })) } : null;
  if (!p && d) counts.refused++;     // found, but not Google's or outside the fence
  info.set(id ?? d?.location?.pano ?? "?", p);
  if (p) info.set(p.id, p);
  return p;
}

function show(p: Pano) {
  here = p;
  pano.setPano(p.id);
  pano.setPov({ heading, pitch: 0 });
}

async function step(dir: 1 | -1) {
  if (!here) return;
  const want = dir === 1 ? heading : (heading + 180) % 360;
  const links = here.links.filter((l) => angleOff(l.heading, want) <= STEP_MAX_OFF)
    .sort((a, b) => angleOff(a.heading, want) - angleOff(b.heading, want));
  for (const l of links) {
    const p = await check({ pano: l.pano }, l.pano);
    if (p) { show(p); walkedAt = performance.now(); note = ""; return; }
  }
  // No link that way (Moss's panoramas don't link to Howard's at the corner): the nearest Google panorama a few
  // metres ahead, if it is ahead and not this one.
  for (const m of AHEAD_M) {
    const at = ahead(here, want, m);
    if (!inScope(at.lat, at.lng)) break;
    const p = await check({ location: at, radius: AHEAD_RADIUS_M, source: google.maps.StreetViewSource.OUTDOOR });
    if (p && p.id !== here.id && angleOff(compass(here, p), want) <= STEP_MAX_OFF) {
      show(p); walkedAt = performance.now(); note = ""; return;
    }
  }
  note = links.length ? "end of Moss St this way (outside the Street View fence)" : "no street this way";
}

function turn(deg: number) {
  heading = (heading + deg + 360) % 360;
  pano?.setPov({ heading, pitch: 0 });
}

async function start() {
  const lib = await loadStreetView(() => { note = "Google rejected the key (is the Maps JavaScript API enabled?)"; });
  svc = new lib.StreetViewService();
  const first = await check({ location: pts[0], radius: 15, source: lib.StreetViewSource.OUTDOOR });
  if (!first) { note = "no Google panorama at the route start"; return; }
  pano = new lib.StreetViewPanorama(view, {
    pano: first.id, pov: { heading, pitch: 0 }, zoom: 0, ...PANO_OPTS,
  });
  counts.panoramaLoads++;
  here = first;
  pano.addListener("pano_changed", () => {
    const id = pano.getPano();
    if (!info.get(id)) { if (here) pano.setPano(here.id); return; }   // never leave the fence
    counts.panosShown++;
  });
  pano.addListener("pov_changed", () => { heading = (pano.getPov().heading + 360) % 360; });  // mouse drags too
  note = "";
}

// --- sockets
let sent = "", pending = "", pendingAt = 0, sentAt = 0;
setInterval(() => {
  if (!here) return;
  const msg = JSON.stringify({ type: "sv_view", pano: here.id, lat: here.lat, lng: here.lng, heading: Math.round(heading) % 360 });
  const now = performance.now();
  if (msg !== pending) { pending = msg; pendingAt = now; }
  if (now - pendingAt >= VIEW_SETTLE_MS && (msg !== sent || now - sentAt >= VIEW_RESEND_MS)) {
    if (msg !== sent) counts.viewsSent++;
    relay.send(msg); sent = msg; sentAt = now;
  }
}, 100);

setInterval(() => {
  if (!here) return;
  ear.json({ type: "position", source: "sim", heading, place: "Moss St", lat: here.lat, lng: here.lng, ...toSim(here) });
  const s = performance.now() - walkedAt < WALKING_S * 1000 ? "walking" : "stopped";
  if (s !== status) { status = s; ear.json({ type: "status", source: "sim", state: s }); }
}, 1000 / POSITION_HZ);

// --- input
window.addEventListener("keydown", (e) => {
  if (e.key.startsWith("Arrow")) e.preventDefault();
  if (e.key === "ArrowUp" && !e.repeat) void step(1);
  if (e.key === "ArrowDown" && !e.repeat) void step(-1);
  if (e.key === "ArrowLeft") turn(-TURN_STEP);
  if (e.key === "ArrowRight") turn(TURN_STEP);
  const k = e.key.toLowerCase();
  if (k === "b") black = true;
  if (k === "v") black = false;
  if (k === "h" && !e.repeat) {
    // DEBUG ONLY (skeleton gate): real hazards come from the detector.
    ear.json({ type: "hazard", source: "sim", kind: "crowd", bearing: 0, distance: 5 });
    lastDebug = performance.now();
  }
  view.classList.toggle("hidden", black);
}, true);   // capture: the panorama never sees the arrow keys

function drawHud() {
  const p = here;
  const dbg = performance.now() - lastDebug < 1500 ? "  (debug hazard sent)" : "";
  hud.textContent =
    `Street View walk, Moss St${p?.date ? `   imagery ${p.date}` : ""}\n` +
    (p ? `lat/lng: ${p.lat.toFixed(6)}, ${p.lng.toFixed(6)}   heading ${heading.toFixed(0)}°\n` : "") +
    `relay ${relay.open ? "on" : "off"}   earpiece ${ear.open ? "on" : "off"}   ${status}\n` +
    (note ? `${note}\n` : "") +
    (black ? "BLACK SCREEN (V to reveal)\n" : "") +
    `\nUp / Down next / previous panorama   Left / Right turn ${TURN_STEP}°\nB black screen   V reveal\n` +
    `H  DEBUG: send one fake crowd hazard${dbg}`;
  requestAnimationFrame(drawHud);
}
requestAnimationFrame(drawHud);

// Dev checks drive the walk and read the request counts.
(window as any).__kiteSv = { counts, step, turn, get here() { return here; }, get heading() { return heading; },
  get note() { return note; } };
start().catch((e) => { note = String(e); });
