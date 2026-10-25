// Same signature the UNO Q already checks: HMAC-SHA256 over canonical JSON.
// Canonical form matches kite_common.canonical: sorted keys, no spaces, `sig` left out.
// The signed fields here are ASCII, so JSON.stringify matches Python's ensure_ascii dumps.

export function canonical(body) {
  const out = {};
  for (const key of Object.keys(body).filter((name) => name !== "sig").sort()) {
    out[key] = body[key];
  }
  return JSON.stringify(out);
}

export async function hmacHex(token, message) {
  const enc = new TextEncoder();
  const key = await crypto.subtle.importKey(
    "raw",
    enc.encode(token),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  const mac = await crypto.subtle.sign("HMAC", key, enc.encode(message));
  return [...new Uint8Array(mac)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

export async function signBody(token, text) {
  const bytes = crypto.getRandomValues(new Uint8Array(8));
  const body = {
    to: "unoq",
    cmd: "send",
    text,
    id: [...bytes].map((byte) => byte.toString(16).padStart(2, "0")).join(""),
    ts: Date.now(),
  };
  return { ...body, sig: await hmacHex(token, canonical(body)) };
}

export function locationText(coords, when = Date.now()) {
  const body = {
    kind: "location_fix",
    lat: round(coords.latitude, 6),
    lon: round(coords.longitude, 6),
  };
  if (Number.isFinite(coords.accuracy)) body.accuracy_m = round(coords.accuracy, 1);
  if (coords.altitude != null && Number.isFinite(coords.altitude)) body.altitude_m = round(coords.altitude, 1);
  if (coords.speed != null && Number.isFinite(coords.speed) && coords.speed >= 0) body.speed_mps = round(coords.speed, 1);
  if (coords.heading != null && Number.isFinite(coords.heading) && coords.heading >= 0) {
    body.heading_deg = round(coords.heading, 0);
  }
  body.timestamp = new Date(when).toISOString().replace(/\.\d{3}Z$/, "Z");
  const text = JSON.stringify(body);
  if (text.length > 500) {
    throw new Error("fix is too long");
  }
  return text;
}

function round(value, digits) {
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}
