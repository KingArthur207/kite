// The phone PWA posts a position while GPS is on. This forwards that row to
// SpacetimeDB and refuses anything else. The UNO Q is not on this path.

const PUSH = "https://maincloud.spacetimedb.com/v1/database/kite-sync-pl6rp/call/push";
const IDENTITY = "https://maincloud.spacetimedb.com/v1/identity";

let identityToken = null;

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === "/api/location") {
      if (request.method !== "POST") {
        return json({ error: "POST only" }, 405);
      }
      return relay(request);
    }
    return env.ASSETS.fetch(request);
  },
};

async function relay(request) {
  const raw = await request.arrayBuffer();
  if (raw.byteLength > 8192) {
    return json({ error: "too large" }, 413);
  }
  let body;
  try {
    body = JSON.parse(new TextDecoder().decode(raw));
  } catch {
    return json({ error: "not json" }, 400);
  }
  const reason = reject(body);
  if (reason) {
    return json({ error: reason }, 400);
  }
  const encoded = JSON.stringify(body);
  let status = 502;
  let text = "";
  for (let attempt = 0; attempt < 2; attempt++) {
    const token = await spacetimeIdentity(attempt > 0);
    const upstream = await fetch(PUSH, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: encoded,
    });
    status = upstream.status;
    text = await upstream.text();
    if (status !== 401 && status !== 403) {
      break;
    }
    identityToken = null;
  }
  return new Response(text || "{}", {
    status,
    headers: {
      "Content-Type": "application/json",
      "Cache-Control": "no-store",
    },
  });
}

export function reject(body) {
  if (!body || body.type !== "position" || body.source !== "phone" || body.priority !== "nav") {
    return "wrong envelope";
  }
  if (typeof body.payload !== "string" || body.payload.length > 2000) {
    return "bad payload";
  }
  let fix;
  try {
    fix = JSON.parse(body.payload);
  } catch {
    return "payload is not json";
  }
  if (!fix || typeof fix !== "object" || Array.isArray(fix)) {
    return "not a position";
  }
  if (!inRange(fix.lat, -90, 90) || !inRange(fix.lng, -180, 180)) {
    return "bad coordinates";
  }
  return "";
}

function inRange(value, lo, hi) {
  return typeof value === "number" && Number.isFinite(value) && value >= lo && value <= hi;
}

async function spacetimeIdentity(force) {
  if (identityToken && !force) {
    return identityToken;
  }
  const response = await fetch(IDENTITY, { method: "POST" });
  if (!response.ok) {
    throw new Error(`identity ${response.status}`);
  }
  const header = response.headers.get("spacetime-identity-token");
  let token = header || "";
  try {
    const parsed = JSON.parse(await response.text());
    token = token || parsed.token || "";
  } catch {
    token = token || "";
  }
  if (!token) {
    throw new Error("no spacetime identity");
  }
  identityToken = token;
  return token;
}

function json(obj, status) {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { "Content-Type": "application/json", "Cache-Control": "no-store" },
  });
}
