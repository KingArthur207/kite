// Content negotiation for agents (Cloudflare Pages Functions).
//
// Every page is also published as Markdown (/index.md, /how.md, ...), generated from the
// same source as the HTML by site/tools/build_pages.py. A request whose Accept header
// prefers text/markdown gets that Markdown; everyone else gets the HTML exactly as before.
// Both carry `Vary: Accept` so caches keep them apart. Unknown paths are a real 404: HTML
// (404.html, which Pages serves with status 404) or, for Markdown clients, a Markdown body.

export const PAGES = {
  "/": "/index.md",
  "/how": "/how.md",
  "/judges": "/judges.md",
  "/about": "/about.md",
  "/contact": "/contact.md",
  "/privacy": "/privacy.md",
};

const NOT_FOUND_MD = "/404.md";

/** The q-value the Accept header gives a media type (exact type, then type/*, then *\/*). */
export function quality(accept, type) {
  if (!accept) return type === "text/html" ? 1 : 0;
  const [major] = type.split("/");
  let exact = null, family = null, any = null;
  for (const part of accept.split(",")) {
    const [range, ...params] = part.trim().toLowerCase().split(";");
    if (!range) continue;
    let q = 1;
    for (const p of params) {
      const [k, v] = p.trim().split("=");
      if (k === "q") q = Number.isFinite(parseFloat(v)) ? Math.min(1, Math.max(0, parseFloat(v))) : 0;
    }
    if (range === type) exact = Math.max(exact ?? 0, q);
    else if (range === `${major}/*`) family = Math.max(family ?? 0, q);
    else if (range === "*/*") any = Math.max(any ?? 0, q);
  }
  return exact ?? family ?? any ?? 0;
}

/** True when the client asked for Markdown at least as much as for HTML. */
export function wantsMarkdown(accept) {
  const md = quality(accept, "text/markdown");
  if (md <= 0) return false;
  // a bare "*/*" (curl's default) or a browser's "text/html,...,*/*;q=0.8" stays HTML
  if (!accept || !/text\/markdown/i.test(accept)) return false;
  return md >= quality(accept, "text/html");
}

/** "/how/", "/how.html", "/index.html" -> the canonical page path, or null if it isn't a page. */
export function pagePath(pathname) {
  let p = pathname.replace(/\/+$/, "") || "/";
  if (p === "/index.html" || p === "/index") p = "/";
  else if (p.endsWith(".html")) p = p.slice(0, -5);
  return Object.prototype.hasOwnProperty.call(PAGES, p) ? p : null;
}

function withVary(headers) {
  const h = new Headers(headers);
  const vary = h.get("Vary");
  if (!vary) h.set("Vary", "Accept");
  else if (!/(^|,)\s*accept\s*(,|$)/i.test(vary)) h.set("Vary", `${vary}, Accept`);
  return h;
}

async function markdown(context, assetPath, status) {
  const url = new URL(assetPath, context.request.url);
  const res = await context.env.ASSETS.fetch(new Request(url, { method: "GET" }));
  const body = res.ok ? await res.text() : "# Not found\n\nThis page has no Markdown version. See https://trykite.pages.dev/llms.txt\n";
  const headers = withVary({
    "Content-Type": "text/markdown; charset=utf-8",
    "Cache-Control": "public, max-age=300",
    "X-Content-Type-Options": "nosniff",
    Link: '<https://trykite.pages.dev/llms.txt>; rel="describedby"; type="text/plain"',
  });
  return new Response(context.request.method === "HEAD" ? null : body, { status, headers });
}

export async function onRequest(context) {
  const { request } = context;
  if (request.method !== "GET" && request.method !== "HEAD") return context.next();
  const url = new URL(request.url);
  const page = pagePath(url.pathname);
  const isAsset = !page && /\.[a-z0-9]{1,8}$/i.test(url.pathname);

  if (wantsMarkdown(request.headers.get("Accept"))) {
    if (page) return markdown(context, PAGES[page], 200);
    if (!isAsset) return markdown(context, NOT_FOUND_MD, 404);
  }

  const res = await context.next();
  if (page || res.status === 404 || (res.headers.get("Content-Type") || "").startsWith("text/html")) {
    return new Response(res.body, { status: res.status, statusText: res.statusText, headers: withVary(res.headers) });
  }
  return res;
}
