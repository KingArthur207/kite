// Tests for site/functions/_middleware.js: Markdown negotiation, Vary, and real 404s.
// Run: node --test site/tests/*.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { onRequest, quality, wantsMarkdown, pagePath, PAGES, SECURITY_HEADERS } from "../functions/_middleware.js";

const PUBLIC = join(dirname(fileURLToPath(import.meta.url)), "..", "public");

// A stand-in for Cloudflare Pages: ASSETS serves files from site/public; next() mimics Pages'
// static routing (/how -> how.html, unknown -> 404.html with status 404).
function context(path, accept, method = "GET") {
  const request = new Request(`https://trykite.pages.dev${path}`, { method, headers: accept ? { Accept: accept } : {} });
  const file = (p) => { try { return readFileSync(join(PUBLIC, p)); } catch { return null; } };
  return {
    request,
    env: { ASSETS: { fetch: async (req) => {
      const body = file(new URL(req.url).pathname);
      return body ? new Response(body, { status: 200 }) : new Response("missing", { status: 404 });
    } } },
    next: async () => {
      const p = new URL(request.url).pathname.replace(/\/+$/, "") || "/";
      const name = p === "/" ? "index.html" : p.endsWith(".html") || p.includes(".") ? p.slice(1) : `${p.slice(1)}.html`;
      const body = file(name);
      if (body) return new Response(body, { status: 200, headers: { "Content-Type": name.endsWith(".html") ? "text/html; charset=utf-8" : "application/octet-stream" } });
      return new Response(file("404.html"), { status: 404, headers: { "Content-Type": "text/html; charset=utf-8" } });
    },
  };
}

test("q-values: exact type beats family beats */*", () => {
  assert.equal(quality("text/markdown", "text/markdown"), 1);
  assert.equal(quality("text/html,*/*;q=0.8", "text/markdown"), 0.8);
  assert.equal(quality("text/*;q=0.5", "text/markdown"), 0.5);
  assert.equal(quality("text/markdown;q=0", "text/markdown"), 0);
  assert.equal(quality("", "text/html"), 1);
});

test("who gets Markdown", () => {
  assert.equal(wantsMarkdown("text/markdown"), true);
  assert.equal(wantsMarkdown("text/markdown, text/html;q=0.9"), true);
  assert.equal(wantsMarkdown("text/markdown;q=0.5, text/html"), false);
  assert.equal(wantsMarkdown("text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"), false); // a browser
  assert.equal(wantsMarkdown("*/*"), false);   // curl's default stays HTML
  assert.equal(wantsMarkdown(null), false);
  assert.equal(wantsMarkdown("text/markdown;q=0"), false);
});

test("page paths are canonicalised", () => {
  assert.equal(pagePath("/"), "/");
  assert.equal(pagePath("/index.html"), "/");
  assert.equal(pagePath("/how/"), "/how");
  assert.equal(pagePath("/judges.html"), "/judges");
  assert.equal(pagePath("/nope"), null);
  assert.equal(pagePath("/styles.css"), null);
});

for (const [path, md] of Object.entries(PAGES)) {
  test(`${path} as Markdown: 200, text/markdown, Vary: Accept, the page's own text`, async () => {
    const res = await onRequest(context(path, "text/markdown"));
    assert.equal(res.status, 200);
    assert.match(res.headers.get("Content-Type"), /^text\/markdown/);
    assert.match(res.headers.get("Vary"), /Accept/);
    const body = await res.text();
    assert.equal(body, readFileSync(join(PUBLIC, md.slice(1)), "utf8"));
    assert.match(body, /^.*#/m);
    assert.ok(body.length > 200);
  });
  test(`${path} as HTML: unchanged page, with Vary: Accept`, async () => {
    const res = await onRequest(context(path, "text/html,application/xhtml+xml,*/*;q=0.8"));
    assert.equal(res.status, 200);
    assert.match(res.headers.get("Content-Type"), /^text\/html/);
    assert.match(res.headers.get("Vary"), /Accept/);
    assert.match(await res.text(), /<!doctype html>/i);
  });
}

test("unknown path as Markdown: real 404 with a helpful Markdown body", async () => {
  const res = await onRequest(context("/some-path-that-does-not-exist", "text/markdown"));
  assert.equal(res.status, 404);
  assert.match(res.headers.get("Content-Type"), /^text\/markdown/);
  assert.match(res.headers.get("Vary"), /Accept/);
  const body = await res.text();
  assert.ok(body.length >= 20);
  assert.match(body, /llms\.txt/);
  assert.match(body, /sitemap\.xml/);
});

test("unknown path as HTML: real 404 with the HTML error page", async () => {
  const res = await onRequest(context("/some-path-that-does-not-exist", "text/html"));
  assert.equal(res.status, 404);
  assert.match(res.headers.get("Vary"), /Accept/);
  assert.match(await res.text(), /Page not/);
});

test("HEAD gets headers and no body", async () => {
  const res = await onRequest(context("/", "text/markdown", "HEAD"));
  assert.equal(res.status, 200);
  assert.match(res.headers.get("Content-Type"), /^text\/markdown/);
  assert.equal(await res.text(), "");
});

test("static assets pass through untouched, even for Markdown clients", async () => {
  const res = await onRequest(context("/styles.css", "text/markdown"));
  assert.equal(res.status, 200);
  assert.equal(res.headers.get("Vary"), null);
});

test("existing Vary values are kept", async () => {
  const ctx = context("/", "text/html");
  const next = ctx.next;
  ctx.next = async () => { const r = await next(); const h = new Headers(r.headers); h.set("Vary", "Accept-Encoding"); return new Response(r.body, { status: r.status, headers: h }); };
  const res = await onRequest(ctx);
  assert.equal(res.headers.get("Vary"), "Accept-Encoding, Accept");
});

// The `/*` block of public/_headers, as {name: value}.
function headersFile() {
  const out = {};
  let inAll = false;
  for (const line of readFileSync(join(PUBLIC, "_headers"), "utf8").split("\n")) {
    if (!line.trim()) continue;
    if (!/^\s/.test(line)) { inAll = line.trim() === "/*"; continue; }
    if (inAll) { const i = line.indexOf(":"); out[line.slice(0, i).trim()] = line.slice(i + 1).trim(); }
  }
  return out;
}

test("the middleware's security headers are exactly public/_headers' `/*` block", () => {
  const file = headersFile();
  assert.ok(Object.keys(file).length >= 6);
  assert.deepEqual(SECURITY_HEADERS, file);
});

for (const [path, accept, status] of [["/how", "text/markdown", 200], ["/nope", "text/markdown", 404],
                                      ["/", "text/html", 200], ["/nope", "text/html", 404]]) {
  test(`${path} (${accept}) carries every security header: _headers never reaches a Function's response`, async () => {
    const res = await onRequest(context(path, accept));
    assert.equal(res.status, status);
    for (const [k, v] of Object.entries(headersFile())) assert.equal(res.headers.get(k), v, k);
  });
}

test("a security header the asset already has is kept, not overwritten", async () => {
  const ctx = context("/", "text/html");
  const next = ctx.next;
  ctx.next = async () => { const r = await next(); const h = new Headers(r.headers); h.set("X-Frame-Options", "SAMEORIGIN"); return new Response(r.body, { status: r.status, headers: h }); };
  const res = await onRequest(ctx);
  assert.equal(res.headers.get("X-Frame-Options"), "SAMEORIGIN");
});

test("other methods go straight to the next handler", async () => {
  const ctx = context("/", "text/markdown", "POST");
  let called = false;
  ctx.next = async () => { called = true; return new Response("no", { status: 405 }); };
  const res = await onRequest(ctx);
  assert.ok(called);
  assert.equal(res.status, 405);
});

test("odd Accept headers never throw and never pick Markdown by accident", () => {
  for (const a of ["", ",,,", ";q=1", "text/markdown;q=abc", "text/markdown;q=-1", "text/markdown;q=5", "*/*;q=", "a".repeat(10000)]) {
    assert.doesNotThrow(() => wantsMarkdown(a), a.slice(0, 20));
  }
  assert.equal(wantsMarkdown("text/markdown;q=abc"), false);   // an unreadable q counts as 0
  assert.equal(wantsMarkdown("text/markdown;q=-1"), false);
  assert.equal(quality("text/markdown;q=5", "text/markdown"), 1);   // clamped to 1
});

test("odd path shapes are not pages", () => {
  for (const p of ["//how", "/how.html.html", "/HOW", "/how%2F", "/constructor", "/__proto__", "/hasOwnProperty", "/toString"]) {
    assert.equal(pagePath(p), null, p);
  }
});
