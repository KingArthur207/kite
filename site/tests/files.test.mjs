// The generated site files: pages, Markdown twins, structured data, sitemap, llms.txt.
// Run: node --test site/tests/*.test.mjs   (after python3 site/tools/build_pages.py)
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { PAGES } from "../functions/_middleware.js";

const PUBLIC = join(dirname(fileURLToPath(import.meta.url)), "..", "public");
const read = (p) => readFileSync(join(PUBLIC, p), "utf8");
const text = (html) => html.replace(/<script[\s\S]*?<\/script>/g, "").replace(/<svg[\s\S]*?<\/svg>/g, "").replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();
const ldjson = (html) => JSON.parse(html.match(/<script type="application\/ld\+json">([\s\S]*?)<\/script>/)[1]);

test("every page has HTML and a Markdown twin; 404 exists for Pages' real-404 behaviour", () => {
  for (const md of Object.values(PAGES)) {
    const name = md.slice(1, -3);
    assert.ok(existsSync(join(PUBLIC, `${name}.html`)), `${name}.html`);
    assert.ok(read(md.slice(1)).trim().length > 200, md);
  }
  assert.ok(existsSync(join(PUBLIC, "404.html")));
  assert.match(read("404.md"), /llms\.txt/);
  assert.match(read("404.html"), /<meta name="robots" content="noindex"/);
});

test("trust pages have real content (500+ characters each)", () => {
  for (const name of ["about", "contact", "privacy"]) {
    const main = read(`${name}.html`).match(/<main[\s\S]*<\/main>/)[0];
    assert.ok(text(main).length >= 500, `${name}: ${text(main).length}`);
  }
});

test("Organization JSON-LD on every page, with a contact point", () => {
  for (const md of [...Object.values(PAGES), "/404.md"]) {
    const html = read(`${md.slice(1, -3)}.html`);
    const graph = ldjson(html)["@graph"];
    const org = graph.find((n) => n["@type"] === "Organization");
    assert.ok(org, md);
    assert.equal(org.name, "Kite");
    assert.equal(org.contactPoint[0]["@type"], "ContactPoint");
    assert.ok(org.contactPoint[0].contactType);
    assert.ok(graph.find((n) => n["@type"] === "WebSite"));
  }
});

test("pages link their Markdown version and llms.txt", () => {
  for (const md of Object.values(PAGES)) {
    const html = read(`${md.slice(1, -3)}.html`);
    assert.match(html, new RegExp(`<link rel="alternate" type="text/markdown" href="${md}" />`));
    assert.match(html, /href="\/llms\.txt"/);
  }
});

test("sitemap lists every indexable page and nothing else", () => {
  const xml = read("sitemap.xml");
  for (const path of Object.keys(PAGES)) assert.match(xml, new RegExp(`<loc>https://trykite\\.pages\\.dev${path === "/" ? "/" : path}</loc>`));
  assert.doesNotMatch(xml, /404/);
});

test("llms.txt follows the format and says when to use the site", () => {
  const t = read("llms.txt");
  assert.match(t, /^# Kite\n\n> /);
  assert.match(t, /^## When to use this site$/m);
  assert.match(t, /^## How to read it$/m);
  assert.match(t, /^## Pages$/m);
  for (const md of Object.values(PAGES)) assert.ok(t.includes(`https://trykite.pages.dev${md}`), md);
  assert.match(read("robots.txt"), /Sitemap: https:\/\/trykite\.pages\.dev\/sitemap\.xml/);
});

test("titles name the product, not just the brand", () => {
  const title = read("index.html").match(/<title>(.*?)<\/title>/)[1];
  assert.match(title, /Kite/);
  assert.match(title, /blind/i);
});

test("security headers stay strict", () => {
  const h = read("_headers");
  assert.match(h, /Cross-Origin-Opener-Policy: same-origin/);
  assert.match(h, /Content-Security-Policy: default-src 'self'; script-src 'self';/);
  assert.match(h, /X-Frame-Options: DENY/);
});
