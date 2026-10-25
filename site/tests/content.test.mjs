// Page content rules for /how and /judges: the comic's kite/drone avatar, track order, evidence and logos.
// Run: node --test site/tests/*.test.mjs   (after python3 site/tools/build_pages.py)
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync, statSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const PUBLIC = join(dirname(fileURLToPath(import.meta.url)), "..", "public");
const read = (p) => readFileSync(join(PUBLIC, p), "utf8");
const css = read("styles.css");

test("every comic panel's Kite is both the kite and the drone", () => {
  const panels = read("how.html").match(/<figure class="panel">[\s\S]*?<\/figure>/g);
  assert.equal(panels.length, 6);
  for (const p of panels) {
    assert.match(p, /class="kb-kite"/);
    assert.match(p, /class="kb-drone"/);
    assert.equal((p.match(/class="kb-prop/g) || []).length, 4, "four propellers");
    assert.match(p, /class="kb-tail"/);
  }
});

test("the kite/drone flip is CSS-only, about 1.8 s a side, with a still fallback for reduced motion", () => {
  assert.match(css, /@keyframes kb-flip/);
  const dur = css.match(/\.kb-kite, \.scene-svg \.kb-drone \{[^}]*animation: kb-flip ([\d.]+)s/);
  assert.ok(dur, "kb-flip animation on both");
  const half = Number(dur[1]) / 2;
  assert.ok(half >= 1.5 && half <= 2, `each side shows for ${half} s`);
  assert.match(css, /\.kb-drone \{ animation-delay: -1\.8s; \}/);
  const reduced = css.match(/@media \(prefers-reduced-motion: reduce\) \{\s*\.scene-svg \.kb-kite \{ display: none; \}\s*\.scene-svg \.kb-tail \{ display: inline; \}/);
  assert.ok(reduced, "reduced motion shows the drone flying a tiny kite");
  assert.doesNotMatch(read("app.js"), /kb-(kite|drone|flip)/, "no JS involved");
});

const judges = read("judges.html");
const SPONSOR = ["cartesia", "spacetime", "anthropic", "spacexai"];
const CALHACKS = ["people", "hardware", "complex", "creative", "overall", "commit"];
const card = (id) => judges.match(new RegExp(`<article class="card track" id="${id}">[\\s\\S]*?</article>`))[0];

test("/judges cards run from most to least likely to win, and the contents list matches", () => {
  const cards = [...judges.matchAll(/<article class="card track" id="(\w+)">/g)].map((m) => m[1]);
  assert.deepEqual(cards, [...SPONSOR, ...CALHACKS]);
  const toc = judges.match(/<nav class="toc"[\s\S]*?<\/nav>/)[0];
  assert.deepEqual([...toc.matchAll(/href="#(\w+)"/g)].map((m) => m[1]), [...SPONSOR, ...CALHACKS]);
  const calhacks = judges.indexOf('id="calhacks"');
  assert.ok(judges.indexOf('id="spacexai"') < calhacks && calhacks < judges.indexOf('id="people"'));
});

test("every sponsor card carries its official logo: local, small, with alt text", () => {
  for (const id of SPONSOR) {
    const img = card(id).match(/<img class="track-logo" src="([^"]+)" alt="([^"]+)" width="\d+" height="\d+" \/>/);
    assert.ok(img, `${id} logo`);
    assert.match(img[1], /^\/assets\/logos\/[\w-]+\.(svg|png)$/);
    assert.match(img[2], / logo$/);
    const file = join(PUBLIC, img[1]);
    assert.ok(existsSync(file), img[1]);
    assert.ok(statSync(file).size < 20 * 1024, `${img[1]} is ${statSync(file).size} bytes`);
    if (file.endsWith(".svg")) assert.doesNotMatch(readFileSync(file, "utf8"), /<script|<foreignObject|href="http/i);
  }
  for (const id of CALHACKS) assert.doesNotMatch(card(id), /track-logo/, `${id} has no sponsor logo`);
});

test("evidence images are real local files, at most 900 px wide, each with alt text", () => {
  const imgs = [...judges.matchAll(/<img src="(\/assets\/evidence\/[^"]+)" alt="([^"]{40,})" width="(\d+)" height="\d+" loading="lazy" \/>/g)];
  assert.deepEqual(imgs.map((m) => m[1]).sort(), ["/assets/evidence/live-dashboard.jpg", "/assets/evidence/yolo-chest-view.jpg", "/assets/evidence/yolo-drone-view.jpg"]);
  for (const [, src, , w] of imgs) {
    assert.ok(Number(w) <= 900, src);
    assert.ok(existsSync(join(PUBLIC, src)), src);
    assert.ok(statSync(join(PUBLIC, src)).size < 200 * 1024, src);
  }
  assert.match(card("spacetime"), /live-dashboard\.jpg/);
  assert.match(card("complex"), /yolo-drone-view\.jpg/);
  assert.match(card("cartesia"), /href="\/#hear"/, "Cartesia links to the walk instead of copying it");
});

test("the SpaceXAI card names Cursor, Grok Bot and Grok Imagine", () => {
  const c = card("spacexai");
  assert.match(c, /built Kite in Cursor and used Grok Bot for research, planning and team collaboration/);
  assert.match(c, /Grok Imagine/);
  assert.equal((c.match(/\/assets\/imagined\//g) || []).length, 3);
});
