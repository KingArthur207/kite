// The UNO Q installer the board fetches with `curl -fsSL https://trykite.pages.dev/unoq/install.sh | sh`.
// site/public/unoq/ must be byte-for-byte copies of hardware/arduino/unoq_link/, and install.sh must carry the
// right SHA-256 for each board file it downloads. Fix drift with: python3 hardware/arduino/unoq_link/publish.py
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { onRequest } from "../functions/_middleware.js";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const PUB = join(ROOT, "site", "public", "unoq");
const SRC = join(ROOT, "hardware", "arduino", "unoq_link");
const PUBLISHED = ["install.sh", "board.py", "kite_common.py", "kite-link.service", "kite_link.py"];
const URL_LINE = "curl -fsSL https://trykite.pages.dev/unoq/install.sh | sh";
const sha = (p) => createHash("sha256").update(readFileSync(p)).digest("hex");

test("the installer and board files are published", () => {
  for (const f of PUBLISHED) assert.ok(existsSync(join(PUB, f)), f);
  assert.match(readFileSync(join(PUB, "install.sh"), "utf8"), /^#!\/bin\/sh\n/);
});

test("published files are identical to hardware/arduino/unoq_link (no drift)", () => {
  for (const f of PUBLISHED) assert.ok(readFileSync(join(PUB, f)).equals(readFileSync(join(SRC, f))), `${f} differs: run publish.py`);
});

test("install.sh pins the SHA-256 of every file it downloads", () => {
  const block = readFileSync(join(PUB, "install.sh"), "utf8").match(/HASHES='\n([\s\S]*?)'\n/)[1];
  const pinned = Object.fromEntries(block.trim().split("\n").map((l) => l.split(/\s+/).reverse()));
  assert.deepEqual(Object.keys(pinned).sort(), ["board.py", "kite-link.service", "kite_common.py"]);
  for (const [f, h] of Object.entries(pinned)) assert.equal(h, sha(join(PUB, f)), `${f}: run publish.py`);
});

test("the READMEs point at the installer URL that exists", () => {
  assert.ok(readFileSync(join(SRC, "README.md"), "utf8").includes(URL_LINE));
  assert.ok(readFileSync(join(PUB, "install.sh"), "utf8").includes(URL_LINE));
});

test("/unoq/install.sh passes through the middleware untouched, even for Markdown clients", async () => {
  for (const accept of ["*/*", "text/markdown"]) {
    const body = readFileSync(join(PUB, "install.sh"));
    const res = await onRequest({
      request: new Request("https://trykite.pages.dev/unoq/install.sh", { headers: { Accept: accept } }),
      env: { ASSETS: { fetch: async () => new Response("wrong", { status: 500 }) } },
      next: async () => new Response(body, { status: 200, headers: { "Content-Type": "application/x-sh" } }),
    });
    assert.equal(res.status, 200);
    assert.ok(Buffer.from(await res.arrayBuffer()).equals(body));
    assert.equal(res.headers.get("Content-Type"), "application/x-sh");
  }
});
