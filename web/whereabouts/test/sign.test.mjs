import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";

import { canonical, hmacHex } from "../public/sign.js";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..", "..");
const link = path.join(root, "hardware", "arduino", "unoq_link");

test("browser signature matches the board's Python signer", async () => {
  const body = {
    to: "unoq",
    cmd: "send",
    text: '{"kind":"location_fix","lat":37.8719,"lon":-122.2585}',
    id: "abcdabcdabcdabcd",
    ts: 1700000000000,
  };
  const token = "test-token-value-1";
  const jsSig = await hmacHex(token, canonical(body));
  const script = [
    "import json, sys",
    `sys.path.insert(0, ${JSON.stringify(link)})`,
    "from kite_common import sign",
    `body = json.loads(${JSON.stringify(JSON.stringify(body))})`,
    `print(sign(${JSON.stringify(token)}, body)["sig"])`,
  ].join("\n");
  const python = process.env.PYTHON || "python";
  const run = spawnSync(python, ["-c", script], { encoding: "utf-8" });
  assert.equal(run.status, 0, run.stderr || run.stdout);
  assert.equal(jsSig, run.stdout.trim());
});
