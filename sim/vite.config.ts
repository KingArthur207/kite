// Dev only: POST /__compare saves a ?compare=1 shot (compare.ts) to out/compare/ at the repo root (gitignored):
// the sim canvas as <stamp>_sim.png and one JSON line in notes.jsonl. Never Street View pixels.
import { defineConfig } from "vite";
import { appendFileSync, mkdirSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const OUT = fileURLToPath(new URL("../out/compare/", import.meta.url));
const MAX_BYTES = 50_000_000;
const PNG_PREFIX = "data:image/png;base64,";

export default defineConfig({
  plugins: [{
    name: "kite-compare-save",
    configureServer(server) {
      server.middlewares.use("/__compare", (req, res) => {
        if (req.method !== "POST") { res.statusCode = 405; res.end(); return; }
        const chunks: Buffer[] = [];
        let size = 0;
        req.on("data", (c: Buffer) => { size += c.length; if (size <= MAX_BYTES) chunks.push(c); });
        req.on("end", () => {
          try {
            if (size > MAX_BYTES) throw new Error("too large");
            const { stamp, png, line } = JSON.parse(Buffer.concat(chunks).toString());
            if (typeof stamp !== "string" || !/^[\w-]{1,64}$/.test(stamp) || typeof png !== "string" || !png.startsWith(PNG_PREFIX)) {
              throw new Error("bad request");
            }
            mkdirSync(OUT, { recursive: true });
            writeFileSync(`${OUT}${stamp}_sim.png`, Buffer.from(png.slice(PNG_PREFIX.length), "base64"));
            appendFileSync(`${OUT}notes.jsonl`, JSON.stringify(line) + "\n");
            res.end("ok");
          } catch (e) {
            res.statusCode = 400;
            res.end(String(e));
          }
        });
      });
    },
  }],
});
