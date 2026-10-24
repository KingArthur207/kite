// ?world=streetview is a separate walk on Google Street View (streetviewWalk.ts); every other world is main.ts.
// With no ?world, the Moss splat is the default when public/local/moss.splat exists (it is gitignored;
// scripts/splat/ makes it). Vite answers a missing file with index.html, hence the content-type check.
const query = new URLSearchParams(location.search);
const LOCAL_SPLAT = "/local/moss.splat";

async function start() {
  if (query.get("world") === "streetview") return import("./streetviewWalk");
  if (!query.has("world") && query.get("metrics") !== "1") {
    const r = await fetch(LOCAL_SPLAT, { method: "HEAD" }).catch(() => null);
    if (r?.ok && !(r.headers.get("content-type") ?? "").includes("html")) {
      query.set("world", "splat");
      history.replaceState(null, "", `?${query}`);
    }
  }
  return import("./main");
}
void start();
