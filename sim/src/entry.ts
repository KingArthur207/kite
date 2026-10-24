// ?world=streetview is a separate walk on Google Street View (streetviewWalk.ts); every other world is main.ts.
if (new URLSearchParams(location.search).get("world") === "streetview") void import("./streetviewWalk");
// main.ts picks its default world from the photo manifests, and ?look=2 reads the building sheet, so they load first.
else void Promise.all([import("./photoWorld").then((m) => m.loadPhotoManifests()), import("./look2").then((m) => m.loadBuildingSheet())]).then(() => import("./main"));
