// ?world=streetview is a separate walk on Google Street View (streetviewWalk.ts); every other world is main.ts.
if (new URLSearchParams(location.search).get("world") === "streetview") void import("./streetviewWalk");
else void import("./main");
