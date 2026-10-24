// ?walker=route: the walker walks the Moss demo route on its own, for drone-in-the-loop runs.
//   walk=0.7   speed in m/s (default 1.0)
//   curb=3     seconds stopped at the curb before the zebra (the route's turn), default 3
//   autostart=1  start at once; otherwise it waits at the start for {"type":"walker_cmd","cmd":"go"} on the relay
// Straight up Moss's NE sidewalk, stop at the curb, left turn across the zebra, then along Howard. No collisions:
// the walker is the drone's target here, not a cane user.
import { route } from "./osmWorld";
import { Path } from "./mossActors";

const TURN_DEG_S = 120;          // how fast the body turns to face the path

export function makeRouteWalker(walker: { x: number; n: number; heading: number }, q: URLSearchParams) {
  const path = new Path(route);
  const speed = Math.max(0.1, Number(q.get("walk")) || 1.0);
  const curbS = Math.max(0, Number(q.get("curb") ?? 3));
  const curbAt = path.s[1];        // route vertex 1: the corner at the zebra's NE end
  let s = 0, going = q.get("autostart") === "1", curbLeft = curbS, done = false, moving = false;

  const place = () => {
    const p = path.at(s);
    walker.x = p.x; walker.n = p.n;
  };
  const p0 = path.at(0);
  walker.heading = ((Math.atan2(p0.dx, p0.dn) * 180) / Math.PI + 360) % 360;
  place();

  return {
    go() { going = true; },
    step(dt: number) {
      moving = false;
      if (!going || done) return;
      if (s >= curbAt && curbLeft > 0) curbLeft -= dt;
      else {
        const next = Math.min(path.len, s + speed * dt);
        if (s < curbAt && next > curbAt) s = curbAt; else s = next;   // stop exactly at the curb
        moving = true;
        if (s >= path.len) done = true;
      }
      place();
      const p = path.at(s + 0.01);
      const want = (Math.atan2(p.dx, p.dn) * 180) / Math.PI;
      const err = ((want - walker.heading + 540) % 360) - 180;
      walker.heading = (walker.heading + Math.sign(err) * Math.min(Math.abs(err), TURN_DEG_S * dt) + 360) % 360;
    },
    get state() { return done ? "done" : !going ? "waiting" : moving ? "walking" : "curb"; },
    get s() { return s; },
    get moving() { return moving; },
    speed,
  };
}
