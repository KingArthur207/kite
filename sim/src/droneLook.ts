// Where the drone flies and looks. It follows 1 m behind the walker's head, looking ahead, except near a corner
// where the route turns: there it scouts the corner. It flies out past the corner and looks up the cross street,
// at the near lane, whose cars (right-hand traffic) come from the walker's right, hidden by the corner building
// from the walker and from any camera behind them. The detector (standin_detect.py) gets the drone's position and
// heading in its pose and warns of a car approaching the corner.
type P = [number, number] | number[];

export const LOOK_BEFORE_M = 20;   // scout from this far before the corner
export const LOOK_AFTER_M = 3;     // until this far onto the next leg (off the curb and onto the zebra)
export const SCOUT_PAST_M = 2;     // hover this far past the corner along the incoming leg (beyond the building line)
export const SCOUT_HEIGHT = 3;     // m
export const LOOK_RIGHT_DEG = 90;  // up the cross street on the right
export const FLY_SPEED = 4;        // m/s, a Tello flies up to ~8
export const YAW_RATE = 90;        // deg/s, about a Tello's
const MIN_TURN_DEG = 45, LANE_M = 3, TELEPORT_M = 30;

export interface DronePose { x: number; n: number; y: number; heading: number }   // heading: compass deg

const wrap = (d: number) => ((d + 540) % 360) - 180;
const compass = (a: P, b: P) => (Math.atan2(b[0] - a[0], b[1] - a[1]) * 180) / Math.PI;

/** Point (x, n) in leg a->b's frame, measured from `from`: metres along the leg and to its side. */
function legFrame(a: P, b: P, from: P, x: number, n: number) {
  const L = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1;
  const ux = (b[0] - a[0]) / L, un = (b[1] - a[1]) / L;
  const ex = x - from[0], en = n - from[1];
  return { along: ex * ux + en * un, lat: ex * un - en * ux, ux, un };
}

/** Where the drone should be: following behind the walker, or scouting the corner ahead. */
export function droneTarget(w: { x: number; n: number; heading: number }, route: P[], height: number, behind: number): DronePose & { scouting: boolean } {
  for (let i = 1; i + 1 < route.length; i++) {
    const [a, v, b] = [route[i - 1], route[i], route[i + 1]];
    const inH = compass(a, v);
    if (Math.abs(wrap(compass(v, b) - inH)) < MIN_TURN_DEG) continue;
    const before = legFrame(a, v, v, w.x, w.n), after = legFrame(v, b, v, w.x, w.n);
    if ((before.along > -LOOK_BEFORE_M && before.along <= 0 && Math.abs(before.lat) < LANE_M)
        || (after.along >= 0 && after.along < LOOK_AFTER_M && Math.abs(after.lat) < LANE_M)) {
      return { x: v[0] + before.ux * SCOUT_PAST_M, n: v[1] + before.un * SCOUT_PAST_M, y: SCOUT_HEIGHT,
               heading: (inH + LOOK_RIGHT_DEG + 360) % 360, scouting: true };
    }
  }
  const h = (w.heading * Math.PI) / 180;
  return { x: w.x - Math.sin(h) * behind, n: w.n - Math.cos(h) * behind, y: height, heading: w.heading, scouting: false };
}

/** One rate-limited step of the drone toward its target (FLY_SPEED, YAW_RATE). Returns a new pose. */
export function stepDrone(cur: DronePose, target: DronePose, dt: number): DronePose {
  const dx = target.x - cur.x, dn = target.n - cur.n, dy = target.y - cur.y;
  const d = Math.hypot(dx, dn, dy);
  if (d > TELEPORT_M) return { x: target.x, n: target.n, y: target.y, heading: target.heading };   // walker was moved
  const k = d > FLY_SPEED * dt ? (FLY_SPEED * dt) / d : 1;
  const dh = wrap(target.heading - cur.heading), max = YAW_RATE * dt;
  return { x: cur.x + dx * k, n: cur.n + dn * k, y: cur.y + dy * k,
           heading: (cur.heading + Math.max(-max, Math.min(max, dh)) + 360) % 360 };
}
