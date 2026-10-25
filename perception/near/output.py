"""Where the priorities go: a plain terminal line, an ASCII radar, and the earpiece's `obstacles` contract.

Earpiece mapping (no sonar change). The sonar (earpiece/earpiece/sonar.py, read-only here) turns an object into
urgency u = max(u_ttc, 0.85 u_dist) * corridor * (0.5 + 0.5 confidence), then pitch = 520 + 730 u Hz and
interval = 1.0 - 0.89 u^0.8 s. Given a closing speed it uses it instead of estimating one. So each priority is
sent as a canonical (distance, closing speed) that lands on a fixed urgency:

    P1  1.3 m at 1.2 m/s   ttc 1.08 s -> u 0.97 -> 1225 Hz every 0.13 s   (not "critical": no spoken "Stop.")
    P2  3.0 m at 1.2 m/s   ttc 2.50 s -> u 0.40 ->  800 Hz every 0.57 s
    P3  4.0 m at 1.2 m/s   u_dist     -> u 0.15 ->  625 Hz every 0.81 s

confidence is sent as 1.0 (the priority state machine already required agreement over frames), the bearing is
the real one but clamped so its lateral offset at the canonical distance stays inside the sonar's 0.55 m
corridor (the sign, left or right, is kept). When the measured time to contact is shorter than the canonical
one, the closing speed is raised to canonical_distance / ttc so the sonar's own ttc equals ours, and below 1 s
the sonar goes critical and says "Stop." on its own. Each object also carries "priority" (ignored by today's
parser) so a future sonar can key on it directly; the PR describes that change.
"""

from __future__ import annotations

import math
import sys

CANONICAL = {1: (1.3, 1.2), 2: (3.0, 1.2), 3: (4.0, 1.2)}
SONAR_HALF_M = 0.5          # stay inside the sonar's corridor_half_width_m (0.55) with a margin
SOURCE = "near"


def to_obstacles(near_event: dict, seq: int, source: str = SOURCE) -> dict:
    """A `near` frame event -> one earpiece `obstacles` message (every in-corridor object with a priority)."""
    objs = []
    for o in near_event.get("objects", []):
        p = o.get("priority")
        if p not in CANONICAL:
            continue
        d, cs = CANONICAL[p]
        ttc = o.get("ttc_s")
        if ttc is not None and ttc > 0 and d / ttc > cs:
            cs = min(15.0, d / ttc)
        b = float(o.get("bearing") or 0.0)
        lim = math.degrees(math.asin(min(1.0, SONAR_HALF_M / d)))
        b = max(-lim, min(lim, b))
        b = math.copysign(math.floor(abs(b) * 10) / 10, b)      # round toward the line, never past the limit
        objs.append({"id": str(o["id"]), "kind": o["kind"], "bearing": b, "distance": d,
                     "closing_speed": round(cs, 2), "confidence": 1.0, "priority": p})
    return {"type": "obstacles", "source": source, "seq": seq, "objects": objs}


# ------------------------------------------------------------------ terminal
COLORS = {1: "\033[1;31m", 2: "\033[33m", 3: "\033[36m", None: "\033[2m"}
RESET = "\033[0m"


def line(ev: dict, color: bool = False) -> str:
    """One plain line per frame: the top priority, why, and every cue that fired."""
    p = ev.get("priority")
    tag = f"P{p}" if p else "--"
    if color:
        tag = f"{COLORS[p]}{tag}{RESET}"
    if p is None:
        return f"t={ev['t']:7.2f}  {tag}  clear ahead"
    cues = " | ".join(f"{k} {v:.1f}" for k, v in (ev.get("cues") or {}).items())
    ttc = f"  ttc {ev['ttc_s']:.1f} s" if ev.get("ttc_s") is not None else ""
    d = ev.get("distance_m")
    n = sum(1 for o in ev.get("objects", []) if o.get("priority"))
    more = f"  (+{n - 1} more)" if n > 1 else ""
    return (f"t={ev['t']:7.2f}  {tag}  {ev.get('kind')} ~{d:.1f} m{ttc}  conf {ev.get('confidence') or 0:.2f}"
            f"  [{cues}]{more}")


def radar(ev: dict, far_m: float = 8.0, rows: int = 12, half_cols: int = 10, lat_m: float = 2.0,
          color: bool = False) -> str:
    """Top-down ASCII radar: the walker at the bottom, the +-0.5 m corridor as two rails, objects as their
    priority digit at (lateral, forward). Rings at 2 m and 5 m are the P1 / P2 boundaries."""
    cols = 2 * half_cols + 1
    grid = [[" "] * cols for _ in range(rows)]
    rail = int(round(0.5 / lat_m * half_cols))
    row_of = lambda d: rows - 1 - min(rows - 1, int(d / far_m * rows))      # the row an object at d lands in
    rings = {row_of(2.0): " 2 m", row_of(5.0): " 5 m"}
    for r in range(rows):
        for c in range(cols):
            grid[r][c] = "-" if r in rings else " "
        grid[r][half_cols - rail] = grid[r][half_cols + rail] = "|"
    for o in sorted(ev.get("objects", []), key=lambda o: -(o.get("priority") or 9)):
        d = o.get("distance_m")
        if d is None or d > far_m:
            continue
        r = row_of(d)
        c = half_cols + int(round(max(-lat_m, min(lat_m, o.get("lateral_m") or 0.0)) / lat_m * half_cols))
        p = o.get("priority")
        ch = str(p) if p else "."
        grid[r][max(0, min(cols - 1, c))] = f"{COLORS[p]}{ch}{RESET}" if color else ch
    labels = {**rings, 0: f" {far_m:g} m"}
    out = [f"  {''.join(row)}{labels.get(i, '')}" for i, row in enumerate(grid)]
    out.append("  " + " " * half_cols + "^" + "  you")
    return "\n".join(out)


class Terminal:
    """Plain lines when piped (one per priority change, plus a heartbeat every 2 s), a redrawn radar on a TTY."""

    def __init__(self, show_radar: bool = False, stream=None):
        self.stream = stream or sys.stdout
        self.tty = hasattr(self.stream, "isatty") and self.stream.isatty()
        self.show_radar = show_radar and self.tty
        self.last_key, self.last_t = None, -1e9

    def show(self, ev: dict) -> None:
        key = (ev.get("priority"), ev.get("kind"), ev.get("id"))
        if self.show_radar:
            self.stream.write("\033[H\033[2J" + line(ev, True) + "\n\n" + radar(ev, color=True) + "\n")
            self.stream.flush()
            return
        if key != self.last_key or ev["t"] - self.last_t >= 2.0:
            self.stream.write(line(ev, self.tty) + "\n")
            self.stream.flush()
            self.last_key, self.last_t = key, ev["t"]
