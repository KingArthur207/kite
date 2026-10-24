"""Offline eval for the Claude judgment layer (earpiece/earpiece/judgment.py), with the real Claude.

    ANTHROPIC_API_KEY=... earpiece/.venv/bin/python scripts/judgment_eval.py            # all scenarios
    earpiece/.venv/bin/python scripts/judgment_eval.py moss_car_turning --repeat 3 --speed 2

Each scenario JSONL is replayed into an in-process earpiece (null audio, no network, no Cartesia:
the offline voice speaks deterministic lines). The real Judgment worker decides when to call Claude,
exactly as in a live run. Per scenario it prints the calls, latency p50/p95, what Claude said,
guardrail rejections, and a rubric verdict: what an O&M instructor would want said at that moment.
Costs a few Haiku calls per scenario (well under a cent).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "earpiece"))

from earpiece.config import Cfg  # noqa: E402  (loads .env)

os.environ.pop("CARTESIA_API_KEY", None)   # never Cartesia: the team plan allows 2 concurrent requests

from earpiece.app import Earpiece  # noqa: E402
from earpiece.events import parse  # noqa: E402
from earpiece.judgment import _FORBIDDEN  # noqa: E402
from earpiece.memory import MemoryStore  # noqa: E402
from earpiece.sim import jsonl_timeline  # noqa: E402

WAIT = r"\b(?:wait|hold|stop|pause|stay)\b"
MAX_CHARS = 70     # ~4 s of speech; a car turning in doesn't wait for a long sentence
# expect: "warn" = at least one line matching every regex in `mention`; "quiet" = nothing said;
# "any" = speaking or not are both fine. `avoid` must never be said.
RUBRIC = {
    "moss_car_turning": dict(
        expect="warn", mention=[r"\bcar\b", WAIT, r"\bright\b"],
        note="Uncontrolled zebra; a car turns in from Howard on the walker's right. A guide dog would "
             "refuse to step off. Say: wait, car turning in from your right."),
    "moss_crowd_crossing": dict(
        expect="warn", mention=[r"\b(?:crowd|people)\b", WAIT],
        note="People on the crossing coming toward the walker. Hold at the curb until it thins."),
    "moss_clear_approach": dict(
        expect="quiet",
        note="Nothing coming. The route already said where the crossing is. Kite never says it is "
             "clear, so the right call is silence."),
    "moss_parked_car": dict(
        expect="quiet",
        note="A parked car at the curb is not a hazard on the sidewalk. A warning here is noise "
             "that teaches the walker to ignore Kite."),
    "howard_cyclist_sidewalk": dict(
        expect="warn", mention=[r"\b(?:cyclist|bike|bicycle)\b"],
        note="A bike on the sidewalk closes fast and quietly; sonar only beeps inside ~5 m. Warn "
             "early: cyclist ahead on the sidewalk, coming toward you."),
    "moss_construction_sign": dict(
        expect="warn", mention=[r"\b(?:closed|cones?|construction|sign)\b"],
        avoid=r"\bother side\b|\bcross (?:the|to)\b|\bdetour\b",
        note="Read what the sign means for the walker (sidewalk closed, cones on the right) and "
             "let them decide; Kite never reroutes."),
    "sample_walk": dict(
        expect="warn", mention=[r"\bturn\w*\b", r"\b(?:cars?|vehicles?|traffic)\b"],
        avoid=r"\bwalk signal\b.*\b(?:go|cross)\b",
        note="Signalised crossing at Lyon/Bay: don't walk, then walk, then a crowd on the right. "
             "Deterministic code speaks all of it. What an instructor adds when the walk signal comes "
             "on: listen for turning cars first. Never a go."),
    "sample_track": dict(
        expect="any",
        note="Don't walk, crowd, then a car backing out (critical, already 'Stop.'). Claude may "
             "repeat a wait; it must not add anything that sounds like permission."),
}
SCENARIOS = [ROOT / "data" / "sample_walk.jsonl", ROOT / "data" / "sample_track.jsonl",
             *sorted((ROOT / "data" / "scenarios").glob("*.jsonl"))]


def pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(p * (len(xs) - 1)))]


def run(path: Path, speed: float) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="kite-eval-"))
    cfg = Cfg()
    cfg.speech.cache_dir = tmp / "cache"
    logs: list[str] = []
    app = Earpiece(cfg, output="null", network=False, log=logs.append,
                   memory=MemoryStore(project_id="", project_key="", store=tmp / "memory.json", log=logs.append))
    j = app.judgment
    if not j.online:
        sys.exit("judgment is offline: set ANTHROPIC_API_KEY")
    calls: list[dict] = []
    said: list[str] = []
    step, say = j.step, j.say

    def timed_step():
        n_log, n_said, t0 = len(logs), len(said), time.monotonic()
        out = step()
        if out not in ("unchanged", "empty"):
            calls.append({"out": out, "ms": (time.monotonic() - t0) * 1000, "said": said[n_said:],
                          "rejected": [m for m in logs[n_log:] if "rejected" in m],
                          "notes": [m for m in logs[n_log:] if "dropped" in m or "no decision" in m]})
        return out

    def spy_say(text, *a, **k):
        said.append(text)
        return say(text, *a, **k)

    j.step, j.say = timed_step, spy_say
    app.start(prewarm=False, announce=False)
    try:
        start = time.monotonic()
        for at, ev in jsonl_timeline(path):
            delay = start + at / speed - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            app.handle(parse(ev))
        time.sleep(j.min_gap_s + j.timeout_s + 1.0)          # let the last decision land
    finally:
        app.stop()
    return {"calls": calls, "said": said, "spoken": [h["text"] for h in app.speaker.history]}


def grade(name: str, said: list[str]) -> tuple[bool, str]:
    r = RUBRIC.get(name, {"expect": "any"})
    avoid = r.get("avoid")
    if avoid and any(re.search(avoid, s, re.I) for s in said):
        return False, "said something to avoid"
    if any(len(s) > MAX_CHARS for s in said):
        return False, f"over {MAX_CHARS} chars: too long to hear before the moment passes"
    if r["expect"] == "quiet":
        return (not said), "quiet" if not said else "should be quiet"
    if r["expect"] == "warn":
        hit = [s for s in said if all(re.search(m, s, re.I) for m in r.get("mention", []))]
        return bool(hit), "warned" if hit else ("said the wrong thing" if said else "missed the warning")
    return True, "ok"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("names", nargs="*", help="scenario names (default: all)")
    p.add_argument("--speed", type=float, default=2.0, help="replay speed (the first live run was 2x)")
    p.add_argument("--repeat", type=int, default=1)
    a = p.parse_args(argv)
    paths = [s for s in SCENARIOS if not a.names or s.stem in a.names]
    rows, all_ms = [], []
    for path in paths:
        for i in range(a.repeat):
            r = run(path, a.speed)
            ms = [c["ms"] for c in r["calls"] if c["out"] not in ("dropped", "late")]
            dropped = sum(c["out"] in ("dropped", "late") for c in r["calls"])
            rejected = sum(len(c["rejected"]) for c in r["calls"])
            ok, why = grade(path.stem, r["said"])
            violations = sum(bool(_FORBIDDEN.search(s)) for s in r["said"])   # spoken despite the guardrail
            all_ms += ms
            rows.append((path.stem, len(r["calls"]), dropped, rejected, ok, violations, ms, r["said"]))
            print(f"\n== {path.stem}{f' #{i + 1}' if a.repeat > 1 else ''}: {'PASS' if ok else 'FAIL'} ({why})")
            print(f"   expected: {RUBRIC.get(path.stem, {}).get('note', '-')}")
            print(f"   calls {len(r['calls'])}, dropped {dropped}, rejected {rejected}, "
                  f"p50 {pct(ms, .5) or 0:.0f} ms, p95 {pct(ms, .95) or 0:.0f} ms")
            for c in r["calls"]:
                print(f"   - {c['out']:8} {c['ms']:5.0f} ms  {' | '.join(c['said']) or ''}")
                for m in c["rejected"] + c["notes"]:
                    print(f"     {m}")
    print("\n| scenario | rubric | calls | dropped | p50 ms | p95 ms | said |\n|---|---|---|---|---|---|---|")
    for name, calls, dropped, _, ok, _, ms, said in rows:
        print(f"| {name} | {'pass' if ok else 'FAIL'} | {calls} | {dropped} | {pct(ms, .5) or 0:.0f} | "
              f"{pct(ms, .95) or 0:.0f} | {' / '.join(said) or '(silent)'} |")
    print(f"\nTOTAL: {sum(r[4] for r in rows)}/{len(rows)} pass, {sum(r[1] for r in rows)} calls, "
          f"{sum(r[2] for r in rows)} dropped, {sum(r[3] for r in rows)} rejected, "
          f"{sum(r[5] for r in rows)} guardrail violations, "
          f"p50 {pct(all_ms, .5) or 0:.0f} ms, p95 {pct(all_ms, .95) or 0:.0f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
