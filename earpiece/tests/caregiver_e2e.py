"""Headless end to end run of the Grok caregiver line against a throwaway local SpacetimeDB.

    python earpiece/tests/caregiver_e2e.py [--out DIR]     # or scripts/caregiver_e2e.sh

Nothing is played, recorded from a microphone, or filmed:
  * the caregiver's questions are synthesised to WAV files with macOS `say -o` (writes a file,
    never plays) and streamed to Grok as the microphone, through `caregiver --input-wav`;
  * Grok's voice goes to WAV files (`--output-wav`), so the speaker is never opened;
  * the walker's earpiece runs `run --null` (no audio device, KITE_CAMERA=off, no stem capture,
    no judgment layer) and logs every line it says as `[said] ...`.

The database is a fresh `spacetime start` on a free 127.0.0.1 port with its own data directory,
JWT keys and CLI config, so Maincloud and the user's spacetime config are never touched.
Live cost: one Grok Voice session with two short turns, and at most one new Cartesia line
(the caregiver message); everything else comes from the phrase cache or is skipped.

Needs XAI_API_KEY, the spacetime CLI, `say`, openssl and node_modules for dashboard/spacetimedb
(taken from this checkout, the main checkout of a git worktree, or `npm install`).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import secrets
import shutil
import signal
import socket
import subprocess  # nosec B404: runs local tools only (spacetime, say, openssl, this package)
import sys
import threading
import time
import wave
from pathlib import Path

import httpx

EARPIECE = Path(__file__).resolve().parents[1]
REPO = EARPIECE.parent
DB = "kite-e2e"
QUESTIONS = ("How's the walk going?", "Tell her I'm running five minutes late.")
WALKER = "Maya"
# words from the seeded walk; a reply naming any of them came from walk_status, not invention
WALK_WORDS = ("lyon", "crowd", "palace", "fine arts", "bay street")
HEAR_WITHIN_S = 10.0


def _log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} [e2e] {msg}", flush=True)


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def main_checkout() -> Path:
    """The main checkout when this is a git worktree (it holds node_modules and the phrase cache)."""
    try:
        common = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--path-format=absolute",  # nosec
                                 "--git-common-dir"], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return REPO
    return Path(common).parent if common else REPO


def load_keys(names=("XAI_API_KEY", "CARTESIA_API_KEY")) -> None:
    """Only the keys this test uses, from the repo-root .env or the main checkout's (worktrees have
    none). Values already in the environment win; nothing is printed."""
    try:
        from dotenv import dotenv_values
    except ImportError:
        return
    for p in (REPO / ".env", main_checkout() / ".env"):
        if p.is_file():
            vals = dotenv_values(p)
            for k in names:
                if vals.get(k) and not os.environ.get(k):
                    os.environ[k] = vals[k]
            return


def _stop(proc: subprocess.Popen | None, sig=signal.SIGINT, wait_s: float = 8.0) -> None:
    """Stop a process group started with start_new_session=True: SIGINT, then SIGTERM, then SIGKILL."""
    if proc is None or proc.poll() is not None:
        return
    for s in (sig, signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, s)
        except ProcessLookupError:
            return
        try:
            proc.wait(wait_s)
            return
        except subprocess.TimeoutExpired:
            continue


class LocalSpacetime:
    """A throwaway SpacetimeDB on 127.0.0.1 with its own data dir, JWT keys and CLI config."""

    def __init__(self, root: Path):
        self.root = root
        self.port = free_port()
        self.host = f"http://127.0.0.1:{self.port}"
        self.cli = root / "cli.toml"
        self.proc: subprocess.Popen | None = None
        self.log_path = root / "spacetime.log"

    def _keys(self) -> tuple[Path, Path]:
        pem, priv, pub = self.root / "ec.pem", self.root / "id_ecdsa", self.root / "id_ecdsa.pub"
        run = lambda *c: subprocess.run(c, check=True, capture_output=True, timeout=30)  # nosec
        run("openssl", "ecparam", "-name", "prime256v1", "-genkey", "-noout", "-out", str(pem))
        run("openssl", "pkcs8", "-topk8", "-nocrypt", "-in", str(pem), "-out", str(priv))
        run("openssl", "ec", "-in", str(pem), "-pubout", "-out", str(pub))
        return priv, pub

    def start(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        priv, pub = self._keys()
        cmd = ["spacetime", "--config-path", str(self.cli), "start", "--listen-addr", f"127.0.0.1:{self.port}",
               "--data-dir", str(self.root / "data"), "--jwt-pub-key-path", str(pub),
               "--jwt-priv-key-path", str(priv), "--in-memory", "--non-interactive"]
        self.proc = subprocess.Popen(cmd, stdout=open(self.log_path, "w"), stderr=subprocess.STDOUT,  # nosec
                                     stdin=subprocess.DEVNULL, start_new_session=True, cwd=self.root)
        t_end = time.time() + 30
        while time.time() < t_end:
            if self.proc.poll() is not None:
                raise RuntimeError(f"spacetime start exited, see {self.log_path}")
            try:
                if httpx.get(f"{self.host}/v1/ping", timeout=1).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.3)
        raise RuntimeError(f"local SpacetimeDB did not come up, see {self.log_path}")

    def publish(self, module: Path, db: str = DB) -> None:
        """Publish a copy of the module (so dashboard/spacetime.json, which names Maincloud, is
        never in the CLI's working directory) to this server only."""
        assert self.host.startswith("http://127.0.0.1:")
        copy = self.root / "module"
        shutil.copytree(module / "src", copy / "src")
        for f in ("package.json", "tsconfig.json"):
            if (module / f).exists():
                shutil.copy(module / f, copy / f)
        for nm in (module / "node_modules", main_checkout() / "dashboard" / "spacetimedb" / "node_modules"):
            if (nm / "spacetimedb").exists():
                (copy / "node_modules").symlink_to(nm)
                break
        else:
            subprocess.run(["npm", "install", "--no-audit", "--no-fund"], cwd=copy, check=True,  # nosec
                           capture_output=True, timeout=300)
        r = subprocess.run(["spacetime", "--config-path", str(self.cli), "publish", "--module-path", str(copy),  # nosec
                            "--server", self.host, db, "-y"], cwd=self.root, capture_output=True, text=True,
                           timeout=300)
        with open(self.root / "publish.log", "w") as fh:
            fh.write(r.stdout + r.stderr)
        if r.returncode != 0:
            raise RuntimeError(f"publish failed ({r.returncode}), see {self.root / 'publish.log'}")

    def stop(self) -> None:
        _stop(self.proc, signal.SIGTERM, 5)


class Db:
    def __init__(self, host: str, db: str = DB):
        self.host, self.db = host, db
        self.http = httpx.Client(timeout=5)
        self.token = self.http.post(f"{host}/v1/identity").json()["token"]

    def _h(self):
        return {"Authorization": f"Bearer {self.token}"}

    def push(self, event: dict) -> None:
        args = {"type": event["type"], "source": event["source"], "priority": event.get("priority", ""),
                "payload": json.dumps(event, separators=(",", ":"))}
        r = self.http.post(f"{self.host}/v1/database/{self.db}/call/push", json=args, headers=self._h())
        r.raise_for_status()

    def rows(self, query: str) -> list[dict]:
        from earpiece.spacetime import sql_rows
        r = self.http.post(f"{self.host}/v1/database/{self.db}/sql", content=query, headers=self._h())
        r.raise_for_status()
        return sql_rows(r.json())


SEED = [
    {"type": "status", "source": "drone", "state": "following", "battery": 41},
    {"type": "guidance", "source": "route", "priority": "nav", "text": "Turn left onto Bay Street."},
    {"type": "beacon", "source": "route", "label": "Palace of Fine Arts", "distance": 131.4, "bearing": -2},
    {"type": "hazard", "source": "drone", "priority": "hazard", "kind": "crowd",
     "text": "Small crowd ahead on the right."},
    {"type": "spoken", "source": "earpiece", "priority": "hazard", "text": "Small crowd ahead on the right."},
    {"type": "position", "source": "sim", "place": "Lyon Street", "lat": 37.80171, "lng": -122.44792},
]


def synth(text: str, path: Path) -> Path:
    """`say -o` renders to a file and does not play; PCM16 mono 24 kHz, Grok's input format."""
    if shutil.which("say") is None:
        raise RuntimeError("macOS `say` not found")
    subprocess.run(["say", "-v", "Samantha", "-r", "190", "-o", str(path), "--file-format=WAVE",  # nosec
                    "--data-format=LEI16@24000", text], check=True, capture_output=True, timeout=30)
    with wave.open(str(path)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 24000), path
        assert w.getnframes() > 24000 * 0.5, f"{path} is too short"
    return path


class Tail:
    """Follows a log file on a thread, stamping every line with the time it was seen."""

    def __init__(self, path: Path):
        self.path, self.lines = path, []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        while not self.path.exists() and not self._stop.is_set():
            time.sleep(0.05)
        with open(self.path, errors="replace") as fh:
            buf = ""
            while not self._stop.is_set():
                chunk = fh.readline()
                if not chunk:
                    time.sleep(0.05)
                    continue
                buf += chunk
                if buf.endswith("\n"):
                    self.lines.append((time.time(), buf.rstrip("\n")))
                    buf = ""

    def find(self, pattern: str, after: float = 0.0):
        rx = re.compile(pattern, re.IGNORECASE)
        return next(((t, line) for t, line in list(self.lines) if t >= after and rx.search(line)), None)

    def wait(self, pattern: str, timeout: float, after: float = 0.0):
        t_end = time.time() + timeout
        while time.time() < t_end:
            if hit := self.find(pattern, after):
                return hit
            time.sleep(0.1)
        return self.find(pattern, after)

    def stop(self):
        self._stop.set()


def run(out: Path, python: str = sys.executable, log=_log) -> dict:
    """The whole scenario. Returns {"checks": [(name, ok, detail)], "replies": [...], "out": dir}."""
    out.mkdir(parents=True, exist_ok=True)
    if not os.environ.get("XAI_API_KEY"):
        raise RuntimeError("XAI_API_KEY is not set")
    if shutil.which("spacetime") is None:
        raise RuntimeError("spacetime CLI not found")
    token = secrets.token_hex(16)              # both sides share it: messages must be signed
    cache = os.environ.get("KITE_TTS_CACHE") or str(main_checkout() / "earpiece" / ".cache" / "tts")
    env = {**os.environ, "PYTHONPATH": str(EARPIECE), "KITE_CAMERA": "off", "KITE_JUDGMENT": "0",
           "KITE_CAREGIVER_TOKEN": token, "SPACETIME_DB": DB, "KITE_TTS_CACHE": cache}
    # only Grok (caregiver) and Cartesia (walker) are part of this test: no Claude, Moss or Maps calls
    for k in ("KITE_SPACETIME", "ANTHROPIC_API_KEY", "MOSS_PROJECT_ID", "MOSS_PROJECT_KEY", "GOOGLE_MAPS_API_KEY"):
        env[k] = ""
    st = LocalSpacetime(out / "spacetime")
    walker = care = None
    tail = None
    checks: list[tuple[str, bool, str]] = []
    result = {"checks": checks, "replies": [], "out": str(out)}
    try:
        log(f"starting local SpacetimeDB in {st.root}")
        st.start()
        st.publish(REPO / "dashboard" / "spacetimedb")
        env["SPACETIME_HOST"] = st.host
        log(f"published {DB} to {st.host}")
        db = Db(st.host)
        t0 = time.time()
        db.rows("SELECT * FROM latest")          # a fresh local server's first SQL read takes ~5 s
        log(f"first SQL read took {time.time() - t0:.1f} s")

        walker_log = out / "walker.log"
        walker = subprocess.Popen(  # nosec: this package, local flags only
            [python, "-m", "earpiece", "run", "--null", "--spacetime", "--no-stem", "--no-judgment",
             "--no-prewarm", "--udp-port", str(free_port()), "--ws-port", str(free_port())],
            cwd=EARPIECE, env={**env, "PYTHONUNBUFFERED": "1"}, stdin=subprocess.DEVNULL,
            stdout=open(walker_log, "w"), stderr=subprocess.STDOUT, start_new_session=True)
        tail = Tail(walker_log)
        # the inbox treats rows from before its first poll as history, so wait for that poll
        if not tail.wait(r"\[caregiver\] inbox ready", 120):    # generous: imports are slow on a busy laptop
            raise RuntimeError(f"the walker's caregiver inbox did not start, see {walker_log}")
        for ev in SEED:
            db.push(ev)
        log(f"seeded {len(SEED)} walk events")

        wavs = [synth(q, out / f"caregiver-q{i}.wav") for i, q in enumerate(QUESTIONS, 1)]
        care_log = out / "caregiver.log"
        transcript = out / "transcript.json"
        t_call = time.time()
        log("caregiver line: two recorded turns to Grok")
        care = subprocess.run(  # nosec
            [python, "-m", "earpiece", "caregiver", "--no-greeting", "--walker", WALKER,
             *[x for w in wavs for x in ("--input-wav", str(w))], "--output-wav", str(out / "grok-reply.wav"),
             "--transcript", str(transcript), "--max-minutes", "3", "--debug"],
            cwd=EARPIECE, env={**env, "PYTHONUNBUFFERED": "1"}, stdin=subprocess.DEVNULL,
            stdout=open(care_log, "w"), stderr=subprocess.STDOUT, timeout=240)
        log(f"caregiver line exited with {care.returncode}")
        tr = json.loads(transcript.read_text()) if transcript.exists() else {"turns": [], "tools": []}
        turns = tr.get("turns", [])
        for t in turns:
            result["replies"].append(" ".join(x["text"] for x in t["said"] if x["who"] == "kite"))
        care_text = care_log.read_text()

        def turn(i):
            return turns[i] if len(turns) > i else {"said": [], "tools": []}

        def check(name, ok, detail=""):
            checks.append((name, bool(ok), detail))
            log(f"{'PASS' if ok else 'FAIL'}  {name}" + (f": {detail}" if detail else ""))

        check("caregiver line exited cleanly", care.returncode == 0, f"exit {care.returncode}")
        check("no audio device opened on the caregiver side",
              not re.search(r"\[caregiver-line\] (speaker|mic):", care_text))
        t1, t2 = turn(0), turn(1)
        names1 = [c["name"] for c in t1["tools"]]
        check("turn 1: Grok called walk_status", "walk_status" in names1, f"tools {names1}")
        reply1 = result["replies"][0] if result["replies"] else ""
        hits = [w for w in WALK_WORDS if w in reply1.lower()]
        check("turn 1: the reply uses the live walk data", hits, f"{hits} in {reply1!r}")
        sends = [c for c in t2["tools"] if c["name"] == "message_walker"]
        sent = [c for c in sends if isinstance(c["result"], dict) and c["result"].get("sent")]
        check("turn 2: Grok called message_walker and it was sent", sent,
              json.dumps([c["arguments"] for c in sends])[:200])
        msg = (sent[0]["result"].get("text") if sent else "") or ""
        check("turn 2: the message is about running late", "late" in msg.lower(), repr(msg))
        t0 = time.time()
        rows = db.rows("SELECT * FROM event WHERE source = 'caregiver'")
        payloads = [json.loads(r["payload"]) for r in rows]
        texts = [p.get("text", "") for p in payloads]
        check("a caregiver say row landed in the local database",
              any("late" in t.lower() for t in texts) and all(r["type"] == "say" for r in rows),
              f"{texts!r} ({time.time() - t0:.1f} s to read)")
        check("the row is signed like the dashboard's", payloads and all({"ts", "sig"} <= set(p) for p in payloads),
              f"keys {sorted(payloads[0]) if payloads else []}")
        t_sent = sent[0]["t"] if sent else t_call
        heard = tail.wait(r"\[said\] \[info/done\] Message from your caregiver: .*late", HEAR_WITHIN_S + 5,
                          after=t_sent - 1) if sent else None
        delay = heard[0] - t_sent if heard else None
        check(f"the walker's earpiece said it within {HEAR_WITHIN_S:.0f} s",
              heard is not None and delay <= HEAR_WITHIN_S,
              (f"{delay:.1f} s after the tool call: {heard[1].split('[said] ', 1)[-1]}" if heard
               else "not heard"))
        result["heard_after_s"] = None if delay is None else round(delay, 2)
        for _ in range(20):                     # the earpiece mirrors `spoken` right after the line ends
            spoken = [json.loads(r["payload"]).get("text", "")
                      for r in db.rows("SELECT * FROM event WHERE type = 'spoken'")]
            if any(t.startswith("Message from your caregiver") for t in spoken):
                break
            time.sleep(0.25)
        check("the dashboard can see it was spoken (earpiece `spoken` row)",
              any(t.startswith("Message from your caregiver") and "late" in t.lower() for t in spoken),
              repr([t for t in spoken if "caregiver" in t.lower()]))
        problems = [line for _, line in tail.lines if re.search(r"ignored row|rejected row|Traceback", line)]
        check("the earpiece accepted the signed message (no ignored or rejected rows)", not problems,
              "; ".join(problems)[:200])
        for i, t in enumerate(turns, 1):
            result.setdefault("audio", []).append(t.get("audio_file"))
            log(f"turn {i} ({t.get('seconds')} s, {t.get('audio_seconds')} s of audio): "
                f"tools {[c['name'] for c in t['tools']]}; Kite said {result['replies'][i - 1]!r}")
    finally:
        _stop(walker)
        if tail:
            time.sleep(0.2)
            tail.stop()
        st.stop()
        with open(out / "summary.json", "w") as fh:
            json.dump(result, fh, indent=1)
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=EARPIECE / ".cache" / "caregiver-e2e" / time.strftime("%Y%m%d-%H%M%S"))
    a = p.parse_args(argv)
    load_keys()
    try:
        res = run(a.out.resolve())
    except Exception as e:
        _log(f"FAIL  setup: {type(e).__name__}: {e}")
        return 2
    ok = sum(1 for _, good, _ in res["checks"] if good)
    print(f"\n{ok}/{len(res['checks'])} checks passed. Audio, transcripts and logs: {res['out']}")
    return 0 if ok == len(res["checks"]) and res["checks"] else 1


if __name__ == "__main__":
    sys.path.insert(0, str(EARPIECE))
    sys.exit(main())
