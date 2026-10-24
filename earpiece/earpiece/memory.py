"""Kite's memory of the wearer's world, on Moss.

What it holds:
  place   "my usual coffee shop: Blue Bottle on Chestnut Street"   (with lat/lng when known)
  hazard  "the curb at Lyon and Bay is very tall"
  note    anything else the wearer or their caregiver wants Kite to know

How it is used:
  * "take me to my coffee shop" -> the personal place is resolved before routing, and
    the navigation side receives the real name and coordinates. When two saved places
    fit equally well ("the bench", and there are two), Kite asks which one rather than
    guess (`resolve` returns the candidates, `pick` understands the answer);
  * every question to Claude carries the few memories most relevant to it;
  * when the wearer reaches a remembered place, a matching hazard note is spoken once,
    more softly the older it is ("from 3 months ago: ...");
  * by voice: "remember this place as the bench", "remember that ...", "forget the bench",
    "rename the bench to the reading bench". Saving the same place or the same note twice
    updates the one memory instead of piling up copies. The caregiver dashboard can add
    memories too.

Matching: words are folded to a common form (plurals, "cafe" = "coffee", "fell" =
"tripped"), rare words count for more than common ones, and the head of a phrase counts
for more than its modifiers ("the BENCH by the park" is a bench first). A destination
is only ever a saved place that shares a real word with what was said, so a semantic
near-miss can never invent somewhere to walk to.

Why Moss: the index is loaded into Moss's local (Rust) runtime, so a lookup runs
in-process in a few milliseconds (see tests/test_memory.py::test_live_moss_latency),
cheap enough to run on every utterance. Writes go to the Moss cloud index and are
mirrored to a local JSON file, so memories survive restarts and the same matching keeps
working with no network. Moss is the only service this module talks to; if it errors
repeatedly it is benched for a while and the local mirror answers on its own.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import threading
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace

from .config import REPO_ROOT

INDEX = "kite-memory"
STORE = REPO_ROOT / "earpiece" / ".cache" / "memory.json"
SEED = Path(__file__).with_name("memory_seed.json")
DAY_S = 86400.0

_STOP = {"the", "a", "an", "my", "to", "of", "at", "on", "in", "and", "is", "me", "take", "go", "i",
         "where", "what", "for", "by", "near", "with", "this", "that", "it", "please", "kite",
         # generic street words: "Union Street" must not match a note about "Lyon Street"
         "street", "st", "avenue", "ave", "boulevard", "blvd", "road", "rd", "lane", "way", "drive",
         "corner", "intersection", "block", "side", "hazards", "hazard",
         # small talk and time words: "where did I fall last week" is about the fall, not the week
         "did", "do", "does", "was", "were", "am", "are", "be", "been", "you", "your", "we", "us", "there",
         "here", "when", "which", "who", "how", "about", "from", "into", "so", "just", "last", "week",
         "weeks", "month", "months", "year", "years", "ago", "yesterday", "today", "tonight", "earlier",
         "before", "time", "day", "days", "get", "got", "going", "want", "need", "like", "would", "could",
         "can", "let", "lets", "let's", "again", "one", "ones", "once", "had", "have", "has", "up", "out",
         # "a place", "that spot": every saved place is one
         "place", "places", "spot", "location", "area", "thing", "somewhere", "tips"}
# Words that say "mine" rather than "which": they help a little, never decide on their own.
_PERSONAL = {"usual", "favorite", "regular", "normal", "own", "our", "always", "usually"}
_SYN = {"cafe": "coffee", "café": "coffee", "coffeeshop": "coffee", "espresso": "coffee",
        "shop": "store", "fell": "fall", "fallen": "fall", "falling": "fall", "tripped": "fall",
        "trip": "fall", "stumbled": "fall", "stumble": "fall", "slipped": "fall", "slip": "fall",
        "drugstore": "pharmacy", "chemist": "pharmacy", "house": "home", "apartment": "home",
        "seat": "bench", "kerb": "curb", "pavement": "sidewalk", "favourite": "favorite",
        "fave": "favorite", "restroom": "bathroom", "toilet": "bathroom", "stair": "stairs"}
_PHRASES = [(re.compile(r"\bmy (?:place|house|apartment|flat|home)\b"), "home")]
_HEAD_SPLIT = re.compile(r"\b(?:by|near|at|on|in|next to|behind|across from|outside|opposite|beside|"
                         r"with|where|that|which)\b")

ACCEPT = 0.5          # share of what was said a memory must cover to count as a match
TIE = 0.12            # candidates this close to the best are equally good: ask, don't guess
SAME_PLACE_M = 30.0   # two saved places with the same name this close are one place
DUP_NOTE = 0.75       # word overlap at which a new note is the same note said again
MOSS_FAILS = 3        # consecutive Moss errors before it is benched...
MOSS_BENCH_S = 30.0   # ...for this long (each failing query would otherwise cost its timeout)


@dataclass
class Memory:
    id: str
    kind: str
    text: str
    meta: dict = field(default_factory=dict)
    score: float = 0.0


@dataclass
class Resolution:
    """What a phrase refers to. `place` is set when one memory clearly fits; when several
    fit about equally, `place` is None and `candidates` holds them (best first)."""
    place: Memory | None
    candidates: list[Memory] = field(default_factory=list)

    @property
    def ambiguous(self) -> bool:
        return self.place is None and len(self.candidates) > 1

    def question(self, lead: str = "Which one") -> str:
        """'Which one: the bench by the 30 bus stop, or the bench by Crissy Field?'"""
        opts = options(self.candidates)
        return f"{lead}: {', '.join(opts[:-1])}, or {opts[-1]}?" if len(opts) > 1 else f"{lead}?"


def _stem(w: str) -> str:
    w = w.strip("'")
    if w.endswith("'s"):
        w = w[:-2]
    w = _SYN.get(w, w)
    if len(w) > 4 and w.endswith("ies"):
        w = w[:-3] + "y"
    elif w.endswith(("ches", "shes", "sses", "xes", "zes")):
        w = w[:-2]
    elif len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        w = w[:-1]
    elif len(w) > 5 and w.endswith("ing"):
        w = w[:-3]
    elif len(w) > 5 and w.endswith("ed"):
        w = w[:-2]
    return _SYN.get(w, w)


def _words(s: str) -> list[str]:
    return [t for w in re.findall(r"[a-z0-9'é]+", s.lower()) if w not in _STOP
            for t in [_stem(w)] if len(t) > 1 and t not in _STOP]


def _tokens(s: str) -> set[str]:
    return set(_words(s))


def name_of(m: Memory) -> str:
    """How Kite says a memory's name: its saved name, or the text before a colon."""
    return str(m.meta.get("name") or m.text.split(":")[0]).strip()


def options(cands: list[Memory], max_words: int = 9) -> list[str]:
    """Short spoken labels that tell the candidates apart: names when they differ,
    otherwise the descriptions."""
    names = [name_of(c) for c in cands]
    toks = [_tokens(n) for n in names]
    # "the bench" vs "the bus stop bench" doesn't tell them apart for the listener either
    if all(a and not (a <= b or b <= a) for i, a in enumerate(toks) for b in toks[i + 1:]):
        return names
    out = []
    for c in cands:
        words = c.text.replace(":", ",").split()[:max_words]
        while len(words) > 2 and words[-1].lower().strip(",.") in {"on", "and", "the", "by", "at", "of", "near",
                                                                     "to", "a", "in", "with"}:
            words.pop()
        s = " ".join(words).rstrip(",.")
        out.append(s[:1].lower() + s[1:] if s[1:2].islower() else s)
    return out


def pick(cands: list[Memory], answer: str) -> int | None:
    """Which candidate the wearer chose: an index, -1 for "neither", None if unclear.
    "the first one", "the second", "the last one", or any word only one of them has
    ("the park one", "Crissy")."""
    a = " ".join(answer.lower().strip(" .!?").split())
    if re.fullmatch(r"(?:no|nope|neither|none|never ?mind|cancel|forget it|nothing|neither (?:of them|one)|"
                    r"none of them|not those|stop|quiet|be quiet)(?:,? thanks| thank you)?", a):
        return -1
    for pat, i in ((r"\b(?:first|1st|number one|former)\b", 0), (r"\b(?:second|2nd|number two)\b", 1),
                   (r"\b(?:third|3rd|number three)\b", 2)):
        if re.search(pat, a) and i < len(cands):
            return i
    if re.search(r"\b(?:last|latter)\b", a) and cands:
        return len(cands) - 1
    toks = [_tokens(c.text + " " + name_of(c)) for c in cands]
    common = set.intersection(*toks) if toks else set()
    said = _tokens(a)
    hits = [len((t - common) & said) for t in toks]
    best = max(hits, default=0)
    return hits.index(best) if best and hits.count(best) == 1 else None


def _moss_types():
    """(DocumentInfo, MutationOptions, QueryOptions) from the SDK, or plain stand-ins so
    a fake client in tests works whether or not the SDK is installed."""
    try:
        from inferedge_moss import DocumentInfo, MutationOptions, QueryOptions
        return DocumentInfo, MutationOptions, QueryOptions
    except ImportError:
        return (lambda **k: SimpleNamespace(**k),) * 3


class MemoryStore:
    SYNC_SETTLE_S = 2.0       # Moss needs a moment after a write before the index reports its new state

    def __init__(self, project_id: str | None = None, project_key: str | None = None,
                 index: str = INDEX, store: Path = STORE, seed: Path | None = SEED, log=print, client=None):
        self.project_id = project_id if project_id is not None else os.environ.get("MOSS_PROJECT_ID")
        self.project_key = project_key if project_key is not None else os.environ.get("MOSS_PROJECT_KEY")
        self.index = index
        self.store = Path(store)
        self.forgotten_path = self.store.with_name(self.store.stem + ".forgotten.json")
        self.log = log
        self.docs: dict[str, Memory] = {}
        self.ready = False                    # Moss local index loaded
        self.status = "local"
        self._client = client
        self._loop: asyncio.AbstractEventLoop | None = None
        self._dirty_since_load = False
        self._lock = threading.RLock()
        self._tok: dict[str, set[str]] = {}
        self._df: Counter = Counter()
        self._inv: dict[str, list[str]] = {}
        self._forgotten: set[str] = set()     # deleted here, not yet deleted in Moss
        self._moss_fails = 0
        self._moss_off_until = 0.0
        self.stats = {"queries": 0, "moss_queries": 0, "last_ms": None, "writes": 0, "errors": 0}
        self._load_local(seed)

    # ------------------------------------------------------------------ local mirror
    def _load_local(self, seed: Path | None) -> None:
        src = self.store if self.store.exists() else seed
        if src is None or not Path(src).exists():
            return
        try:
            rows = json.loads(Path(src).read_text())
        except (OSError, ValueError) as e:
            self.log(f"[memory] could not read {src}: {e}")
            rows = None
            if src == self.store:
                # keep the damaged file for a human to look at, and start again from the seed
                try:
                    os.replace(self.store, self.store.with_suffix(".corrupt.json"))
                except OSError:
                    pass
                if seed is not None and Path(seed).exists():
                    rows = json.loads(Path(seed).read_text())
        for d in rows or []:
            try:
                m = Memory(d["id"], d.get("kind", "note"), d["text"], d.get("meta") or {})
            except (KeyError, TypeError):
                continue
            self.docs[m.id] = m
        try:
            self._forgotten = set(json.loads(self.forgotten_path.read_text()))
        except (OSError, ValueError):
            self._forgotten = set()
        if src == self.store and seed is not None and Path(seed).exists():
            # A seed entry the wearer never edited (no "t") follows the seed, so a correction
            # there (the demo home moved off a real house) reaches mirrors made before it.
            changed = False
            try:
                for d in json.loads(Path(seed).read_text()):
                    m = self.docs.get(d.get("id"))
                    if m is not None and "t" not in m.meta and (m.text != d["text"] or m.meta != d.get("meta", {})):
                        m.kind, m.text, m.meta = d.get("kind", m.kind), d["text"], dict(d.get("meta") or {})
                        changed = True
            except (OSError, ValueError, KeyError, TypeError):
                pass
            if changed:
                self._save_local()
        self._reindex()

    def _write(self, path: Path, data) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1))
        os.chmod(tmp, 0o600)                  # the wearer's places: readable by this user only
        os.replace(tmp, path)

    def _save_local(self) -> None:
        try:
            with self._lock:
                rows = [{k: v for k, v in asdict(m).items() if k != "score"} for m in self.docs.values()]
                gone = sorted(self._forgotten)
            self._write(self.store, rows)
            if gone or self.forgotten_path.exists():
                self._write(self.forgotten_path, gone)
        except OSError as e:
            self.log(f"[memory] could not save locally: {e}")

    def _reindex(self) -> None:
        with self._lock:
            self._tok = {i: _tokens(m.text + " " + str(m.meta.get("name") or "") + " " + str(m.meta.get("aka") or ""))
                         for i, m in self.docs.items()}
            inv: dict[str, list[str]] = {}
            for i, t in self._tok.items():
                for w in t:
                    inv.setdefault(w, []).append(i)
            self._inv = inv                   # word -> memories that have it
            self._df = Counter({w: len(ids) for w, ids in inv.items()})

    # ------------------------------------------------------------------ Moss lifecycle
    def start(self) -> None:
        if self._client is None:
            if not (self.project_id and self.project_key):
                self.log("[memory] no Moss keys: using the local memory only")
                return
            try:
                from inferedge_moss import MossClient
            except ImportError:
                self.log("[memory] inferedge-moss not installed: using the local memory only")
                return
            self._client = MossClient(self.project_id, self.project_key)
        threading.Thread(target=self._run_loop, name="memory", daemon=True).start()
        t0 = time.time()
        while self._loop is None and time.time() - t0 < 2:
            time.sleep(0.005)
        asyncio.run_coroutine_threadsafe(self._bootstrap(), self._loop)

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        loop.run_forever()

    def _doc(self, m: Memory):
        doc_info = _moss_types()[0]
        meta = {"kind": m.kind, **{k: str(v) for k, v in m.meta.items() if v is not None}}
        return doc_info(id=m.id, text=m.text, metadata=meta)

    async def _wait_ready(self, timeout: float = 90) -> bool:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            try:
                info = await self._client.get_index(self.index)
                if str(info.status).lower() == "ready":
                    return True
            except Exception:
                pass
            await asyncio.sleep(1.5)
        return False

    async def _bootstrap(self) -> None:
        t0 = time.monotonic()
        try:
            self.status = "connecting"
            names = {i.name for i in await self._client.list_indexes()}
            if self.index not in names:
                self.status = "creating"
                await self._client.create_index(self.index, [self._doc(m) for m in list(self.docs.values())])
            else:
                remote = {d.id: d for d in await self._client.get_docs(self.index)}
                # Forgotten while Moss was out of reach: delete them there too, so they can't come back.
                gone = [i for i in self._forgotten if i in remote]
                if gone:
                    await self._client.delete_docs(self.index, gone)
                with self._lock:
                    self._forgotten.clear()
                    # Saved on another device (or by an earlier install): pull them into the mirror.
                    for i, d in remote.items():
                        if i not in self.docs and i not in gone:
                            meta = {k: v for k, v in dict(getattr(d, "metadata", None) or {}).items()}
                            kind = meta.pop("kind", "note")
                            self.docs[i] = Memory(i, kind, d.text, _typed(meta))
                    # The mirror is the source of truth for content: anything edited here (a rename,
                    # a re-saved place, a scrubbed seed entry) replaces the older copy in Moss.
                    missing = [m for m in self.docs.values() if m.id not in remote
                               or _remote_differs(remote[m.id], self._doc(m))]
                self._reindex()
                self._save_local()
                if missing:
                    await self._add(missing)
            if not await self._wait_ready():
                raise RuntimeError("index did not become ready")
            self.status = "loading"
            await self._client.load_index(self.index)
            self.ready = True
            self.status = "moss"
            self._dirty_since_load = False
            self.log(f"[memory] Moss index '{self.index}' loaded locally ({len(self.docs)} memories, "
                     f"{time.monotonic() - t0:.1f}s)")
        except Exception as e:
            self.stats["errors"] += 1
            self.status = "local"
            self.log(f"[memory] Moss unavailable ({type(e).__name__}: {e}); local memory only")

    async def _add(self, ms: list[Memory]) -> None:
        mutation = _moss_types()[1]
        await self._client.add_docs(self.index, [self._doc(m) for m in ms], mutation(upsert=True))

    async def _sync(self, m: Memory) -> None:
        try:
            await self._add([m])
            await asyncio.sleep(self.SYNC_SETTLE_S)
            if await self._wait_ready(60):
                await self._client.load_index(self.index)       # refresh the local copy
                self._dirty_since_load = False
        except Exception as e:
            self.stats["errors"] += 1
            self.log(f"[memory] Moss write failed ({type(e).__name__}); kept locally")

    async def _sync_delete(self, mem_id: str) -> None:
        try:
            await self._client.delete_docs(self.index, [mem_id])
            with self._lock:
                self._forgotten.discard(mem_id)
            self._save_local()
            if await self._wait_ready(60):
                await self._client.load_index(self.index)
                self._dirty_since_load = False
        except Exception as e:
            self.stats["errors"] += 1
            self.log(f"[memory] Moss delete failed ({type(e).__name__}); will retry on next start")

    def _push(self, coro_fn, *args) -> None:
        if self._client is not None and self._loop is not None:
            asyncio.run_coroutine_threadsafe(coro_fn(*args), self._loop)

    # ------------------------------------------------------------------ writing
    def save(self, kind: str, text: str, meta: dict | None = None, now: float | None = None) -> tuple[Memory, str]:
        """Add a memory, or refresh the one it repeats. Returns (memory, status):
        "new"; "updated" (same place saved again: new coordinates, one memory);
        "same_name" (new place whose name another saved place already has, far from it);
        "duplicate" (a note or hazard Kite already had: marked as still true)."""
        kind = kind if kind in ("place", "hazard", "note") else "note"
        text = " ".join(text.split())[:300]
        meta = {k: v for k, v in (meta or {}).items() if v is not None}
        now = time.time() if now is None else now
        status, target = "new", None
        with self._lock:
            if kind == "place":
                name = _norm_name(meta.get("name") or text)
                same = [m for m in self.docs.values() if m.kind == "place" and _norm_name(name_of(m)) == name]
                near = [m for m in same if _distance(m.meta, meta) is None or _distance(m.meta, meta) <= SAME_PLACE_M]
                if near:
                    target, status = near[0], "updated"
                    if not _tokens(text) <= self._tok.get(target.id, set()):
                        target.text = text          # a fuller description wins; a bare name doesn't
                    target.meta.update(meta)
                elif same:
                    status = "same_name"
            else:
                toks = _tokens(text)
                for m in self.docs.values():
                    if m.kind in ("hazard", "note") and _jaccard(toks, self._tok.get(m.id, set())) >= DUP_NOTE:
                        target, status = m, "duplicate"
                        target.meta["confirmed"] = int(target.meta.get("confirmed", 1)) + 1
                        break
            if target is None:
                target = Memory(f"{kind}-{uuid.uuid4().hex[:8]}", kind, text, meta)
                self.docs[target.id] = target
            target.meta["t"] = round(now)
            self.stats["writes"] += 1
            self._dirty_since_load = True
        self._reindex()
        self._save_local()
        self._push(self._sync, target)
        return target, status

    def remember(self, kind: str, text: str, meta: dict | None = None) -> Memory:
        return self.save(kind, text, meta)[0]

    def forget(self, mem_id: str) -> bool:
        with self._lock:
            if mem_id not in self.docs:
                return False
            del self.docs[mem_id]
            self._forgotten.add(mem_id)            # so a copy still in Moss can't come back
            self._dirty_since_load = True
        self._reindex()
        self._save_local()
        self._push(self._sync_delete, mem_id)
        return True

    def restore(self, m: Memory) -> Memory:
        """Undo a forget: the same memory, same id, back in place."""
        with self._lock:
            self.docs[m.id] = m
            self._forgotten.discard(m.id)
            self._dirty_since_load = True
        self._reindex()
        self._save_local()
        self._push(self._sync, m)
        return m

    def rename(self, mem_id: str, new_name: str) -> Memory | None:
        """A saved place gets a new spoken name; the old name still finds it."""
        new_name = " ".join(new_name.strip().rstrip(".!?").split())
        with self._lock:
            m = self.docs.get(mem_id)
            if m is None or m.kind != "place" or not new_name:
                return None
            old = name_of(m)
            desc = m.meta.get("description") or m.text
            aka = [a for a in str(m.meta.get("aka") or "").split("; ") if a]
            if old.lower() != new_name.lower() and old not in aka:
                aka.insert(0, old)
            m.meta.update({"name": new_name, "description": desc, "aka": "; ".join(aka[:4]),
                           "t": round(time.time())})
            m.text = f"{new_name}: {desc}"[:300]
            self._dirty_since_load = True
        self._reindex()
        self._save_local()
        self._push(self._sync, m)
        return m

    # ------------------------------------------------------------------ reading
    def _moss_ok(self) -> bool:
        return self.ready and self._loop is not None and time.monotonic() >= self._moss_off_until

    def search(self, query: str, kind: str | None = None, top_k: int = 3, timeout: float = 0.25) -> list[Memory]:
        """Most relevant memories. Moss local index when loaded; keyword fallback otherwise.
        Memories added since the last load are always merged in from the local mirror."""
        self.stats["queries"] += 1
        t0 = time.perf_counter()
        hits: list[Memory] = []
        fut = None
        if self._moss_ok():
            try:
                query_opts = _moss_types()[2]
                fut = asyncio.run_coroutine_threadsafe(
                    self._client.query(self.index, query, query_opts(top_k=top_k + 4, alpha=0.7)), self._loop)
                res = fut.result(timeout)
                for d in res.docs:
                    m = self.docs.get(d.id)
                    if m is None:
                        continue                  # forgotten here: never let Moss bring it back
                    hits.append(Memory(d.id, m.kind, m.text, m.meta, float(d.score)))
                self.stats["moss_queries"] += 1
                self._moss_fails = 0
            except Exception as e:
                if fut is not None:
                    fut.cancel()
                self.stats["errors"] += 1
                self._moss_fails += 1
                if self._moss_fails >= MOSS_FAILS:
                    self._moss_off_until = time.monotonic() + MOSS_BENCH_S
                    self._moss_fails = 0
                    self.log(f"[memory] Moss failing ({type(e).__name__}); local memory for {MOSS_BENCH_S:.0f}s")
                else:
                    self.log(f"[memory] query failed ({type(e).__name__}); keyword fallback")
                hits = []
        if not hits or self._dirty_since_load:
            seen = {h.id for h in hits}
            hits += [h for h in self._keyword(query) if h.id not in seen]
            hits.sort(key=lambda h: -h.score)
        if kind:
            hits = [h for h in hits if h.kind == kind]
        self.stats["last_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return hits[:top_k]

    def _weights(self, phrase: str) -> dict[str, float]:
        """Query words and how much each matters: rare words more than common ones, the head
        of the phrase ("the BENCH by the park") twice its modifiers, "usual" barely."""
        low = phrase.lower()
        for pat, rep in _PHRASES:
            low = pat.sub(rep, low)
        m = _HEAD_SPLIT.search(low)
        head, tail = (low[:m.start()], low[m.start():]) if m and _tokens(low[:m.start()]) else (low, "")
        n = len(self.docs) + 1
        w: dict[str, float] = {}
        for part, k in ((tail, 1.0), (head, 2.0)):
            for t in _tokens(part):
                w[t] = (0.3 if t in _PERSONAL else k) * math.log(1 + n / (self._df.get(t, 0) + 0.5))
        return w

    def _scored(self, phrase: str) -> list[tuple[float, Memory]]:
        w = self._weights(phrase)
        total = sum(w.values())
        if not total:
            return []
        got: dict[str, float] = {}
        with self._lock:
            for t, v in w.items():
                for i in self._inv.get(t, ()):
                    got[i] = got.get(i, 0.0) + v
            out = [(round(g / total, 4), self.docs[i]) for i, g in got.items() if i in self.docs]
        return sorted(out, key=lambda x: (-x[0], -float(x[1].meta.get("t") or 0)))

    def _keyword(self, query: str) -> list[Memory]:
        return [Memory(m.id, m.kind, m.text, m.meta, round(min(0.99, 0.5 + 0.5 * s), 3))
                for s, m in self._scored(query)]

    def resolve(self, phrase: str, kinds: tuple[str, ...] | None = ("place",)) -> Resolution:
        """The memory a phrase means, or the few it might mean. A match needs real shared
        words (ACCEPT): a destination is somewhere to walk to, so a semantic near-miss from
        the index is never enough on its own. Moss serves the open questions instead (what
        to tell Claude, which hazard notes fit a place), where a near-miss costs nothing."""
        t0 = time.perf_counter()
        cands = [x for x in self._scored(phrase)
                 if x[0] >= ACCEPT and (kinds is None or x[1].kind in kinds)]
        self.stats["last_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        if not cands:
            return Resolution(None, [])
        best, top = cands[0]
        rivals = []
        for s, m in cands[1:]:
            if s >= best - TIE and not _same_place(m, top) and not any(_same_place(m, r) for r in rivals):
                rivals.append(m)
        if rivals:
            return Resolution(None, [top] + rivals[:2])
        return Resolution(top, [top])

    def resolve_place(self, phrase: str) -> Memory | None:
        """'my coffee shop' -> the remembered place, only when the match is unambiguous."""
        return self.resolve(phrase).place

    def notes_for(self, place: str, now: float | None = None) -> list[Memory]:
        """Hazard notes that mention this place (shared content word required), freshest
        and most relevant first."""
        pt = _tokens(place)
        now = time.time() if now is None else now
        q = f"hazards and tips near {place}"
        hits = self.search(q, top_k=6)
        seen = {h.id for h in hits}
        hits += [h for h in self._keyword(q) if h.id not in seen]   # Moss's top few may miss one
        hits = [h for h in hits if h.kind in ("hazard", "note") and pt & self._tok.get(h.id, _tokens(h.text))]
        return sorted(hits, key=lambda h: -h.score * (0.7 + 0.3 * freshness(h, now)))[:4]

    def heads_up(self, m: Memory, now: float | None = None) -> tuple[str, float]:
        """(what to say, how long it may wait) for a remembered hazard. Older notes are
        said more softly and give up sooner: the curb may have been fixed since."""
        days = age_days(m, time.time() if now is None else now)
        if days is None or days < 30:
            return f"Heads up: {m.text}", 30.0
        if days < 180:
            months = max(1, round(days / 30))
            when = "last month" if months == 1 else f"{months} months ago"
            return f"From {when}: {m.text}", 20.0
        return f"An old note, it may have changed: {m.text}", 15.0

    def summary(self) -> dict:
        return {"status": self.status, "count": len(self.docs), **self.stats}


# ---------------------------------------------------------------------- helpers
def _typed(meta: dict) -> dict:
    """Moss stores metadata as strings; numbers come back as numbers."""
    out = {}
    for k, v in meta.items():
        try:
            out[k] = float(v) if k in ("lat", "lng") else (int(v) if k in ("t", "confirmed") else v)
        except (TypeError, ValueError):
            out[k] = v
    return out


def _remote_differs(remote, local) -> bool:
    return (getattr(remote, "text", None) != local.text
            or dict(getattr(remote, "metadata", None) or {}) != dict(local.metadata or {}))


def _norm_name(s: str) -> str:
    return " ".join(sorted(_tokens(s)))


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def _distance(a: dict, b: dict) -> float | None:
    try:
        la, ga, lb, gb = (float(a["lat"]), float(a["lng"]), float(b["lat"]), float(b["lng"]))
    except (KeyError, TypeError, ValueError):
        return None
    from .geo import haversine_m
    return haversine_m(la, ga, lb, gb)


def _same_place(a: Memory, b: Memory) -> bool:
    d = _distance(a.meta, b.meta)
    return (d is not None and d <= SAME_PLACE_M and _norm_name(name_of(a)) == _norm_name(name_of(b)))


def age_days(m: Memory, now: float) -> float | None:
    t = m.meta.get("t")
    try:
        return max(0.0, (now - float(t)) / DAY_S) if t else None
    except (TypeError, ValueError):
        return None


def freshness(m: Memory, now: float) -> float:
    """1.0 for a fresh or undated note, halving every six months, never below 0.2."""
    d = age_days(m, now)
    return 1.0 if d is None else max(0.2, 0.5 ** (d / 180.0))
