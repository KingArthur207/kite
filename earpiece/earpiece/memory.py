"""Kite's memory of the wearer's world, on Moss.

What it holds:
  place   "my usual coffee shop: Blue Bottle on Chestnut Street"   (with lat/lng when known)
  hazard  "the curb at Lyon and Bay is very tall"
  note    anything else the wearer or their caregiver wants Kite to know

How it is used:
  * "take me to my coffee shop" -> the personal place is resolved before routing, and
    the navigation side receives the real name and coordinates;
  * every question to Claude carries the few memories most relevant to it;
  * when the wearer reaches a remembered place, a matching hazard note is spoken once;
  * "Kite, remember this place as the bench" / "remember that ..." add memories by voice;
    the caregiver dashboard can add them too.

Why Moss: the index is loaded into Moss's local (Rust) runtime, so a lookup takes
3-12 ms in-process (measured), cheap enough to run on every utterance. Writes go
to the Moss cloud index and are mirrored to a local JSON file, so memories
survive restarts and a simple keyword search keeps working with no network.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import REPO_ROOT

INDEX = "kite-memory"
STORE = REPO_ROOT / "earpiece" / ".cache" / "memory.json"
SEED = Path(__file__).with_name("memory_seed.json")
_STOP = {"the", "a", "an", "my", "to", "of", "at", "on", "in", "and", "is", "me", "take", "go", "i",
         "where", "what", "for", "by", "near", "with", "this", "that", "it", "please", "kite",
         # generic street words: "Union Street" must not match a note about "Lyon Street"
         "street", "st", "avenue", "ave", "boulevard", "blvd", "road", "rd", "lane", "way", "drive",
         "corner", "intersection", "block", "side", "hazards", "hazard"}


@dataclass
class Memory:
    id: str
    kind: str
    text: str
    meta: dict = field(default_factory=dict)
    score: float = 0.0


def _tokens(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", s.lower()) if w not in _STOP and len(w) > 1}


class MemoryStore:
    def __init__(self, project_id: str | None = None, project_key: str | None = None,
                 index: str = INDEX, store: Path = STORE, seed: Path | None = SEED, log=print):
        self.project_id = project_id if project_id is not None else os.environ.get("MOSS_PROJECT_ID")
        self.project_key = project_key if project_key is not None else os.environ.get("MOSS_PROJECT_KEY")
        self.index = index
        self.store = Path(store)
        self.log = log
        self.docs: dict[str, Memory] = {}
        self.ready = False                    # Moss local index loaded
        self.status = "local"
        self._client = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._dirty_since_load = False
        self.stats = {"queries": 0, "moss_queries": 0, "last_ms": None, "writes": 0, "errors": 0}
        self._load_local(seed)

    # ------------------------------------------------------------------ local mirror
    def _load_local(self, seed: Path | None) -> None:
        src = self.store if self.store.exists() else seed
        if src is None or not Path(src).exists():
            return
        try:
            for d in json.loads(Path(src).read_text()):
                m = Memory(d["id"], d.get("kind", "note"), d["text"], d.get("meta") or {})
                self.docs[m.id] = m
        except (OSError, ValueError, KeyError) as e:
            self.log(f"[memory] could not read {src}: {e}")

    def _save_local(self) -> None:
        self.store.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.store.with_suffix(".tmp")
        tmp.write_text(json.dumps([{k: v for k, v in asdict(m).items() if k != "score"}
                                   for m in self.docs.values()], indent=1))
        os.replace(tmp, self.store)

    # ------------------------------------------------------------------ Moss lifecycle
    def start(self) -> None:
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
        from inferedge_moss import DocumentInfo
        meta = {"kind": m.kind, **{k: str(v) for k, v in m.meta.items() if v is not None}}
        return DocumentInfo(id=m.id, text=m.text, metadata=meta)

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
                await self._client.create_index(self.index, [self._doc(m) for m in self.docs.values()])
            else:
                remote = {d.id for d in await self._client.get_docs(self.index)}
                missing = [m for m in self.docs.values() if m.id not in remote]
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
        from inferedge_moss import MutationOptions
        await self._client.add_docs(self.index, [self._doc(m) for m in ms], MutationOptions(upsert=True))

    async def _sync_new(self, m: Memory) -> None:
        try:
            await self._add([m])
            await asyncio.sleep(2.0)
            if await self._wait_ready(60):
                await self._client.load_index(self.index)       # refresh the local copy
                self._dirty_since_load = False
        except Exception as e:
            self.stats["errors"] += 1
            self.log(f"[memory] Moss write failed ({type(e).__name__}); kept locally")

    # ------------------------------------------------------------------ API
    def remember(self, kind: str, text: str, meta: dict | None = None) -> Memory:
        kind = kind if kind in ("place", "hazard", "note") else "note"
        text = " ".join(text.split())[:300]
        m = Memory(f"{kind}-{uuid.uuid4().hex[:8]}", kind, text, {k: v for k, v in (meta or {}).items()
                                                                   if v is not None})
        self.docs[m.id] = m
        self.stats["writes"] += 1
        self._dirty_since_load = True
        try:
            self._save_local()
        except OSError as e:
            self.log(f"[memory] could not save locally: {e}")
        if self._client is not None and self._loop is not None:
            asyncio.run_coroutine_threadsafe(self._sync_new(m), self._loop)
        return m

    def forget(self, mem_id: str) -> bool:
        if mem_id not in self.docs:
            return False
        del self.docs[mem_id]
        self._dirty_since_load = True
        self._save_local()
        if self._client is not None and self._loop is not None:
            async def _del():
                try:
                    await self._client.delete_docs(self.index, [mem_id])
                    if await self._wait_ready(60):
                        await self._client.load_index(self.index)
                        self._dirty_since_load = False
                except Exception as e:
                    self.log(f"[memory] Moss delete failed ({type(e).__name__})")
            asyncio.run_coroutine_threadsafe(_del(), self._loop)
        return True

    def search(self, query: str, kind: str | None = None, top_k: int = 3, timeout: float = 0.25) -> list[Memory]:
        """Most relevant memories. Moss local index when loaded; keyword fallback otherwise.
        Memories added since the last load are always merged in from the local mirror."""
        self.stats["queries"] += 1
        t0 = time.perf_counter()
        hits: list[Memory] = []
        if self.ready and self._loop is not None:
            try:
                from inferedge_moss import QueryOptions
                fut = asyncio.run_coroutine_threadsafe(
                    self._client.query(self.index, query, QueryOptions(top_k=top_k + 4, alpha=0.7)), self._loop)
                res = fut.result(timeout)
                for d in res.docs:
                    m = self.docs.get(d.id)
                    meta = dict(d.metadata or {})
                    k = meta.pop("kind", m.kind if m else "note")
                    hits.append(Memory(d.id, k, d.text, m.meta if m else meta, float(d.score)))
                self.stats["moss_queries"] += 1
            except Exception as e:
                self.stats["errors"] += 1
                self.log(f"[memory] query failed ({type(e).__name__}); keyword fallback")
                hits = []
        if not hits or self._dirty_since_load:
            seen = {h.id for h in hits}
            hits += [h for h in self._keyword(query) if h.id not in seen]
            hits.sort(key=lambda h: -h.score)
        if kind:
            hits = [h for h in hits if h.kind == kind]
        self.stats["last_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return hits[:top_k]

    def _keyword(self, query: str) -> list[Memory]:
        q = _tokens(query)
        out = []
        for m in self.docs.values():
            t = _tokens(m.text)
            if q and t:
                s = len(q & t) / len(q)
                if s > 0:
                    out.append(Memory(m.id, m.kind, m.text, m.meta, round(min(0.99, 0.5 + 0.5 * s), 3)))
        return sorted(out, key=lambda m: -m.score)

    def resolve_place(self, phrase: str) -> Memory | None:
        """'my coffee shop' -> the remembered place, only when the match is unambiguous.
        Moss scores are relative, so we also require a shared content word (or an explicitly
        personal phrase like 'my ...' / 'home') to avoid inventing a destination."""
        hits = self.search(phrase, kind="place", top_k=2)
        if not hits:
            return None
        top = hits[0]
        personal = bool(re.search(r"\b(my|home|usual|favou?rite|our)\b", phrase.lower()))
        shared = _tokens(phrase) & _tokens(top.text + " " + str(top.meta.get("name", "")))
        if not shared and not (personal and top.score >= 0.9):
            return None
        if len(hits) > 1 and hits[1].score >= top.score - 0.005 and not shared:
            return None
        return top

    def notes_for(self, place: str) -> list[Memory]:
        """Hazard notes that mention this place (shared content word required)."""
        pt = _tokens(place)
        return [h for h in self.search(f"hazards and tips near {place}", top_k=4)
                if h.kind in ("hazard", "note") and pt & _tokens(h.text)]

    def summary(self) -> dict:
        return {"status": self.status, "count": len(self.docs), **self.stats}
