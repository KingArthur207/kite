"""Shared by board.py (on the UNO Q) and kite_link.py (on the computer). Standard library only.

* Spacetime: the three SpacetimeDB HTTP calls Kite uses (identity, the `push` reducer, SQL reads).
* sign / verify: HMAC-SHA256 with the board's token over a canonical JSON body. The SpacetimeDB tables are
  public, so every heartbeat, command and reply that goes through them is signed: a stranger can neither
  point the computer at their own IP (and collect the token) nor send the board commands.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import urllib.error
import urllib.request

SPACETIME_HOST = "https://maincloud.spacetimedb.com"
SPACETIME_DB = "kite-sync-pl6rp"
DEFAULT_PORT = 8765
DEFAULT_SOURCE = "unoq"
TAILSCALE_NAME = "kite-unoq"          # the board's MagicDNS name when it runs Tailscale
MAX_TEXT = 500                        # longest text message, in characters
FRESH_S = 120                         # a signed command older than this (by the server's clock) is ignored
_IDENT = re.compile(r"^[A-Za-z0-9_.:-]{1,48}$")   # the module allows 64; we append ".link" / ".board"


def check_source(source: str) -> str:
    if not _IDENT.match(source or ""):
        raise ValueError("source: 1-48 letters, digits, _ . : - only")
    return source


def link_source(source: str) -> str:
    """Where the computer pushes commands for this board."""
    return f"{source}.link"


def reply_source(source: str) -> str:
    """Where the board pushes its replies."""
    return f"{source}.board"


def clean_text(text) -> str:
    """A message as the board accepts it: a string, printable, at most MAX_TEXT characters."""
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    text = "".join(ch for ch in text if ch.isprintable() or ch == " ").strip()
    if not text:
        raise ValueError("text is empty")
    if len(text) > MAX_TEXT:
        raise ValueError(f"text is longer than {MAX_TEXT} characters")
    return text


def canonical(body: dict) -> bytes:
    return json.dumps({k: v for k, v in body.items() if k != "sig"}, sort_keys=True,
                      separators=(",", ":"), ensure_ascii=True).encode()


def sign(token: str, body: dict) -> dict:
    """body plus `sig` = hex HMAC-SHA256(token, canonical JSON of everything else)."""
    out = dict(body)
    out["sig"] = hmac.new(token.encode(), canonical(body), hashlib.sha256).hexdigest()
    return out


def verify(token: str, body) -> bool:
    if not isinstance(body, dict) or not isinstance(body.get("sig"), str) or not token:
        return False
    want = hmac.new(token.encode(), canonical(body), hashlib.sha256).hexdigest()
    return hmac.compare_digest(want, body["sig"])


def row_time(v) -> float | None:
    """A row's `t` column (a Timestamp, sent as [micros since the epoch]) in epoch seconds."""
    while isinstance(v, (list, tuple)) and v:
        v = v[0]
    if isinstance(v, dict):
        v = next(iter(v.values()), None)
    try:
        return int(v) / 1e6
    except (TypeError, ValueError):
        return None


def sql_rows(reply) -> list[dict]:
    """The HTTP SQL endpoint's reply (one statement per entry) as plain dicts keyed by column."""
    out = []
    for stmt in reply:
        names = [(e.get("name") or {}).get("some") for e in stmt["schema"]["elements"]]
        out += [dict(zip(names, row)) for row in stmt["rows"]]
    return out


class Spacetime:
    """Kite's SpacetimeDB database over plain HTTP: an anonymous identity, `push`, and SQL reads."""

    def __init__(self, host: str = SPACETIME_HOST, db: str = SPACETIME_DB, timeout: float = 8.0) -> None:
        self.host, self.db, self.timeout = host.rstrip("/"), db, timeout
        self.token: str | None = None
        self.calls = 0

    def _post(self, path: str, body: bytes, auth: bool = True, json_body: bool = False) -> bytes:
        headers = {"Content-Type": "application/json"} if json_body else {}
        if auth:
            if self.token is None:
                self.token = json.loads(self._post("/v1/identity", b"", auth=False))["token"]
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(self.host + path, data=body, headers=headers, method="POST")
        self.calls += 1
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:   # http(s) host from config  # nosec B310
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):          # identity expired or revoked: mint a new one next time
                self.token = None
            raise

    def push(self, type_: str, source: str, payload: dict, priority: str = "system") -> None:
        args = {"type": type_, "source": source, "priority": priority, "payload": json.dumps(payload)}
        self._post(f"/v1/database/{self.db}/call/push", json.dumps(args).encode(), json_body=True)

    def sql(self, query: str) -> list[dict]:
        return sql_rows(json.loads(self._post(f"/v1/database/{self.db}/sql", query.encode())))

    def rows(self, type_: str, source: str, after_id: int | None = None) -> list[dict]:
        """Every `event` row of this type and source (newer than after_id), oldest first.
        type_ and source are checked against the module's identifier rule, so nothing is injectable."""
        check_source(source)
        if not _IDENT.match(type_):
            raise ValueError("bad type")
        q = f"SELECT * FROM event WHERE type = '{type_}' AND source = '{source}'"  # checked above  # nosec B608
        if after_id is not None and after_id >= 0:      # ids are unsigned: "id > -1" would be rejected
            q += f" AND id > {int(after_id)}"
        return sorted(self.sql(q), key=lambda r: int(r["id"]))

    def latest(self, type_: str, source: str) -> dict | None:
        """The newest row for (type, source) from the `latest` table, or None."""
        check_source(source)
        if not _IDENT.match(type_):
            raise ValueError("bad type")
        got = self.sql(f"SELECT * FROM latest WHERE key = '{type_}:{source}'")  # checked above  # nosec B608
        return got[0] if got else None

    def last_id(self, type_: str, source: str) -> int:
        rows = self.rows(type_, source)
        return int(rows[-1]["id"]) if rows else -1
