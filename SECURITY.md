# Security

Kite guides a blind pedestrian through traffic, so a security bug can become a safety bug. Treat both the same way.

## Reporting a problem

Open a [private security advisory](https://github.com/KingArthur207/kite/security/advisories/new) on this repository, or message the team directly. Please don't open a public issue. Include what you found, how to reproduce it, and what an attacker (or a buggy component) could make the wearer hear or miss.

## Secrets

- API keys live only in the repo-root `.env`, which is git-ignored. `.env.example` lists the variable names with no values.
- Keys are read from the environment at runtime and are never logged, sent to the dashboard, or embedded in client code.
- If a key is ever pasted somewhere public (a chat, an issue, a commit), rotate it at the provider straight away, then update `.env`. Removing it from git history is not enough.
- [`scripts/check.sh`](scripts/check.sh) runs [gitleaks](https://github.com/gitleaks/gitleaks) over the **full git history**; run it before merging.

| Variable | Used by | Where to rotate |
|---|---|---|
| `CARTESIA_API_KEY` | earpiece voice (TTS and STT) | play.cartesia.ai, API keys |
| `ANTHROPIC_API_KEY` | earpiece voice brain (Claude) | console.anthropic.com |
| `MOSS_PROJECT_ID`, `MOSS_PROJECT_KEY` | earpiece memory | portal.usemoss.dev, API keys |
| `GOOGLE_MAPS_API_KEY` | destination lookup (Places API New, Routes API v2) | Google Cloud console; restrict it to those APIs and add a daily quota |

## How the earpiece limits damage

- **Network exposure.** The UDP (5010) and WebSocket (5011) inputs bind to `127.0.0.1` by default. Binding to the LAN (`--bind 0.0.0.0`) is an explicit choice, so only do it on a trusted network.
- **Other web pages.** Browsers let any page open a WebSocket to `127.0.0.1`, so binding to loopback is not enough on its own. The earpiece WebSocket (and the sim relay on 5020) check the `Origin` header: pages served from this machine and non-browser clients (no `Origin`) may connect, and every other origin gets 403. A sim page served from another machine needs `earpiece run --allow-origin http://HOST:5173` (and `--origin` on the relay).
- **Untrusted input.** Every message is parsed and validated in `earpiece/events.py`, which raises only `BadMessage`: Hypothesis fuzz tests throw arbitrary bytes, arbitrary JSON, huge and non-finite numbers, deep nesting, unicode and junk base64 at it. Numbers must be finite and are clamped, coordinates outside the globe count as unknown, text is length-capped, identifiers are sanitised, frames are capped at 64 objects, and oversized messages are rejected. One bad object, or one garbled optional field, never drops a whole frame.
- **Flooding.** Each sender is rate-limited with a token bucket (the table of buckets is itself bounded), and stale or out-of-order perception frames are ignored. What a sender can grow is capped too: the speech queue (32 lines, critical lines only shed past 128), active hazards (64), inbound memories (30 a minute), and memory lookups (one at a time). A WebSocket client too slow to take a message in 0.5 s is closed so it reconnects. The stress tests send 20,000 datagrams (a third of them malformed), reconnect storms, preemption storms and the sample walk at 200x.
- **Audio safety.** A peak limiter caps output at -1 dBFS, so no message can make the AirPods dangerously loud. Critical warnings cannot be muted or paused by any remote command.
- **Model tools.** Claude's tool calls are validated against an allow-list before anything runs (`earpiece/brain.py`). The voice brain can navigate, change settings, or save a memory, and nothing else; settings come from a fixed list and every string is length-capped.
- **Prompt injection.** Scene descriptions, sign text, hazards, memories and place names all reach Claude as JSON inside a tagged block in which `<` is escaped, so street text can't close the block and pose as the wearer. Both prompts say that text is untrusted data. The judgment layer also drops any line that could be heard as permission to move.

## Automated checks

All of them run locally with `scripts/check.sh` (or one at a time: `scripts/check.sh secrets audit`). The team runs it before merging; there are no hosted CI runners.

| Check | What it catches | Step |
|---|---|---|
| gitleaks | committed keys and tokens, anywhere in history | `secrets` |
| pip-audit | known CVEs in Python dependencies (earpiece and drone venvs) | `audit` |
| bandit (medium and above) | risky Python patterns in `earpiece/`, `kite/`, `perception/`, `earpiece/evals/`, `site/tools/` and `scripts/`; each accepted finding carries `# nosec <id>` and its reason | `audit` |
| npm audit (high and above) | known CVEs in the dashboard and SpacetimeDB module (the site has no npm dependencies) | `audit` |
| cargo audit | RustSec advisories for the Rust audio engine | `audit` |
| ruff (bug-finding rules, config in `pyproject.toml`) | unused and undefined names, dropped asyncio tasks, likely bugs, in every Python directory | `lint` |
| shellcheck | quoting and logic bugs in the shell scripts | `lint` |
| bidi scan | Trojan Source: bidirectional control characters hidden in source | `lint` |
| `node --check` | syntax errors in the site's Functions and browser JS, which have no build step | `lint` |

`scripts/check.sh hooks` installs a pre-commit hook that runs ruff and gitleaks on what you stage.

## The site

- `site/public/_headers` sets HSTS, a strict Content-Security-Policy (own scripts and media only, Google Fonts for type), `X-Frame-Options: DENY`, `nosniff`, a Referrer-Policy and a Permissions-Policy that turns off camera, microphone and location.
- Cloudflare Pages does not apply `_headers` to responses a Pages Function builds, so `site/functions/_middleware.js` adds the same headers to the Markdown pages and 404s it serves; a test checks the two sets are identical.

## SpacetimeDB and the dashboard

- The `push` reducer checks type, source, priority and a 16 KB JSON-object payload. The event log is held to about 500 rows even when ids skip, and `latest` to 200 (type, source) pairs, so no caller can grow either table without bound.
- `clear` runs at most once a minute.
- The dashboard renders all row text through React (no raw HTML) and ships a Content-Security-Policy in production builds: its own scripts and styles only, network access to the SpacetimeDB host only, and images from itself and OpenStreetMap tiles.

## Accepted risks

These are known and accepted for the hackathon demo. Revisit them before Kite is used for real.

- **No auth on SpacetimeDB.** `push` and `clear` accept any identity, so anyone who knows the database name (it is in this repo) can add fake dashboard rows, clear the view once a minute, or read the walker's live position. The earpiece only writes to it and never acts on its rows. Use a fresh database for anything beyond the demo.
- **The CSP needs `'unsafe-eval'`**, because the SpacetimeDB SDK builds its serializers with `Function()` (from the schema, never from row data). It only applies to production builds; `scripts/demo.sh` runs the Vite dev server, which has none.
- **The sim's Google key ships in its page** (`VITE_GOOGLE_MAPS_API_KEY`) by design. Restrict it by HTTP referrer in the Google Cloud console.
- **LAN mode trusts the LAN.** With `--bind 0.0.0.0`, any machine on the network can send UDP events, which carry no Origin and no signature. Use it only on a network you control.
- **Hugging Face depth weights are not pinned to a revision.** `python -m perception --download` fetches them by name; at run time they load with `local_files_only`, so nothing is fetched during a walk. Pinning would invalidate the caches already downloaded; do it when the weights are next refreshed.
- **Some dev scripts cache in `/tmp`** (Street View tiles, open-data pulls). They run by hand on a single-user laptop; bandit's /tmp, fixed-URL and all-interfaces checks are skipped for `scripts/` only.
- **Claude can still be talked into a setting change.** Content can't pose as the wearer, but a model can still be swayed by what it reads, and the `control` tool can pause guidance or mute the beeps. Neither setting silences a critical warning, and "resume" or "beeps on" undoes it.
