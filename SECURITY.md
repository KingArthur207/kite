# Security

Kite guides a blind pedestrian through traffic, so a security bug can become a safety bug. Treat both the same way.

## Reporting a problem

Open a [private security advisory](https://github.com/KingArthur207/kite/security/advisories/new) on this repository, or message the team directly. Please don't open a public issue. Include what you found, how to reproduce it, and what an attacker (or a buggy component) could make the wearer hear or miss.

## Secrets

- API keys live only in the repo-root `.env`, which is git-ignored. `.env.example` lists the variable names with no values.
- Keys are read from the environment at runtime and are never logged, sent to the dashboard, or embedded in client code.
- If a key is ever pasted somewhere public (a chat, an issue, a commit), rotate it at the provider straight away, then update `.env`. Removing it from git history is not enough.
- CI runs [gitleaks](https://github.com/gitleaks/gitleaks) over the **full git history** on every push and pull request.

| Variable | Used by | Where to rotate |
|---|---|---|
| `CARTESIA_API_KEY` | earpiece voice (TTS and STT) | play.cartesia.ai, API keys |
| `ANTHROPIC_API_KEY` | earpiece voice brain (Claude) | console.anthropic.com |
| `MOSS_PROJECT_ID`, `MOSS_PROJECT_KEY` | earpiece memory | portal.usemoss.dev, API keys |
| `GOOGLE_MAPS_API_KEY` | destination lookup (Places API New, Routes API v2) | Google Cloud console; restrict it to those APIs and add a daily quota |
| `GEMINI_API_KEY` | drone scene description | aistudio.google.com |

## How the earpiece limits damage

- **Network exposure.** The UDP (5010) and WebSocket (5011) inputs bind to `127.0.0.1` by default. Binding to the LAN (`--bind 0.0.0.0`) is an explicit choice, so only do it on a trusted network.
- **Untrusted input.** Every message is parsed and validated in `earpiece/events.py`. Numbers must be finite and are clamped, text is length-capped, identifiers are sanitised, frames are capped at 64 objects, and oversized datagrams are rejected. One bad object never drops a whole frame.
- **Flooding.** Each sender is rate-limited with a token bucket, and stale or out-of-order perception frames are ignored. The stress tests send 20,000 datagrams, a third of them malformed.
- **Audio safety.** A peak limiter caps output at -1 dBFS, so no message can make the AirPods dangerously loud. Critical warnings cannot be muted or paused by any remote command.
- **Model tools.** Claude's tool calls are validated against an allow-list before anything runs (`earpiece/brain.py`). The voice brain can navigate, change settings, save a memory or alert a caregiver, and nothing else.

## Automated checks

| Check | What it catches | Where |
|---|---|---|
| gitleaks | committed keys and tokens, anywhere in history | `.github/workflows/security.yml` |
| pip-audit | known CVEs in Python dependencies | security workflow |
| bandit (medium and above) | risky Python patterns | security workflow |
| npm audit (high and above) | known CVEs in the dashboard and SpacetimeDB module | security workflow |
| cargo audit | RustSec advisories for the Rust audio engine | security workflow |
| Dependabot | outdated dependencies and actions, grouped weekly | `.github/dependabot.yml` |

The security workflow also runs weekly, so newly published advisories surface without a code change.
