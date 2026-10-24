# Kite sync and dashboard

Real-time shared state for Kite, built on [SpacetimeDB](https://spacetimedb.com). The drone laptop, the wearable laptop and the caregiver dashboard all read and write the same database, so they stay in sync even though the two cameras are on different WiFi networks.

- `spacetimedb/src/index.ts`: the server module (tables and reducers)
- `src/`: the React dashboard, which subscribes to tables and re-renders live
- `src/module_bindings/`: generated client code; regenerate after changing the module

The hosted database is `kite-sync-pl6rp` on Maincloud ([dashboard](https://spacetimedb.com/kite-sync-pl6rp)).

## Run the dashboard

```bash
npm install
npm run dev        # connects to Maincloud by default
```

![Dashboard](docs/reveal-normal.png)

The map (Leaflet, OpenStreetMap tiles darkened in CSS) shows the walker from the newest `position` event with usable `lat`/`lng` (from either table, so one bad payload can't hide the walker), and draws the recent trail. A chip in the corner names the place, the direction the walker faces and how old the position is. Active hazards are placed from their walker-frame `bearing` and `distance` plus the walker's compass `heading`, coloured by priority. The latest `beacon` shows as a green marker with a dashed line. Two buttons pick the view: "Follow walker" keeps the walker centred, "Whole walk" frames the trail and every timeline moment. Dragging the map leaves both. On the right: the last spoken line with who said it, the latest stop, the walker, active hazards (with distance and direction in words, such as "8 m ahead right"), the caregiver message box, the devices and the live feed.

Under the map, the walk timeline lists the last walk's key moments in order: stops (spoken "Stop." lines and critical hazards, merged into one stop worded as the walker heard it), help requests (`caregiver_alert`), hazards, Claude's judgments, answers and scene text, caregiver messages and route guidance. A walk is everything after the last 45 s gap in the event table. A ribbon above the cards shows the whole walk at a glance, with stops marked in red. Each card has a badge for who acted: Kite (deterministic code, no AI), Claude, or Caregiver, and the header counts each. Click a card or a ribbon mark to fly the map to where the walker was at that moment; stops stay marked on the map. Times come from the producer's own `t` when it is an epoch time, so a burst delivered late by a busy bridge still spreads out correctly.

"Devices" counts sources that wrote in the last 15 s and events in the last minute, with a sparkline per source, all from the `event` and `latest` tables. Each source is green under 5 s since its last write, amber under 15 s, red ("silent") after. The event table keeps only the newest 500 rows, so when every row falls inside the last minute the rate shows as a lower bound (≥).

When the walker asks Kite to tell their caregiver (a `caregiver_alert` row), a red banner shows the request until the caregiver presses "Got it". The header always says whether the page is live, connecting, reconnecting (the SDK reconnects by itself; the last data stays on screen) or offline, and a banner says so when no device has written for 30 s.

Press `P` (or the Present button) for presentation mode at the reveal; `Esc` leaves it. It hides the raw feed, the message box, the walker panel and the Clear button, frames the whole walk on the map, and makes the timeline, the last spoken line ("Kite just said") and the latest stop bigger. `#present` in the URL opens it that way.

Accessibility: every control works from the keyboard with a visible focus ring, section titles are real headings, the timeline cards are buttons, and text colours meet WCAG AA contrast (checked with axe-core). New spoken lines are announced politely to screen readers; stops and help requests are announced assertively. Only changes after the page loads are announced, never the backlog. Motion is reduced when the system asks for it, and `prefers-contrast: more` brightens the secondary text. Payloads are untrusted: they are parsed defensively, clipped, and rendered as text, never as HTML.

![Presentation mode](docs/reveal-presentation.png)

The pure logic (timeline, payload parsing, freshness, distances, message delivery and the blocked-word mirror) has self-checks with no extra dependencies: `npm run check` (Node 22+).

To see it move without hardware, from the repo root:

```bash
earpiece/.venv/bin/python scripts/replay.py data/sample_track.jsonl --spacetime --speed 2
```

`data/sample_track.jsonl` is a 45 s synthetic walk up Lyon Street (source `test-dashboard`) with a beacon, three hazards and two status sources.

## Change the module

```bash
npm run spacetime:publish    # needs `spacetime login` as the database owner
npm run spacetime:generate   # refresh src/module_bindings
```

## Tables

- `event`: every Kite event, newest 500 kept. Columns are `id`, `type`, `source`, `priority`, `payload` (the full event as a JSON string, same shapes as `earpiece/earpiece/events.py`) and `t` (server timestamp).
- `latest`: the newest event per type and source, keyed `"type:source"` (for example `"position:sim"`), so a dashboard that joins late sees current state right away.

Reducers: `push(type, source, priority, payload)` writes to both tables, and `clear()` wipes them before a new run. The dashboard reads both tables. It calls `clear` from its "Clear run" button and `push` from "Message the walker".

## Caregiver messages

"Message the walker" sends free text (trimmed, 200 characters max, with a counter) or one of the quick replies as a `say` event, source `caregiver`, priority `info`, payload `{"type":"say","source":"caregiver","priority":"info","text":"...","key":"caregiver-<ms>"}`. The last three sent show under the box with their state: sending, sent and waiting for the earpiece, "Heard by the walker" once the earpiece logs a `spoken` row with the same words after "Message from your caregiver:", or a hint to check the earpiece if nothing was spoken after 20 s. They also appear in the live feed (in blue).

The box mirrors the earpiece's permission filter (`earpiece/earpiece/judgment.py`): a message with a word like "go", "cross", "clear" or "you can" would never be spoken, so the dashboard explains why and doesn't send it. The earpiece stays the authority.

An earpiece run with `--spacetime` polls for these about once a second and reads them aloud, prefixed "Message from your caregiver:". It uses the HTTP SQL endpoint:

```bash
curl -X POST $HOST/v1/database/kite-sync-pl6rp/sql -H "Authorization: Bearer $TOKEN" \
  -d "SELECT * FROM event WHERE source = 'caregiver'"
```

The earpiece clamps them to `info`, ignores anything other than `say`, skips rows that were already there when it started, and doesn't mirror them back into Spacetime.

## Write from Python (drone or vision laptop)

No SDK needed. Reducers are plain HTTP calls with a JSON object of arguments. `payload` is itself a JSON string:

```bash
HOST=https://maincloud.spacetimedb.com   # or http://127.0.0.1:3000 for a local server
TOKEN=$(curl -s -X POST $HOST/v1/identity | jq -r .token)
curl -X POST $HOST/v1/database/kite-sync-pl6rp/call/push \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"type":"position","source":"sim","priority":"info",
       "payload":"{\"type\":\"position\",\"lat\":37.80,\"lng\":-122.45,\"heading\":270,\"place\":\"Lyon St\"}"}'
```

## Work offline against a local server

```bash
spacetime start --listen-addr 127.0.0.1:3000
npm run spacetime:publish:local
printf 'VITE_SPACETIMEDB_HOST=ws://localhost:3000\n' > .env.local
npm run dev
```

## Caregiver message signing

The `event` table is public and `push` takes calls from any identity, so the earpiece checks caregiver
messages itself. Put one random value in the repo `.env` as `KITE_CAREGIVER_TOKEN` and the same value in
`dashboard/.env.local` as `VITE_KITE_CAREGIVER_TOKEN`. The dashboard then adds `ts` and `sig`
(HMAC-SHA256 of `ts:text`) to each message, and the earpiece speaks only signed messages under 2 minutes
old. The value ends up in the page's JS, so serve the dashboard only on the demo laptop. If neither side
sets a token, messages are still accepted.

Either way, the earpiece drops any message that could be heard as permission to move ("go", "cross",
"you can", "clear" and so on), always speaks at priority `info`, and ignores the row's `key` and `ttl`.
The module is unchanged, so there is nothing to republish.
