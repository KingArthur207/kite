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

The map (Leaflet, OpenStreetMap tiles darkened in CSS) follows the walker from the newest `position` event that has `lat`/`lng`, and draws the recent trail. Active hazards are placed from their walker-frame `bearing` and `distance` plus the walker's compass `heading`, coloured by priority. The latest `beacon` shows as a green marker with a dashed line. Dragging the map stops auto-follow; the button turns it back on. On the right: the last spoken line with who said it, the walker, active hazards, shared live state (each source is green under 5 s since its last write, amber under 15 s, red after), the caregiver message box and the live feed.

Under the map, the walk timeline lists the last walk's key moments in order: stops (spoken "Stop." lines and critical hazards), hazards, Claude's judgments, answers and scene text, caregiver messages and route guidance. A walk is everything after the last 45 s gap in the event table. Each card has a badge for who acted: Kite (deterministic code, no AI), Claude, or Caregiver. Click a card to fly the map to where the walker was at that moment; stops stay marked on the map. Times come from the producer's own `t` when it is an epoch time, so a burst delivered late by a busy bridge still spreads out correctly.

"Shared live state" counts sources that wrote in the last 15 s and events in the last minute, with a sparkline per source, all from the `event` and `latest` tables. The event table keeps only the newest 500 rows, so when every row falls inside the last minute the rate shows as a lower bound (≥).

Press `P` (or the Present button) for presentation mode at the reveal. It hides the raw feed, the message box, the walker panel and the Clear button, and makes the map, timeline and last spoken line bigger. `#present` in the URL opens it that way.

![Presentation mode](docs/reveal-presentation.png)

The timeline logic has a self-check: `node --experimental-strip-types src/timeline.check.ts` (Node 22+).

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

"Message the walker" sends free text (trimmed, 200 characters max) or one of the quick buttons as a `say` event, source `caregiver`, priority `info`, payload `{"type":"say","source":"caregiver","priority":"info","text":"...","key":"caregiver-<ms>"}`. The last three sent show under the box and in the live feed (in blue).

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
