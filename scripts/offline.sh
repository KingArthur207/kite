#!/usr/bin/env bash
# Offline fallback (spec 5.2): local SpacetimeDB with the Kite module, dashboard pointed at it.
#   scripts/offline.sh        start (if needed) + publish + write dashboard/.env.local
#   scripts/offline.sh off    point the dashboard back at Maincloud (the server keeps running)
# Safe to run again. Python producers read SPACETIME_HOST from .env; this prints the line.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DASH="$ROOT/dashboard"
ENV_LOCAL="$DASH/.env.local"
ADDR=127.0.0.1:3000
LINE="VITE_SPACETIMEDB_HOST=ws://$ADDR"
export PATH="$HOME/.local/bin:$PATH"

drop_line() {  # remove our line from dashboard/.env.local, keep anything else a teammate put there
  [ -f "$ENV_LOCAL" ] || return 0
  grep -v '^VITE_SPACETIMEDB_HOST=' "$ENV_LOCAL" > "$ENV_LOCAL.tmp" || true
  if [ -s "$ENV_LOCAL.tmp" ]; then mv "$ENV_LOCAL.tmp" "$ENV_LOCAL"; else rm -f "$ENV_LOCAL.tmp" "$ENV_LOCAL"; fi
}

if [ "${1:-}" = "off" ]; then
  drop_line
  echo "dashboard back on Maincloud (restart npm run dev). In .env put:"
  echo "  SPACETIME_HOST=https://maincloud.spacetimedb.com"
  echo "The local server is still running; stop it with: pkill -f 'spacetimedb-standalone start'"
  exit 0
fi

command -v spacetime >/dev/null || { echo "spacetime CLI not found (expected ~/.local/bin/spacetime)"; exit 1; }

if curl -sf "http://$ADDR/v1/ping" >/dev/null; then
  echo "local SpacetimeDB already running on $ADDR"
else
  echo "starting local SpacetimeDB on $ADDR (log: /tmp/kite-spacetime.log)"
  nohup spacetime start --listen-addr "$ADDR" > /tmp/kite-spacetime.log 2>&1 &
  for _ in $(seq 1 30); do curl -sf "http://$ADDR/v1/ping" >/dev/null && break; sleep 1; done
  curl -sf "http://$ADDR/v1/ping" >/dev/null || { echo "server did not come up, see /tmp/kite-spacetime.log"; exit 1; }
fi

cd "$DASH"
[ -d spacetimedb/node_modules ] || (cd spacetimedb && npm install --no-audit --no-fund)
npm run spacetime:publish:local

drop_line
echo "$LINE" >> "$ENV_LOCAL"
echo
echo "dashboard: wrote $LINE to dashboard/.env.local (restart npm run dev)"
echo "Python producers: put this in .env"
echo "  SPACETIME_HOST=http://$ADDR"
