#!/bin/sh
# Writes sim/.env.local (gitignored) with the Google Maps key from the repo-root .env, for ?world=tiles.
#   sh sim/tiles_env.sh [path/to/.env]      # default: the repo-root .env
# Never prints the key.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
src=${1:-$here/../.env}
key=$(sed -n 's/^GOOGLE_MAPS_API_KEY=//p' "$src" | tail -n 1 | tr -d "\"' \r")
[ -n "$key" ] || { echo "no GOOGLE_MAPS_API_KEY in $src" >&2; exit 1; }
umask 077
printf 'VITE_GOOGLE_MAPS_API_KEY=%s\n' "$key" > "$here/.env.local"
echo "wrote $here/.env.local"
