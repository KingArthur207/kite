#!/usr/bin/env bash
# Deploy the site to https://trykite.pages.dev.
# Rebuilds the pages from site/src, runs the site tests, then deploys site/public together with the
# Pages Functions in site/functions (Markdown negotiation and real 404s; wrangler picks up ./functions
# from the directory it runs in, hence the cd).
# Uses your own `wrangler login` session, or CLOUDFLARE_API_TOKEN / CLOUDFLARE_ACCOUNT_ID if set.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
python3 "$ROOT/site/tools/build_pages.py"
node --test "$ROOT"/site/tests/*.test.mjs
cd "$ROOT/site"
npx -y wrangler@4 pages deploy public --project-name trykite --branch main --commit-dirty=true
echo "live at https://trykite.pages.dev"
