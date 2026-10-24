#!/usr/bin/env bash
# Deploy the landing site (site/public) to https://trykite.pages.dev.
# Uses your own `wrangler login` session, or CLOUDFLARE_API_TOKEN / CLOUDFLARE_ACCOUNT_ID if set.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
npx -y wrangler@4 pages deploy site/public --project-name trykite --branch main --commit-dirty=true
echo "live at https://trykite.pages.dev"
