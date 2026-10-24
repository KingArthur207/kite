# Kite site

The site at **https://trykite.pages.dev**: three static pages (`/`, `/how`, `/judges`), plain CSS and a little JS.

- `public/` is what gets deployed (with `_headers` for security headers and caching).
- `tools/make_audio.py` renders every sound on the page from the real earpiece code:
  `earpiece/.venv/bin/python site/tools/make_audio.py`
- `src/*.body.html` hold each page's content. `python3 tools/build_pages.py` wraps each in the shared head (with Organization JSON-LD), nav and footer, writes a Markdown twin of every page (`public/<page>.md`), and regenerates `sitemap.xml` and `llms.txt`, so none of them drift. Never edit the generated files by hand.
- `functions/_middleware.js` (Cloudflare Pages Functions): a request whose `Accept` prefers `text/markdown` gets the page's Markdown with `Vary: Accept`; everyone else gets the HTML. Unknown paths are a real 404 (`public/404.html`, or a Markdown body for Markdown clients).
- `tests/` checks the middleware and the generated files: `node --test tests/*.test.mjs` (also run by `scripts/check.sh site` and by `scripts/deploy-site.sh`).
- `og/og.html` is the social preview card; screenshot it to `public/assets/og.png` at 1200x630. `og/social.html` is the same card at 1280x640 for the GitHub repo preview (`docs/img/social-preview.png`).

Deploy with `scripts/deploy-site.sh`: it rebuilds, runs the tests, and deploys `public/` with `functions/` from inside `site/` (wrangler only picks up `./functions` from the directory it runs in).
There is no automatic deploy: the repo has no GitHub Actions (they cost Arthur real money), so deploy by hand with the command above.

## Evidence images

Evidence images in `public/assets/evidence/` are copies of real artifacts from this repo, resized with `sips`: `caregiver-dashboard.jpg` is `dashboard/docs/reveal-normal.png` (a replayed test walk), and `yolo-drone-view.jpg` and `yolo-chest-view.jpg` are `sim/docs/yolo_drone.jpg` and `sim/docs/yolo_chest.jpg` (YOLO on the browser simulation, scored in `sim/README.md`). Only add an image to a card when a real artifact proves the claim.

## Sponsor logos

Logos on `/judges` are the companies' own marks, fetched from their sites and stored in `public/assets/logos/` (the CSP only allows local images). Only site-specific CSS classes were stripped and `currentColor` was set to the colour each site renders it in. If a company has no official mark we can fetch, use a text wordmark in the card's own type; never draw an imitation.

- `cartesia.svg`: header logo on https://cartesia.ai (icon green `#309d4b` from the site's own CSS)
- `anthropic.svg`: header wordmark on https://www.anthropic.com
- `spacetimedb.svg`: https://spacetimedb.com/images/brand.svg, recoloured from light grey to near-black for the paper background
- `moss.svg`: `moss_wordmark_dark.svg` from the Moss brand kit, https://www.moss.dev/brand-kit (viewBox trimmed to the mark)
- `microsoft.png`: header logo on https://www.microsoft.com/en-us, served from https://img-prod-cms-rt-microsoft-com.akamaized.net/cms/api/am/imageFileData/RE1Mu3b
- `meta.svg`: header logo on https://www.meta.com/about/, served from https://static.xx.fbcdn.net/rsrc.php/y9/r/tL_v571NdZ0.svg
- `xai.svg`: header mark on https://x.ai (SpaceXAI)
