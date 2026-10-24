# Kite site

The landing page at **https://trykite.pages.dev**: static HTML, CSS and a little JS, no build step.

- `public/` is what gets deployed (with `_headers` for security headers and caching).
- `tools/make_audio.py` renders every sound on the page from the real earpiece code:
  `earpiece/.venv/bin/python site/tools/make_audio.py`
- `og/og.html` is the social preview card; screenshot it to `public/assets/og.png` at 1200x630. `og/social.html` is the same card at 1280x640 for the GitHub repo preview (`docs/img/social-preview.png`).

Deploy by hand: `npx wrangler pages deploy site/public --project-name trykite --branch main`.
Pushes to `main` that touch `site/public/` deploy automatically once the repo has the
`CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID` secrets.
