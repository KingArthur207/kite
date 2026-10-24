# Kite site

The site at **https://trykite.pages.dev**: three static pages (`/`, `/how`, `/judges`), plain CSS and a little JS.

- `public/` is what gets deployed (with `_headers` for security headers and caching).
- `tools/make_audio.py` renders every sound on the page from the real earpiece code:
  `earpiece/.venv/bin/python site/tools/make_audio.py`
- `src/*.body.html` hold each page's content; `python3 tools/build_pages.py` wraps them in the shared head, nav and footer.
- `og/og.html` is the social preview card; screenshot it to `public/assets/og.png` at 1200x630. `og/social.html` is the same card at 1280x640 for the GitHub repo preview (`docs/img/social-preview.png`).

Deploy by hand: `npx wrangler pages deploy site/public --project-name trykite --branch main`.
There is no automatic deploy: the repo has no GitHub Actions (they cost Arthur real money), so deploy by hand with the command above.
