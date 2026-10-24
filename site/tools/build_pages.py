"""Build site/public/{index,how,judges}.html from site/src/*.body.html with one shared head, nav and footer.

    python3 site/tools/build_pages.py
"""
import re, pathlib
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent / 'public'
SRC = HERE.parent / 'src'
FONTS = 'https://fonts.googleapis.com/css2?family=Caveat:wght@500;600&family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600&family=JetBrains+Mono:wght@400;500&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,500;1,8..60,400;1,8..60,500&display=swap'

def head(title, desc, path):
    url = 'https://trykite.pages.dev' + path
    return f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <meta name="description" content="{desc}" />
  <link rel="canonical" href="{url}" />
  <meta name="theme-color" content="#F5F2EC" />
  <link rel="icon" href="/assets/kite-icon.svg" type="image/svg+xml" />
  <link rel="apple-touch-icon" href="/assets/kite-touch-icon.png" />
  <meta property="og:type" content="website" />
  <meta property="og:site_name" content="Kite" />
  <meta property="og:title" content="{title}" />
  <meta property="og:description" content="{desc}" />
  <meta property="og:url" content="{url}" />
  <meta property="og:image" content="https://trykite.pages.dev/assets/og.png" />
  <meta property="og:image:width" content="1200" />
  <meta property="og:image:height" content="630" />
  <meta name="twitter:card" content="summary_large_image" />
  <meta name="twitter:title" content="{title}" />
  <meta name="twitter:description" content="{desc}" />
  <meta name="twitter:image" content="https://trykite.pages.dev/assets/og.png" />
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link rel="stylesheet" href="{FONTS}" />
  <link rel="stylesheet" href="/styles.css" />
  <script type="application/ld+json">
  {{"@context":"https://schema.org","@type":"SoftwareApplication","name":"Kite","applicationCategory":"AccessibilityApplication","operatingSystem":"macOS","description":"A guide dog that flies, for blind and low-vision pedestrians: a drone and a chest camera watch the street, and AirPods give directional beeps and a calm voice.","url":"https://trykite.pages.dev/"}}
  </script>
</head>
<body>
'''

def nav(active):
    items = [('/', 'Hear it'), ('/how', 'How it works'), ('/judges', 'For judges')]
    lis = ''.join(f'<li><a href="{h}"{" aria-current=\"page\"" if h == active else ""}>{t}</a></li>' for h, t in items)
    return f'''  <a class="skip" href="#main">Skip to content</a>
  <header class="nav-wrap">
    <nav class="nav" aria-label="Main">
      <a class="brand" href="/" aria-label="Kite home"><img src="/assets/kite-mark.svg" alt="" width="30" height="30" /><span>kite</span></a>
      <ul class="links">{lis}</ul>
      <a class="badge" href="https://github.com/KingArthur207/kite" target="_blank" rel="noopener"><svg viewBox="0 0 16 16" width="15" height="15" aria-hidden="true"><path fill="currentColor" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>Source <span aria-hidden="true">↗</span></a>
    </nav>
  </header>
'''

FOOT = '''  <footer class="footer">
    <div class="foot-inner">
      <a class="brand" href="/"><img src="/assets/kite-mark.svg" alt="" width="26" height="26" /><span>kite</span></a>
      <p>Built in 36 hours at Cal Hacks 13.0, San Francisco.</p>
      <p class="mono fine">MIT licensed · <a href="/">Hear it</a> · <a href="/how">How it works</a> · <a href="/judges">For judges</a></p>
    </div>
  </footer>
  <script src="/app.js" defer></script>
</body>
</html>
'''

for name in ['index', 'how', 'judges']:
    body = (SRC / f'{name}.body.html').read_text()
    meta = {'index': ('Kite · a guide dog that flies', "Kite is a guide dog that flies, for blind and low-vision people who walk city streets on their own: a drone and a chest camera watch the street, and AirPods tell them what's coming before they reach it.", '/', '/'),
            'how': ('How Kite works · sound, sight and voice', 'How Kite turns a drone and a chest camera into sound: directional beeps that speed up as things get closer, a beacon toward your next turn, and a voice you can just talk to.', '/how', '/how'),
            'judges': ('Kite for judges · every track, criterion by criterion', 'How Kite meets each Cal Hacks 13.0 prize and sponsor track it enters, answered against each track\'s own judging criteria.', '/judges', '/judges')}[name]
    (ROOT / f'{name}.html').write_text(head(meta[0], meta[1], meta[2]) + nav(meta[3]) + body + FOOT)
print('built')
