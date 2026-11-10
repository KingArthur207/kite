# Accessibility

Kite is built for people who are blind or have low vision, so its screens aim at **WCAG 2.2 Level AA**:
the live dashboard (`dashboard/`) and the walker's phone app (`phone-app/`).

## How it was checked (November 2026)

- **Dashboard:** axe-core 4.14 (tags `wcag2a`, `wcag2aa`, `wcag21a`, `wcag21aa`, `wcag22aa`) in headless Chrome at
  1440×900, 390×844 and 320×800, on 16 states: the analytics view (live, Feeds open, a camera feed failing, Follow,
  presentation, an empty database, offline) and the walk view (live, a moment picked, following a phone, the live
  feed open, a blocked word typed, a message sent, presentation, an empty database, offline). The data came from a
  local SpacetimeDB seeded with `data/sample_track.jsonl` and a few extra moments. axe reports no violations in any
  of them. By hand: a keyboard walk through every view, the accessibility tree (names, roles, landmarks, headings,
  live regions), contrast of every text element against its painted background, text spacing, reflow at 320 px,
  and target sizes.
- **Phone app:** read against the React Native accessibility props (one full-screen button with a label, hint and
  state; text that scales to 200% with Dynamic Type; 4.5:1 text; motion that settles and follows Reduce Motion).
  It has no web build, so it was not run through axe.

## Known limitations

- The map itself (tiles, trails, accuracy circles) is visual. Everything it shows is also in text: the Phones
  list, the hazards list with distance and direction in words, the timeline cards and the devices table.
- The timeline ribbon's dots and the map's pins are small pointer targets with no keyboard stop of their own.
  The timeline cards do the same thing and are full-size buttons (WCAG 2.5.8's equivalent-control exception).
- The camera panes are live video with no audio and no text alternative beyond their label and state.
- The phone app was not tested with VoiceOver on a device in this pass.

## Report a barrier

Open an issue on this repository with the page or screen, what you were trying to do, and the browser or device
and assistive technology you used.
