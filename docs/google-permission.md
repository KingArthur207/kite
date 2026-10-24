# Google Maps Platform permission

On Sat Oct 24 2026, about 3 AM PT, Google representatives at Cal Hacks 13.0 told Eddy Zhao that Kite may use Google Maps Platform (Photorealistic 3D Tiles and Street View) for this hackathon, including running our object detector on rendered frames, "only to the extent we truly need it, which is just this one street": Moss St in San Francisco, between Folsom St and Howard St, and the Moss/Howard corner.

Representative: [TODO: name and role]. Written confirmation: [TODO: email or Slack link, or "verbal only"].

How we keep to it:
- The sim requests and renders 3D Tiles only inside a geofence around that block, and keeps the walker and cameras inside it.
- `kite/sources/streetview.py` fetches only along the planned route on that block.
- The Blender street (`?world=real`) uses facade textures made from Street View images of this block. They and the exported model live only on Eddy's laptop in `sim/public/local/`, which git ignores; they are never committed, published or used to train a model, and we delete them after the hackathon. A `npm run build` made on that laptop must not be published.
- No other Google imagery is stored or cached beyond the browser's normal cache. Google's attribution is shown on screen.
- Without this permission, the sim's detector cameras render the OpenStreetMap world instead (the hybrid path).
