# Kite sim

A browser street corner with an arrow-key walker and two virtual cameras. A drone camera sits 0.3 m above the walker's head and 1 m behind; a chest camera sits at 1.2 m. Both render 960x720 with the Tello's vertical FOV, `2*atan(360/FX)`. Frames go to perception through a local relay. Walker position and status go to the earpiece.

## World

The world is Moss Street in SoMa, San Francisco, running from Howard St at its NW end, where the demo crossing is, to Folsom. It is built from OpenStreetMap: building footprints with textured facades, roads at estimated widths, sidewalks raised 15 cm, the zebra across Moss at Howard, and Howard's cycle track (see OSM look below). Building faces and curbs block the walker, and each marked crossing has a gap in the curb. With a Google Maps key the sim draws it from Google Photorealistic 3D Tiles instead, geofenced to this one street (`?world=tiles`; see below). `?world=moss` forces the OSM world everywhere, and `?world=placeholder` loads the old box world. Locally, a Blender-built street with Street View facades replaces the OSM look when its file exists (`?world=real`, see Real facades below). Without Blender, `?world=photo` puts the Street View walls straight onto the OSM buildings (see Photo facades below).

- `scripts/osm_corner.py` fetches OSM and writes `public/osm/moss.json` in metres (x east, y north), with the origin at the Moss zebra, OSM node 12144593726. Run it again to refresh. `python scripts/osm_corner.py raw.json` reuses a saved Overpass response.
- `moss.json` also has `lanes`: car lane polylines in travel order with `speed_mps`. Moss flows SE (Howard to Folsom) and Howard flows SW. `src/osmWorld.ts` exports them as `lanes`, and the traffic drives on them (see Cars and people).
- The demo walk is `route` in `moss.json`, and the same points as lat/lng are in `public/osm/route.json`. It starts on the NE sidewalk of Moss 70 m before the zebra, facing NW, turns left across the zebra, and ends 20 m SW along Howard's south sidewalk, 98 m in all. The sim draws it faintly on screen but never in the camera frames.
- The numbers are under `measurements` in `moss.json`. OSM has no width tag for Moss, so the 6.1 m roadway is an assumption. Tape-measure it on site.
- `moss.json` keeps each building's OSM tags that matter for its look (`building`, `building:levels`, `height`, `building:material`, `building:colour`, `roof:*`, `shop`, `amenity`, `addr:housenumber`) and a `storefront` flag (a shop or amenity tag, or a shop/amenity node inside the footprint). `furniture` lists OSM street furniture nodes in the box: 3 hydrants, 2 signals at 7th and Howard, the stop sign at Moss and Folsom, 2 SFMTA pay stations, 2 power poles and 8 lowered kerbs with tactile paving. OSM has no trees, lamps, bins or meters here.

### OSM look

`src/osmLook.ts` (materials, sky, sun, renderer settings), `src/osmFacades.ts` (buildings) and `src/osmStreet.ts` (streets and furniture) turn `moss.json` into something that reads as a SoMa alley. This is also what the detector's capture cameras see in `?world=moss` and `?world=tiles&detector=osm`. Screenshots from the capture renderer: `docs/osm_facades_spawn.jpg` (spawn, facing NW up Moss), `docs/osm_facades_corner.jpg` and `docs/osm_facades_corner_chest.jpg` (the corner, facing the zebra), `docs/osm_facades_howard.jpg` (Howard's south sidewalk, facing SW).

- Textures are CC0 from Poly Haven, 9.8 MB in `public/textures/` (list in `public/textures/CREDITS.md`). Every surface has UVs in metres, so bricks, siding and sidewalk slabs are real size.
- Each building gets a facade material from `building:material` if OSM has one, otherwise from its `building` type and a random pick seeded by its OSM id: painted stucco or plaster, wood siding, brick, concrete or corrugated metal, tinted from an SF palette (or `building:colour`). No building here has a material or colour tag, so every look is the seeded pick.
- Storeys come from `building:levels`, else height / 3.2 m. Every storey gets a row of windows with frames, sills and a meeting rail, and dark glass that reflects the sky. Walls shared with a neighbour stay blank. Street fronts (the wall's normal reaches a roadway within 20 m) get a ground floor: shop glass and a sign band for storefronts, a garage door and a front door per 7.6 m lot on residential fronts, a roll-up door on industrial ones. About two thirds of residential lots with 2+ storeys get an angled SF bay window from the second floor up. A cornice runs along every roofline. Windows and doors are instanced, about 3,500 of them in 14 draw calls.
- Streets: asphalt, lane lines from the OSM lanes (white dashes 3 m on, 9 m off between same-direction lanes, double yellow between opposite ones), zebra stripes, and a stop line before each marked crossing on a one-way road where it doesn't fall inside the junction. Sidewalks have concrete slabs on 15 cm curbs, an apron ramps from the curb into the gutter in front of every garage, and the OSM lowered kerbs get yellow tactile pads.
- Furniture: the OSM nodes above, built from primitives. Not in OSM, added by us: wooden utility poles every 32 m along Moss's SW curb, each with a cobra-head light, and three overhead wires; steel streetlights every 36 m and street trees every 14 m along both Howard curbs. None of it collides; the walker still collides with building faces and curbs only.
- Light: the Poly Haven `urban_street_01` HDRI lights everything and shows in the glass, a partly cloudy sky is the background, and a sun from the SW casts shadows over a 160 m square around the route. A light haze fades the far edge of the model. ACES tone mapping and sRGB output are set on both the screen renderer and the capture renderer through `world.tune()`, so the screen and the frames match. The tiles world inherits the same renderer settings and sky.
- Speed: 60 fps (vsync cap) in headless Chromium with Metal on an M-series Mac, capture renderer running.

YOLO (yolo11n, `python sim/yolo_check.py --url 'http://localhost:5699/?world=moss'`, the walk described under Cars and people, both runs back to back on Oct 24 with the same script):

| Capture frames | Drone car | Drone person | Chest car | Chest person | Car 10 to 20 m | Person 10 to 20 m | Person over 20 m |
|---|---|---|---|---|---|---|---|
| flat-colour OSM boxes (main before this) | 55% | 59% | 63% | 59% | 97% | 88% | 46% |
| textured OSM facades | 67% | 51% | 72% | 54% | 100% | 85% | 36% |

Cars are found more often. People past 20 m are found less: in front of textured walls with garage doors, windows and poles they stand out less than against a flat pastel box. Close in they are about the same (95% under 10 m, 90% before). The traffic differs between runs, so a few points either way is noise.

#### Mapillary facades (not done)

The plan was to texture each street-facing wall with a rectified Mapillary photo (CC BY-SA 4.0). There was no `MAPILLARY_TOKEN` in `.env`, so only the geometry exists: `scripts/facade_rectify.py` projects a wall into a Mapillary camera (position, compass angle, focal from `camera_parameters`) and warps it into a texture with a homography. `uv run --no-project --with numpy python scripts/facade_rectify.py` runs its self-check, which renders a synthetic wall through a known camera and rectifies it back. The docstring has the Graph API query and the image selection rule for when a token exists.

### Photo facades (`?world=photo`, local only)

`?world=photo` is the default when any of `public/local/facades_ne/`, `facades_sw/` or `facades_howard/` has a `manifest.json`. It's the OSM world (`src/photoWorld.ts`) with each wall listed in a manifest drawn as one quad carrying its Street View photo. The other walls keep the procedural `osmFacades` look, and streets, curbs and furniture still come from `osmStreet`. Collision, lanes, route, spawn and traffic don't change. The screen renderer and the capture renderer both draw it. It was built for the demo video's street-level shots, so it only has to look right along the route.

- Manifest: a JSON list (or `{"walls": [...]}`) of `building_id`, `wall_index` (the OSM footprint edge, `pts[i]` to `pts[i+1]`), `corners_local_m` (4 points in `moss.json` metres, bottom-left as seen from the street first, `[x, y, z]` or `[x, y]`), `width_m`, `height_m`, `texture` (a file in the same folder) and `score`. When two folders cover the same wall, the higher score wins. If `wall_index` doesn't sit under the corners, the sim uses the closest edge of that building and logs it. The image's left edge goes on the bottom-left corner and its top edge at the top.
- The photo darkens toward the bottom over its lowest 0.6 m so it meets the sidewalk like a contact shadow. A faint plaster normal map and roughness 0.92 keep it from looking flat and shiny.
- L toggles labels (`building_id/wall_index`) on every footprint edge of 2 m or more, on screen only: green means it has a photo and white means it's procedural. Use it to check that a photo landed on the right wall.
- The console logs how many walls are photo and how many are procedural, and so does `window.__kitePhotoStats`. `window.__kitePhoto` reads `ready` once every photo has loaded.
- The textures sit in the gitignored `public/local/` of whichever checkout runs vite. The facade agents write them into the main checkout, so run vite there, or link the folders into a worktree's `sim/public/local/` (vite lists public files at startup, so restart it after a facade agent adds files there). Screenshots go in `public/local/compare_photo/`. Never commit or publish them.

### Real facades (`?world=real`, local only)

`?world=real` is the default when `public/local/moss_real.glb` exists and no photo manifest does. It is Moss St built in Blender, with each street-facing wall textured from Google Street View (our permission covers this one street, see `docs/google-permission.md`). The GLB is only the look: collisions, lanes, the route, the spawn and the traffic still come from `moss.json`, and both the screen and capture renderers draw it. Without the file the sim logs a warning and shows the textured OSM street. `?world=moss` forces the OSM street.

Street View-derived pixels are never committed or published. The textures, the GLB and any screenshots of it live in `public/local/`, which is gitignored. `npm run build` copies `public/` into `dist/`, so don't publish a build made on a machine that has `public/local/`.

To rebuild (Moss and the Howard corner: 27 Google panoramas, about 90 billed Static images on the first run):

```
set -a; . ./.env; set +a           # from the repo root: GOOGLE_MAPS_API_KEY (in a worktree, source the main checkout's)
uv run --no-project --with numpy --with opencv-python-headless python scripts/streetview_facades.py
/Applications/Blender.app/Contents/MacOS/Blender -b -P sim/blender/build_real_moss.py -- --export
rm -rf /tmp/kite_sv                # raw Street View responses cached for reruns
```

- `scripts/streetview_facades.py` picks the OSM footprint edges that face Moss (Folsom to Howard) or Howard within the corner, bay faces included (72 walls). It asks the metadata endpoint for Google panoramas every 4 m along those roads, inside the `kite/sources/streetview.py` SCOPE, then fetches one 640x640 view per panorama per group of walls within 15 m, aimed square at them and pitched up to mid-wall. Each wall texel (40 px/m) is projected into every view with `scripts/facade_rectify.py` (camera 2.5 m up at the panorama, now with pitch) and blended by how square and close each view is. Views where another footprint is in the way are skipped. It then finds window and door rectangles in each texture (closed Canny contours) and writes `public/local/facades/*.jpg` plus `facades.json`. `--openings-only` redoes the detection without fetching.
- `sim/blender/build_real_moss.py` extrudes every footprint and maps the photos onto the street walls (u along the wall, v up). Detected windows are recessed 12 cm with glossier glass, and doors and garages 25 to 30 cm, with a step in front of each door. Each photographed wall gets a cornice in the colour of the top of its photo. Side and back walls of photographed buildings are plain stucco in that colour. Other buildings get the Poly Haven materials and a plain window grid. The street side has asphalt, 15 cm sidewalks and curbs, zebras, lane lines, the OSM furniture, wooden poles with three wires along Moss, and steel streetlights and simple trees along Howard. It also runs inside a live Blender (`exec(open(...).read(), {"KITE_ROOT": "<repo>", "__name__": "x"})`, then `build()` and `export(*build())`), which adds cameras at the drone spawn, chest corner and Howard viewpoints.
- Oct 24 run: 72 of 72 walls textured, 18 MB GLB, 60 fps in headless Chromium with Metal (vsync cap) with the capture renderer running. YOLO (yolo11n, same script and session, traffic differs between runs): `?world=real` drone car 71%, person 39%, chest car 74%, person 36%; `?world=moss` drone car 68%, person 26%, chest car 69%, person 30%. Person recall was lower for both worlds in this session than in the table above, so compare within a session.
- Still off: parked cars, street trees and wires are baked into the lower walls; close walls are soft (about 50 px/m at the source); OSM heights don't always match the real rooflines, so some walls show sky at the top; window detection finds about half the windows and some false ones on murals and trees; the Howard trees and streetlights are placed every 18 m and 36 m, not where the real ones stand; buildings outside the corner have no photos.

### Google 3D tiles (`?world=tiles`)

`src/tilesWorld.ts` loads Google Photorealistic 3D Tiles with the `3d-tiles-renderer` package. Collisions, lanes, the route, the spawn and the traffic still come from `moss.json`.

Key: the sim reads `VITE_GOOGLE_MAPS_API_KEY` from `sim/.env.local`, which is gitignored. Write it from the repo-root `.env` (`GOOGLE_MAPS_API_KEY=...`, with the Map Tiles API enabled):

```
sh sim/tiles_env.sh                       # or: sh sim/tiles_env.sh /path/to/.env
```

In a git worktree with no `.env` of its own, the script reads the main checkout's. It never prints the key. Without a key the sim falls back to `?world=moss` and logs a warning. Vite bakes the key into `npm run build` output (`dist/`, gitignored), so don't publish a build made with an unrestricted key.

#### Permission, geofence and the detector

Google's Map Tiles policy allows tile content for visualization only and lists object detection as a prohibited use. At Cal Hacks (Oct 24) Google's reps gave us permission to use Maps Platform, including running our detector on rendered tile frames, "only to the extent we truly need it, which is just this one street" (Moss St at Howard). The record is `docs/google-permission.md`. The sim enforces that scope in code:

- Geofence (`src/fence.ts`): a box along Moss from Folsom to Howard plus the Howard corner, built from Moss's lane ends and the demo route in `moss.json` with a 30 m margin. It is about 260 m along Moss and 140 m across. A `LoadRegionPlugin` mask keeps the renderer from requesting tiles whose bounds miss the box, with sibling and ancestor preloading turned off because those ignored the mask. Every tile download is also checked against the box, and a miss counts in `tilesStats().fenceViolations` and logs an error. The tile shader discards pixels outside the box, the walker is clamped 10 m inside it, and the debug top view sits at 120 m so its footprint stays inside (`docs/hybrid_debug_topdown.jpg`).
- Detector frames: by default the hidden 960x720 capture renderer, which sends the drone and chest frames to `relay.py`, draws the tiles (`docs/hybrid_capture_tiles_drone.jpg`, `docs/hybrid_capture_tiles_chest.jpg`).
- `?detector=osm` is the fallback if that permission doesn't hold. Tiles show on screen only and the capture frames show the OSM Moss meshes with the same cars and people (`docs/hybrid_capture_osm_drone.jpg`, `docs/hybrid_capture_osm_chest.jpg`). `main.ts` calls `view("capture")` before each capture render and `view("screen")` before the screen render. Each tile mesh's `onBeforeRender` counts any draw by the capture renderer in `tilesStats().captureLeaks`.

`sim/yolo_check.py` fails if either counter is above zero.

#### What's on screen

The big view is a chase camera 4 m behind and 3 m above the walker, looking 10 m ahead (`?chase=behind,up,ahead` tunes it). At drone height Moss's narrow alley is a wall of melted photogrammetry. A little higher and further back it reads as a street. The chase camera is never a capture camera. The chest view stays in the inset. Screenshots: `docs/hybrid_moss_spawn.jpg` (spawn, facing NW), `docs/hybrid_corner_zebra.jpg` (the corner, facing the zebra) and `docs/hybrid_howard.jpg` (Howard's south sidewalk at the end of the route, looking back NE).

Street trees and parked cars are solid blobs in the photogrammetry, sometimes down to the ground, and the OSM colliders let the walker through them. The tile shader discards geometry more than 0.3 m above the ground within 3 m of the rendering camera, and in a cone from the camera to the walker (2.5 m wide at the camera, 1.2 m at the walker), so the walker stays visible under trees. A gray plane 0.4 m down shows through the holes. The corner looks good. The spawn is under a tree next to a building, so the cut opens holes into the building that show the gray plane, and the end of the route on Howard is under a large tree. The photogrammetry also has its own parked cars and the odd person baked in, which YOLO sometimes detects. Actors stand on y = 0 with a blob shadow.

Placement: the tiles are reoriented to east-north-up at the Moss zebra (`ORIGIN`) and turned so x is east and -z is north, like the rest of the sim. Moss drops about 2 m from Howard to the spawn, so the sim fits a plane to the tile road under the lanes within 12 m of the route (lowest hit, so trees and photogrammetry cars don't count) and tilts the tiles 1.6° to make that plane y = 0. It refits every 2 s for the first 90 s as finer tiles arrive. The road then sits within 0.2 to 0.3 m of y = 0 (90th percentile). Horizontally, OSM building footprints overlap tile geometry taller than 4 m with an IoU of 0.84, so the two agree to about 1 m. `?debug=1` draws the OSM curbs (white) and building outlines (orange) over everything on screen, and T toggles the north-up top view.

Attribution: the Google Maps logo and the visible tiles' data credits sit at the bottom left of the screen, with a note that the city is Google's and the cars and people are the sim's (Maps Platform policy for 3D Tiles). The logo is Google's outlined asset in `public/google/`. Tiles are cached only by the browser and the renderer's in-memory LRU.

Cost, measured on this Mac: walking the whole 98 m route in real time makes one `root.json` request (one session) and 114 tile requests, with no fence violations. Before the fence it was 640 to 670. The screenshot runs, including the debug top view, made 114 to 126. Each page load starts a new session.

Speed: 60 fps (the vsync cap) in headless Chromium with Metal on an M-series Mac, with the capture renderer sending both cameras at 8 fps each.

YOLO (yolo11n, `python sim/yolo_check.py --url '<sim>/?world=...'`, same walk as below, about 650 frames per run, all three runs in one session on Oct 24). When the capture frames show tiles, the ground truth drops actors whose middle and top are both behind tile geometry from that camera, and the check reports how many.

| Capture frames | Drone car | Drone person | Chest car | Chest person | Car 10 to 20 m | Person 10 to 20 m |
|---|---|---|---|---|---|---|
| `?world=moss` (OSM everywhere) | 59% | 60% | 63% | 63% | 98% | 98% |
| `?world=tiles&detector=osm` | 55% | 56% | 65% | 60% | 99% | 89% |
| `?world=tiles` (tiles in capture) | 36% | 10% | 42% | 9% | 54% | 61% |

The `detector=osm` row matches OSM Moss within run-to-run noise (the traffic differs between runs). These person numbers are higher than the 32 to 38% measured earlier with the same script and the same sim code; the yolo11n weights were downloaded fresh for these runs, so compare rows within this table. With tiles in the capture frames the tiles hid 168 car and 506 person instances, which are left out of that row. Recall drops further because trees and photogrammetry cars partly cover the rest, and people past 20 m are found 3% of the time.

### Street View walk (`?world=streetview`)

Real imagery instead of a 3D world. `src/streetviewWalk.ts` puts a Google Street View panorama (Maps JavaScript API) full screen and walks it along the demo route on Moss St. It starts at the route start facing NW. Up and Down step to the linked panorama nearest the view heading, or behind it. At the corner Moss's panoramas don't link to Howard's, so when no link points the way you face, it takes the nearest Google panorama 8 to 22 m ahead. Left and Right turn 20°, and B, V and H work as in the 3D sim. Google's logo, copyright and terms stay on screen. Every other control is hidden, and clicking the image does nothing.

Scope: the page shows only Google's own panoramas (copyright "Google") inside the same box as `SCOPE` in `kite/sources/streetview.py`. If a panorama change lands anywhere else, the page puts the last good one back.

Earpiece: `position` with the panorama's lat/lng, the view heading and `place: "Moss St"`, plus `status`, like the 3D sim, so the dashboard map follows the walk.

Detector: the page sends `{type: "sv_view", pano, lat, lng, heading}` to the relay and no pixels. `sim/streetview_bridge.py` gets the panorama's metadata (free), checks that it is in `SCOPE` and is Google's, then fetches two Static API images through `kite/sources/streetview.py`. The drone image is at pitch 0 and the sim cameras' 51.3° FOV, and the chest image is at pitch -10 and 69°. It publishes both on the relay as the drone and chest frames at 5 fps, with a `pose` per camera that carries `fx` and `pitch`. The stand-in detector reads them like any sim frames. Images stay in memory (the last 32 views), and nothing is written to disk.

```
scripts/demo.sh streetview          # earpiece, relay, bridge, sim, dashboard, detector; opens the walk in a new Chrome window
```

Measured on Oct 24, walking the whole route in headless Chromium (10 panoramas, from the spawn to Howard's south side): the page loaded the Maps JavaScript API once and created one panorama, and made 14 `StreetViewService` lookups. The bridge made 10 metadata requests and 30 Static images (2 per view, 15 views counting the turns). Screenshots without imagery: `docs/streetview_walk_black_start.jpg` and `docs/streetview_walk_black_corner.jpg` (black screen and HUD).

## Run

```
cd sim && npm install && npm run dev        # http://localhost:5173
python sim/relay.py                         # frames, ws://127.0.0.1:5020 (needs websockets)
cd earpiece && python -m earpiece run       # earpiece, ws://127.0.0.1:5011
python sim/test_relay.py                    # relay self-check
```

Start them in any order. The page reconnects to the relay and the earpiece every 2 s, and the HUD shows which ones are connected.

## Keys

| Key | Does |
|---|---|
| Up / Down | walk forward / back at 1.2 m/s |
| Left / Right | turn |
| B | black screen (HUD stays) |
| V | reveal the view |
| L | `?world=photo`: label walls `building_id/wall_index` (green = photo) |
| H | DEBUG: send one fake `crowd` hazard to the earpiece, only to prove sim to earpiece to beep for the skeleton gate |

The HUD counts collisions with buildings, cars, pedestrians and the bin.

## Metrics mode

`?metrics=1&mode=blind|kite|oracle&trials=20&seed=1` runs an autopilot with no keyboard input, always in the Moss world. A walker at 1.2 m/s follows the demo `route`: up Moss's NE sidewalk, left across the zebra, then along Howard's south sidewalk. On the way it meets a pedestrian crossing Moss into a doorway, a bin, a head-height sign, a car turning in from Howard's left lane through the zebra without yielding, and a pedestrian stepping out of a doorway on Howard. Their timing is seeded, so each trial is different and every run with the same seed is identical. The street's own traffic and people are not spawned in this mode. Building faces and curbs still block the walker, and touching one counts as a collision.

- `blind`: the walker never reacts. Runs time-accelerated (`speed`, default 300).
- `kite`: the walker reacts only to the earpiece on `ws://127.0.0.1:5011`. It stops 0.7 s after a `spoken` event with priority `critical` or `hazard`, a `state` with a critical sonar track, or a hazard in `dont_walk`/`wait`/`red`/`stop`. It walks on after 1.5 s with no cue and never detours. Runs in real time, since the pipeline is real time.
- `oracle`: an upper bound that reacts to the true obstacles in a 5 m corridor ahead, looking up to 3 s into the future. It is not Kite and must never be presented as Kite.

In every mode, a walker that stays blocked for 1 s or waits 8 s feels around the obstacle with a sidestep, the way a cane user would: 1 m toward the buildings on Moss's narrow sidewalk, 1.5 m elsewhere. A sidestep that runs into a wall or curb stops there. Each obstacle counts as one collision per walk at most; walls count every new contact. Results show in the HUD and in `window.kiteMetrics`, get logged as one `KITE_METRICS {...}` console line, and can be downloaded as JSON. Blind and oracle open no sockets, so they never feed a shared earpiece or relay.

Headless (starts its own vite on port 5287; pass `--url` to reuse one):

```
uv run --with playwright playwright install chromium        # first time
uv run --with playwright python scripts/sim_metrics.py --mode blind
uv run --with playwright python scripts/sim_metrics.py --mode oracle
```

Kite mode: start the relay, the detector on sim frames and the earpiece, then open `?metrics=1&mode=kite` or run `--mode kite`. That takes about 90 s a walk. Run it on a machine where nothing else is sending to the earpiece.

Results on Moss, seed 1, 20 walks: blind 61 collisions (3.05 a walk, every walk hit the bin, the sign and the car; one also hit the second pedestrian), oracle 0 collisions and 40 near misses (the bin and sign on the 1 m sidestep). Kite mode has not been run on Moss yet.

## Cars and people

`src/actors.ts` loads glTF models from `public/models/` (credits and licenses in `public/models/CREDITS.md`) and scales them to real size: cars are 4.3 to 5.3 m long, the bus 12 m, and people 1.65 to 1.82 m tall.

Moss world (`src/mossActors.ts`): cars drive the OSM `lanes`. Five cars and the bus run Howard's three SW lanes at about 10 m/s. Four cars come down Howard's left lane, turn left into Moss across the zebra at 4 m/s, and drive Moss SE at about 5 m/s; one passes the zebra every 15 to 20 s. Cars brake smoothly for slower segments, keep a 3 m gap to the car ahead, stop for anyone within 1.5 m of their path, and wait at the stop line while anyone (a pedestrian or the walker) is on or within 0.6 m of a zebra's stripes. At the end of its lane a car starts over once the start is clear. Five people walk back and forth on Howard's south sidewalk (across the zebra), the demo route (across the zebra), and both Moss sidewalks, waiting if a car is right in front of them. One more stands on Howard's south sidewalk NE of the corner. Collision boxes: a car is a chain of axis-aligned squares along its body, sized to stay inside its width at any heading; people get a 0.5 m box.

Placeholder world (`?world=placeholder`): three cars and the bus loop the main road at 8 to 11 m/s. Two cars loop the cross street at 7 m/s and wait at a stop line while someone is on the crosswalk or main-road traffic is within 18 m. Faster cars hold back behind slower ones in the same lane. Five people walk a loop on the sidewalk at 1.25 to 1.5 m/s with their walk animation, crossing the cross street on the crosswalk both ways. One more stands near the corner. Collision boxes come from each car model's bounds; people get a 0.5 m box.

YOLO detects them. `python sim/yolo_check.py` (usage at the top of the file) drives the sim in headless Chromium with a frozen clock, mocks the relay and earpiece sockets, and scores YOLO against the actors' projected boxes. In the placeholder world (`--url 'http://localhost:5199/?world=placeholder'`), on 168 frames (83 drone, 85 chest), counting instances at least 30 px tall:

| Model | Drone car | Drone person | Chest car | Chest person |
|---|---|---|---|---|
| yolo11n | 91% | 99% | 85% | 96% |
| yolo26n | 90% | 99% | 93% | 97% |

The ground truth ignores buildings, so a car hidden behind one counts as a miss and these numbers are a floor. The pickup crossing at a distance is the weakest model (about 70%). Examples: `docs/yolo_drone.jpg`, `docs/yolo_chest.jpg`, `docs/yolo_drone_road.jpg` (thick boxes are detections, thin white boxes are found ground truth, thin red ones are misses).

In the Moss world the check walks up Moss's NE sidewalk to Howard, turns to face the zebra and waits. With yolo11n on 651 frames: drone car 62%, drone person 34%, chest car 63%, chest person 37%. At 10 to 20 m it finds 98% of cars and 90% of people. The low totals are mostly occlusion the ground truth doesn't model: from Moss, the people on Howard's sidewalk and much of Howard's traffic are behind the corner buildings (the standing man 0/167, the two Howard walkers 2/231, the bus 4/100), and cars crossing right in front at the curb are cut off by the frame edge. Examples: `docs/yolo_moss_drone.jpg`, `docs/yolo_moss_chest.jpg`.

## What the sim sends

To the earpiece (`ws://127.0.0.1:5011`):
- `position` at 5 Hz: `heading` (compass degrees, 0 = north), `lat`, `lng`, `x`/`y` (meters east/north of the origin), `place` (`"Moss St at Howard St"`).
- `status` when the walker starts or stops: `state` is `walking` or `stopped`.

The sim emits no hazards from its own object list. Hazards come from the real detector running on the sim's frames (team decision).

To the relay (`ws://127.0.0.1:5020/pub`):
- Binary frames, 8 fps per camera: 1 byte camera id (0 drone, 1 chest), then JPEG bytes.
- Pose as JSON text at 5 Hz, once per camera: `{"type":"pose","camera":"drone","x":..,"y":..,"heading":..,"cam_height":..,"pitch":0,"cam_model":"tello"}`. `cam_model` is `raw` when the page has `?cam=raw`.

Subscribers on `/sub` get the same messages. On connect they first get the latest frame and pose for each camera.

## Frames in Python

```python
import sys; sys.path.insert(0, "sim")
from frames import SimFrameSource

src = SimFrameSource("drone")      # or "chest"; url defaults to ws://127.0.0.1:5020/sub
ok, bgr = src.read()               # like cv2.VideoCapture: 960x720 BGR, blocks up to 2 s for a new frame
pose = src.pose                    # latest pose for this camera, or None
```

Needs `websockets`, `numpy` and `opencv-python`.

## Tello camera model

The capture renderer draws clean, sharp, perfectly exposed frames, and a Tello doesn't. `SimFrameSource("drone")` now runs every drone frame through `tello_cam.py` before handing it over: lens distortion, motion blur from the turn rate, a soft lens, vignetting, auto-exposure that adapts over about 1.5 s, shot and read noise, a washed-out, slightly green colour response, and one JPEG pass at quality 40 for the 8x8 blockiness of the Tello's H.264. The link is modelled too: frames arrive 100 to 200 ms late, about 3% are dropped and 2% repeat the previous frame. `.pose` is not delayed, the same way the Tello's state stream runs ahead of its video. Rolling-shutter skew is in the model but off by default.

The screen view is never modelled, and neither is the chest camera (it isn't a Tello). `?cam=raw` on the sim URL turns the model off; the flag travels in the pose messages. `SimFrameSource(..., raw=True)` does the same from Python. The detector code doesn't change.

All parameters are in `public/tello_camera_model.json`. Lens distortion comes from `data/tello_camera.json` once `kite.calibrate` has written one with `dist`. Until then it uses the community Tello calibration (fx 921, k1 -0.033, k2 0.105) that circulates in Tello ArUco projects, which we haven't verified. The distortion is applied around the sim's own focal length (`config.ts` FX = 1000) so the field of view stays the same; set FX from the calibration separately. Every other number is a guess until it is fitted to real footage (below). `python sim/tello_cam.py` runs its self-check.

### YOLO with the model off and on

`yolo_check.py --cam both` scores the same grab twice, so both columns see identical scenes. Moss world, yolo11n, Oct 24, 657 frames (drone 328, chest 329), recall of visible instances:

| drone camera | model off | model on |
|---|---|---|
| car | 130/212 = 61%, mean conf 0.74 | 133/212 = 63%, mean conf 0.76 |
| person | 103/403 = 26%, mean conf 0.55 | 111/403 = 28%, mean conf 0.54 |
| car < 10 m | 11/19 | 17/19 |
| person > 20 m | 83/382 | 90/382 |

The chest camera is unmodelled, so its numbers are identical in both columns (car 63%, person 31%). Detection went up slightly instead of down. The likely reason is that the raw renders are aliased and over-sharp, and the softness and the exposure gain bring the low-poly people and cars closer to what YOLO was trained on. So the current guesses don't make the sim harder for the detector. Whether the real Tello does is the open question, and only a real recording can answer it.

## Sim-to-real gap (`scripts/sim_real_gap.py`)

```
# raw sim drone frames (the tool applies the camera model itself)
uv run --no-project --with playwright --with ultralytics --with opencv-python-headless \
    python sim/yolo_check.py --url "http://localhost:5199/?cam=raw" --frames out/sim_raw --out out
# real vs sim raw vs sim + model: image stats and YOLO confidence by estimated distance
uv run --no-project --with ultralytics --with opencv-python-headless \
    python scripts/sim_real_gap.py --real recordings/<take> --sim out/sim_raw
# tune the model JSON to the real stats
... python scripts/sim_real_gap.py --real recordings/<take> --sim out/sim_raw --fit
```

It reports sharpness (Laplacian variance), noise (Immerkaer estimate), brightness, contrast, saturation, R/G and B/G, vignetting (corner/centre), blockiness (step across 8 px block edges / step elsewhere) and the colour-histogram distance to the real frames. It also runs YOLO person/car on both sides, binned by a distance estimated from box height, since real footage has no ground truth. `--fit` runs coordinate descent over 11 parameters (exposure target, contrast, black level, saturation, R and B gains, softness, read and shot noise, JPEG quality, vignette) and writes the JSON (`--fit-out` to write somewhere else).

No Tello recording exists yet, so it was tried on a stock clip standing in for real footage (Mixkit 134, "A man being followed walking in a city", 1280x720 cropped to 960x720, in the gitignored `clips/`). Medians over 40 frames each:

| stat | stock clip | sim raw | sim + model | sim + fitted |
|---|---|---|---|---|
| brightness | 75.7 | 84.9 | 114.2 | 102.9 |
| sharpness | 17.4 | 689.9 | 96.2 | 58.6 |
| noise | 0.43 | 1.47 | 0.45 | 0.27 |
| blockiness | 1.08 | 1.22 | 1.88 | 1.52 |
| saturation | 95.5 | 65.0 | 37.1 | 75.0 |
| R/G | 0.77 | 1.07 | 1.00 | 0.93 |
| colour hist distance | 0 | 0.36 | 0.46 | 0.33 |
| gap score (fit objective) | 0 | 395 | 442 | 156 |

The stock clip is a warm, saturated, smooth H.264 phone video and doesn't look like a Tello. The hand-set model moves the sim away from it on colour, and the fit pulls it back. That shows the tool works. It doesn't show the model is right. The stats also depend on the scene, so the fit means the most on Tello footage of a street like Moss. **Re-run `sim_real_gap.py --fit` on the first Tello recording** (a `kite.record` take works as `--real` directly), commit the fitted `public/tello_camera_model.json`, then re-run `yolo_check.py --cam both`.

## Scene descriptions (Claude)

`sim/scene_describe.py` sends one frame to Claude every 6 s and forwards the reply to the earpiece as a `scene` event, with source `scene-chest` or `scene-drone`. The earpiece doesn't speak scenes. When the wearer asks "Kite, what's in front of me?", the brain answers from the latest scene under 90 s old. Run it from the repo root:

```
uv run --no-project --with anthropic --with opencv-python-headless --with websockets \
  --with python-dotenv --with numpy python sim/scene_describe.py --source sim-drone
```

`--source` takes `sim-chest`, `sim-drone`, a camera index (`0`), a video file, or a `kite.record` folder. Other flags are `--interval 6`, `--model` (default `claude-haiku-5-5`), `--camera chest|drone` to rename the source, and `--host`, which defaults to `KITE_EARPIECE_HOST`. `ANTHROPIC_API_KEY` is read from the repo-root `.env`.

Each request carries one JPEG, 768 px wide at quality 70, with a 4 s timeout and no retries. If the previous call is still running, the next one is skipped. The system prompt asks for one or two sentences about the path, people, vehicles, obstacles, crossings and readable signs. It forbids safety judgments like "safe to cross", guesses beyond the image, and descriptions of faces or identity. Frames stay in memory and are never written to disk.

The script also listens on the earpiece WebSocket (`ws://HOST:5011`). A `heard` event that matches the brain's "what's around/ahead/in front" intent, or says "describe" or "see", starts a description right away. The brain answers that question at once, offline, from the scene it already has, so the fresh description is ready for the next question. The 6 s interval keeps the first answer at most about 6 s old.

`python sim/test_scene_describe.py` runs the self-check with a fake client and no network.

## Stand-in detector (until streetseg lands)

`sim/standin_detect.py` runs YOLO11n on the chest and drone frames and sends the events perception owes the earpiece (spec 5.1), so the sim to AirPods path works before Rex's `streetseg` exists. It reads frames through `SimFrameSource`, or any video file with `--source`. Replace it with streetseg; nothing downstream changes.

- Chest camera: `obstacles` for every frame (kind, bearing from the box centre, distance from the box bottom on flat ground at the pose's camera height, falling back to a typical object height), plus `heartbeat` when nothing is in view.
- Drone camera: a `hazard` for `crowd` (3 or more people 1 to 15 m ahead of the walker) or `vehicle` (a car within 12 m and 3 m of the walker's path). A person cut off by the bottom edge is the walker and never counts. A hazard, or a new direction for one, must hold 3 frames before it is spoken. It is refreshed every 2 s with the same text and expires on its 4 s ttl, with no clear message. A `scene` count goes out every 10 s.

From the repo root:

```
uv run --no-project --with ultralytics --with opencv-python-headless --with lap --with websockets --with python-dotenv --with numpy --with scipy python sim/standin_detect.py
```

That runs on sim chest + drone frames and sends to `KITE_EARPIECE_HOST` (`.env`, default 127.0.0.1). Append flags to the same command:

```
--download                                    # once: fetches weights/yolo11n.pt (gitignored); never fetched at runtime
--source clips/mixkit-134.mp4 --camera drone --jsonl out/clip.jsonl --overlay out/clip.mp4
```

`python3 sim/test_standin_geometry.py` checks the geometry and the hazard gate without a model. `--jsonl` writes the same events with a `t` field for `scripts/replay.py`. `--overlay` writes an mp4 with boxes and bearing/distance labels. Other flags: `--fx` (default 1000), `--cam-height`, `--device`, `--fast` (don't pace a video to real time), `--no-send`, `--max-seconds`. It uses `mps` when available.

On Moss, with the walker stopped 4 m short of the corner facing NW, every car that turned in from Howard during a 60 s wait drew a drone `hazard` as it came down Moss toward the walker ("Car slightly left, 7 meters" when it was 8 to 9 m away at about -25 degrees). Two gaps showed up. On the walk up Moss, oncoming Moss cars also drew "Car slightly left, 6 meters": the 3 m lane test couldn't tell that the walker is on the sidewalk of a 6 m alley. Now a car counts only within 1.5 m of the walker's path, or when its sideways offset (tracked per car over the last second) is shrinking fast enough to reach the path within 3 s, so a car crossing ahead still draws a hazard and one driving past on the road doesn't. Not re-measured on Moss yet. Once the walker turns to face the zebra (SW), a turning car comes from behind-right and is only in frame when it is about 3 m away and cut off by the edge, and the stand-in gave no hazard for it.

On an M-series Mac, YOLO11n with tracking takes about 25 ms a frame on `mps` with the machine idle, and 75 to 90 ms with headless Chrome rendering the sim alongside. The sim's placeholder boxes and capsules are rarely detected; the glTF cars and people are.

Test clips come from Mixkit (free license, mixkit.co): "A man being followed walking in a city" (`https://assets.mixkit.co/videos/134/134-720.mp4`) is close to the drone's view, and "Times Square during a sunny day" (`.../4442/4442-720.mp4`) has lots of cars. Download them into the gitignored `clips/`.

## Live walk (spec section 4)

The judge wears a phone on their chest as the camera, headphones and a sleep mask, and walks 3 to 4 m. A teammate steps in front. Kite says "Stop. Person ahead." Then the reveal window replays what the chest camera saw, with boxes and the STOP banner.

The live path is the stand-in detector with a camera index as `--source`. `sim/live_walk.py` holds the chest-only parts, and they run on sim chest frames too:

- Distances are a median over 3 frames, and closing speed is a least-squares fit over the last second.
- Anything inside 1.6 m (`--stop-m`) and within 0.7 m of the walker's path is a stop. So is anything under 1 s from contact. A stop goes out with a closing speed that puts the earpiece's time to contact at 0.8 s. Its sonar then goes critical and says "Stop. Person ahead." at about 1.5 m instead of its own 0.75 m.
- When a box is cut off by the bottom or the top edge (within 16 px; a person close to a chest camera), the distance comes from the top of the head, assumed 1.7 m (`--person-height`). When both the head and the feet are out of frame, no row can say how close they are, so the box width does, taking a person as 0.45 m wide. At the live settings (960x540, 15 degrees down) that happens inside about 1.65 m; before this, every distance in there read 1.65 m and "Stop" never fired.
- `--pitch` is the camera's downward tilt in degrees. It goes into the ground-plane distance, and a phone clipped to a shirt usually points 10 to 20 degrees down.
- `--depth` adds Depth Anything V2 Small (metric, indoor) at 336x252 as a fallback for things YOLO has no class for, such as walls, doors and boxes. It reports something closer than 2.5 m that fills the centre of the frame and isn't floor. It takes about 18 ms a frame on `mps`.
- `--show` opens the reveal window: live view, labels (kind, bearing, distance, closing speed), and a red banner such as `STOP - person 1.4 m` that stays up 1.5 s. Press `r` to replay the last 20 s, `l` for live, `q` to quit. Heads are pixelated (`--no-blur` turns that off). The 20 s buffer is JPEG bytes in RAM and is never written to disk, and `--overlay` is refused for a live camera (spec 5.3).

Commands, from the repo root. `--show` needs `opencv-python` (with GUI), not the headless build:

```
RUN="uv run --no-project --with ultralytics --with opencv-python --with lap --with websockets --with python-dotenv --with numpy --with scipy --with transformers --with av"
$RUN python sim/standin_detect.py --download --depth          # once, with internet: YOLO weights and the depth model
$RUN python sim/standin_detect.py --list-cameras              # index, name and frame size of each camera
$RUN python sim/standin_detect.py --source webcam:iPhone --show --cam-height 1.2 --pitch 15 --depth
```

Frames are resized to 960 wide. A live camera defaults to `--fx 700`, which fits an iPhone at 1x or a MacBook camera at 960 wide. Calibrate on the day: stand someone on a tape mark 2 m from the judge and change `--pitch` (then `--fx`) until the label reads 2.0 m.

To use the iPhone as the camera, sign it into the same Apple ID as the Mac, with Wi-Fi and Bluetooth on, and lock it in landscape near the Mac. Continuity Camera then shows it in `--list-cameras`, usually as index 1 when the built-in camera is 0. `--source` takes the index or any part of the name (`webcam:iPhone`); both go through `kite.sources.open_source()`, which can also open a Continuity Camera that OpenCV doesn't list, by name through PyAV. Turn off Center Stage in Control Center > Video Effects so the field of view doesn't change mid-walk. To check you have the right camera, run with `--show` and cover the phone's lens.

macOS asks for Camera access the first time; grant it to the terminal app you run from. If `--list-cameras` finds nothing, check System Settings > Privacy & Security > Camera.

Measured on an M5 MacBook with `--show` and nothing else running, on a stock clip: about 22 fps end to end with YOLO alone, and about 15 fps with `--depth`. The STOP line printed at 1.3 m in `clips/approach.mp4` (below), and the earpiece said "Stop. Person ahead." at critical priority.

Table checklist:

- [ ] Earpiece running with the headphones as output: `cd earpiece && .venv/bin/python -m earpiece run --output "<device name>"` (default is any AirPods)
- [ ] `--list-cameras`, then the detector running with `--show` on the phone's index
- [ ] Phone clipped flat to the judge's chest, about 1.2 m up, landscape, lens clear of jacket and lanyard; `--cam-height` and `--pitch` match
- [ ] Calibrated on the 2 m tape mark
- [ ] 4 m of clear floor taped out, with nothing at head height
- [ ] One spotter walks beside the judge, within arm's reach, the whole time
- [ ] One teammate steps in at about 2 m and stays put until Kite says "Stop"
- [ ] Sleep mask on after the headphones are in; take it off for the reveal
- [ ] Reveal: press `r` on the laptop, and turn the screen to the judge

Test clip for the live path without a second person: Pexels "Woman walking inside a building with camera" (`https://www.pexels.com/download/video/2795407/`, free under the Pexels license), with its last 9 s reversed so she walks toward a waist-height camera:

```
ffmpeg -ss 9 -i clips/pexels-2795407.mp4 -vf "fps=30,scale=960:-2,reverse" -an clips/approach.mp4
$RUN python sim/standin_detect.py --source clips/approach.mp4 --camera chest --show
```

### Live walk bench (40 walks, no camera)

`scripts/live_walk_bench.py` replays the live walk 40 times without a judge or a phone. Each run plays a clip through the stand-in detector (chest settings, real-time pacing that drops frames when it falls behind, as a camera would) into a headless earpiece on its own ports (UDP 6210, WS 6211, no Cartesia key), then reads what the earpiece said over the WebSocket and scores it against the clip's truth.

```
RUN="uv run --no-project --with ultralytics --with opencv-python-headless --with lap --with websockets --with python-dotenv --with numpy --with scipy"
$RUN python scripts/live_walk_bench.py prepare                 # once, with internet: clips/bench/ (gitignored)
$RUN python scripts/live_walk_bench.py run                     # 40 runs, about 20 min; table, JSON in out/bench/
$RUN python scripts/live_walk_bench.py run --runs 6 --only syn_stepin,syn_cross   # a quick check
$RUN python scripts/live_walk_bench.py rescore out/bench/<report>.json            # new scoring rules, same runs
python3 scripts/live_walk_bench.py selftest                    # the scoring rules
```

It needs `earpiece/.venv` (`scripts/check.sh earpiece`) and `weights/yolo11n.pt`. It refuses to start if 6210 or 6211 is taken and only ever stops the processes it started. `--detector <path>` runs another copy of the detector (a `sim/` folder from another branch, copied to the repo root) for a before/after on the same seed.

The clips:

- Stock, from Pexels and Mixkit (free licenses, downloaded by `prepare`, never committed): `approach` (Pexels 2795407 reversed, then forward again, so she walks up and away), `corridor` (Mixkit 33396, the same up-and-back), `group` (Mixkit 23809, four people walking toward the camera), and two empty corridors that must stay silent (`hotel_empty`, `datacenter_empty`). Their truth is the detector's own geometry run on every frame and median-smoothed, so they test flicker, tracking, crowds and timing, not range.
- Synthetic, with exact distances: a person cut out of a stock frame and drawn into the empty hotel corridor along a scripted path with the live geometry (960x540, fx 700, 1.2 m up, 15 degrees down). `syn_walkup` walks up to 1.0 to 1.3 m, waits 2 s and steps aside. `syn_stepin` is the demo: the teammate steps in from the side 2.4 m ahead. `syn_slow` and `syn_fast` walk at 0.6 and 1.8 m/s, `syn_occluded` has something dark over the lower left of the lens, `syn_table` hides the legs behind a low table, `syn_two` adds a bystander off the path, `syn_cross` crosses 2.3 to 2.8 m ahead (must not stop), and `syn_empty` is the corridor alone.

Every run gets new jitter from the seed: speed, stop point, sideways offset, brightness and a chest bob of up to 1.2 degrees.

Scoring: a stop is expected when the nearest person within 0.7 m of the path comes inside 1.5 m. Latency is from that frame to the start of "Stop. Person ahead." (negative means it started before 1.5 m). A stop while the truth is beyond 2.2 m, or with nobody on the path, is a false stop. A stop more than 1 s after the 1.5 m frame is late and counts as missed. "Path is clear" must come after the person is beyond 2.5 m or off the path and before the clip ends; one that comes while they are still there is an early continue.

Before and after the fixes in this branch, same seed, back to back on an M5 MacBook (detector p50 34 and 26 fps):

| | main | this branch |
|---|---|---|
| passed | 16 / 40 | 37 / 40 |
| missed or late stops | 24 | 3 (all `syn_table`) |
| false stops | 0 | 0 |
| stop latency p50 / p95 | -0.43 / +3.26 s | -0.01 / +0.32 s |
| missed continues | 3 | 0 |
| early continues | 3 | 0 |

What it found:

1. At the live settings every synthetic walk-up missed the stop (21 of 21). With the phone 15 degrees down on a 960x540 frame, the head leaves the top of the frame at about 4.7 m and the feet leave the bottom at 1.65 m. Inside that, the top row, the bottom row and the box height all bound the distance from above, so every distance read 1.65 m, which never got under the 1.6 m stop line, and the closing speed fell to zero. A box that loses both head and feet now takes its distance from its width as well.
2. A box only counted as cut off within 8 px of the bottom edge. YOLO boxes often stop a few pixels short, so in `approach` the same person read 3.2 m on one frame and 1.3 m on the next, the stop came 3 s late, and "Path is clear" came while she was still standing there. Cut off now means within 16 px, and a head out of the top of the frame bounds the distance too.
3. Known limit: legs hidden behind something low (`syn_table`) make the box bottom look like feet farther away, so the person reads too far and Kite never stops. `--depth` should see the table itself; nothing in the YOLO path can.

Bench fixes on the way: video files are now paced like a camera (skip ahead when behind, so a slow laptop can't fall further and further behind the clip), the detector warms YOLO up before frame 0 so the first frame doesn't stall 4 s, and it prints `[standin] t0_wall=` so the bench can line frames up with speech.

## Files and placeholders

- `src/tilesWorld.ts` is the Google 3D tiles world over the Moss colliders and traffic, and `src/fence.ts` is its geofence. `?detector=osm` keeps the capture frames on OSM Moss. `?relay=` and `?ear=` override the relay and earpiece URLs (empty means offline). `window.__kiteWalker` is the walker, for scripted checks.
- `src/world.ts` is the placeholder box world, with its glTF traffic from `src/actors.ts`. `src/osmWorld.ts` is Moss St, with its traffic from `src/mossActors.ts` and its look from `src/osmLook.ts`, `src/osmFacades.ts` and `src/osmStreet.ts`. Both export `scene`, `update` and `solids`. The Moss world also exports `spawn`, `walls`, `overlay`, `place` and `tune` (see `WorldExtras`).
- `ORIGIN` in `src/config.ts` is the Moss zebra (37.778352, -122.408768) and maps to sim (0, 0), so the route and the dashboard use the same points.
- `FX` in `src/config.ts` starts at 1000. Arthur's Tello calibration replaces it.
