# About *Kite.*

who, why and what's real

Kite is an assistive navigation aid for blind and low-vision people who walk city streets on their own. A four-person student team built it in 36 hours at Cal Hacks 13.0 in San Francisco, in October 2026.

## What it is

A white cane reports the street one step at a time. A guide dog looks ahead, stops at curbs and refuses to step out in front of a car, but fewer than 10,000 guide dogs work in the US for about a million blind Americans. Kite tries to do a guide dog's jobs with things you can buy: a small DJI Tello drone flies about 30 cm above and behind the walker's head and watches the street ahead, a camera on the chest watches the ground at their feet, and AirPods turn both into sound. Beeps come from the direction of each obstacle and speed up as it gets closer; a calm voice says "Stop." first and only says the way is clear when plain, tested code is sure.

## What works today, and what doesn't

- **Real and running:** the earpiece (spatial beeps, the beacon toward the next turn, the speech queue, stop and wait), Cartesia voice in and out with offline fallbacks, Claude's careful judgment calls, scene descriptions and reading text aloud, Moss memory of the walker's places, walking directions, and the SpacetimeDB caregiver dashboard with two-way messages.
- **A stand-in:** obstacle detection uses an off-the-shelf detector (YOLO11n) while a dedicated street model is finished.
- **Simulated:** the 41-second walk on the home page is a simulated street rendered by the real earpiece code. Drones aren't allowed to fly at the venue, so the drone's view at our table is recorded footage.
- **Not a medical device:** Kite is a hackathon prototype. It has not been tested with blind travellers or orientation and mobility instructors yet, and it should not be relied on for a real walk.

## How it's built

Python and a real-time Rust audio engine on the walker's laptop; TypeScript for the browser simulation and the caregiver dashboard. Cartesia is Kite's voice and ears, Claude its judgment, SpacetimeDB the live state every device and the caregiver share, and Moss its memory of the walker's places. Grok powers a voice line for caregivers and generates street scenes we can't safely film. Every decision that can hurt someone runs in deterministic code first; AI may only make Kite more careful.

## Open source

The code is MIT licensed at [github.com/KingArthur207/kite](https://github.com/KingArthur207/kite). Questions or problems: see [Contact](https://trykite.pages.dev/contact). What the site and the earpiece collect: see [Privacy](https://trykite.pages.dev/privacy).

---

Kite · https://trykite.pages.dev/about · Pages: [Home](https://trykite.pages.dev/) · [How it works](https://trykite.pages.dev/how) · [For judges](https://trykite.pages.dev/judges) · [About](https://trykite.pages.dev/about) · [Contact](https://trykite.pages.dev/contact) · [Privacy](https://trykite.pages.dev/privacy) · [llms.txt](https://trykite.pages.dev/llms.txt) · [Sitemap](https://trykite.pages.dev/sitemap.xml)
