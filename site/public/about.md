# About *Kite.*

who, why and what's real

Kite is an assistive navigation aid for blind and low-vision people who walk city streets on their own. A four-person student team built it in 36 hours at Cal Hacks 13.0 in San Francisco, in October 2026.

## What it is

A white cane reports the street one step at a time. A guide dog looks ahead, stops at curbs and refuses to step out in front of a car, but fewer than 10,000 guide dogs work in the US for about a million blind Americans. Kite tries to do a guide dog's jobs with things you can buy: a small DJI Tello drone flies about 30 cm above and behind the walker's head and watches the street ahead, a camera on the chest watches the ground at their feet, and AirPods turn both into sound. Beeps come from the direction of each obstacle and speed up as it gets closer; a calm voice says "Stop." first and only says the way is clear when plain, tested code is sure.

## What works today

- **The earpiece:** spatial beeps, the beacon toward the next turn, the speech queue, and stop and wait, on a real-time Rust audio engine. Cartesia voice in and out, with every safety phrase pre-rendered so it speaks even with no signal, Claude's careful judgment calls, scene descriptions and reading text aloud, and a memory of the walker's places.
- **The body camera:** a Logitech webcam on an Arduino UNO Q streams over Wi-Fi to the laptop, which ranks what's straight ahead into three priorities, each with its own tone. Distance comes from five fused cues, and on hand-labelled real frames it is right 92.7% of the time.
- **The drone:** a DJI Tello that locks onto the walker and follows, guarded by a safety governor and a preflight GO/NO-GO check.
- **The phone and the live map:** a phone app sends GPS to the board and the laptop for walking directions, the SpacetimeDB dashboard shows the walk live, and a message typed there reaches the walker's AirPods seconds later.
- **The demo:** the 41-second walk on the home page is rendered by the real earpiece code on a simulated street. Drones can't fly inside the venue, so at our table the drone's view comes from recorded flights.
- **Next:** walks with blind travellers and orientation and mobility instructors, who should shape when Kite beeps, speaks and stays quiet.

## How it's built

Python and a real-time Rust audio engine on the walker's laptop; TypeScript for the browser simulation and the dashboard. Cartesia is Kite's voice and ears, Claude its judgment, and SpacetimeDB the live state every device and the dashboard share. Grok Imagine generates the street scenes we can't safely film, and Kite is tested against them. Every decision that can hurt someone runs in deterministic code first; AI may only make Kite more careful.

## Open source

The code is MIT licensed at [github.com/KingArthur207/kite](https://github.com/KingArthur207/kite). Questions or problems: see [Contact](https://trykite.pages.dev/contact). What the site and the earpiece collect: see [Privacy](https://trykite.pages.dev/privacy).

---

Kite · https://trykite.pages.dev/about · Pages: [Home](https://trykite.pages.dev/) · [How it works](https://trykite.pages.dev/how) · [For judges](https://trykite.pages.dev/judges) · [About](https://trykite.pages.dev/about) · [Contact](https://trykite.pages.dev/contact) · [Privacy](https://trykite.pages.dev/privacy) · [llms.txt](https://trykite.pages.dev/llms.txt) · [Sitemap](https://trykite.pages.dev/sitemap.xml)
