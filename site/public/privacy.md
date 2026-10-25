# Privacy.

short version: very little

What this website collects, and what the Kite earpiece does with camera frames, audio and location. Last updated October 24, 2026.

## This website

- No accounts, no forms, no cookies, no analytics and no advertising trackers.
- The site is hosted on Cloudflare Pages, which processes standard request data (such as IP address and browser user agent) to serve and protect the site, under Cloudflare's own privacy policy.
- Fonts load from Google Fonts, so your browser sends a request to Google when a page loads.
- The audio clips and images are served from this site; nothing you play or click is recorded.

## The Kite earpiece

- **Camera frames** are processed on the walker's laptop. The body camera's video goes straight from the Arduino UNO Q to that laptop behind a token, and is never stored or sent to SpacetimeDB. A single frame is sent to Anthropic's Claude API only when the walker asks "what's in front of me?" or "read this", or when scene descriptions are turned on. The dashboard's walk view never shows video; its live view can show the body camera and the drone feeds, which go straight from the UNO Q board and the drone laptop to that screen on the local network.
- **Location:** the phone app sends its GPS position to the Kite receiver (the UNO Q or the laptop) over Bluetooth or Wi-Fi. The newest position is kept in SpacetimeDB, so walking directions and the live map can use it.
- **Voice:** after an AirPod press (or the optional wake word), the walker's speech is streamed to Cartesia to be transcribed, and Kite's replies are sent to Cartesia as text to be spoken. Questions that need an answer go to Claude as text.
- **Memories** such as "remember this place as the bench" are stored in a local file on the laptop.
- **Dashboard sharing** is off unless it is turned on for the demo. When it is on, the walker's position, the hazards Kite reports, what Kite says and the text of what the walker says are shared through SpacetimeDB so the walk can be followed on a map. Video is never shared through SpacetimeDB. "Clear run" on the dashboard deletes the walk's events.
- API keys stay on the device in a local file and are never sent anywhere except to the service they belong to.

## Questions

Ask through [Contact](https://trykite.pages.dev/contact). As Kite grows, this page will change before any new data is collected.

---

Kite · https://trykite.pages.dev/privacy · Pages: [Home](https://trykite.pages.dev/) · [How it works](https://trykite.pages.dev/how) · [For judges](https://trykite.pages.dev/judges) · [About](https://trykite.pages.dev/about) · [Contact](https://trykite.pages.dev/contact) · [Privacy](https://trykite.pages.dev/privacy) · [llms.txt](https://trykite.pages.dev/llms.txt) · [Sitemap](https://trykite.pages.dev/sitemap.xml)
