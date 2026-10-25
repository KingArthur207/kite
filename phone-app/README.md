# Kite phone app

The walker's iPhone app. Tap the screen (or open `kite://start` from the Action button) and it sends a GPS fix to
Kite's SpacetimeDB about once a second over the phone's own internet, including when the screen is locked. Any
computer reads it from there: `python -m phone.cloud` forwards it to the earpiece, and the live dashboard shows it on
the map. No Bluetooth and no pairing. It speaks every change and gives a haptic for each one.

Each fix is one HTTPS call to the `push` reducer (`cloud.ts`), a `position` event with source `phone`:
`{"type":"position","source":"phone","lat","lng","fix_t","accuracy","speed","course"}`, unknown values left out.
While standing still it sends the newest fix again every 2 s, so Kite knows the phone is still there. After two
failed calls in a row it says "Lost connection. Reconnecting." and keeps retrying, waiting up to 5 s between tries.

## Install on an iPhone

You need a Mac with Xcode, Node 22, a free or paid Apple ID added in Xcode (Settings > Accounts), and the phone on a USB cable.

1. On the iPhone, turn on Developer Mode: Settings > Privacy & Security > Developer Mode. The phone restarts.
2. Install and generate the native project:
   ```sh
   cd phone-app
   npm install
   npx expo prebuild -p ios
   ```
3. Set the signing team once: `open ios/Kite.xcworkspace`, select the Kite target, go to Signing & Capabilities, and pick your Team. The default bundle id is `com.eddyzhao.kite` (Eddy's team, 4T3U9F8XAD); on another team, rerun prebuild with `KITE_BUNDLE_ID=com.yourname.kite npx expo prebuild -p ios`.
4. Build and install. A Release build bundles the JS, so it runs without Metro:
   ```sh
   npx expo run:ios --device --configuration Release
   ```
   Pick the phone when prompted. Use `npx expo run:ios --device` for a Debug build that reloads from Metro, with the phone and Mac on the same Wi-Fi.
5. The first time, the phone refuses to open the app. Go to Settings > General > VPN & Device Management, tap your Apple ID, and tap Trust.
6. Open Kite and tap the screen. Allow location. Choose "Allow While Using App", then "Change to Always Allow" when iOS asks later.

## Action button

1. In the Shortcuts app, tap + and add the "Open URLs" action with the URL `kite://start`. Name the shortcut "Kite".
2. Go to Settings > Action Button, swipe to Shortcut, and pick "Kite".

Pressing the button then opens Kite and starts it straight away. A `kite://stop` shortcut stops it, and so does a tap anywhere on the screen.

## Develop

```sh
npm run typecheck
npm test            # bun test: the fix format and the SpacetimeDB link (fake fetch, no network)
npx expo run:ios    # simulator: set a location in Features > Location and it sends it to SpacetimeDB
xcrun simctl openurl booted kite://start   # the simulator asks "Open in Kite?"; press Return
```

`ios/` is generated. Change `app.config.ts` and run `npx expo prebuild -p ios --clean`. Don't edit the native files.
