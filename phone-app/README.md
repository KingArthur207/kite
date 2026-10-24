# Kite phone app

The walker's iPhone app. It finds the Kite computer over Bluetooth LE (`python -m phone.ble`), connects, and sends a GPS fix twice a second, including when the screen is locked. Open it with `kite://start` from the Action button and it starts on its own. It speaks every change and gives a haptic for each one.

Fixes go to characteristic `4b495445-0001-…` as one JSON line each, `{"t","lat","lng","acc","alt","speed","course"}`. Unknown values are `null`. Each write is cut to fit the MTU.

## Install on an iPhone

You need a Mac with Xcode, Node 22, a free or paid Apple ID added in Xcode (Settings > Accounts), and the phone on a USB cable.

1. On the iPhone, turn on Developer Mode: Settings > Privacy & Security > Developer Mode. The phone restarts.
2. Install and generate the native project:
   ```sh
   cd phone-app
   npm install
   npx expo prebuild -p ios
   ```
3. Set the signing team once: `open ios/Kite.xcworkspace`, select the Kite target, go to Signing & Capabilities, and pick your Team. If `com.kite.phone` is already taken, rerun prebuild with `KITE_BUNDLE_ID=com.yourname.kite npx expo prebuild -p ios`.
4. Build and install. A Release build bundles the JS, so it runs without Metro:
   ```sh
   npx expo run:ios --device --configuration Release
   ```
   Pick the phone when prompted. Use `npx expo run:ios --device` for a Debug build that reloads from Metro, with the phone and Mac on the same Wi-Fi.
5. The first time, the phone refuses to open the app. Go to Settings > General > VPN & Device Management, tap your Apple ID, and tap Trust.
6. Open Kite and tap the screen. Allow Bluetooth and location. Choose "Allow While Using App", then "Change to Always Allow" when iOS asks later.

## Action button

1. In the Shortcuts app, tap + and add the "Open URLs" action with the URL `kite://start`. Name the shortcut "Kite".
2. Go to Settings > Action Button, swipe to Shortcut, and pick "Kite".

Pressing the button then opens Kite and starts it straight away. A `kite://stop` shortcut stops it, and so does a tap anywhere on the screen.

## Develop

```sh
npm run typecheck
npm test            # bun test: framing and chunking
npx expo run:ios    # simulator. There is no Bluetooth there, so it stays on "Looking for Kite"
xcrun simctl openurl booted kite://start   # the simulator asks "Open in Kite?"; press Return
```

`ios/` is generated. Change `app.config.ts` and run `npx expo prebuild -p ios --clean`. Don't edit the native files.
