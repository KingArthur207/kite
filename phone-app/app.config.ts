import type { ExpoConfig } from 'expo/config';

// KITE_BUNDLE_ID overrides the bundle id, e.g. if com.kite.phone is taken on your Apple team.
const bundleIdentifier = process.env.KITE_BUNDLE_ID ?? 'com.eddyzhao.kite';

const config: ExpoConfig = {
  name: 'Kite',
  slug: 'kite-phone',
  scheme: 'kite',
  version: '1.0.0',
  orientation: 'portrait',
  icon: './assets/icon.png',
  userInterfaceStyle: 'dark',
  ios: {
    bundleIdentifier,
    supportsTablet: false,
    infoPlist: {
      // the plugins below add these too; spelled out so the background modes are visible here
      UIBackgroundModes: ['location', 'bluetooth-central'],
    },
  },
  plugins: [
    [
      'react-native-ble-plx',
      {
        modes: ['central'],
        bluetoothAlwaysPermission:
          'Kite sends your location over Bluetooth to the Kite computer that flies your guide drone.',
      },
    ],
    [
      'expo-location',
      {
        isIosBackgroundLocationEnabled: true,
        locationWhenInUsePermission: 'Kite sends your location to the guide drone so it can lead you.',
        locationAlwaysAndWhenInUsePermission:
          'Kite keeps sending your location to the guide drone while your phone is locked.',
        locationAlwaysPermission: 'Kite keeps sending your location to the guide drone while your phone is locked.',
      },
    ],
  ],
};

export default config;
