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
      // expo-location adds this too; spelled out so the background mode is visible here
      UIBackgroundModes: ['location'],
    },
  },
  plugins: [
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
