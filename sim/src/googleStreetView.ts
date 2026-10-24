// Shared by ?world=streetview (streetviewWalk.ts) and ?compare=1 (compare.ts): the Street View fence and the
// Maps JavaScript API loader. Only Google's own panoramas inside SCOPE are ever shown (docs/google-permission.md).

// ponytail: no @types/google.maps; the handful of calls are checked by hand
// eslint-disable-next-line @typescript-eslint/no-explicit-any
declare const google: any;

// Same box as SCOPE in kite/sources/streetview.py: Moss St from Folsom to Howard plus the corner.
export const SCOPE = { latMin: 37.7768, latMax: 37.7788, lngMin: -122.4095, lngMax: -122.4068 };
export const inScope = (lat: number, lng: number) =>
  lat >= SCOPE.latMin && lat <= SCOPE.latMax && lng >= SCOPE.lngMin && lng <= SCOPE.lngMax;
const KEY: string = import.meta.env.VITE_GOOGLE_MAPS_API_KEY ?? "";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let loading: Promise<any> | null = null;
/** Loads the Maps JavaScript API once and resolves to its streetView library. `onAuthFail` hears Google reject the key. */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function loadStreetView(onAuthFail: () => void): Promise<any> {
  if (!KEY) return Promise.reject(new Error("no VITE_GOOGLE_MAPS_API_KEY: run sh sim/tiles_env.sh"));
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const w = window as any;
  w.gm_authFailure = onAuthFail;
  loading ??= new Promise<void>((resolve, reject) => {
    w.__kiteSvReady = resolve;
    const s = document.createElement("script");
    s.src = `https://maps.googleapis.com/maps/api/js?key=${encodeURIComponent(KEY)}&v=weekly&loading=async&callback=__kiteSvReady`;
    s.onerror = () => reject(new Error("could not load the Maps JavaScript API"));
    document.head.appendChild(s);
  }).then(() => google.maps.importLibrary("streetView"));
  return loading;
}

/** Panorama options for both pages: Google's logo, copyright and terms stay (required); every control goes. */
export const PANO_OPTS = {
  disableDefaultUI: true, clickToGo: false, linksControl: false, panControl: false, zoomControl: false,
  addressControl: false, fullscreenControl: false, motionTracking: false, motionTrackingControl: false,
  showRoadLabels: false, keyboardShortcuts: false, scrollwheel: false, disableDoubleClickZoom: true,
  imageDateControl: false,
};
