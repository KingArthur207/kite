// Sim constants. World frame: x = meters east, y = up, z = meters south (north is -z).

export const FX = 1000;                 // Tello focal length in px; the Tello calibration replaces this
export const CAM_W = 960;
export const CAM_H = 720;
export const VFOV_DEG = (2 * Math.atan(CAM_H / 2 / FX) * 180) / Math.PI;

export const EYE_HEIGHT = 1.6;
export const HEAD_TOP = 1.7;
export const DRONE_ABOVE_HEAD = 0.3;
export const DRONE_BEHIND = 1.0;
export const CHEST_HEIGHT = 1.2;
export const WALKER_RADIUS = 0.3;

export const WALK_SPEED = 1.2;          // m/s
export const TURN_SPEED = 90;           // deg/s

// Sim (0,0) maps here: OSM node 12144593726, the zebra across Moss St at Howard St, SF.
// scripts/osm_corner.py uses the same origin and projection.
export const ORIGIN = { lat: 37.778352, lng: -122.408768 };

export const RELAY_URL = "ws://127.0.0.1:5020/pub";
export const EARPIECE_URL = "ws://127.0.0.1:5011";
export const FRAME_FPS = 8;             // per camera
export const POSITION_HZ = 5;

export function toLatLng(east: number, north: number) {
  const mPerDegLat = 111320;
  return {
    lat: ORIGIN.lat + north / mPerDegLat,
    lng: ORIGIN.lng + east / (mPerDegLat * Math.cos((ORIGIN.lat * Math.PI) / 180)),
  };
}
