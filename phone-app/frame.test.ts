// bun test (from phone-app/)
import { expect, test } from 'bun:test';
import { toFix, toPosition } from './frame';

const loc = {
  timestamp: 1761330000123,
  coords: { latitude: 37.77835212, longitude: -122.40876834, accuracy: 4.83, altitude: -2.04, altitudeAccuracy: 3, speed: -1, heading: -1 },
};

test('unknown iOS values become null, negative altitude is kept', () => {
  expect(toFix(loc)).toEqual({ t: 1761330000123, lat: 37.7783521, lng: -122.4087683, acc: 4.8, alt: -2, speed: null, course: null });
  expect(toFix({ ...loc, coords: { ...loc.coords, altitudeAccuracy: -1 } }).alt).toBeNull();
});

test('a fix becomes an earpiece position event; unknown values are left out', () => {
  const fix = toFix(loc);
  expect(toPosition(fix)).toEqual({ type: 'position', source: 'phone', lat: 37.7783521, lng: -122.4087683, fix_t: 1761330000123, accuracy: 4.8 });
  const moving = toFix({ ...loc, coords: { ...loc.coords, speed: 1.234, heading: 271.26 } });
  expect(toPosition(moving, 'phone-a')).toMatchObject({ source: 'phone-a', speed: 1.23, course: 271.3 });
  expect(toPosition({ ...fix, speed: 250 }).speed).toBe(100);
});
