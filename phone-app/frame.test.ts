// bun test (from phone-app/)
import { expect, test } from 'bun:test';
import { chunk, frameFix, toFix, writeSize } from './frame';

const loc = {
  timestamp: 1761330000123,
  coords: { latitude: 37.77835212, longitude: -122.40876834, accuracy: 4.83, altitude: -2.04, altitudeAccuracy: 3, speed: -1, heading: -1 },
};

test('unknown iOS values become null, negative altitude is kept', () => {
  expect(toFix(loc)).toEqual({ t: 1761330000123, lat: 37.7783521, lng: -122.4087683, acc: 4.8, alt: -2, speed: null, course: null });
  expect(toFix({ ...loc, coords: { ...loc.coords, altitudeAccuracy: -1 } }).alt).toBeNull();
});

test('one newline-terminated JSON line per fix', () => {
  const line = frameFix(toFix(loc));
  expect(line.endsWith('\n')).toBe(true);
  expect(line.indexOf('\n')).toBe(line.length - 1);
  expect(JSON.parse(line).speed).toBeNull();
});

test('chunks fit the write size and reassemble exactly', () => {
  const line = frameFix(toFix(loc));
  for (const size of [20, 182]) {
    const parts = chunk(line, size);
    expect(parts.every((p) => p.length <= size)).toBe(true);
    expect(parts.join('')).toBe(line);
  }
  expect(chunk(line, 20).length).toBe(Math.ceil(line.length / 20));
});

test('write size from MTU', () => {
  expect(writeSize(undefined)).toBe(20);
  expect(writeSize(23)).toBe(20);
  expect(writeSize(185)).toBe(182);
  expect(writeSize(1000)).toBe(512);
});
