// A live GPS fix as the position row SpacetimeDB already stores for the phone.
// Sent only while the browser watch is running.

export function positionEnvelope(coords, when = Date.now()) {
  const lat = round(coords.latitude, 7);
  const lng = round(coords.longitude, 7);
  if (!Number.isFinite(lat) || !Number.isFinite(lng)) {
    throw new Error("no coordinates");
  }
  const payload = { type: "position", source: "phone", lat, lng };
  if (Number.isFinite(coords.accuracy) && coords.accuracy >= 0) payload.accuracy = round(coords.accuracy, 1);
  if (coords.altitude != null && Number.isFinite(coords.altitude)) payload.alt = round(coords.altitude, 1);
  if (coords.speed != null && Number.isFinite(coords.speed) && coords.speed >= 0) payload.speed = round(coords.speed, 2);
  if (coords.heading != null && Number.isFinite(coords.heading) && coords.heading >= 0) {
    payload.course = round(coords.heading, 1);
  }
  payload.t = Math.round(when);
  return {
    type: "position",
    source: "phone",
    priority: "nav",
    payload: JSON.stringify(payload),
  };
}

function round(value, digits) {
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}
