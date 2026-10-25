import assert from "node:assert/strict";
import test from "node:test";

import { positionEnvelope } from "../public/position.js";
import { reject } from "../src/worker.js";

const coords = { latitude: 37.8719, longitude: -122.2585, accuracy: 12.04, altitude: 15.2, speed: 1.14, heading: 274.2 };

test("a live fix is a phone position row", () => {
  const body = positionEnvelope(coords, 1_700_000_000_000);
  assert.equal(reject(body), "");
  assert.equal(body.type, "position");
  assert.equal(body.source, "phone");
  assert.equal(body.priority, "nav");
  const payload = JSON.parse(body.payload);
  assert.equal(payload.lat, 37.8719);
  assert.equal(payload.lng, -122.2585);
  assert.equal(payload.accuracy, 12);
  assert.equal(payload.speed, 1.14);
  assert.equal(payload.course, 274.2);
  assert.equal(payload.t, 1_700_000_000_000);
});

test("a command for the board is not forwarded", () => {
  const reason = reject({
    type: "control",
    source: "unoq.link",
    priority: "system",
    payload: JSON.stringify({ to: "unoq", cmd: "send", text: "{}", sig: "ab" }),
  });
  assert.equal(reason, "wrong envelope");
});

test("coordinates off the globe are refused", () => {
  const body = positionEnvelope(coords, 1);
  const payload = JSON.parse(body.payload);
  payload.lat = 91;
  body.payload = JSON.stringify(payload);
  assert.equal(reject(body), "bad coordinates");
});
