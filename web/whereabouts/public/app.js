import { positionEnvelope } from "./position.js";

const SEND_MS = 3000;

const statusEl = document.querySelector("#status");
const enableEl = document.querySelector("#enable");

const map = L.map("map", { zoomControl: true }).setView([20, 0], 2);
L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
}).addTo(map);

const marker = L.marker([20, 0], {
  icon: L.divIcon({ className: "", html: '<div class="dot"></div>', iconSize: [16, 16], iconAnchor: [8, 8] }),
  interactive: false,
});
const accuracy = L.circle([20, 0], { radius: 1, color: "#d97757", weight: 1, fillOpacity: 0.15 });

let watchId = null;
let latest = null;
let sending = false;
let placed = false;

enableEl.addEventListener("click", () => {
  if (watchId != null) {
    navigator.geolocation.clearWatch(watchId);
    watchId = null;
    latest = null;
    enableEl.textContent = "Enable location";
    setStatus("Location off", "");
    return;
  }
  if (!navigator.geolocation) {
    setStatus("This browser has no location", "bad");
    return;
  }
  enableEl.textContent = "Stop";
  setStatus("Acquiring a fix…", "warn");
  watchId = navigator.geolocation.watchPosition(onFix, onFixError, {
    enableHighAccuracy: true,
    maximumAge: 2000,
    timeout: 20000,
  });
});

window.setInterval(() => {
  if (latest && watchId != null) send(latest);
}, SEND_MS);

if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("/sw.js").catch(() => {});
}

function onFix(position) {
  latest = position;
  const { latitude, longitude, accuracy: metres } = position.coords;
  const here = [latitude, longitude];
  if (!placed) {
    marker.addTo(map);
    accuracy.addTo(map);
    map.setView(here, 16);
    placed = true;
  } else {
    marker.setLatLng(here);
  }
  accuracy.setLatLng(here);
  accuracy.setRadius(Number.isFinite(metres) ? metres : 1);
  const acc = Number.isFinite(metres) ? `±${Math.round(metres)} m` : "fix";
  if (!statusEl.classList.contains("ok") && !statusEl.classList.contains("bad")) {
    setStatus(acc, "");
  }
}

function onFixError(error) {
  const denied = error && error.code === 1;
  setStatus(denied ? "Location permission denied" : "No location fix", "bad");
}

async function send(position) {
  if (sending || watchId == null) return;
  sending = true;
  try {
    const response = await fetch("/api/location", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(positionEnvelope(position.coords, position.timestamp || Date.now())),
    });
    if (!response.ok) {
      let detail = "";
      try {
        detail = (await response.json()).error || "";
      } catch {
        detail = "";
      }
      throw new Error(detail || `spacetime ${response.status}`);
    }
    if (watchId == null) return;
    const metres = position.coords.accuracy;
    const acc = Number.isFinite(metres) ? `±${Math.round(metres)} m` : "fix";
    setStatus(`Sent ${acc}`, "ok");
  } catch (error) {
    if (watchId != null) setStatus(error.message || "Send failed", "bad");
  } finally {
    sending = false;
  }
}

function setStatus(text, kind) {
  statusEl.textContent = text;
  statusEl.className = kind || "";
}
