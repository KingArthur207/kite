import { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import { MapContainer, TileLayer, Marker, Polyline, CircleMarker, Tooltip, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { meters, type Fresh } from './view';

export type LatLng = [number, number];
export type Pose = { lat: number; lng: number; heading?: number };
// bearing is in the walker frame (0 ahead, +90 right), distance in meters.
export type Relative = { key: string; bearing: number; distance: number; label: string; priority: string };
// A timeline moment placed where the walker was at that time.
export type Pin = { id: string; at: LatLng; color: string; label: string; stop: boolean };
// follow: keep the walker centred. walk: frame the whole walk (trail and moments). free: the user moved the map.
export type MapView = 'follow' | 'walk' | 'free';

const M_PER_DEG_LAT = 110_540;
const M_PER_DEG_LNG = 111_320;
const ZOOM = 20;      // hazards are 5-20 m away, so start close
const MAX_ZOOM = 21;
export const PRIORITY_COLOR: Record<string, string> = {
  critical: '#ef4444', hazard: '#f59e0b', nav: '#60a5fa', answer: '#a78bfa', info: '#9ca3af', system: '#6b7280',
};
const reducedMotion = () => typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;

// Walker-frame bearing + distance -> lat/lng. Heading is compass degrees (0 north, 90 east).
export function project(pose: Pose, bearing: number, distance: number): LatLng {
  const b = (((pose.heading ?? 0) + bearing) * Math.PI) / 180;
  const dLat = (distance * Math.cos(b)) / M_PER_DEG_LAT;
  const dLng = (distance * Math.sin(b)) / (M_PER_DEG_LNG * Math.cos((pose.lat * Math.PI) / 180));
  return [pose.lat + dLat, pose.lng + dLng];
}

// The heading is a finite number from the payload (num()), so it is safe inside the icon's HTML.
const walkerIcon = (heading: number | undefined) => L.divIcon({
  className: 'walker-icon',
  iconSize: [44, 44],
  iconAnchor: [22, 22],
  // An SVG transform attribute, not a style="" attribute: the production CSP (vite.config.ts) blocks inline styles.
  html: `<svg viewBox="-22 -22 44 44" width="44" height="44" aria-hidden="true"><g transform="rotate(${heading ?? 0})">
    <circle r="11" fill="#facc15" stroke="#0e1116" stroke-width="3"/>
    ${heading == null ? '' : '<path d="M0 -21 L8 -10 L-8 -10 Z" fill="#facc15" stroke="#0e1116" stroke-width="2"/>'}
  </g></svg>`,
});

const beaconIcon = L.divIcon({
  className: 'beacon-icon',
  iconSize: [26, 26],
  iconAnchor: [13, 13],
  html: '<svg viewBox="-13 -13 26 26" width="26" height="26" aria-hidden="true"><path d="M0 -12 L12 0 L0 12 L-12 0 Z" fill="#34d399" stroke="#0e1116" stroke-width="2"/></svg>',
});

function Camera({ pose, view, setView, frame }: { pose: Pose; view: MapView; setView: (v: MapView) => void; frame: LatLng[] }) {
  const map = useMap();
  const framed = useRef(false);
  useEffect(() => {
    if (view === 'follow') map.panTo([pose.lat, pose.lng], { animate: !reducedMotion() });
  }, [map, view, pose.lat, pose.lng]);
  // Whole walk: refit only when something would fall outside the view, so the map doesn't twitch every second.
  useEffect(() => {
    if (view !== 'walk') { framed.current = false; return; }
    const bounds = L.latLngBounds([...frame, [pose.lat, pose.lng]]);
    if (framed.current && map.getBounds().pad(-0.08).contains(bounds)) return;
    framed.current = true;
    map.fitBounds(bounds, { padding: [70, 70], maxZoom: 19, animate: !reducedMotion() });
  }, [map, view, frame, pose.lat, pose.lng]);
  useEffect(() => {
    const stop = () => setView('free');  // dragging the map means the user wants to look around
    map.on('dragstart', stop);
    return () => { map.off('dragstart', stop); };
  }, [map, setView]);
  return null;
}

// Clicking a timeline moment flies there and pauses auto-follow so the map stays on it.
function FocusOn({ at, setView }: { at?: LatLng; setView: (v: MapView) => void }) {
  const map = useMap();
  const lat = at?.[0], lng = at?.[1];
  useEffect(() => {
    if (lat == null || lng == null) return;
    setView('free');
    if (reducedMotion()) map.setView([lat, lng], Math.max(map.getZoom(), ZOOM - 1));
    else map.flyTo([lat, lng], Math.max(map.getZoom(), ZOOM - 1), { duration: 0.6 });
  }, [map, lat, lng, setView]);
  return null;
}

export function WalkMap({ pose, trail, hazards, beacon, place, where, freshness, fresh, pins, focusId, onPick, initialView, waiting }: {
  pose?: Pose; trail: LatLng[]; hazards: Relative[]; beacon?: Relative; place: string; where: string; freshness: string; fresh: Fresh;
  pins: Pin[]; focusId?: string; onPick: (id: string) => void; initialView: MapView; waiting: string;
}) {
  const [view, setView] = useState<MapView>(initialView);
  useEffect(() => setView(initialView), [initialView]);
  if (!pose) {
    return (
      <div className="map empty" role="region" aria-label="Map">
        <div>
          <div className="big">Waiting for the walker</div>
          <p className="muted">{waiting}</p>
        </div>
      </div>
    );
  }
  const here: LatLng = [pose.lat, pose.lng];
  const beaconAt = beacon && project(pose, beacon.bearing, beacon.distance);
  const frame = [...trail, ...pins.map(p => p.at)];
  return (
    <div className="map" role="region" aria-label={`Map. The walker is at ${place || 'an unnamed place'}${where ? `, ${where}` : ''}.`}>
      {/* Before the map in the DOM, so the view buttons come first in the tab order. */}
      <div className="map-where">
        <strong>{place || 'Walker'}</strong>
        {where && <span>{where}</span>}
        <span className={`fresh-${fresh}`}>position {freshness}</span>
      </div>
      <div className="map-overlay" role="group" aria-label="Map view">
        <button type="button" aria-pressed={view === 'follow'} onClick={() => setView('follow')}>Follow walker</button>
        <button type="button" aria-pressed={view === 'walk'} onClick={() => setView('walk')}>Whole walk</button>
      </div>
      <MapContainer center={here} zoom={ZOOM} maxZoom={MAX_ZOOM} zoomControl={false} style={{ height: '100%', width: '100%' }}>
        {/* OSM standard tiles, darkened in CSS (.leaflet-tile-pane). CARTO's dark tiles now need an API key. */}
        <TileLayer
          url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          maxNativeZoom={19}
          maxZoom={MAX_ZOOM}
        />
        <Camera pose={pose} view={view} setView={setView} frame={frame} />
        {trail.length > 1 && <Polyline positions={trail} pathOptions={{ color: '#4da3ff', weight: 5, opacity: 0.8 }} />}
        {beacon && beaconAt && (
          <>
            <Polyline positions={[here, beaconAt]} pathOptions={{ color: '#34d399', weight: 3, dashArray: '8 8' }} />
            <Marker position={beaconAt} icon={beaconIcon} keyboard={false}>
              <Tooltip permanent direction="top" offset={[0, -12]} className="map-label beacon">
                {beacon.label} · {meters(beacon.distance)}
              </Tooltip>
            </Marker>
          </>
        )}
        {hazards.map(h => {
          const color = PRIORITY_COLOR[h.priority] ?? PRIORITY_COLOR.hazard;
          return (
            <CircleMarker key={h.key} center={project(pose, h.bearing, h.distance)} radius={14} interactive={false}
              pathOptions={{ color, fillColor: color, fillOpacity: 0.45, weight: 3 }}>
              <Tooltip permanent direction={h.bearing < 0 ? 'left' : 'right'} offset={[h.bearing < 0 ? -14 : 14, 0]} className={`map-label p-${h.priority}`}>
                {h.label} · {meters(h.distance)}
              </Tooltip>
            </CircleMarker>
          );
        })}
        {pins.map(pin => {
          const on = pin.id === focusId;
          return (
            <CircleMarker key={`${pin.id}-${on}`} center={pin.at} radius={on ? 13 : pin.stop ? 10 : 6}
              eventHandlers={{ click: () => onPick(pin.id) }}
              pathOptions={{ color: on ? '#fff' : '#0e1116', weight: on ? 3 : 2, fillColor: pin.color, fillOpacity: 1 }}>
              <Tooltip permanent={on || pin.stop} direction="top" offset={[0, -10]}
                className={`map-label ${pin.stop ? 'p-critical' : ''} ${on ? 'focus' : ''}`}>
                {on || !pin.stop ? pin.label : 'Stop'}
              </Tooltip>
            </CircleMarker>
          );
        })}
        <FocusOn at={pins.find(p => p.id === focusId)?.at} setView={setView} />
        <Marker position={here} icon={walkerIcon(pose.heading)} zIndexOffset={1000} keyboard={false} />
      </MapContainer>
    </div>
  );
}
