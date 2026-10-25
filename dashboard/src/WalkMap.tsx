import { Fragment, useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import { MapContainer, TileLayer, Marker, Polyline, Circle, CircleMarker, Tooltip, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { meters } from './view';
import type { Client } from './clients';

export type LatLng = [number, number];
export type Pose = { lat: number; lng: number; heading?: number };
// bearing is in the walker frame (0 ahead, +90 right), distance in meters.
export type Relative = { key: string; bearing: number; distance: number; label: string; priority: string };
// A timeline moment placed where the walker was at that time.
export type Pin = { id: string; at: LatLng; color: string; label: string; stop: boolean };
// all: frame every live client. follow: keep one client centred. walk: frame every trail and moment.
// free: the user moved the map.
export type MapView = 'all' | 'follow' | 'walk' | 'free';

const M_PER_DEG_LAT = 110_540;
const M_PER_DEG_LNG = 111_320;
const ZOOM = 19;
const MAX_ZOOM = 21;
const FIT_MAX_ZOOM = 19;
export const PRIORITY_COLOR: Record<string, string> = {
  critical: '#D92D20', hazard: '#D98A00', nav: '#2F80ED', answer: '#7B5CD6', info: '#8A90A8', system: '#8A90A8',
};
const INK = '#1D2B6B';
const reducedMotion = () => typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches;

// Walker-frame bearing + distance -> lat/lng. Heading is compass degrees (0 north, 90 east).
export function project(pose: Pose, bearing: number, distance: number): LatLng {
  const b = (((pose.heading ?? 0) + bearing) * Math.PI) / 180;
  const dLat = (distance * Math.cos(b)) / M_PER_DEG_LAT;
  const dLng = (distance * Math.sin(b)) / (M_PER_DEG_LNG * Math.cos((pose.lat * Math.PI) / 180));
  return [pose.lat + dLat, pose.lng + dLng];
}

// Colour comes from the fixed palette and heading from num(), so both are safe inside the icon's HTML.
// An SVG transform attribute, not a style="" attribute: the production CSP (vite.config.ts) blocks inline styles.
const clientIcon = (color: string, heading: number | undefined, on: boolean, dim: boolean) => L.divIcon({
  className: `client-icon${dim ? ' dim' : ''}`,
  iconSize: [44, 44],
  iconAnchor: [22, 22],
  html: `<svg viewBox="-22 -22 44 44" width="44" height="44" aria-hidden="true">
    ${on ? `<circle r="17" fill="none" stroke="${color}" stroke-width="1.5" opacity=".5"/>` : ''}
    <g transform="rotate(${heading ?? 0})">
      ${heading == null ? '' : `<path d="M0 -15 L6 -7 L-6 -7 Z" fill="${color}"/>`}
      <circle r="8" fill="${color}" stroke="#fff" stroke-width="3"/>
    </g></svg>`,
});

const beaconIcon = L.divIcon({
  className: 'beacon-icon',
  iconSize: [22, 22],
  iconAnchor: [11, 11],
  html: `<svg viewBox="-11 -11 22 22" width="22" height="22" aria-hidden="true"><path d="M0 -9 L9 0 L0 9 L-9 0 Z" fill="#fff" stroke="${INK}" stroke-width="2"/></svg>`,
});

function Camera({ view, setView, target, live, frame }: {
  view: MapView; setView: (v: MapView) => void; target?: LatLng; live: LatLng[]; frame: LatLng[];
}) {
  const map = useMap();
  const framed = useRef<MapView>();
  const fit = (pts: LatLng[]) => {
    if (pts.length === 0) return;
    const bounds = L.latLngBounds(pts);
    // Refit only when something would fall outside the view, so the map doesn't twitch every second.
    if (framed.current === view && map.getBounds().pad(-0.1).contains(bounds)) return;
    framed.current = view;
    map.fitBounds(bounds, { padding: [80, 80], maxZoom: FIT_MAX_ZOOM, animate: !reducedMotion() });
  };
  useEffect(() => {
    if (view === 'follow' && target) map.panTo(target, { animate: !reducedMotion() });
  }, [map, view, target?.[0], target?.[1]]);
  useEffect(() => {
    if (view === 'all') fit(live);
    else if (view === 'walk') fit(frame);
    else framed.current = undefined;
  }, [map, view, live, frame]);
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
    if (reducedMotion()) map.setView([lat, lng], Math.max(map.getZoom(), ZOOM));
    else map.flyTo([lat, lng], Math.max(map.getZoom(), ZOOM), { duration: 0.6 });
  }, [map, lat, lng, setView]);
  return null;
}

export function WalkMap({ clients, pose, follow, onFollow, hazards, beacon, pins, focusId, onPick, initialView, waiting }: {
  clients: Client[]; pose?: Pose; follow?: string; onFollow: (source: string | undefined) => void;
  hazards: Relative[]; beacon?: Relative; pins: Pin[]; focusId?: string; onPick: (id: string) => void;
  initialView: MapView; waiting: string;
}) {
  const [view, setView] = useState<MapView>(initialView);
  useEffect(() => setView(initialView), [initialView]);
  useEffect(() => { if (follow) setView('follow'); }, [follow]);
  const liveRef = useRef<LatLng[]>([]);
  const frameRef = useRef<LatLng[]>([]);
  if (!pose || clients.length === 0) {
    return (
      <div className="map empty" role="region" aria-label="Map">
        <div>
          <p className="empty-word">Waiting for a phone</p>
          <p className="muted">{waiting}</p>
        </div>
      </div>
    );
  }
  const followed = clients.find(c => c.source === follow) ?? clients.reduce((a, b) => (b.fix.t > a.fix.t ? b : a));
  const target: LatLng = [followed.fix.lat, followed.fix.lng];
  const liveClients = clients.filter(c => c.fresh !== 'dead');
  const live = (liveClients.length ? liveClients : clients).map(c => [c.fix.lat, c.fix.lng] as LatLng);
  const frame = [...clients.flatMap(c => c.trail), ...pins.map(p => p.at)];
  // Stable array identities while the points don't change, so Camera's effects don't run every render.
  if (JSON.stringify(live) !== JSON.stringify(liveRef.current)) liveRef.current = live;
  if (frame.length !== frameRef.current.length || frame[frame.length - 1] !== frameRef.current[frameRef.current.length - 1]) frameRef.current = frame;
  const here: LatLng = [pose.lat, pose.lng];
  const beaconAt = beacon && project(pose, beacon.bearing, beacon.distance);
  const pick = (v: MapView) => { if (v !== 'follow') onFollow(undefined); setView(v); };
  return (
    <div className="map" role="region" aria-label={`Map with ${clients.length} ${clients.length === 1 ? 'phone' : 'phones'}`}>
      {/* Before the map in the DOM, so the view buttons come first in the tab order. */}
      <div className="segmented" role="group" aria-label="Map view">
        <button type="button" aria-pressed={view === 'all'} onClick={() => pick('all')}>{clients.length > 1 ? 'All phones' : 'Phone'}</button>
        <button type="button" aria-pressed={view === 'follow'} onClick={() => { onFollow(followed.source); setView('follow'); }}>Follow</button>
        <button type="button" aria-pressed={view === 'walk'} onClick={() => pick('walk')}>Whole walk</button>
      </div>
      <MapContainer center={target} zoom={ZOOM} maxZoom={MAX_ZOOM} zoomControl={false} style={{ height: '100%', width: '100%' }}>
        {/* OSM standard tiles, muted toward cream in CSS (.leaflet-tile-pane). CARTO's light and dark tiles now need an API key. */}
        <TileLayer
          url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          maxNativeZoom={19}
          maxZoom={MAX_ZOOM}
        />
        <Camera view={view} setView={setView} target={target} live={liveRef.current} frame={frameRef.current} />
        {clients.map(c => {
          const at: LatLng = [c.fix.lat, c.fix.lng];
          const dim = c.fresh === 'dead';
          return (
            <Fragment key={c.source}>
              {c.trail.length > 1 && <Polyline positions={c.trail} interactive={false}
                pathOptions={{ color: c.color, weight: 4, opacity: dim ? 0.25 : 0.7, lineCap: 'round', lineJoin: 'round' }} />}
              {c.fix.accuracy != null && c.fix.accuracy <= 200 && <Circle center={at} radius={c.fix.accuracy} interactive={false}
                pathOptions={{ color: c.color, weight: 1, opacity: 0.35, fillColor: c.color, fillOpacity: dim ? 0.04 : 0.1 }} />}
              {c.ahead && <Polyline positions={[at, c.ahead]} interactive={false}
                pathOptions={{ color: c.color, weight: 3, opacity: 0.9, dashArray: '1 8', lineCap: 'round' }} />}
              <Marker position={at} icon={clientIcon(c.color, c.fix.heading, c.source === follow, dim)}
                zIndexOffset={dim ? 500 : 1000} keyboard={false} eventHandlers={{ click: () => onFollow(c.source) }}>
                {clients.length > 1 && <Tooltip permanent direction="right" offset={[12, 0]} className="map-label">{c.name}</Tooltip>}
              </Marker>
            </Fragment>
          );
        })}
        {beacon && beaconAt && (
          <>
            <Polyline positions={[here, beaconAt]} interactive={false} pathOptions={{ color: INK, weight: 2, opacity: 0.6, dashArray: '6 6' }} />
            <Marker position={beaconAt} icon={beaconIcon} keyboard={false}>
              <Tooltip permanent direction="top" offset={[0, -10]} className="map-label">{beacon.label} · {meters(beacon.distance)}</Tooltip>
            </Marker>
          </>
        )}
        {hazards.map(h => {
          const color = PRIORITY_COLOR[h.priority] ?? PRIORITY_COLOR.hazard;
          return (
            <CircleMarker key={h.key} center={project(pose, h.bearing, h.distance)} radius={11} interactive={false}
              pathOptions={{ color, fillColor: color, fillOpacity: 0.25, weight: 2 }}>
              <Tooltip permanent direction={h.bearing < 0 ? 'left' : 'right'} offset={[h.bearing < 0 ? -12 : 12, 0]} className={`map-label p-${h.priority}`}>
                {h.label} · {meters(h.distance)}
              </Tooltip>
            </CircleMarker>
          );
        })}
        {pins.map(pin => {
          const on = pin.id === focusId;
          return (
            <CircleMarker key={`${pin.id}-${on}`} center={pin.at} radius={on ? 10 : pin.stop ? 7 : 4}
              eventHandlers={{ click: () => onPick(pin.id) }}
              pathOptions={{ color: '#fff', weight: on ? 3 : 2, fillColor: pin.color, fillOpacity: 1 }}>
              <Tooltip permanent={on || pin.stop} direction="top" offset={[0, -8]}
                className={`map-label ${pin.stop ? 'p-critical' : ''} ${on ? 'focus' : ''}`}>
                {on || !pin.stop ? pin.label : 'Stop'}
              </Tooltip>
            </CircleMarker>
          );
        })}
        <FocusOn at={pins.find(p => p.id === focusId)?.at} setView={setView} />
      </MapContainer>
    </div>
  );
}
