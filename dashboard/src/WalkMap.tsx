import { useEffect, useState } from 'react';
import L from 'leaflet';
import { MapContainer, TileLayer, Marker, Polyline, CircleMarker, Tooltip, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

export type LatLng = [number, number];
export type Pose = { lat: number; lng: number; heading?: number };
// bearing is in the walker frame (0 ahead, +90 right), distance in meters.
export type Relative = { key: string; bearing: number; distance: number; label: string; priority: string };

const M_PER_DEG_LAT = 110_540;
const M_PER_DEG_LNG = 111_320;
const ZOOM = 20;      // hazards are 5-20 m away, so start close
const MAX_ZOOM = 21;
export const PRIORITY_COLOR: Record<string, string> = {
  critical: '#ef4444', hazard: '#f59e0b', nav: '#60a5fa', answer: '#a78bfa', info: '#9ca3af', system: '#6b7280',
};

// Walker-frame bearing + distance -> lat/lng. Heading is compass degrees (0 north, 90 east).
export function project(pose: Pose, bearing: number, distance: number): LatLng {
  const b = (((pose.heading ?? 0) + bearing) * Math.PI) / 180;
  const dLat = (distance * Math.cos(b)) / M_PER_DEG_LAT;
  const dLng = (distance * Math.sin(b)) / (M_PER_DEG_LNG * Math.cos((pose.lat * Math.PI) / 180));
  return [pose.lat + dLat, pose.lng + dLng];
}

const walkerIcon = (heading: number | undefined) => L.divIcon({
  className: 'walker-icon',
  iconSize: [44, 44],
  iconAnchor: [22, 22],
  html: `<svg viewBox="-22 -22 44 44" width="44" height="44" style="transform: rotate(${heading ?? 0}deg)">
    <circle r="11" fill="#facc15" stroke="#0e1116" stroke-width="3"/>
    ${heading == null ? '' : '<path d="M0 -21 L8 -10 L-8 -10 Z" fill="#facc15" stroke="#0e1116" stroke-width="2"/>'}
  </svg>`,
});

const beaconIcon = L.divIcon({
  className: 'beacon-icon',
  iconSize: [26, 26],
  iconAnchor: [13, 13],
  html: '<svg viewBox="-13 -13 26 26" width="26" height="26"><path d="M0 -12 L12 0 L0 12 L-12 0 Z" fill="#34d399" stroke="#0e1116" stroke-width="2"/></svg>',
});

function Follow({ pose, follow, setFollow }: { pose: Pose; follow: boolean; setFollow: (on: boolean) => void }) {
  const map = useMap();
  useEffect(() => {
    if (follow) map.panTo([pose.lat, pose.lng], { animate: true });
  }, [map, follow, pose.lat, pose.lng]);
  useEffect(() => {
    const stop = () => setFollow(false);  // dragging the map means the user wants to look around
    map.on('dragstart', stop);
    return () => { map.off('dragstart', stop); };
  }, [map, setFollow]);
  return null;
}

export function WalkMap({ pose, trail, hazards, beacon, freshness }: {
  pose?: Pose; trail: LatLng[]; hazards: Relative[]; beacon?: Relative; freshness: string;
}) {
  const [follow, setFollow] = useState(true);
  if (!pose) {
    return (
      <div className="map empty">
        <div>
          <div className="big">Waiting for the walker</div>
          <p className="muted">No position with coordinates yet. The map appears as soon as a position event with lat/lng arrives.</p>
        </div>
      </div>
    );
  }
  const here: LatLng = [pose.lat, pose.lng];
  const beaconAt = beacon && project(pose, beacon.bearing, beacon.distance);
  return (
    <div className="map">
      <MapContainer center={here} zoom={ZOOM} maxZoom={MAX_ZOOM} zoomControl={false} style={{ height: '100%', width: '100%' }}>
        {/* OSM standard tiles, darkened in CSS (.leaflet-tile-pane). CARTO's dark tiles now need an API key. */}
        <TileLayer
          url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          maxNativeZoom={19}
          maxZoom={MAX_ZOOM}
        />
        <Follow pose={pose} follow={follow} setFollow={setFollow} />
        {trail.length > 1 && <Polyline positions={trail} pathOptions={{ color: '#4da3ff', weight: 5, opacity: 0.8 }} />}
        {beacon && beaconAt && (
          <>
            <Polyline positions={[here, beaconAt]} pathOptions={{ color: '#34d399', weight: 3, dashArray: '8 8' }} />
            <Marker position={beaconAt} icon={beaconIcon}>
              <Tooltip permanent direction="top" offset={[0, -12]} className="map-label beacon">
                {beacon.label} · {Math.round(beacon.distance)} m
              </Tooltip>
            </Marker>
          </>
        )}
        {hazards.map(h => {
          const color = PRIORITY_COLOR[h.priority] ?? PRIORITY_COLOR.hazard;
          return (
            <CircleMarker key={h.key} center={project(pose, h.bearing, h.distance)} radius={14}
              pathOptions={{ color, fillColor: color, fillOpacity: 0.45, weight: 3 }}>
              <Tooltip permanent direction={h.bearing < 0 ? 'left' : 'right'} offset={[h.bearing < 0 ? -14 : 14, 0]} className={`map-label p-${h.priority}`}>
                {h.label} · {Math.round(h.distance)} m
              </Tooltip>
            </CircleMarker>
          );
        })}
        <Marker position={here} icon={walkerIcon(pose.heading)} zIndexOffset={1000} />
      </MapContainer>
      <div className="map-overlay">
        <span>position {freshness}</span>
        <button className={follow ? 'on' : ''} onClick={() => setFollow(f => !f)}>
          {follow ? 'Following walker' : 'Follow walker'}
        </button>
      </div>
    </div>
  );
}
