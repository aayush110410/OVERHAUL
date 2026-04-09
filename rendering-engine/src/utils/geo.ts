/* ──────────────────────────────────────────────────────────────────────────
 * OVERHAUL — Geospatial Utilities
 * ────────────────────────────────────────────────────────────────────────── */

import type { GeoPosition, GeoBounds } from '../core/types';

const DEG2RAD = Math.PI / 180;
const RAD2DEG = 180 / Math.PI;
const EARTH_RADIUS = 6_371_000;

/** Haversine distance in metres. */
export function haversine(a: GeoPosition, b: GeoPosition): number {
  const dLat = (b.latitude - a.latitude) * DEG2RAD;
  const dLon = (b.longitude - a.longitude) * DEG2RAD;
  const lat1 = a.latitude * DEG2RAD;
  const lat2 = b.latitude * DEG2RAD;

  const h =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLon / 2) ** 2;
  return 2 * EARTH_RADIUS * Math.asin(Math.sqrt(h));
}

/** Camera height → approximate zoom level. */
export function heightToZoom(height: number): number {
  return Math.max(0, Math.min(20, Math.log2(4e7 / Math.max(height, 1))));
}

/** Zoom level → approximate camera height in metres. */
export function zoomToHeight(zoom: number): number {
  return 4e7 / Math.pow(2, zoom);
}

/** Expand bounds by a margin in degrees. */
export function expandBounds(bounds: GeoBounds, margin: number): GeoBounds {
  return {
    west: bounds.west - margin,
    south: bounds.south - margin,
    east: bounds.east + margin,
    north: bounds.north + margin,
  };
}

/** Test if a position is inside bounds. */
export function positionInBounds(pos: GeoPosition, bounds: GeoBounds): boolean {
  return (
    pos.longitude >= bounds.west &&
    pos.longitude <= bounds.east &&
    pos.latitude >= bounds.south &&
    pos.latitude <= bounds.north
  );
}

/** Great-circle interpolation. */
export function interpolateGreatCircle(
  from: GeoPosition,
  to: GeoPosition,
  t: number,
): GeoPosition {
  const φ1 = from.latitude * DEG2RAD;
  const λ1 = from.longitude * DEG2RAD;
  const φ2 = to.latitude * DEG2RAD;
  const λ2 = to.longitude * DEG2RAD;

  const d = 2 * Math.asin(
    Math.sqrt(
      Math.sin((φ2 - φ1) / 2) ** 2 +
      Math.cos(φ1) * Math.cos(φ2) * Math.sin((λ2 - λ1) / 2) ** 2,
    ),
  );

  if (d < 1e-10) return { ...from };

  const A = Math.sin((1 - t) * d) / Math.sin(d);
  const B = Math.sin(t * d) / Math.sin(d);

  const x = A * Math.cos(φ1) * Math.cos(λ1) + B * Math.cos(φ2) * Math.cos(λ2);
  const y = A * Math.cos(φ1) * Math.sin(λ1) + B * Math.cos(φ2) * Math.sin(λ2);
  const z = A * Math.sin(φ1) + B * Math.sin(φ2);

  return {
    latitude: Math.atan2(z, Math.sqrt(x * x + y * y)) * RAD2DEG,
    longitude: Math.atan2(y, x) * RAD2DEG,
    altitude: from.altitude + (to.altitude - from.altitude) * t,
  };
}
