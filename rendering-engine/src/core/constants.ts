/* ──────────────────────────────────────────────────────────────────────────
 * OVERHAUL Rendering Engine — Constants
 * ────────────────────────────────────────────────────────────────────────── */

import type { GeoPosition, LODLevel, ShaderPreset } from './types';

// ─── Defaults ────────────────────────────────────────────────────────────

/** Centre on New Delhi / NCR — OVERHAUL's primary region. */
export const DEFAULT_VIEW: GeoPosition = {
  longitude: 77.209,
  latitude: 28.6139,
  altitude: 15_000_000, // start zoomed-out to see the full globe
};

export const DEFAULT_HEADING = 0;
export const DEFAULT_PITCH = -90;
export const DEFAULT_FOV = 60;

export const MAX_FPS = 60;
export const TILE_SIZE = 256;
export const MAX_TILE_ZOOM = 18;
export const MIN_TILE_ZOOM = 0;

// ─── LOD Levels ──────────────────────────────────────────────────────────

export const LOD_LEVELS: LODLevel[] = [
  { distance: 500,      geometryDetail: 1.0,  textureSize: 1024, shadowsEnabled: true  },
  { distance: 2_000,    geometryDetail: 0.7,  textureSize: 512,  shadowsEnabled: true  },
  { distance: 10_000,   geometryDetail: 0.4,  textureSize: 256,  shadowsEnabled: false },
  { distance: 50_000,   geometryDetail: 0.15, textureSize: 128,  shadowsEnabled: false },
  { distance: Infinity, geometryDetail: 0.05, textureSize: 64,   shadowsEnabled: false },
];

// ─── Shader Presets ──────────────────────────────────────────────────────

export const SHADER_PRESETS: Record<string, ShaderPreset> = {
  standard: {
    name: 'standard',
    bloom:    { enabled: false, strength: 0.3,  radius: 0.4,  threshold: 0.85 },
    contrast: { enabled: false, amount: 1.0,    brightness: 0.0 },
    sharpness:{ enabled: false, amount: 0.0 },
    glow:     { enabled: false, intensity: 0.0, color: [1, 1, 1] },
    fog:      { enabled: false, near: 1000,     far: 50000, color: [0.7, 0.8, 0.9] },
  },
  night: {
    name: 'night',
    bloom:    { enabled: true,  strength: 1.2,  radius: 0.6,  threshold: 0.3 },
    contrast: { enabled: true,  amount: 1.4,    brightness: -0.15 },
    sharpness:{ enabled: false, amount: 0.0 },
    glow:     { enabled: true,  intensity: 0.4, color: [0.3, 0.5, 1.0] },
    fog:      { enabled: true,  near: 500,      far: 20000, color: [0.02, 0.02, 0.08] },
  },
  thermal: {
    name: 'thermal',
    bloom:    { enabled: true,  strength: 0.5,  radius: 0.3,  threshold: 0.5 },
    contrast: { enabled: true,  amount: 1.8,    brightness: 0.05 },
    sharpness:{ enabled: true,  amount: 0.6 },
    glow:     { enabled: true,  intensity: 0.6, color: [1.0, 0.3, 0.0] },
    fog:      { enabled: false, near: 1000,     far: 50000, color: [0.1, 0.0, 0.1] },
  },
  satellite: {
    name: 'satellite',
    bloom:    { enabled: false, strength: 0.15, radius: 0.2,  threshold: 0.9 },
    contrast: { enabled: true,  amount: 1.15,   brightness: 0.02 },
    sharpness:{ enabled: true,  amount: 0.4 },
    glow:     { enabled: false, intensity: 0.0, color: [1, 1, 1] },
    fog:      { enabled: false, near: 5000,     far: 100000, color: [0.7, 0.8, 0.9] },
  },
};

// ─── Layer colours ───────────────────────────────────────────────────────

export const LAYER_COLORS: Record<string, string> = {
  satellite:  '#3b82f6',
  flight:     '#f59e0b',
  traffic:    '#ef4444',
  weather:    '#06b6d4',
  simulation: '#8b5cf6',
  intelligence: '#fb7185',
  building:   '#6b7280',
  terrain:    '#22c55e',
};
