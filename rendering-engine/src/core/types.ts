/* ──────────────────────────────────────────────────────────────────────────
 * OVERHAUL Rendering Engine — Core Types
 * ────────────────────────────────────────────────────────────────────────── */

import * as Cesium from 'cesium';
import * as THREE from 'three';

// ─── Geospatial ──────────────────────────────────────────────────────────

export interface GeoPosition {
  longitude: number;  // degrees
  latitude: number;   // degrees
  altitude: number;   // meters above WGS84 ellipsoid
}

export interface GeoBounds {
  west: number;
  south: number;
  east: number;
  north: number;
}

// ─── Engine ──────────────────────────────────────────────────────────────

export interface EngineConfig {
  container: HTMLElement;
  cesiumToken?: string;
  baseUrl?: string;
  terrainProvider?: 'cesium-world' | 'ellipsoid' | 'custom';
  imageryProvider?: 'osm' | 'bing' | 'mapbox' | 'custom';
  initialView?: GeoPosition;
  initialHeading?: number;
  initialPitch?: number;
  msaaSamples?: number;
  maxFps?: number;
  enableLighting?: boolean;
  enableAtmosphere?: boolean;
  enableFog?: boolean;
  debugMode?: boolean;
}

export interface EngineState {
  viewer: Cesium.Viewer | null;
  scene: THREE.Scene | null;
  renderer: THREE.WebGLRenderer | null;
  camera: THREE.PerspectiveCamera | null;
  running: boolean;
  fps: number;
  frameTime: number;
  objectCount: number;
  drawCalls: number;
  triangles: number;
}

// ─── Layers ──────────────────────────────────────────────────────────────

export type LayerType =
  | 'satellite'
  | 'flight'
  | 'traffic'
  | 'weather'
  | 'simulation'
  | 'intelligence'
  | 'building'
  | 'terrain'
  | 'custom';

export interface LayerConfig {
  id: string;
  type: LayerType;
  name: string;
  visible: boolean;
  opacity: number;
  minZoom?: number;
  maxZoom?: number;
  zIndex?: number;
  dataUrl?: string;
  refreshInterval?: number;  // ms, 0 = no refresh
  metadata?: Record<string, unknown>;
}

export interface LayerStats {
  objectCount: number;
  visibleCount: number;
  lastUpdate: number;
  loadTimeMs: number;
  memoryBytes: number;
  updateFrequencyHz?: number;
  source?: string;
  status?: 'active' | 'idle' | 'error';
}

export interface LayerEvent {
  type: 'loaded' | 'updated' | 'error' | 'visibility-changed' | 'removed';
  layerId: string;
  timestamp: number;
  data?: unknown;
}

// ─── Camera ──────────────────────────────────────────────────────────────

export type CameraMode = 'orbit' | 'pan' | 'tilt' | 'free' | 'cinematic';

export interface CameraState {
  position: GeoPosition;
  heading: number;   // degrees
  pitch: number;     // degrees
  roll: number;      // degrees
  fov: number;       // degrees
  mode: CameraMode;
}

export interface CameraTransition {
  target: Partial<CameraState>;
  duration: number;     // seconds
  easing: EasingFunction;
  onComplete?: () => void;
}

export type EasingFunction =
  | 'linear'
  | 'ease-in'
  | 'ease-out'
  | 'ease-in-out'
  | 'ease-in-cubic'
  | 'ease-out-cubic'
  | 'ease-in-out-cubic';

// ─── Shaders / Post-processing ───────────────────────────────────────────

export type VisualizationMode = 'standard' | 'night' | 'thermal' | 'satellite';

export interface ShaderUniforms {
  [name: string]: {
    value: number | number[] | THREE.Texture | THREE.Vector2 | THREE.Vector3;
  };
}

export interface PostProcessPass {
  name: string;
  enabled: boolean;
  order: number;
  vertexShader: string;
  fragmentShader: string;
  uniforms: ShaderUniforms;
}

export interface ShaderPreset {
  name: VisualizationMode;
  bloom: { enabled: boolean; strength: number; radius: number; threshold: number };
  contrast: { enabled: boolean; amount: number; brightness: number };
  sharpness: { enabled: boolean; amount: number };
  glow: { enabled: boolean; intensity: number; color: [number, number, number] };
  fog: { enabled: boolean; near: number; far: number; color: [number, number, number] };
}

// ─── Performance ─────────────────────────────────────────────────────────

export interface LODLevel {
  distance: number;        // camera distance threshold
  geometryDetail: number;  // 0.0–1.0
  textureSize: number;     // pixels
  shadowsEnabled: boolean;
}

export interface SpatialNode<T = unknown> {
  bounds: GeoBounds;
  items: T[];
  children: SpatialNode<T>[];
  depth: number;
}

export interface PerformanceMetrics {
  fps: number;
  frameTime: number;
  gpuTime: number;
  cpuTime: number;
  objectCount: number;
  visibleObjects: number;
  drawCalls: number;
  triangles: number;
  textureMemory: number;
  geometryMemory: number;
}

// ─── Data ────────────────────────────────────────────────────────────────

export interface TileCoord {
  x: number;
  y: number;
  z: number; // zoom level
}

export interface DataSource<T = unknown> {
  id: string;
  url: string;
  type: 'geojson' | 'tiles' | 'stream' | 'csv' | 'binary';
  transform?: (raw: unknown) => T[];
  refreshInterval?: number;
}

// ─── Object representation ──────────────────────────────────────────────

export interface RenderableObject {
  id: string;
  position: GeoPosition;
  mesh?: THREE.Mesh;
  billboard?: Cesium.Billboard;
  label?: string;
  properties: Record<string, unknown>;
  lodLevel: number;
  visible: boolean;
  lastUpdate: number;
}
