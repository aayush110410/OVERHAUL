import * as THREE from 'three';
import * as Cesium from 'cesium';
import { BaseLayer } from './BaseLayer';
import type { LayerConfig } from '../core/types';

interface TrafficRecord {
  id: string;
  path?: { coordinates: Array<{ longitude: number; latitude: number; altitude?: number }> };
  properties?: Record<string, unknown>;
}

interface TrafficSegment {
  id: string;
  start: { longitude: number; latitude: number; altitude: number };
  end: { longitude: number; latitude: number; altitude: number };
  congestion: number;
  density: number;
  speedKph: number;
  laneCapacity: number;
}

const MAX_PARTICLES = 24_000;

const PARTICLE_VERTEX = `
  uniform float uTime;
  uniform float uPointScale;
  attribute vec3 aStart;
  attribute vec3 aEnd;
  attribute float aOffset;
  attribute float aSpeed;
  attribute float aDensity;
  varying float vDensity;

  void main() {
    float t = fract(aOffset + uTime * aSpeed);
    vec3 position3D = mix(aStart, aEnd, t);
    vec4 mvPosition = modelViewMatrix * vec4(position3D, 1.0);
    float pointSize = mix(2.5, 7.0, aDensity);
    gl_PointSize = pointSize * uPointScale / max(1.0, -mvPosition.z);
    gl_Position = projectionMatrix * mvPosition;
    vDensity = aDensity;
  }
`;

const PARTICLE_FRAGMENT = `
  varying float vDensity;

  void main() {
    vec2 centered = gl_PointCoord - vec2(0.5);
    float falloff = smoothstep(0.26, 0.0, length(centered));
    vec3 cool = vec3(0.17, 0.95, 0.52);
    vec3 warm = vec3(1.0, 0.42, 0.12);
    vec3 color = mix(cool, warm, clamp(vDensity, 0.0, 1.0));
    gl_FragColor = vec4(color, falloff * (0.28 + vDensity * 0.58));
  }
`;

export class TrafficLayer extends BaseLayer {
  private roadMesh: THREE.LineSegments | null = null;
  private particleSystem: THREE.Points | null = null;
  private particleGeometry: THREE.BufferGeometry | null = null;
  private particleMaterial: THREE.ShaderMaterial | null = null;
  private segments: TrafficSegment[] = [];
  private elapsed = 0;

  constructor(config: LayerConfig, viewer: Cesium.Viewer, scene: THREE.Scene) {
    super(config, viewer, scene);
  }

  protected async onLoad(): Promise<void> {
    this.particleGeometry = new THREE.BufferGeometry();
    this.particleGeometry.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(MAX_PARTICLES * 3), 3));
    this.particleGeometry.setAttribute('aStart', new THREE.Float32BufferAttribute(new Float32Array(MAX_PARTICLES * 3), 3));
    this.particleGeometry.setAttribute('aEnd', new THREE.Float32BufferAttribute(new Float32Array(MAX_PARTICLES * 3), 3));
    this.particleGeometry.setAttribute('aOffset', new THREE.Float32BufferAttribute(new Float32Array(MAX_PARTICLES), 1));
    this.particleGeometry.setAttribute('aSpeed', new THREE.Float32BufferAttribute(new Float32Array(MAX_PARTICLES), 1));
    this.particleGeometry.setAttribute('aDensity', new THREE.Float32BufferAttribute(new Float32Array(MAX_PARTICLES), 1));
    this.particleGeometry.setDrawRange(0, 0);

    this.particleMaterial = new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: 0 },
        uPointScale: { value: 1200 },
      },
      vertexShader: PARTICLE_VERTEX,
      fragmentShader: PARTICLE_FRAGMENT,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });

    this.particleSystem = new THREE.Points(this.particleGeometry, this.particleMaterial);
    this.particleSystem.frustumCulled = false;
    this.scene.add(this.particleSystem);

    this.applyFallbackRoadNetwork();
    this.rebuildRoadMesh();
    this.reseedParticles();
  }

  protected onUpdate(dt: number): void {
    this.elapsed += dt;
    if (this.particleMaterial) {
      this.particleMaterial.uniforms.uTime.value = this.elapsed;
    }
  }

  protected onUnload(): void {
    if (this.roadMesh) {
      this.scene.remove(this.roadMesh);
      this.roadMesh.geometry.dispose();
      (this.roadMesh.material as THREE.Material).dispose();
      this.roadMesh = null;
    }
    if (this.particleSystem) {
      this.scene.remove(this.particleSystem);
      this.particleSystem.geometry.dispose();
      (this.particleSystem.material as THREE.Material).dispose();
      this.particleSystem = null;
    }
    this.particleGeometry = null;
    this.particleMaterial = null;
    this.segments = [];
  }

  protected onVisibilityChange(visible: boolean): void {
    if (this.roadMesh) this.roadMesh.visible = visible;
    if (this.particleSystem) this.particleSystem.visible = visible;
  }

  protected onOpacityChange(opacity: number): void {
    if (this.roadMesh) {
      (this.roadMesh.material as THREE.LineBasicMaterial).opacity = opacity * 0.75;
    }
    if (this.particleMaterial) {
      this.particleMaterial.transparent = opacity < 1;
    }
  }

  protected onData(data: unknown): void {
    const payload = data as {
      items?: TrafficRecord[];
      stats?: { object_count?: number; update_frequency_hz?: number; source?: string };
      meta?: { particleBudget?: number };
    };
    const items = payload?.items ?? [];
    const nextSegments: TrafficSegment[] = [];

    for (const item of items) {
      const coordinates = item.path?.coordinates ?? [];
      if (coordinates.length < 2) continue;
      for (let i = 0; i < coordinates.length - 1; i++) {
        const start = coordinates[i];
        const end = coordinates[i + 1];
        nextSegments.push({
          id: `${item.id}-${i}`,
          start: {
            longitude: start.longitude,
            latitude: start.latitude,
            altitude: start.altitude ?? 20,
          },
          end: {
            longitude: end.longitude,
            latitude: end.latitude,
            altitude: end.altitude ?? 20,
          },
          congestion: Number(item.properties?.congestion ?? 0.45),
          density: Math.min(1, Number(item.properties?.density ?? 0.5) / Math.max(1, Number(item.properties?.lane_capacity ?? 1200))),
          speedKph: Number(item.properties?.speed_kph ?? 42),
          laneCapacity: Number(item.properties?.lane_capacity ?? 1200),
        });
      }
    }

    if (nextSegments.length > 0) {
      this.segments = nextSegments;
      this.rebuildRoadMesh();
      this.reseedParticles(payload?.meta?.particleBudget);
    }

    this.setExternalStats({
      objectCount: payload?.stats?.object_count ?? this.segments.length,
      updateFrequencyHz: payload?.stats?.update_frequency_hz,
      source: payload?.stats?.source,
    });
  }

  private rebuildRoadMesh(): void {
    if (this.roadMesh) {
      this.scene.remove(this.roadMesh);
      this.roadMesh.geometry.dispose();
      (this.roadMesh.material as THREE.Material).dispose();
      this.roadMesh = null;
    }

    const positions: number[] = [];
    const colors: number[] = [];
    for (const segment of this.segments) {
      const start = Cesium.Cartesian3.fromDegrees(segment.start.longitude, segment.start.latitude, segment.start.altitude);
      const end = Cesium.Cartesian3.fromDegrees(segment.end.longitude, segment.end.latitude, segment.end.altitude);
      positions.push(start.x, start.y, start.z, end.x, end.y, end.z);
      const color = new THREE.Color().setHSL(0.33 * (1 - segment.congestion), 0.85, 0.48);
      colors.push(color.r, color.g, color.b, color.r, color.g, color.b);
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    const material = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: this.config.opacity * 0.7,
    });

    this.roadMesh = new THREE.LineSegments(geometry, material);
    this.roadMesh.frustumCulled = false;
    this.scene.add(this.roadMesh);
  }

  private reseedParticles(particleBudget = MAX_PARTICLES): void {
    if (!this.particleGeometry || this.segments.length === 0) return;
    const particleCount = Math.min(MAX_PARTICLES, particleBudget);

    const startAttr = this.particleGeometry.getAttribute('aStart') as THREE.BufferAttribute;
    const endAttr = this.particleGeometry.getAttribute('aEnd') as THREE.BufferAttribute;
    const offsetAttr = this.particleGeometry.getAttribute('aOffset') as THREE.BufferAttribute;
    const speedAttr = this.particleGeometry.getAttribute('aSpeed') as THREE.BufferAttribute;
    const densityAttr = this.particleGeometry.getAttribute('aDensity') as THREE.BufferAttribute;

    const weightedSegments = [...this.segments].sort((a, b) => b.congestion - a.congestion);

    for (let i = 0; i < particleCount; i++) {
      const segment = weightedSegments[i % weightedSegments.length];
      const start = Cesium.Cartesian3.fromDegrees(segment.start.longitude, segment.start.latitude, segment.start.altitude);
      const end = Cesium.Cartesian3.fromDegrees(segment.end.longitude, segment.end.latitude, segment.end.altitude);

      startAttr.setXYZ(i, start.x, start.y, start.z);
      endAttr.setXYZ(i, end.x, end.y, end.z);
      offsetAttr.setX(i, (i * 0.61803398875) % 1);
      speedAttr.setX(i, 0.02 + (segment.speedKph / 1200) * 0.04);
      densityAttr.setX(i, Math.min(1, Math.max(0.05, segment.congestion * 0.9 + segment.density * 0.4)));
    }

    this.particleGeometry.setDrawRange(0, particleCount);
    startAttr.needsUpdate = true;
    endAttr.needsUpdate = true;
    offsetAttr.needsUpdate = true;
    speedAttr.needsUpdate = true;
    densityAttr.needsUpdate = true;
  }

  private applyFallbackRoadNetwork(): void {
    this.segments = [];
    const centerLon = 77.209;
    const centerLat = 28.614;
    for (let xi = -12; xi <= 12; xi++) {
      this.segments.push({
        id: `fallback-v-${xi}`,
        start: { longitude: centerLon + xi * 0.01, latitude: centerLat - 0.12, altitude: 20 },
        end: { longitude: centerLon + xi * 0.01, latitude: centerLat + 0.12, altitude: 20 },
        congestion: 0.25 + Math.abs(xi) * 0.03,
        density: 0.35 + Math.abs(xi) * 0.02,
        speedKph: 36,
        laneCapacity: 1200,
      });
    }
    for (let yi = -12; yi <= 12; yi++) {
      this.segments.push({
        id: `fallback-h-${yi}`,
        start: { longitude: centerLon - 0.12, latitude: centerLat + yi * 0.01, altitude: 20 },
        end: { longitude: centerLon + 0.12, latitude: centerLat + yi * 0.01, altitude: 20 },
        congestion: 0.3 + Math.abs(yi) * 0.025,
        density: 0.4 + Math.abs(yi) * 0.02,
        speedKph: 34,
        laneCapacity: 1100,
      });
    }
    this.setExternalStats({ objectCount: this.segments.length, source: 'procedural' });
  }
}
