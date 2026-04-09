/* ──────────────────────────────────────────────────────────────────────────
 * OVERHAUL — Satellite Layer
 *
 * Renders orbiting satellites as GPU-instanced spheres.
 * Generates a realistic LEO / MEO / GEO constellation.
 * ────────────────────────────────────────────────────────────────────────── */

import * as THREE from 'three';
import * as Cesium from 'cesium';
import { BaseLayer } from './BaseLayer';
import type { LayerConfig } from '../core/types';

interface SatelliteRecord {
  id: string;
  position?: { longitude: number; latitude: number; altitude?: number };
  properties?: Record<string, unknown>;
}

const SAT_COUNT = 1800;
const EARTH_RADIUS = 6_371_000; // metres

export class SatelliteLayer extends BaseLayer {
  private instancedMesh: THREE.InstancedMesh | null = null;
  private orbits: { inclination: number; raan: number; altitude: number; phase: number; speed: number }[] = [];
  private dummy = new THREE.Object3D();
  private liveSatellites: SatelliteRecord[] = [];

  constructor(config: LayerConfig, viewer: Cesium.Viewer, scene: THREE.Scene) {
    super(config, viewer, scene);
  }

  protected async onLoad(): Promise<void> {
    const geometry = new THREE.SphereGeometry(30_000, 6, 6);
    const material = new THREE.MeshBasicMaterial({ color: 0x3b82f6, transparent: true, opacity: 0.85 });

    this.instancedMesh = new THREE.InstancedMesh(geometry, material, SAT_COUNT);
    this.instancedMesh.frustumCulled = true;
    this.instancedMesh.count = 0;
    this.scene.add(this.instancedMesh);

    for (let i = 0; i < 800; i++) {
      const alt = this.randomAltitude();
      this.orbits.push({
        inclination: Math.random() * Math.PI,
        raan: Math.random() * Math.PI * 2,
        altitude: alt,
        phase: Math.random() * Math.PI * 2,
        speed: 0.0002 + Math.random() * 0.0003,
      });
    }
    this.setExternalStats({ objectCount: this.orbits.length, source: 'procedural' });
  }

  protected onUpdate(dt: number): void {
    if (!this.instancedMesh) return;

    if (this.liveSatellites.length > 0) {
      for (let i = 0; i < this.liveSatellites.length && i < SAT_COUNT; i++) {
        const sat = this.liveSatellites[i];
        const position = sat.position;
        if (!position) continue;
        const cart = Cesium.Cartesian3.fromDegrees(
          position.longitude,
          position.latitude,
          position.altitude ?? 450_000,
        );
        this.dummy.position.set(cart.x, cart.y, cart.z);
        this.dummy.scale.setScalar(1);
        this.dummy.updateMatrix();
        this.instancedMesh.setMatrixAt(i, this.dummy.matrix);
      }
      this.instancedMesh.count = Math.min(this.liveSatellites.length, SAT_COUNT);
      this.instancedMesh.instanceMatrix.needsUpdate = true;
      return;
    }

    const now = performance.now() / 1000;
    for (let i = 0; i < this.orbits.length; i++) {
      const o = this.orbits[i];
      const angle = now * o.speed + o.phase;
      const r = EARTH_RADIUS + o.altitude;

      // Simplified Keplerian orbit
      const x = r * Math.cos(angle);
      const y = r * Math.sin(angle) * Math.cos(o.inclination);
      const z = r * Math.sin(angle) * Math.sin(o.inclination);

      // Rotate by RAAN
      const cx = x * Math.cos(o.raan) - y * Math.sin(o.raan);
      const cy = x * Math.sin(o.raan) + y * Math.cos(o.raan);

      this.dummy.position.set(cx, cy, z);
      this.dummy.updateMatrix();
      this.instancedMesh.setMatrixAt(i, this.dummy.matrix);
    }
    this.instancedMesh.count = this.orbits.length;
    this.instancedMesh.instanceMatrix.needsUpdate = true;
  }

  protected onUnload(): void {
    if (this.instancedMesh) {
      this.scene.remove(this.instancedMesh);
      this.instancedMesh.geometry.dispose();
      (this.instancedMesh.material as THREE.Material).dispose();
      this.instancedMesh = null;
    }
  }

  protected onVisibilityChange(visible: boolean): void {
    if (this.instancedMesh) this.instancedMesh.visible = visible;
  }

  protected onOpacityChange(opacity: number): void {
    if (this.instancedMesh) {
      (this.instancedMesh.material as THREE.MeshBasicMaterial).opacity = opacity;
    }
  }

  protected onData(data: unknown): void {
    const payload = data as {
      items?: SatelliteRecord[];
      stats?: { object_count?: number; update_frequency_hz?: number; source?: string };
    };
    this.liveSatellites = payload?.items ?? [];
    this.setExternalStats({
      objectCount: payload?.stats?.object_count ?? this.liveSatellites.length,
      updateFrequencyHz: payload?.stats?.update_frequency_hz,
      source: payload?.stats?.source,
    });
  }

  private randomAltitude(): number {
    const r = Math.random();
    if (r < 0.6) return 400_000 + Math.random() * 1_600_000;     // LEO
    if (r < 0.85) return 2_000_000 + Math.random() * 18_000_000; // MEO
    return 35_786_000 + (Math.random() - 0.5) * 2_000_000;       // GEO
  }
}
