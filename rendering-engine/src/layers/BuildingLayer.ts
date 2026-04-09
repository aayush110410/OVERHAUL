/* ──────────────────────────────────────────────────────────────────────────
 * OVERHAUL — Building Layer
 *
 * Renders 3D extruded buildings from OSM data (or procedural generation).
 * Uses GPU-instanced box geometry for performance.
 * ────────────────────────────────────────────────────────────────────────── */

import * as THREE from 'three';
import * as Cesium from 'cesium';
import { BaseLayer } from './BaseLayer';
import type { LayerConfig } from '../core/types';

const BUILDING_COUNT = 4000;

interface BuildingData {
  lon: number;
  lat: number;
  width: number;
  depth: number;
  height: number;
}

interface BuildingRecord {
  id: string;
  position?: { longitude: number; latitude: number; altitude?: number };
  properties?: Record<string, unknown>;
}

export class BuildingLayer extends BaseLayer {
  private instancedMesh: THREE.InstancedMesh | null = null;
  private buildings: BuildingData[] = [];
  private liveBuildings: BuildingData[] = [];
  private dummy = new THREE.Object3D();

  constructor(config: LayerConfig, viewer: Cesium.Viewer, scene: THREE.Scene) {
    super(config, viewer, scene);
  }

  protected async onLoad(): Promise<void> {
    // Procedural building generation around Delhi NCR
    this.generateBuildings();

    const geometry = new THREE.BoxGeometry(1, 1, 1);
    const material = new THREE.MeshPhongMaterial({
      color: 0x8899aa,
      transparent: true,
      opacity: 0.85,
      flatShading: true,
    });

    this.instancedMesh = new THREE.InstancedMesh(geometry, material, BUILDING_COUNT);
    this.instancedMesh.frustumCulled = true;
    this.instancedMesh.count = 0;
    this.rebuildInstances();
    this.scene.add(this.instancedMesh);
  }

  protected onUpdate(_dt: number, cameraHeight: number): void {
    // Buildings only visible at city-level zoom
    if (this.instancedMesh) {
      this.instancedMesh.visible = cameraHeight < 100_000;
    }
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

  protected onData(data: unknown): void {
    const payload = data as {
      items?: BuildingRecord[];
      stats?: { object_count?: number; update_frequency_hz?: number; source?: string };
    };
    const items = payload?.items ?? [];
    this.liveBuildings = items.map((item) => {
      const position = item.position;
      const height = Number(item.properties?.height_m ?? position?.altitude ?? 30);
      return {
        lon: position?.longitude ?? 0,
        lat: position?.latitude ?? 0,
        width: Number(item.properties?.width_m ?? 18 + (height % 40)),
        depth: Number(item.properties?.depth_m ?? 18 + ((height * 0.7) % 40)),
        height,
      };
    });
    this.rebuildInstances();
    this.setExternalStats({
      objectCount: payload?.stats?.object_count ?? this.liveBuildings.length,
      updateFrequencyHz: payload?.stats?.update_frequency_hz,
      source: payload?.stats?.source,
    });
  }

  /* ── Procedural city ────────────────────────────────────────────────── */

  private generateBuildings(): void {
    const centres = [
      { lon: 77.209, lat: 28.614, density: 1.0 },  // Connaught Place
      { lon: 77.220, lat: 28.630, density: 0.8 },  // Chandni Chowk
      { lon: 77.068, lat: 28.493, density: 0.9 },  // Gurgaon
      { lon: 77.350, lat: 28.570, density: 0.85 }, // Noida
      { lon: 77.280, lat: 28.550, density: 0.7 },  // South Delhi
    ];

    let count = 0;
    while (count < BUILDING_COUNT) {
      const centre = centres[count % centres.length];
      const spread = 0.03 + (1 - centre.density) * 0.04;

      const lon = centre.lon + (Math.random() - 0.5) * spread * 2;
      const lat = centre.lat + (Math.random() - 0.5) * spread * 2;

      // Taller near centre
      const distFromCentre = Math.sqrt(
        (lon - centre.lon) ** 2 + (lat - centre.lat) ** 2,
      );
      const maxHeight = Math.max(10, (1 - distFromCentre / spread) * 150 * centre.density);
      const height = 5 + Math.random() * maxHeight;

      this.buildings.push({
        lon,
        lat,
        width: 20 + Math.random() * 60,
        depth: 20 + Math.random() * 60,
        height,
      });
      count++;
    }
  }

  private rebuildInstances(): void {
    if (!this.instancedMesh) return;
    const sourceBuildings = this.liveBuildings.length > 0 ? this.liveBuildings : this.buildings;
    const count = Math.min(sourceBuildings.length, BUILDING_COUNT);

    for (let i = 0; i < count; i++) {
      const b = sourceBuildings[i];
      const cart = Cesium.Cartesian3.fromDegrees(b.lon, b.lat, b.height / 2);
      this.dummy.position.set(cart.x, cart.y, cart.z);
      this.dummy.scale.set(b.width, b.height, b.depth);

      const up = new THREE.Vector3(cart.x, cart.y, cart.z).normalize();
      this.dummy.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), up);

      this.dummy.updateMatrix();
      this.instancedMesh.setMatrixAt(i, this.dummy.matrix);

      const shade = 0.48 + (i % 9) * 0.03;
      this.instancedMesh.setColorAt(i, new THREE.Color(shade, shade, shade + 0.06));
    }

    this.instancedMesh.count = count;
    this.instancedMesh.instanceMatrix.needsUpdate = true;
    if (this.instancedMesh.instanceColor) this.instancedMesh.instanceColor.needsUpdate = true;
    this.setExternalStats({
      objectCount: count,
      source: this.liveBuildings.length > 0 ? 'stream' : 'procedural',
    });
  }
}
