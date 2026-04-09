import * as THREE from 'three';
import * as Cesium from 'cesium';
import { BaseLayer } from './BaseLayer';
import type { LayerConfig } from '../core/types';

interface SimulationRecord {
  id: string;
  position?: { longitude: number; latitude: number; altitude?: number };
  path?: { coordinates: Array<{ longitude: number; latitude: number; altitude?: number }> };
  properties?: Record<string, unknown>;
}

const MAX_HEAT_POINTS = 1024;

export class SimulationLayer extends BaseLayer {
  private heatMesh: THREE.InstancedMesh | null = null;
  private flowLines: THREE.LineSegments | null = null;
  private infrastructureLines: THREE.LineSegments | null = null;
  private dummy = new THREE.Object3D();
  private heatItems: SimulationRecord[] = [];
  private flowItems: SimulationRecord[] = [];
  private infrastructureItems: SimulationRecord[] = [];

  constructor(config: LayerConfig, viewer: Cesium.Viewer, scene: THREE.Scene) {
    super(config, viewer, scene);
  }

  protected async onLoad(): Promise<void> {
    const geometry = new THREE.SphereGeometry(9000, 8, 8);
    const material = new THREE.MeshBasicMaterial({
      color: 0xff7a18,
      transparent: true,
      opacity: this.config.opacity * 0.65,
      depthWrite: false,
    });
    this.heatMesh = new THREE.InstancedMesh(geometry, material, MAX_HEAT_POINTS);
    this.heatMesh.count = 0;
    this.heatMesh.frustumCulled = false;
    this.scene.add(this.heatMesh);

    this.applyFallbackState();
    this.rebuildLines();
  }

  protected onUpdate(dt: number): void {
    if (!this.heatMesh) return;
    const pulse = 1 + Math.sin(performance.now() * 0.0025) * 0.22;
    for (let i = 0; i < this.heatItems.length && i < MAX_HEAT_POINTS; i++) {
      const item = this.heatItems[i];
      const position = item.position;
      if (!position) continue;
      const cart = Cesium.Cartesian3.fromDegrees(
        position.longitude,
        position.latitude,
        position.altitude ?? 180,
      );
      const intensity = Number(item.properties?.intensity ?? 0.4);
      this.dummy.position.set(cart.x, cart.y, cart.z);
      this.dummy.scale.setScalar(0.5 + intensity * 1.8 * pulse);
      this.dummy.updateMatrix();
      this.heatMesh.setMatrixAt(i, this.dummy.matrix);
      this.heatMesh.setColorAt(i, new THREE.Color().setHSL(0.12 - intensity * 0.1, 0.92, 0.45 + intensity * 0.2));
    }
    this.heatMesh.count = Math.min(this.heatItems.length, MAX_HEAT_POINTS);
    this.heatMesh.instanceMatrix.needsUpdate = true;
    if (this.heatMesh.instanceColor) this.heatMesh.instanceColor.needsUpdate = true;
  }

  protected onUnload(): void {
    if (this.heatMesh) {
      this.scene.remove(this.heatMesh);
      this.heatMesh.geometry.dispose();
      (this.heatMesh.material as THREE.Material).dispose();
      this.heatMesh = null;
    }
    if (this.flowLines) {
      this.scene.remove(this.flowLines);
      this.flowLines.geometry.dispose();
      (this.flowLines.material as THREE.Material).dispose();
      this.flowLines = null;
    }
    if (this.infrastructureLines) {
      this.scene.remove(this.infrastructureLines);
      this.infrastructureLines.geometry.dispose();
      (this.infrastructureLines.material as THREE.Material).dispose();
      this.infrastructureLines = null;
    }
  }

  protected onVisibilityChange(visible: boolean): void {
    if (this.heatMesh) this.heatMesh.visible = visible;
    if (this.flowLines) this.flowLines.visible = visible;
    if (this.infrastructureLines) this.infrastructureLines.visible = visible;
  }

  protected onOpacityChange(opacity: number): void {
    if (this.heatMesh) {
      (this.heatMesh.material as THREE.MeshBasicMaterial).opacity = opacity * 0.65;
    }
    if (this.flowLines) {
      (this.flowLines.material as THREE.LineBasicMaterial).opacity = opacity * 0.75;
    }
    if (this.infrastructureLines) {
      (this.infrastructureLines.material as THREE.LineBasicMaterial).opacity = opacity;
    }
  }

  protected onData(data: unknown): void {
    const payload = data as {
      items?: SimulationRecord[];
      stats?: { object_count?: number; update_frequency_hz?: number; source?: string };
      meta?: Record<string, unknown>;
    };
    const items = payload?.items ?? [];
    this.heatItems = items.filter((item) => item.properties?.kind === 'heat' || item.position);
    this.flowItems = items.filter((item) => item.properties?.kind === 'flow');
    this.infrastructureItems = items.filter((item) => item.properties?.kind === 'infrastructure');
    this.rebuildLines();
    this.setExternalStats({
      objectCount: payload?.stats?.object_count ?? items.length,
      updateFrequencyHz: payload?.stats?.update_frequency_hz,
      source: payload?.stats?.source,
    });
  }

  private rebuildLines(): void {
    if (this.flowLines) {
      this.scene.remove(this.flowLines);
      this.flowLines.geometry.dispose();
      (this.flowLines.material as THREE.Material).dispose();
      this.flowLines = null;
    }
    if (this.infrastructureLines) {
      this.scene.remove(this.infrastructureLines);
      this.infrastructureLines.geometry.dispose();
      (this.infrastructureLines.material as THREE.Material).dispose();
      this.infrastructureLines = null;
    }

    const buildLineSet = (
      items: SimulationRecord[],
      colorFrom: (item: SimulationRecord) => THREE.Color,
      opacity: number,
    ) => {
      const positions: number[] = [];
      const colors: number[] = [];

      for (const item of items) {
        const coords = item.path?.coordinates ?? [];
        if (coords.length < 2) continue;
        for (let i = 0; i < coords.length - 1; i++) {
          const start = coords[i];
          const end = coords[i + 1];
          const c0 = Cesium.Cartesian3.fromDegrees(start.longitude, start.latitude, start.altitude ?? 50);
          const c1 = Cesium.Cartesian3.fromDegrees(end.longitude, end.latitude, end.altitude ?? 50);
          positions.push(c0.x, c0.y, c0.z, c1.x, c1.y, c1.z);
          const color = colorFrom(item);
          colors.push(color.r, color.g, color.b, color.r, color.g, color.b);
        }
      }

      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
      geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

      const material = new THREE.LineBasicMaterial({
        vertexColors: true,
        transparent: true,
        opacity,
      });

      const lines = new THREE.LineSegments(geometry, material);
      lines.frustumCulled = false;
      this.scene.add(lines);
      return lines;
    };

    this.flowLines = buildLineSet(
      this.flowItems,
      (item) => {
        const intensity = Number(item.properties?.intensity ?? 0.4);
        return new THREE.Color().setHSL(0.6 - intensity * 0.28, 0.88, 0.55);
      },
      this.config.opacity * 0.75,
    );
    this.infrastructureLines = buildLineSet(
      this.infrastructureItems,
      () => new THREE.Color(0.6, 0.92, 1),
      this.config.opacity,
    );
  }

  private applyFallbackState(): void {
    const centerLon = 77.209;
    const centerLat = 28.614;
    this.heatItems = [];
    this.flowItems = [];
    this.infrastructureItems = [];

    for (let i = 0; i < 120; i++) {
      const angle = (i / 120) * Math.PI * 2;
      const radius = 0.09 + (i % 8) * 0.003;
      const lon = centerLon + Math.cos(angle) * radius;
      const lat = centerLat + Math.sin(angle) * radius * 0.8;
      this.heatItems.push({
        id: `fallback-heat-${i}`,
        position: { longitude: lon, latitude: lat, altitude: 180 },
        properties: { kind: 'heat', intensity: 0.35 + (i % 12) * 0.04 },
      });
    }

    for (let i = 0; i < 48; i++) {
      const lon0 = centerLon - 0.12 + i * 0.005;
      const lat0 = centerLat - 0.08;
      this.flowItems.push({
        id: `fallback-flow-${i}`,
        path: {
          coordinates: [
            { longitude: lon0, latitude: lat0, altitude: 80 },
            { longitude: lon0 + 0.04, latitude: lat0 + 0.12, altitude: 80 },
          ],
        },
        properties: { kind: 'flow', intensity: 0.45 + (i % 6) * 0.06 },
      });
    }

    this.setExternalStats({ objectCount: this.heatItems.length + this.flowItems.length, source: 'procedural' });
  }
}
