import * as THREE from 'three';
import * as Cesium from 'cesium';
import { BaseLayer } from './BaseLayer';
import type { LayerConfig } from '../core/types';

interface IntelligenceRecord {
  id: string;
  position?: { longitude: number; latitude: number; altitude?: number };
  properties?: Record<string, unknown>;
}

const MAX_EVENTS = 512;

export class IntelligenceLayer extends BaseLayer {
  private mesh: THREE.InstancedMesh | null = null;
  private dummy = new THREE.Object3D();
  private records: IntelligenceRecord[] = [];

  constructor(config: LayerConfig, viewer: Cesium.Viewer, scene: THREE.Scene) {
    super(config, viewer, scene);
  }

  protected async onLoad(): Promise<void> {
    const geometry = new THREE.SphereGeometry(12000, 8, 8);
    const material = new THREE.MeshBasicMaterial({
      color: 0xfb7185,
      transparent: true,
      opacity: this.config.opacity,
    });

    this.mesh = new THREE.InstancedMesh(geometry, material, MAX_EVENTS);
    this.mesh.count = 0;
    this.mesh.frustumCulled = true;
    this.scene.add(this.mesh);
  }

  protected onUpdate(dt: number): void {
    if (!this.mesh) return;
    const pulse = 1 + Math.sin(performance.now() * 0.003) * 0.18;
    for (let i = 0; i < this.records.length && i < MAX_EVENTS; i++) {
      const record = this.records[i];
      const position = record.position;
      if (!position) continue;
      const cart = Cesium.Cartesian3.fromDegrees(
        position.longitude,
        position.latitude,
        position.altitude ?? 1200,
      );
      const magnitude = Number(record.properties?.mag ?? record.properties?.severity ?? 1);
      this.dummy.position.set(cart.x, cart.y, cart.z);
      this.dummy.scale.setScalar(0.7 + magnitude * 0.16 * pulse);
      this.dummy.updateMatrix();
      this.mesh.setMatrixAt(i, this.dummy.matrix);
      const intensity = Math.min(1, magnitude / 7);
      this.mesh.setColorAt(i, new THREE.Color(1, 0.2 + intensity * 0.4, 0.4 + intensity * 0.2));
    }
    this.mesh.count = Math.min(this.records.length, MAX_EVENTS);
    this.mesh.instanceMatrix.needsUpdate = true;
    if (this.mesh.instanceColor) this.mesh.instanceColor.needsUpdate = true;
  }

  protected onUnload(): void {
    if (!this.mesh) return;
    this.scene.remove(this.mesh);
    this.mesh.geometry.dispose();
    (this.mesh.material as THREE.Material).dispose();
    this.mesh = null;
    this.records = [];
  }

  protected onVisibilityChange(visible: boolean): void {
    if (this.mesh) this.mesh.visible = visible;
  }

  protected onOpacityChange(opacity: number): void {
    if (this.mesh) {
      (this.mesh.material as THREE.MeshBasicMaterial).opacity = opacity;
    }
  }

  protected onData(data: unknown): void {
    const payload = data as {
      items?: IntelligenceRecord[];
      stats?: { object_count?: number; update_frequency_hz?: number; source?: string };
    };
    this.records = payload?.items ?? [];
    this.setExternalStats({
      objectCount: payload?.stats?.object_count ?? this.records.length,
      updateFrequencyHz: payload?.stats?.update_frequency_hz,
      source: payload?.stats?.source,
    });
  }
}
