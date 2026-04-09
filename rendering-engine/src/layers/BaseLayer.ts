/* ──────────────────────────────────────────────────────────────────────────
 * OVERHAUL — Base Layer
 *
 * Abstract base class for every visualization layer.
 * Concrete layers (satellite, flight, traffic, …) extend this.
 * ────────────────────────────────────────────────────────────────────────── */

import * as Cesium from 'cesium';
import * as THREE from 'three';
import type { LayerConfig, LayerStats, GeoPosition, RenderableObject } from '../core/types';

export abstract class BaseLayer {
  readonly config: LayerConfig;
  protected viewer: Cesium.Viewer;
  protected scene: THREE.Scene;
  protected objects: Map<string, RenderableObject> = new Map();
  protected loaded = false;
  protected loadTime = 0;
  protected lastUpdate = Date.now();
  protected updateFrequencyHz = 0;
  protected source = 'local';
  protected status: 'active' | 'idle' | 'error' = 'idle';

  private refreshTimer: ReturnType<typeof setInterval> | null = null;

  constructor(config: LayerConfig, viewer: Cesium.Viewer, scene: THREE.Scene) {
    this.config = { ...config };
    this.viewer = viewer;
    this.scene = scene;
  }

  /* ── Public API ─────────────────────────────────────────────────────── */

  async load(): Promise<void> {
    const t0 = performance.now();
    await this.onLoad();
    this.loadTime = performance.now() - t0;
    this.loaded = true;
    this.lastUpdate = Date.now();
    this.status = 'active';

    if (this.config.refreshInterval && this.config.refreshInterval > 0) {
      this.refreshTimer = setInterval(() => this.refresh(), this.config.refreshInterval);
    }
  }

  async refresh(): Promise<void> {
    await this.onRefresh();
    this.lastUpdate = Date.now();
    this.status = 'active';
  }

  applyData(data: unknown): void {
    this.onData(data);
    this.lastUpdate = Date.now();
    this.status = 'active';
  }

  setVisible(visible: boolean): void {
    this.config.visible = visible;
    this.onVisibilityChange(visible);
    this.objects.forEach((obj) => {
      obj.visible = visible;
      if (obj.mesh) obj.mesh.visible = visible;
    });
  }

  setOpacity(opacity: number): void {
    this.config.opacity = Math.max(0, Math.min(1, opacity));
    this.onOpacityChange(this.config.opacity);
  }

  update(dt: number, cameraHeight: number): void {
    if (!this.loaded || !this.config.visible) return;

    // Zoom-based visibility
    if (this.config.minZoom !== undefined || this.config.maxZoom !== undefined) {
      const zoom = this.heightToZoom(cameraHeight);
      const inRange =
        (this.config.minZoom === undefined || zoom >= this.config.minZoom) &&
        (this.config.maxZoom === undefined || zoom <= this.config.maxZoom);
      if (!inRange) {
        this.objects.forEach((o) => {
          if (o.mesh) o.mesh.visible = false;
        });
        return;
      }
    }

    this.onUpdate(dt, cameraHeight);
    this.lastUpdate = Date.now();
  }

  unload(): void {
    if (this.refreshTimer) clearInterval(this.refreshTimer);
    this.onUnload();
    this.objects.forEach((obj) => {
      if (obj.mesh) {
        this.scene.remove(obj.mesh);
        obj.mesh.geometry.dispose();
        const mat = obj.mesh.material;
        if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
        else mat.dispose();
      }
    });
    this.objects.clear();
    this.loaded = false;
  }

  getStats(): LayerStats {
    let visibleCount = 0;
    this.objects.forEach((o) => { if (o.visible) visibleCount++; });
    return {
      objectCount: this.objects.size,
      visibleCount,
      lastUpdate: this.lastUpdate,
      loadTimeMs: this.loadTime,
      memoryBytes: 0,
      updateFrequencyHz: this.updateFrequencyHz,
      source: this.source,
      status: this.status,
    };
  }

  isLoaded(): boolean { return this.loaded; }
  getObjects(): Map<string, RenderableObject> { return this.objects; }
  getVisibleCount(): number {
    let c = 0;
    this.objects.forEach((o) => { if (o.visible) c++; });
    return c;
  }

  /* ── Protected helpers ──────────────────────────────────────────────── */

  protected addObject(obj: RenderableObject): void {
    this.objects.set(obj.id, obj);
    if (obj.mesh) {
      obj.mesh.visible = this.config.visible;
      this.scene.add(obj.mesh);
    }
  }

  protected removeObject(id: string): void {
    const obj = this.objects.get(id);
    if (!obj) return;
    if (obj.mesh) {
      this.scene.remove(obj.mesh);
      obj.mesh.geometry.dispose();
    }
    this.objects.delete(id);
  }

  protected setExternalStats(stats?: { objectCount?: number; updateFrequencyHz?: number; source?: string }): void {
    if (!stats) return;
    if (typeof stats.updateFrequencyHz === 'number') this.updateFrequencyHz = stats.updateFrequencyHz;
    if (typeof stats.source === 'string') this.source = stats.source;
    if (typeof stats.objectCount === 'number' && stats.objectCount >= 0) {
      // Keep reported object count in sync for streamed layers whose GPU objects are pooled.
      while (this.objects.size > stats.objectCount) {
        const firstId = this.objects.keys().next().value;
        if (!firstId) break;
        this.objects.delete(firstId);
      }
      while (this.objects.size < stats.objectCount) {
        const id = `external-${this.objects.size}`;
        this.objects.set(id, {
          id,
          position: { longitude: 0, latitude: 0, altitude: 0 },
          visible: this.config.visible,
          properties: {},
          lodLevel: 0,
          lastUpdate: Date.now(),
        });
      }
    }
  }

  /** Very rough altitude → zoom-level mapping. */
  protected heightToZoom(height: number): number {
    return Math.max(0, Math.min(20, Math.log2(4e7 / Math.max(height, 1))));
  }

  /** Convert WGS84 to Cesium Cartesian3. */
  protected toCartesian(pos: GeoPosition): Cesium.Cartesian3 {
    return Cesium.Cartesian3.fromDegrees(pos.longitude, pos.latitude, pos.altitude);
  }

  /* ── Abstract hooks — implemented by concrete layers ────────────────── */

  protected abstract onLoad(): Promise<void>;
  protected abstract onUpdate(dt: number, cameraHeight: number): void;
  protected abstract onUnload(): void;

  protected onRefresh(): Promise<void> { return Promise.resolve(); }
  protected onData(_data: unknown): void {}
  protected onVisibilityChange(_visible: boolean): void {}
  protected onOpacityChange(_opacity: number): void {}
}
