/* ──────────────────────────────────────────────────────────────────────────
 * OVERHAUL — Layer Manager
 *
 * Registry + lifecycle controller for all visualisation layers.
 * Supports dynamic loading / unloading at runtime.
 * ────────────────────────────────────────────────────────────────────────── */

import * as Cesium from 'cesium';
import { SceneManager } from '../core/SceneManager';
import { BaseLayer } from './BaseLayer';
import type { LayerConfig, LayerEvent, LayerStats } from '../core/types';

// Concrete layer imports
import { SatelliteLayer } from './SatelliteLayer';
import { FlightLayer } from './FlightLayer';
import { TrafficLayer } from './TrafficLayer';
import { WeatherLayer } from './WeatherLayer';
import { SimulationLayer } from './SimulationLayer';
import { BuildingLayer } from './BuildingLayer';
import { IntelligenceLayer } from './IntelligenceLayer';

type LayerEventHandler = (event: LayerEvent) => void;

export class LayerManager {
  private layers = new Map<string, BaseLayer>();
  private order: string[] = [];
  private viewer: Cesium.Viewer;
  private sceneManager: SceneManager;
  private listeners: LayerEventHandler[] = [];

  constructor(viewer: Cesium.Viewer, sceneManager: SceneManager) {
    this.viewer = viewer;
    this.sceneManager = sceneManager;
  }

  /* ── Add / remove ───────────────────────────────────────────────────── */

  async addLayer(config: LayerConfig): Promise<BaseLayer> {
    if (this.layers.has(config.id)) {
      throw new Error(`Layer "${config.id}" already exists`);
    }

    const layer = this.createLayer(config);
    this.layers.set(config.id, layer);
    this.order.push(config.id);
    this.sortOrder();

    await layer.load();
    this.emit({ type: 'loaded', layerId: config.id, timestamp: Date.now() });
    return layer;
  }

  async removeLayer(id: string): Promise<void> {
    const layer = this.layers.get(id);
    if (!layer) return;
    layer.unload();
    this.layers.delete(id);
    this.order = this.order.filter((lid) => lid !== id);
    this.emit({ type: 'removed', layerId: id, timestamp: Date.now() });
  }

  removeAll(): void {
    for (const id of [...this.layers.keys()]) {
      const layer = this.layers.get(id);
      layer?.unload();
    }
    this.layers.clear();
    this.order = [];
  }

  /* ── Queries ────────────────────────────────────────────────────────── */

  getLayer(id: string): BaseLayer | undefined { return this.layers.get(id); }
  getAllLayers(): BaseLayer[] { return this.order.map((id) => this.layers.get(id)!).filter(Boolean); }

  getStats(): Map<string, LayerStats> {
    const map = new Map<string, LayerStats>();
    this.layers.forEach((l, id) => map.set(id, l.getStats()));
    return map;
  }

  getVisibleObjectCount(): number {
    let total = 0;
    this.layers.forEach((l) => { total += l.getVisibleCount(); });
    return total;
  }

  /* ── Per-frame update ───────────────────────────────────────────────── */

  update(dt: number, cameraHeight: number): void {
    for (const id of this.order) {
      this.layers.get(id)?.update(dt, cameraHeight);
    }
  }

  /* ── Visibility ─────────────────────────────────────────────────────── */

  setLayerVisible(id: string, visible: boolean): void {
    const layer = this.layers.get(id);
    if (!layer) return;
    layer.setVisible(visible);
    this.emit({ type: 'visibility-changed', layerId: id, timestamp: Date.now(), data: { visible } });
  }

  setLayerOpacity(id: string, opacity: number): void {
    this.layers.get(id)?.setOpacity(opacity);
  }

  applyLayerData(id: string, data: unknown): void {
    this.layers.get(id)?.applyData(data);
  }

  /* ── Events ─────────────────────────────────────────────────────────── */

  onEvent(handler: LayerEventHandler): () => void {
    this.listeners.push(handler);
    return () => { this.listeners = this.listeners.filter((h) => h !== handler); };
  }

  private emit(event: LayerEvent): void {
    for (const h of this.listeners) h(event);
  }

  /* ── Factory ────────────────────────────────────────────────────────── */

  private createLayer(config: LayerConfig): BaseLayer {
    const scene = this.sceneManager.scene;
    switch (config.type) {
      case 'satellite':  return new SatelliteLayer(config, this.viewer, scene);
      case 'flight':     return new FlightLayer(config, this.viewer, scene);
      case 'traffic':    return new TrafficLayer(config, this.viewer, scene);
      case 'weather':    return new WeatherLayer(config, this.viewer, scene);
      case 'simulation': return new SimulationLayer(config, this.viewer, scene);
      case 'building':   return new BuildingLayer(config, this.viewer, scene);
      case 'intelligence': return new IntelligenceLayer(config, this.viewer, scene);
      default:
        throw new Error(`Unknown layer type: ${config.type}`);
    }
  }

  private sortOrder(): void {
    this.order.sort((a, b) => {
      const la = this.layers.get(a)!.config;
      const lb = this.layers.get(b)!.config;
      return (la.zIndex ?? 0) - (lb.zIndex ?? 0);
    });
  }
}
