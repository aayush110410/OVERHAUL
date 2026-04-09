import type { LayerConfig, VisualizationMode } from '../core/types';
import type { Demo4LayerId, Demo4LayerSnapshot, Demo4LayerState, Demo4SimulationReport, Demo4TimelineState } from './types';

export const DEMO4_API_BASE_URL = import.meta.env.VITE_DEMO4_API_URL || 'http://localhost:8014';

export const DEMO4_LAYER_CONFIGS: Demo4LayerState[] = [
  { id: 'satellites', type: 'satellite', name: 'Satellite layer', visible: true, opacity: 0.9, zIndex: 10, loaded: true, objectCount: 0, updateFrequencyHz: 0, source: 'boot' },
  { id: 'flights', type: 'flight', name: 'Flight layer', visible: true, opacity: 0.88, zIndex: 20, loaded: true, objectCount: 0, updateFrequencyHz: 0, source: 'boot' },
  { id: 'traffic', type: 'traffic', name: 'Traffic layer', visible: true, opacity: 0.95, zIndex: 30, loaded: true, objectCount: 0, updateFrequencyHz: 0, source: 'boot' },
  { id: 'weather', type: 'weather', name: 'Weather layer', visible: true, opacity: 0.7, zIndex: 5, loaded: true, objectCount: 0, updateFrequencyHz: 0, source: 'boot' },
  { id: 'simulation', type: 'simulation', name: 'Simulation layer', visible: true, opacity: 0.78, zIndex: 40, loaded: true, objectCount: 0, updateFrequencyHz: 0, source: 'boot' },
  { id: 'buildings', type: 'building', name: '3D building extrusion', visible: true, opacity: 0.85, zIndex: 35, loaded: true, objectCount: 0, updateFrequencyHz: 0, source: 'boot' },
  { id: 'intelligence', type: 'intelligence', name: 'Intelligence layer', visible: true, opacity: 0.92, zIndex: 45, loaded: true, objectCount: 0, updateFrequencyHz: 0, source: 'boot' },
];

export const VISUALIZATION_MODES: Array<{ id: VisualizationMode; label: string }> = [
  { id: 'standard', label: 'Standard mode' },
  { id: 'night', label: 'Night operations mode' },
  { id: 'thermal', label: 'Thermal vision mode' },
  { id: 'satellite', label: 'Satellite intelligence mode' },
];

export function applySnapshotToLayerState(
  current: Demo4LayerState[],
  snapshots: Partial<Record<Demo4LayerId, Demo4LayerSnapshot>>,
): Demo4LayerState[] {
  return current.map((layer) => {
    const snapshot = snapshots[layer.id as Demo4LayerId];
    if (!snapshot) return layer;
    return {
      ...layer,
      objectCount: snapshot.stats.object_count,
      updateFrequencyHz: snapshot.stats.update_frequency_hz,
      source: snapshot.stats.source,
    };
  });
}

export function timelineStateToLayerSnapshot(state: Demo4TimelineState): Demo4LayerSnapshot {
  return {
    layer_id: 'simulation',
    label: state.label,
    kind: 'hybrid',
    stats: {
      object_count: state.heatmap.length + state.flows.length + state.infrastructure.length,
      update_frequency_hz: 0.25,
      active: true,
      source: 'timeline',
      last_updated: new Date().toISOString(),
    },
    items: [...state.heatmap, ...state.flows, ...state.infrastructure],
    meta: {
      metrics: state.metrics,
      label: state.label,
    },
  };
}

export function simulationReportToLayerSnapshot(report: Demo4SimulationReport): Demo4LayerSnapshot | null {
  const first = report.timeline[0];
  return first ? timelineStateToLayerSnapshot(first) : null;
}
