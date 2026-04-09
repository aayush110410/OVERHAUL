import type { LayerConfig, VisualizationMode } from '../core/types';

export type Demo4LayerId =
  | 'satellites'
  | 'flights'
  | 'traffic'
  | 'weather'
  | 'simulation'
  | 'buildings'
  | 'intelligence';

export interface Demo4GeoPoint {
  longitude: number;
  latitude: number;
  altitude: number;
}

export interface Demo4DatasetRecord {
  id: string;
  position?: Demo4GeoPoint;
  path?: { coordinates: Demo4GeoPoint[] };
  properties: Record<string, unknown>;
}

export interface Demo4LayerStats {
  object_count: number;
  update_frequency_hz: number;
  active: boolean;
  source: string;
  last_updated: string;
}

export interface Demo4LayerSnapshot {
  layer_id: Demo4LayerId;
  label: string;
  kind: 'points' | 'lines' | 'grid' | 'hybrid';
  stats: Demo4LayerStats;
  items: Demo4DatasetRecord[];
  meta: Record<string, unknown>;
}

export interface Demo4WorldSnapshot {
  sequence: number;
  timestamp: string;
  focus: {
    name: string;
    longitude: number;
    latitude: number;
    altitude: number;
    zoom_city?: boolean;
  };
  budget: {
    target_fps: number;
    preferred_resolution_scale: number;
    max_particles: number;
    max_instanced_objects: number;
  };
  layers: Record<Demo4LayerId, Demo4LayerSnapshot>;
}

export interface Demo4TimelineState {
  label: string;
  year_offset: number;
  metrics: Record<string, number>;
  heatmap: Demo4DatasetRecord[];
  flows: Demo4DatasetRecord[];
  infrastructure: Demo4DatasetRecord[];
}

export interface Demo4SimulationReport {
  summary: string;
  traffic_improvement_pct: number;
  travel_time_delta_pct: number;
  congestion_delta_pct: number;
  pollution_delta_pct: number;
  active_models: string[];
  timeline: Demo4TimelineState[];
  recommendations: string[];
}

export interface Demo4AgentTrace {
  agent: string;
  model: string;
  summary: string;
  confidence: number;
}

export interface Demo4OrchestrationResponse {
  prompt: string;
  parsed_location: Record<string, unknown>;
  selected_models: string[];
  agent_trace: Demo4AgentTrace[];
  simulation: Demo4SimulationReport;
  visualization_commands: Record<string, unknown>;
}

export interface Demo4BootstrapResponse {
  snapshot: Demo4WorldSnapshot;
  available_layers: Array<{ id: Demo4LayerId; name: string; type: LayerConfig['type'] }>;
  presets: Array<{ id: VisualizationMode; name: string }>;
  service_status: Record<string, unknown>;
}

export interface Demo4Intervention {
  kind: 'new_road' | 'lane_expansion' | 'flyover' | 'road_closure' | 'signal_optimization';
  name: string;
  coordinates: Demo4GeoPoint[];
  lane_delta?: number;
  capacity_delta?: number;
  speed_delta?: number;
  notes?: string;
}

export interface Demo4SimulationRequest {
  prompt: string;
  focus_name?: string;
  focus?: Demo4GeoPoint;
  years?: number[];
  interventions?: Demo4Intervention[];
}

export interface Demo4LayerState extends LayerConfig {
  loaded: boolean;
  objectCount: number;
  updateFrequencyHz: number;
  source: string;
}
