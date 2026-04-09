/* ──────────────────────────────────────────────────────────────────────────
 * OVERHAUL — Zustand Store
 *
 * Global state for the rendering engine UI.
 * ────────────────────────────────────────────────────────────────────────── */

import { create } from 'zustand';
import type { VisualizationMode, CameraMode, PerformanceMetrics, LayerConfig } from '../core/types';

export interface SimulationResult {
  avgSpeed: number;
  travelTime: number;
  co2Emissions: number;
  congestionEnergy: number;
  aqiImpact: number;
  improvementPercent: number;
  timestamp: number;
}

export interface APIState {
  apiBaseURL: string;
  setAPIBaseURL: (url: string) => void;
  isConnected: boolean;
  setConnected: (connected: boolean) => void;
  lastChatPrompt: string;
  setLastChatPrompt: (prompt: string) => void;
  simulationResults: SimulationResult | null;
  setSimulationResults: (result: SimulationResult | null) => void;
  liveAQI: { value: number; pm25: number; pm10: number } | null;
  setLiveAQI: (aqi: any) => void;
  selectedLocation: string;
  setSelectedLocation: (loc: string) => void;
}

interface EngineStore extends APIState {
  // Layers
  layers: LayerConfig[];
  addLayerConfig: (config: LayerConfig) => void;
  removeLayerConfig: (id: string) => void;
  toggleLayerVisibility: (id: string) => void;
  setLayerOpacity: (id: string, opacity: number) => void;

  // Shaders
  visualizationMode: VisualizationMode;
  setVisualizationMode: (mode: VisualizationMode) => void;

  // Camera
  cameraMode: CameraMode;
  setCameraMode: (mode: CameraMode) => void;

  // Performance
  metrics: PerformanceMetrics;
  setMetrics: (metrics: PerformanceMetrics) => void;
  showPerformance: boolean;
  togglePerformance: () => void;
}

export const useEngineStore = create<EngineStore>((set) => ({
  // API State
  apiBaseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  setAPIBaseURL: (url) => set({ apiBaseURL: url }),
  isConnected: false,
  setConnected: (connected) => set({ isConnected: connected }),
  lastChatPrompt: '',
  setLastChatPrompt: (prompt) => set({ lastChatPrompt: prompt }),
  simulationResults: null,
  setSimulationResults: (result) => set({ simulationResults: result }),
  liveAQI: null,
  setLiveAQI: (aqi) => set({ liveAQI: aqi }),
  selectedLocation: 'Sector 61, Noida',
  setSelectedLocation: (loc) => set({ selectedLocation: loc }),

  // Layers
  layers: [],
  addLayerConfig: (config) =>
    set((s) => ({ layers: [...s.layers, config] })),
  removeLayerConfig: (id) =>
    set((s) => ({ layers: s.layers.filter((l) => l.id !== id) })),
  toggleLayerVisibility: (id) =>
    set((s) => ({
      layers: s.layers.map((l) =>
        l.id === id ? { ...l, visible: !l.visible } : l,
      ),
    })),
  setLayerOpacity: (id, opacity) =>
    set((s) => ({
      layers: s.layers.map((l) =>
        l.id === id ? { ...l, opacity } : l,
      ),
    })),

  // Shaders
  visualizationMode: 'standard',
  setVisualizationMode: (mode) => set({ visualizationMode: mode }),

  // Camera
  cameraMode: 'orbit',
  setCameraMode: (mode) => set({ cameraMode: mode }),

  // Performance
  metrics: {
    fps: 0,
    frameTime: 0,
    gpuTime: 0,
    cpuTime: 0,
    objectCount: 0,
    visibleObjects: 0,
    drawCalls: 0,
    triangles: 0,
    textureMemory: 0,
    geometryMemory: 0,
  },
  setMetrics: (metrics) => set({ metrics }),
  showPerformance: false,
  togglePerformance: () => set((s) => ({ showPerformance: !s.showPerformance })),
}));
