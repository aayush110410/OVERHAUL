/* ──────────────────────────────────────────────────────────────────────────
 * OVERHAUL — GlobeView Component
 *
 * Main React component that mounts the Cesium + Three.js rendering
 * engine and manages its lifecycle.
 * ────────────────────────────────────────────────────────────────────────── */

import { useEffect, useRef, useCallback } from 'react';
import { Engine } from '../core/Engine';
import { useEngineStore } from '../store/engineStore';
import { initializeAPIClient, getAPIClient } from '../api/client';
import type { EngineConfig, LayerConfig } from '../core/types';

interface GlobeViewProps {
  cesiumToken?: string;
  onEngineReady?: (engine: Engine) => void;
}

// Default layers to load on start
const DEFAULT_LAYERS: LayerConfig[] = [
  { id: 'satellites', type: 'satellite',  name: 'Satellites',  visible: true,  opacity: 0.85, zIndex: 10 },
  { id: 'flights',    type: 'flight',     name: 'Flights',     visible: true,  opacity: 0.9,  zIndex: 20 },
  { id: 'traffic',    type: 'traffic',    name: 'Traffic',     visible: false, opacity: 0.8,  zIndex: 30 },
  { id: 'weather',    type: 'weather',    name: 'Weather',     visible: false, opacity: 0.5,  zIndex: 5  },
  { id: 'simulation', type: 'simulation', name: 'Simulation',  visible: false, opacity: 0.6,  zIndex: 40 },
  { id: 'buildings',  type: 'building',   name: '3D Buildings', visible: true,  opacity: 0.85, zIndex: 35 },
];

export function GlobeView({ cesiumToken, onEngineReady }: GlobeViewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const engineRef = useRef<Engine | null>(null);
  const {
    addLayerConfig,
    setMetrics,
    visualizationMode,
    cameraMode,
    layers,
    apiBaseURL,
    setConnected,
    selectedLocation,
    setLiveAQI,
  } = useEngineStore();

  /* ── Init engine ───────────────────────────────────────────────────── */

  useEffect(() => {
    if (!containerRef.current) return;

    // Initialize API client
    const apiClient = initializeAPIClient(apiBaseURL);

    // Check API connection
    (async () => {
      try {
        await apiClient.health();
        setConnected(true);

        // Load live AQI data for selected location
        const aqi = await apiClient.getLiveAQI(selectedLocation);
        setLiveAQI(aqi);
      } catch (error) {
        console.warn('API connection failed, running in demo mode:', error);
        setConnected(false);
        // Use fallback data
        setLiveAQI({
          aqi: Math.floor(Math.random() * 150) + 50,
          pm25: Math.random() * 50 + 20,
          pm10: Math.random() * 80 + 40,
        });
      }
    })();

    const config: EngineConfig = {
      container: containerRef.current,
      cesiumToken,
      terrainProvider: cesiumToken ? 'cesium-world' : 'ellipsoid',
      imageryProvider: 'osm',
      enableLighting: true,
      enableAtmosphere: true,
      enableFog: true,
      msaaSamples: 4,
      debugMode: false,
    };

    try {
      const engine = new Engine(config);
      engineRef.current = engine;

      engine.init().then(async () => {
        // Load default layers
        const layerMgr = engine.getLayers();
        for (const lc of DEFAULT_LAYERS) {
          try {
            await layerMgr.addLayer(lc);
            addLayerConfig(lc);
          } catch (e) {
            console.warn(`Failed to load layer ${lc.id}:`, e);
          }
        }

        // Start render loop
        engine.start();

        // Metrics polling
        const metricsInterval = setInterval(() => {
          if (engine.state.running) {
            setMetrics(engine.getMetrics());
          }
        }, 500);

        onEngineReady?.(engine);

        // Cleanup on unmount will clear this interval
        (engine as unknown as Record<string, unknown>)._metricsInterval = metricsInterval;
      }).catch((err) => {
        console.error('Engine initialization failed:', err);
      });
    } catch (err) {
      console.error('Engine creation failed:', err);
    }

    return () => {
      const eng = engineRef.current;
      if (eng) {
        const interval = (eng as unknown as Record<string, unknown>)._metricsInterval as ReturnType<typeof setInterval> | undefined;
        if (interval) clearInterval(interval);
        eng.destroy();
        engineRef.current = null;
      }
    };
  }, []);

  /* ── Sync visualization mode ────────────────────────────────────────── */

  useEffect(() => {
    engineRef.current?.getShaders().setPreset(visualizationMode);
  }, [visualizationMode]);

  /* ── Sync camera mode ───────────────────────────────────────────────── */

  useEffect(() => {
    engineRef.current?.getCamera().setMode(cameraMode);
  }, [cameraMode]);

  /* ── Sync layer visibility / opacity ────────────────────────────────── */

  useEffect(() => {
    const engine = engineRef.current;
    if (!engine) return;
    const mgr = engine.getLayers();
    for (const lc of layers) {
      mgr.setLayerVisible(lc.id, lc.visible);
      mgr.setLayerOpacity(lc.id, lc.opacity);
    }
  }, [layers]);

  return (
    <div
      ref={containerRef}
      style={{
        width: '100%',
        height: '100%',
        position: 'relative',
        overflow: 'hidden',
        background: '#0a0a0a',
      }}
    />
  );
}
