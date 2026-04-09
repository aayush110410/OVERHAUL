import { useEffect, useRef, useState, startTransition } from 'react';
import type { FormEvent } from 'react';
import type { VisualizationMode, PerformanceMetrics, LayerConfig } from '../core/types';
import { Engine } from '../core/Engine';
import { Demo4ApiClient } from '../demo4/api';
import {
  DEMO4_API_BASE_URL,
  DEMO4_LAYER_CONFIGS,
  applySnapshotToLayerState,
  simulationReportToLayerSnapshot,
  timelineStateToLayerSnapshot,
} from '../demo4/defaults';
import type {
  Demo4AgentTrace,
  Demo4LayerState,
  Demo4OrchestrationResponse,
  Demo4SimulationReport,
  Demo4TimelineState,
  Demo4WorldSnapshot,
} from '../demo4/types';
import { CodexLayerPanel } from '../demo4/components/CodexLayerPanel';
import { CodexHud } from '../demo4/components/CodexHud';
import { CodexTimeline } from '../demo4/components/CodexTimeline';
import { CodexShaderPanel } from '../demo4/components/CodexShaderPanel';
import './Demo4CodexPage.css';

const apiClient = new Demo4ApiClient(DEMO4_API_BASE_URL);

const EMPTY_METRICS: PerformanceMetrics = {
  fps: 60,
  frameTime: 16.7,
  gpuTime: 0,
  cpuTime: 0,
  objectCount: 0,
  visibleObjects: 0,
  drawCalls: 0,
  triangles: 0,
  textureMemory: 0,
  geometryMemory: 0,
};

const FALLBACK_FOCUS = {
  name: 'Tower Bridge, London',
  longitude: -0.0754,
  latitude: 51.5055,
  altitude: 14000,
};

export default function Demo4CodexPage() {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const engineRef = useRef<Engine | null>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const monitoringIntervalRef = useRef<number | null>(null);

  const [layers, setLayers] = useState<Demo4LayerState[]>(DEMO4_LAYER_CONFIGS);
  const [snapshot, setSnapshot] = useState<Demo4WorldSnapshot | null>(null);
  const [metrics, setMetrics] = useState<PerformanceMetrics>(EMPTY_METRICS);
  const [focusQuery, setFocusQuery] = useState(FALLBACK_FOCUS.name);
  const [prompt, setPrompt] = useState('Reduce congestion near Tower Bridge');
  const [report, setReport] = useState<Demo4SimulationReport | null>(null);
  const [timeline, setTimeline] = useState<Demo4TimelineState[]>([]);
  const [timelineIndex, setTimelineIndex] = useState(0);
  const [agentTrace, setAgentTrace] = useState<Demo4AgentTrace[]>([]);
  const [connectionState, setConnectionState] = useState<'connecting' | 'live' | 'offline'>('connecting');
  const [visualizationMode, setVisualizationMode] = useState<VisualizationMode>('standard');
  const [shaderSettings, setShaderSettings] = useState<Record<string, number>>({
    exposure: 1,
    bloomStrength: 0,
    contrast: 1,
    sharpness: 0,
    glowIntensity: 0,
    fogNear: 1000,
    fogFar: 50000,
  });
  const [cameraState, setCameraState] = useState<{ latitude: number; longitude: number; altitude: number } | null>(null);
  const [adaptiveNote, setAdaptiveNote] = useState('Adaptive budget nominal');
  const [busyState, setBusyState] = useState<'idle' | 'bootstrapping' | 'running'>('bootstrapping');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function init() {
      try {
        await ensureEngine();
        if (cancelled) return;
        await refreshBootstrap(focusQuery, true);
        if (cancelled) return;
        connectSocket();
        startMonitoring();
        setBusyState('idle');
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
        setBusyState('idle');
      }
    }

    init();

    return () => {
      cancelled = true;
      if (monitoringIntervalRef.current) {
        window.clearInterval(monitoringIntervalRef.current);
      }
      if (socketRef.current) {
        socketRef.current.close();
        socketRef.current = null;
      }
      engineRef.current?.destroy();
      engineRef.current = null;
    };
  }, []);

  async function ensureEngine() {
    if (engineRef.current || !containerRef.current) return;

    const engine = new Engine({
      container: containerRef.current,
      cesiumToken: import.meta.env.VITE_CESIUM_TOKEN,
      terrainProvider: import.meta.env.VITE_CESIUM_TOKEN ? 'cesium-world' : 'ellipsoid',
      imageryProvider: 'osm',
      enableLighting: true,
      enableAtmosphere: true,
      enableFog: true,
      msaaSamples: 4,
      initialView: {
        longitude: FALLBACK_FOCUS.longitude,
        latitude: FALLBACK_FOCUS.latitude,
        altitude: FALLBACK_FOCUS.altitude,
      },
    });

    await engine.init();
    for (const layer of DEMO4_LAYER_CONFIGS) {
      await engine.getLayers().addLayer(layer);
    }
    engine.start();
    engineRef.current = engine;
    setShaderSettings(engine.getShaders().getSettings());
  }

  async function refreshBootstrap(nextFocus: string, focusCamera = false) {
    setBusyState('bootstrapping');
    setError(null);
    try {
      const bootstrap = await apiClient.bootstrap({ focusName: nextFocus });
      applySnapshot(bootstrap.snapshot, focusCamera);
      setConnectionState('live');
      setBusyState('idle');
    } catch (err) {
      setConnectionState('offline');
      setBusyState('idle');
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  function applySnapshot(nextSnapshot: Demo4WorldSnapshot, focusCamera = false) {
    const engine = engineRef.current;
    if (!engine) return;

    for (const [layerId, layerSnapshot] of Object.entries(nextSnapshot.layers)) {
      if (engine.getLayers().getLayer(layerId)) {
        engine.getLayers().applyLayerData(layerId, layerSnapshot);
      }
    }

    if (focusCamera) {
      engine.getCamera().setMode('cinematic');
      engine.getCamera().flyTo(
        {
          longitude: nextSnapshot.focus.longitude,
          latitude: nextSnapshot.focus.latitude,
          altitude: nextSnapshot.focus.altitude,
        },
        2.6,
      );
    }

    const timelineFromSnapshot = nextSnapshot.layers.simulation?.meta?.timeline as Demo4TimelineState[] | undefined;
    startTransition(() => {
      setSnapshot(nextSnapshot);
      setLayers((current) => applySnapshotToLayerState(current, nextSnapshot.layers));
      if (!report && timelineFromSnapshot?.length) {
        setTimeline(timelineFromSnapshot);
        setTimelineIndex(0);
      }
    });
  }

  function connectSocket() {
    if (socketRef.current) return;
    socketRef.current = apiClient.connectToStream(
      (nextSnapshot) => applySnapshot(nextSnapshot, false),
      (status) => setConnectionState(status),
    );
  }

  function startMonitoring() {
    if (monitoringIntervalRef.current) return;
    monitoringIntervalRef.current = window.setInterval(() => {
      const engine = engineRef.current;
      if (!engine) return;

      const nextMetrics = engine.getMetrics();
      const nextCamera = engine.getCamera().getState();
      const nextLayerStats = engine.getLayers().getStats();

      setMetrics(nextMetrics);
      setCameraState({
        latitude: nextCamera.position.latitude,
        longitude: nextCamera.position.longitude,
        altitude: nextCamera.position.altitude,
      });
      setLayers((current) =>
        current.map((layer) => {
          const stats = nextLayerStats.get(layer.id);
          return stats
            ? {
                ...layer,
                objectCount: Math.max(layer.objectCount, stats.objectCount),
                updateFrequencyHz: stats.updateFrequencyHz ?? layer.updateFrequencyHz,
                source: stats.source ?? layer.source,
              }
            : layer;
        }),
      );

      const viewer = engine.getGlobe().viewer;
      const currentShaderSettings = engine.getShaders().getSettings();
      if (nextMetrics.fps < 58) {
        viewer.resolutionScale = Math.max(0.72, viewer.resolutionScale - 0.04);
        engine.getShaders().setBloom(
          Math.min(currentShaderSettings.bloomStrength ?? 0.3, 0.45),
          currentShaderSettings.bloomRadius ?? 0.35,
          currentShaderSettings.bloomThreshold ?? 0.5,
        );
        setAdaptiveNote(`Adaptive quality engaged at ${Math.round(nextMetrics.fps)} FPS`);
      } else if (nextMetrics.fps >= 60 && viewer.resolutionScale < 1) {
        viewer.resolutionScale = Math.min(1, viewer.resolutionScale + 0.02);
        setAdaptiveNote(`Maintaining ${Math.round(nextMetrics.fps)} FPS target`);
      }
    }, 500);
  }

  async function handleRunLdrago(event: FormEvent) {
    event.preventDefault();
    setBusyState('running');
    setError(null);
    try {
      const response: Demo4OrchestrationResponse = await apiClient.orchestrate({
        prompt,
        focus_name: focusQuery,
        years: [0, 1, 5],
      });
      const nextReport = response.simulation;
      const layerSnapshot = simulationReportToLayerSnapshot(nextReport);
      setReport(nextReport);
      setTimeline(nextReport.timeline);
      setTimelineIndex(0);
      setAgentTrace(response.agent_trace);
      if (layerSnapshot && engineRef.current?.getLayers().getLayer('simulation')) {
        engineRef.current.getLayers().applyLayerData('simulation', layerSnapshot);
      }
      if (
        typeof response.parsed_location.longitude === 'number'
        && typeof response.parsed_location.latitude === 'number'
      ) {
        engineRef.current?.getCamera().cinematicZoom(
          cameraState
            ? {
                longitude: cameraState.longitude,
                latitude: cameraState.latitude,
                altitude: cameraState.altitude,
              }
            : {
                longitude: FALLBACK_FOCUS.longitude,
                latitude: FALLBACK_FOCUS.latitude,
                altitude: FALLBACK_FOCUS.altitude,
              },
          {
            longitude: response.parsed_location.longitude as number,
            latitude: response.parsed_location.latitude as number,
            altitude: Number(response.parsed_location.altitude ?? 12000),
          },
          4,
        );
      }
      setBusyState('idle');
    } catch (err) {
      setBusyState('idle');
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  function handleTimelineChange(index: number) {
    setTimelineIndex(index);
    const step = timeline[index];
    if (!step || !engineRef.current?.getLayers().getLayer('simulation')) return;
    engineRef.current.getLayers().applyLayerData('simulation', timelineStateToLayerSnapshot(step));
  }

  async function handleToggleLoaded(id: string) {
    const engine = engineRef.current;
    if (!engine) return;
    const target = layers.find((layer) => layer.id === id);
    if (!target) return;

    if (target.loaded) {
      await engine.getLayers().removeLayer(id);
      setLayers((current) =>
        current.map((layer) => (layer.id === id ? { ...layer, loaded: false, visible: false } : layer)),
      );
      return;
    }

    const config = DEMO4_LAYER_CONFIGS.find((layer) => layer.id === id);
    if (!config) return;

    await engine.getLayers().addLayer(config as LayerConfig);
    const layerSnapshot = snapshot?.layers[id as keyof Demo4WorldSnapshot['layers']];
    if (layerSnapshot) {
      engine.getLayers().applyLayerData(id, layerSnapshot);
    }
    setLayers((current) =>
      current.map((layer) => (layer.id === id ? { ...layer, loaded: true, visible: true } : layer)),
    );
  }

  function handleToggleVisible(id: string) {
    const engine = engineRef.current;
    if (!engine) return;
    setLayers((current) =>
      current.map((layer) => {
        if (layer.id !== id) return layer;
        const nextVisible = !layer.visible;
        engine.getLayers().setLayerVisible(id, nextVisible);
        return { ...layer, visible: nextVisible };
      }),
    );
  }

  function handleOpacityChange(id: string, opacity: number) {
    const engine = engineRef.current;
    if (!engine) return;
    engine.getLayers().setLayerOpacity(id, opacity);
    setLayers((current) =>
      current.map((layer) => (layer.id === id ? { ...layer, opacity } : layer)),
    );
  }

  function handleModeChange(mode: VisualizationMode) {
    const engine = engineRef.current;
    if (!engine) return;
    engine.getShaders().setPreset(mode);
    setVisualizationMode(mode);
    setShaderSettings(engine.getShaders().getSettings());
  }

  function handleShaderChange(setting: string, value: number) {
    const engine = engineRef.current;
    if (!engine) return;
    const shaderPipeline = engine.getShaders();
    const nextSettings = { ...shaderSettings, [setting]: value };
    setShaderSettings(nextSettings);

    if (setting === 'exposure') shaderPipeline.setToneMappingExposure(value);
    if (setting === 'bloomStrength') shaderPipeline.setBloom(value, nextSettings.bloomRadius ?? 0.4, nextSettings.bloomThreshold ?? 0.85);
    if (setting === 'contrast') shaderPipeline.setContrast(value, nextSettings.brightness ?? 0);
    if (setting === 'sharpness') shaderPipeline.setSharpness(value);
    if (setting === 'glowIntensity') shaderPipeline.setGlow(value);
    if (setting === 'fogFar') shaderPipeline.setFog(nextSettings.fogNear ?? 1000, value);
  }

  function handleCameraMode(mode: 'orbit' | 'pan' | 'tilt' | 'cinematic') {
    const engine = engineRef.current;
    if (!engine) return;
    engine.getCamera().setMode(mode);
    if (mode === 'cinematic' && snapshot) {
      engine.getCamera().flyTo(
        {
          longitude: snapshot.focus.longitude,
          latitude: snapshot.focus.latitude,
          altitude: snapshot.focus.altitude,
        },
        2.4,
      );
    }
  }

  const activeDatasets = layers.filter((layer) => layer.loaded && layer.visible).length;

  return (
    <div className="demo4-shell">
      <div className="demo4-header">
        <div>
          <p className="demo4-eyebrow">OVERHAUL / Demo 4 Codex</p>
          <h1>Planet-scale rendering and simulation command page</h1>
        </div>
        <div className="demo4-header-meta">
          <span>{adaptiveNote}</span>
          <span>{connectionState}</span>
          <span>{Math.round(metrics.fps)} FPS</span>
        </div>
      </div>

      <div className="demo4-layout">
        <aside className="demo4-column">
          <CodexLayerPanel
            layers={layers}
            onToggleLoaded={handleToggleLoaded}
            onToggleVisible={handleToggleVisible}
            onOpacityChange={handleOpacityChange}
          />

          <section className="demo4-card">
            <div className="demo4-section-head">
              <span>Camera</span>
              <span>controls</span>
            </div>
            <div className="demo4-mode-grid">
              {(['orbit', 'pan', 'tilt', 'cinematic'] as const).map((mode) => (
                <button key={mode} type="button" className="demo4-mode-btn" onClick={() => handleCameraMode(mode)}>
                  {mode}
                </button>
              ))}
            </div>
          </section>

          <CodexShaderPanel
            mode={visualizationMode}
            settings={shaderSettings}
            onModeChange={handleModeChange}
            onSettingChange={handleShaderChange}
          />
        </aside>

        <main className="demo4-globe-stage">
          <div ref={containerRef} className="demo4-globe-canvas" />
          <CodexHud
            focusName={snapshot?.focus.name || FALLBACK_FOCUS.name}
            coordinates={cameraState}
            metrics={metrics}
            activeDatasets={activeDatasets}
            connectionState={connectionState}
          />
          <div className="demo4-overview">
            <span>{layers.reduce((sum, layer) => sum + layer.objectCount, 0).toLocaleString()} streamed objects</span>
            <span>{metrics.drawCalls.toLocaleString()} draw calls</span>
            <span>{Math.round(metrics.triangles / 1000)}K triangles</span>
          </div>
          {error && <div className="demo4-error">{error}</div>}
        </main>

        <aside className="demo4-column">
          <section className="demo4-card">
            <div className="demo4-section-head">
              <span>LDRAGO orchestration</span>
              <span>{busyState}</span>
            </div>

            <form className="demo4-form" onSubmit={handleRunLdrago}>
              <label>
                <span>Focus area</span>
                <input
                  type="text"
                  value={focusQuery}
                  onChange={(event) => setFocusQuery(event.target.value)}
                  placeholder="Tower Bridge, London"
                />
              </label>

              <label>
                <span>Prompt</span>
                <textarea
                  rows={5}
                  value={prompt}
                  onChange={(event) => setPrompt(event.target.value)}
                  placeholder="Reduce congestion near Tower Bridge"
                />
              </label>

              <div className="demo4-form-actions">
                <button type="button" className="demo4-primary ghost" onClick={() => refreshBootstrap(focusQuery, true)}>
                  Focus scene
                </button>
                <button type="submit" className="demo4-primary" disabled={busyState !== 'idle'}>
                  Run LDRAGO
                </button>
              </div>
            </form>
          </section>

          <section className="demo4-card">
            <div className="demo4-section-head">
              <span>Impact report</span>
              <span>{report ? report.active_models.join(', ') : 'Waiting'}</span>
            </div>

            <p className="demo4-summary">
              {report?.summary || 'Run an orchestration pass to generate a before/after report, timeline, and infrastructure animation.'}
            </p>

            {report && (
              <div className="demo4-metric-grid">
                <div className="demo4-metric-card">
                  <span>Traffic improvement</span>
                  <strong>+{report.traffic_improvement_pct.toFixed(1)}%</strong>
                </div>
                <div className="demo4-metric-card">
                  <span>Travel time delta</span>
                  <strong>-{report.travel_time_delta_pct.toFixed(1)}%</strong>
                </div>
                <div className="demo4-metric-card">
                  <span>Congestion delta</span>
                  <strong>-{report.congestion_delta_pct.toFixed(1)}%</strong>
                </div>
                <div className="demo4-metric-card">
                  <span>Pollution delta</span>
                  <strong>-{report.pollution_delta_pct.toFixed(1)}%</strong>
                </div>
              </div>
            )}

            <div className="demo4-list">
              {(report?.recommendations || []).map((recommendation) => (
                <div key={recommendation} className="demo4-list-item">{recommendation}</div>
              ))}
            </div>
          </section>

          <section className="demo4-card">
            <div className="demo4-section-head">
              <span>Agent trace</span>
              <span>{agentTrace.length}</span>
            </div>
            <div className="demo4-list">
              {agentTrace.length === 0 && (
                <div className="demo4-list-item">Qwen, Llama, and Gemini traces will appear here after orchestration.</div>
              )}
              {agentTrace.map((trace) => (
                <div key={`${trace.agent}-${trace.model}`} className="demo4-trace">
                  <div className="demo4-trace-head">
                    <strong>{trace.agent}</strong>
                    <span>{trace.model}</span>
                    <span>{Math.round(trace.confidence * 100)}%</span>
                  </div>
                  <p>{trace.summary}</p>
                </div>
              ))}
            </div>
          </section>
        </aside>
      </div>

      <CodexTimeline
        timeline={timeline}
        currentIndex={timelineIndex}
        report={report}
        onChange={handleTimelineChange}
      />
    </div>
  );
}
