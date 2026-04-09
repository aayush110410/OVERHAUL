# Demo 4 Codex Architecture

## Frontend folder structure

```text
rendering-engine/
  src/
    App.tsx
    core/
      Engine.ts
      Globe.ts
      SceneManager.ts
      types.ts
      constants.ts
    layers/
      BaseLayer.ts
      LayerManager.ts
      SatelliteLayer.ts
      FlightLayer.ts
      TrafficLayer.ts
      WeatherLayer.ts
      SimulationLayer.ts
      BuildingLayer.ts
      IntelligenceLayer.ts
    shaders/
      ShaderPipeline.ts
      glsl/
        bloom.frag
        contrast.frag
        fog.frag
        glow.frag
        passthrough.vert
        sharpness.frag
    demo4/
      api.ts
      defaults.ts
      types.ts
      components/
        CodexHud.tsx
        CodexLayerPanel.tsx
        CodexShaderPanel.tsx
        CodexTimeline.tsx
    pages/
      Demo4CodexPage.tsx
      Demo4CodexPage.css
```

## Backend folder structure

```text
services/
  rendering_demo_codex/
    __init__.py
    app.py
    config.py
    schemas.py
    repository.py
    providers.py
    simulation.py
    ldrago.py
    requirements.txt
```

## Frontend architecture

- `Demo4CodexPage.tsx` owns the page shell, LDRAGO workflow, websocket subscription, timeline state, HUD state, and adaptive quality loop.
- `Engine.ts` still orchestrates Cesium, Three.js, camera control, layer updates, and post-processing.
- `LayerManager.ts` now accepts live snapshot payloads with `applyLayerData`, so streamed datasets stay modular.
- `TrafficLayer.ts` was upgraded to a shader-driven GPU particle system over streamed OSM road segments.
- `SimulationLayer.ts` now renders time-stepped heatmap points, flow vectors, and infrastructure geometry.
- `ShaderPipeline.ts` now exposes live controls for HDR exposure, bloom, contrast, sharpness, glow, and fog depth.

## Backend architecture

- `providers.py` ingests Celestrak, OpenSky, USGS, Open-Meteo, and Overpass data, normalizes everything into globe-friendly payloads, and emits unified `WorldSnapshot` objects.
- `repository.py` persists snapshots and simulation runs with PostGIS-aware SQL when PostgreSQL is configured, and falls back to in-memory or SQLite-friendly storage locally.
- `simulation.py` builds a `NetworkX` road graph, applies interventions, computes congestion/travel-time states, and emits timeline geometry for animation.
- `ldrago.py` parses prompts, extracts locations, selects models, coordinates Qwen/Llama/Gemini style agent traces, and returns visualization commands plus simulation output.
- `app.py` exposes bootstrap, simulation, orchestration, and websocket endpoints for the new page.

## Main page route

- Default route: `rendering-engine/#demo-4-codex`
