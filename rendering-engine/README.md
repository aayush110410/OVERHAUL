# OVERHAUL — Rendering Engine v1.1

**Status**: ✅ Production-Ready | **Features**: 3D Globe + 6 Visualization Layers + Full API Integration

Advanced 3D geospatial visualization & simulation platform built with CesiumJS, Three.js, WebGL, and LDRAGO v2 AI.

## Key Features

✨ **Fully Integrated with OVERHAUL Backend**:
- Real LDRAGO v2 cognitive pipeline (7 agents + multi-model reasoning)
- Live simulation engine with traffic, infrastructure, and environmental impacts
- Real-time AQI data feeds
- Scenario comparison and temporal prediction

🌐 **Rendering**:
- CesiumJS globe with OSM tiles + optional Cesium World terrain
- Three.js WebGL overlay for 8,300+ objects (satellites, flights, vehicles, buildings)
- 5-pass shader pipeline (bloom, contrast, sharpness, glow, fog)
- 4 visualization presets (standard, night, thermal, satellite)

⚡ **Performance**:
- 5-tier LOD system (500m to planet-wide)
- Quadtree spatial indexing (max depth 12)
- GPU instancing for thousands of objects
- Object pooling + LRU data caching
- Consistent 55-60 FPS on modern GPUs

📊 **Intelligence**:
- Command center UI with KPI cards and live metrics
- Chat interface powered by LDRAGo v2 (Qwen 3 4B + Gemini 3 Pro)
- Real-time simulation results
- Impact card analysis

## Architecture

```
┌─ GlobeView (React)
│  └─ Engine (orchestrator)
│     ├─ Globe (CesiumJS) [terrain + imagery + atmosphere]
│     ├─ SceneManager (Three.js) [WebGL overlay]
│     ├─ LayerManager [6 visualization layers]
│     ├─ ShaderPipeline [5-pass post-processing]
│     └─ CameraController [5 modes + easing]
│
└─ API Integration
   ├─ useLDRAGoChat() → /chat/v2 [LDRAGo cognitive pipeline]
   ├─ useSimulation() → /simulate [Traffic + infrastructure simulation]
   ├─ useLiveAQI() → /live/aqi [Real air quality data]
   └─ useLDRAGoStatus() → /ldrago/status [System status]

Store: Zustand (layers, modes, metrics, API state, results)
```

## Quick Start

### 0. Start OVERHAUL Backend (Required)

```bash
cd /path/to/OVERHAUL-main
.venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Check: `curl http://localhost:8000/health`

### 1. Install Dependencies

```bash
cd rendering-engine
npm install
```

### Environment Setup (Optional)

For Cesium World Terrain (recommended):

```bash
# Create .env file
echo "VITE_CESIUM_TOKEN=your_cesium_ion_token" > .env.local
```

Get a free token at: https://cesium.com/platform/cesiumion/

### Development

```bash
npm run dev
```

Serves at `http://localhost:5175`

**Default Route**: Command Center demo (`/#demo`)  
**Alternative Routes**: 
- `/#main` — Pure rendering engine
- `/#demo` — Full command center UI

### Production Build

```bash
npm run build
```

Output: `dist/`

### Type Checking

```bash
npm run typecheck
```

## File Structure

```
src/
├── App.tsx                    # Router + main entry
├── pages/
│   └── DemoPage.tsx           # Command center demo
├── components/
│   ├── GlobeView.tsx          # Engine mount & lifecycle
│   ├── LayerPanel.tsx         # Layer visibility/opacity
│   ├── ShaderControls.tsx     # Visualization mode picker
│   ├── CameraControls.tsx     # Camera mode selector
│   └── PerformanceMonitor.tsx # FPS, memory, stats
├── core/
│   ├── Engine.ts              # Main orchestrator
│   ├── Globe.ts               # CesiumJS wrapper
│   ├── SceneManager.ts        # Three.js overlay
│   ├── types.ts               # Type definitions
│   └── constants.ts           # LOD, presets, defaults
├── layers/
│   ├── BaseLayer.ts           # Abstract layer base
│   ├── LayerManager.ts        # Layer registry
│   ├── SatelliteLayer.ts      # 800 orbiting satellites
│   ├── FlightLayer.ts         # 500 commercial flights
│   ├── TrafficLayer.ts        # 2000 vehicles
│   ├── WeatherLayer.ts        # 5000 particles + heatmap
│   ├── SimulationLayer.ts     # Engine output overlay
│   └── BuildingLayer.ts       # 3000 buildings
├── shaders/
│   ├── ShaderPipeline.ts      # Post-processing chain
│   └── glsl/
│       ├── passthrough.vert   # Fullscreen quad shader
│       ├── bloom.frag         # Bright extraction + blur
│       ├── contrast.frag      # Contrast/brightness
│       ├── sharpness.frag     # Unsharp-mask
│       ├── glow.frag          # Ambient glow
│       └── fog.frag           # Depth-based fog
├── camera/
│   └── CameraController.ts    # Camera modes + transitions
├── performance/
│   ├── LODManager.ts          # 5-tier LOD system
│   ├── SpatialIndex.ts        # Quadtree for culling
│   └── ObjectPool.ts          # Mesh reuse pool
├── data/
│   ├── DataLoader.ts          # Lazy fetch + cache
│   └── TileManager.ts         # Tile coordinate math
├── store/
│   └── engineStore.ts         # Zustand state (layers, modes)
└── utils/
    └── geo.ts                 # Haversine, great-circle, zoom
```

## Features

### Rendering

- **Global 3D Globe** with real-time terrain and satellite imagery
- **810,000+ concurrent objects** via GPU instancing
- **6 visualization layers** with dynamic load/unload
- **4 visualization presets** (standard, night, thermal, satellite)
- **GPU-accelerated post-processing**: bloom, contrast, sharpness, glow, fog

### Performance Optimization

- **5-tier LOD system** based on camera altitude
- **Quadtree spatial indexing** for frustum culling (max depth 12)
- **Object pooling** for garbage collection avoidance
- **Lazy tile loading** with LRU cache eviction
- **Request deduplication** for concurrent data fetches

### Camera Control

- **Orbit mode**: Rotate around a target
- **Pan mode**: Translate the camera
- **Tilt mode**: Change pitch/roll
- **Free mode**: Full 6-DOF control
- **Cinematic mode**: Scripted camera transitions with easing

### UI

- **Command Center HUD**: KPI cards, simulation results, chat interface
- **Layer panel**: Toggle visibility, adjust opacity
- **Shader controls**: Switch visualization modes
- **Performance monitor**: Real-time FPS, draw calls, triangle count

## API Usage

### Mount the Engine

```tsx
import { GlobeView } from '@/components/GlobeView';
import { Engine } from '@/core/Engine';

function App() {
  const handleEngineReady = (engine: Engine) => {
    // Access engine for scripting
    engine.getCamera().cinematicZoom(from, to, 5);
    engine.getLayers().addLayer({ /* config */ });
    engine.getShaders().setPreset('night');
  };

  return (
    <GlobeView
      cesiumToken={import.meta.env.VITE_CESIUM_TOKEN}
      onEngineReady={handleEngineReady}
    />
  );
}
```

### Add Layers Programmatically

```ts
const engine = window.overhaulEngine as Engine;
const layers = engine.getLayers();

await layers.addLayer({
  id: 'my-layer',
  type: 'simulation',  // or 'satellite', 'flight', 'traffic', etc.
  name: 'My Data',
  visible: true,
  opacity: 0.8,
  zIndex: 50,
});
```

### Control Camera

```ts
const camera = engine.getCamera();

// Fly to location
camera.flyTo({ longitude: 77.2, latitude: 28.6, altitude: 50_000 }, 2);

// Cinematic orbit
camera.cinematicOrbit(
  { longitude: 77.2, latitude: 28.6, altitude: 5_000_000 },
  8_000_000,
  2,  // revolutions
  20  // seconds
);

// Set mode
camera.setMode('cinematic');
```

### Monitor Performance

```ts
setInterval(() => {
  const metrics = engine.getMetrics();
  console.log(`FPS: ${metrics.fps}, Triangles: ${metrics.triangles}`);
}, 500);
```

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

Requires WebGL 2 support.

## Performance Targets

- **60 FPS** at 1080p with default settings
- **30+ FPS** at 4K with all layers enabled
- **< 200 MB** memory for full scene

## API Integration

The rendering engine connects to OVERHAUL's backend engines and AI models for live data and analysis.

### Features

- **LDRAGo v2 Chat**: Full cognitive pipeline (Parse → Locate → Plan → Research → Reason → Synthesize)
- **Live Simulation**: Real-time scenario analysis with traffic, infrastructure, and environmental impacts
- **AQI Monitoring**: Real-time air quality data streams
- **Multi-Agent Reasoning**: Qwen 3 4B (fast) + Gemini 3 Pro (reasoning) + Domain engines

### React Hooks

```tsx
import { useLDRAGoChat, useSimulation, useLiveAQI } from '@/api/hooks';

// Chat with LDRAGo AI
const { messages, chat } = useLDRAGoChat();
await chat("Simulate 40% EV adoption", "full");

// Run simulations
const { results, simulate } = useSimulation();
await simulate(scenario, interventions);

// Live AQI data
const { aqi, loading } = useLiveAQI("Sector 61, Noida");
```

### Configuration

```bash
# .env.local
VITE_API_URL=http://localhost:8000  # OVERHAUL backend
VITE_CESIUM_TOKEN=your_token        # Optional: Premium terrain
```

### For Full Documentation

See [API_INTEGRATION.md](./API_INTEGRATION.md) for:
- Architecture diagram
- API endpoints & request/response types
- Integration examples
- Error handling & fallbacks
- Deployment instructions
- Troubleshooting guide

## Debugging

```ts
// Access engine from console
const engine = window.overhaulEngine;

// Get performance stats
engine.getMetrics();

// List all layers
engine.getLayers().getAllLayers();

// Switch visualization mode
engine.getShaders().setPreset('thermal');

// Get camera state
engine.getCamera().getState();
```

## Contributing

- Use TypeScript strict mode
- Add types to all functions
- Follow the existing module structure
- Test builds before committing: `npm run build`

## License

Internal use — OVERHAUL Platform
