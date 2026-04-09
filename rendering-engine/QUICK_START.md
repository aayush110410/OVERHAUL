# OVERHAUL Rendering Engine — Quick Integration Guide

**TL;DR**: Real AI, real data, real simulations — all in the browser

---

## Demo 4 Codex Quick Start

### One command

```bash
cd /Users/aayushsharma/Desktop/Overhaul/OVERHAUL-main
./scripts/start_demo4_codex.sh
```

### What starts

- Demo 4 backend: `http://127.0.0.1:8014`
- Demo 4 page: `http://127.0.0.1:5175/#demo-4-codex`

### What to test first

1. Click **Focus scene** with `Tower Bridge, London`.
2. Toggle the live layers in the left panel.
3. Switch between `Standard`, `Night`, `Thermal`, and `Satellite` modes.
4. Run `Reduce congestion near Tower Bridge`.
5. Drag the timeline slider through `Current state`, `1 year later`, and `5 years later`.

---

## 60-Second Setup

### 1. Start Backend (Terminal 1)

```bash
cd /Users/aayushsharma/Desktop/Overhaul/OVERHAUL-main
.venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8000

# Should print:
# ✓ LDRAGo Hybrid Brain loaded
# ✓ Engines loaded
# Uvicorn running on 0.0.0.0:8000
```

### 2. Start Renderer (Terminal 2)

```bash
cd rendering-engine
npm run dev

# Should print:
# ✓ vite v6.4.1
# Local: http://localhost:5175/
```

### 3. Open Browser

Go to **http://localhost:5175**

You should see:
- 🌐 3D globe with satellites, flights, buildings
- 💬 Chat input ("Type scenario or command...")
- 📊 KPI cards (Travel Time, PM2.5, VKT, Congestion)
- 🟢 Status indicator (LIVE = connected to backend)

---

## Test the Integration

### Chat Test (Real LDRAGo AI)

1. Type: `"Simulate 40% EV adoption with signal optimization"`
2. Click **"▶ Send"**
3. Watch the chat panel → Assistant response appears after 8-12 seconds
4. Results show real impact predictions

**What's happening**:
- Your prompt sent to `/chat/v2`
- Backend LDRAGo pipeline runs ( 7 agents, 7 engines)
- Response includes impact cards with KPI deltas
- React updates KPI cards with real numbers

### Simulation Test (Real Traffic Model)

1. Click **"Results"** tab
2. Click **"▶ Run Simulation"**
3. Watch the Results tab → Simulation runs in 2-4 seconds
4. View impact cards

**What's happening**:
- Simulation sent to `/simulate` endpoint
- Backend runs BPR traffic model with interventions
- Returns: avg_speed, co2_emissions, travel_time, congestion
- React displays real simulation results

### Status Check

Check **"Analytics"** tab:
- 🤖 Pipeline: ldrago_v2
- 👁️ Agents: 7
- ⚙️ Modes: full / fast / temporal
- 📡 Status: ✅ Connected

If you see 🔴 Demo instead of ✅ Connected:
- Backend is not running or API_URL is wrong
- Check Terminal 1 backend logs
- Verify `.env.local` has `VITE_API_URL=http://localhost:8000`

---

## Code Examples

### Use Chat in Your App

```tsx
import { useLDRAGoChat } from '@/api/hooks';

function MyComponent() {
  const { messages, chat } = useLDRAGoChat();

  const handleAsk = async () => {
    // This calls the real LDRAGo pipeline
    const response = await chat(
      "Simulate EV adoption",
      "full"  // Use "fast" for quick analysis
    );
    
    // response contains:
    // - summary: narrative text
    // - outputs.impactCards: KPI deltas
    // - viz_data: GeoJSON for globe
  };

  return (
    <>
      <button onClick={handleAsk}>Ask AI</button>
      {messages.map((msg) => (
        <div>{msg.role}: {msg.content}</div>
      ))}
    </>
  );
}
```

### Use Simulation in Your App

```tsx
import { useSimulation } from '@/api/hooks';

function SimulationPanel() {
  const { results, simulate } = useSimulation();

  const handleSimulate = async () => {
    // This calls the real traffic simulation engine
    const response = await simulate(
      {
        demand: 1200,
        weather_factor: 0.3,
        od_from: "A",
        od_to: "F",
      },
      [
        { type: "flyover", params: { segments: 3 } },
        { type: "signal", params: { junctions: 700 } },
      ]
    );

    // response.result contains:
    // avg_speed, total_time, co2_emissions, congestion_ratio, aqi_impact
  };

  return (
    <>
      <button onClick={handleSimulate}>Run Simulation</button>
      {results && (
        <div>
          Speed: {results.result.avg_speed} km/h
          CO2: {results.result.co2_emissions} t/day
        </div>
      )}
    </>
  );
}
```

### Monitor Live AQI

```tsx
import { useLiveAQI } from '@/api/hooks';

function AQIMonitor() {
  const { aqi, loading } = useLiveAQI("Sector 61, Noida");

  if (loading) return <p>Loading AQI...</p>;
  if (!aqi) return <p>No data</p>;

  return (
    <div>
      <p>PM2.5: {aqi.pm25.toFixed(1)} µg/m³</p>
      <p>AQI: {aqi.value}</p>
    </div>
  );
}
```

### Direct API Client

```tsx
import { getAPIClient } from '@/api/client';

async function testAPI() {
  const client = getAPIClient();

  // Health check
  const health = await client.health();
  console.log('API Status:', health);

  // Chat
  const chatResp = await client.chatV2({
    prompt: "Test",
    mode: "fast",
  });
  console.log('Chat:', chatResp);

  // Simulation
  const simResp = await client.simulate({
    scenario: { demand: 1200, weather_factor: 0.3 },
  });
  console.log('Sim:', simResp.result);
}
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│              Browser (Localhost:5175)                   │
│                                                         │
│  React Component (DemoPage.tsx)                         │
│         ↓                                               │
│  useLDRAGoChat / useSimulation / useLiveAQI hooks      │
│         ↓                                               │
│  API Client (src/api/client.ts)                        │
│         ↓                                               │
│  HTTP POST/GET to http://localhost:8000                │
│                                                         │
└─────────────────────────────────────────────────────────┘
                        ↓ (Network)
┌─────────────────────────────────────────────────────────┐
│         OVERHAUL Backend (Localhost:8000)               │
│                                                         │
│  Endpoints:                                             │
│  ├─ /chat/v2 → LDRAGo pipeline                         │
│  ├─ /simulate → Traffic simulation                      │
│  ├─ /live/aqi → Real air quality                       │
│  └─ /ldrago/status → System status                     │
│                                                         │
│  Behind the Scenes:                                    │
│  ├─ 7 AI Agents (parser, planner, reasoner, etc)      │
│  ├─ 7 Domain Engines (transport, environment, etc)     │
│  ├─ Live data APIs (OSRM, OpenAQ, etc)                │
│  └─ Simulation models (BPR traffic model, AQI pred)   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Common Tasks

### Display Simulation Results on Globe

```tsx
// In GlobeView or SimulationLayer
const engine = useRef<Engine | null>(null);

const handleShowResults = (geojson) => {
  // Add GeoJSON overlay to globe
  engine.current?.getLayers().addLayer({
    id: 'simulation-results',
    type: 'simulation',
    visible: true,
    data: geojson,
  });
};
```

### Filter Results by Domain

```tsx
const { outputs } = chatResponse;

const transportImpact = outputs.domains.transport;
const environmentImpact = outputs.domains.environment;
const infrastructureImpact = outputs.domains.infrastructure;

// Each domain has impact metrics
```

### Export Chat to Markdown

```tsx
const { messages } = useLDRAGoChat();

const markdown = messages
  .map((m) => `### ${m.role.toUpperCase()}\n\n${m.content}`)
  .join('\n\n');

// Save to file or display
```

### Check System Capacity

```tsx
const { status } = useLDRAGoStatus();

const activeAgents = status.agents.filter((a) => a.status === 'active').length;
const availableModes = status.modes; // ['full', 'fast', 'temporal']

// Use this info to decide on mode selection
if (activeAgents < 5) {
  // Use fast mode
}
```

---

## Troubleshooting

### "API connection failed"
```
✅ Checklist:
- Backend running: curl http://localhost:8000/health
- Port 8000 is accessible
- VITE_API_URL=http://localhost:8000 in .env.local
- No typos in backend URL
```

### Chat returns dummy data
```
✅ Check:
- Backend logs (Terminal 1) for errors
- Browser console (F12) for network errors
- API status shows 🔴 DEMO instead of 🟢 LIVE
- Try /health endpoint: curl http://localhost:8000/health
```

### Slow chat responses (>15s)
```
✅ Normal behavior:
- Full mode: 8-12 seconds (expected)
- Running 7 agents in parallel + 7 engines
- Fast mode: 5-8 seconds (if time is critical)
```

### Simulation won't run
```
✅ Check:
- /simulate endpoint available: curl http://localhost:8000/simulate
- Intervention params match backend schema
- No validation errors in browser console
```

### Memory usage high
```
✅ Solutions:
- Disable satellite/flight layers if not needed
- Reduce object counts (TrafficLayer: 2000 → 500)
- Use fast chat mode to reduce processing
- Close other browser tabs
```

---

## Performance Tips

### Optimize for Fast Response

```tsx
// Use fast mode for quick feedback
const { chat } = useLDRAGoChat();
await chat(prompt, "fast");  // 5-8 seconds instead of 8-12
```

### Batch API Calls

```tsx
// Don't:
for (const scenario of scenarios) {
  await simulate(scenario);  // Slow, sequential
}

// Do:
const results = await Promise.all(
  scenarios.map((s) => simulate(s))  // Parallel
);
```

### Cache Results

```tsx
// useSimulation stores last 10 results in history
const { history } = useSimulation();
const cachedResult = history.find((r) => r.manifest.params === params);
```

---

## File Navigation

| File | Purpose | When to Edit |
|------|---------|--------------|
| `src/api/client.ts` | API endpoints | Adding new endpoints |
| `src/api/hooks.ts` | React hooks | Custom retry logic |
| `src/pages/DemoPage.tsx` | UI + integration | UI changes, new components |
| `src/store/engineStore.ts` | Global state | New state needed |
| `.env.local` | Configuration | Change API URL |
| `API_INTEGRATION.md` | Documentation | Full integration details |

---

## Next: Building Features

### Add a New API Endpoint

1. **Backend** (app.py): Create endpoint
2. **Client** (src/api/client.ts): Add method
3. **Hook** (src/api/hooks.ts): Create custom hook
4. **Component** (src/pages/DemoPage.tsx): Use hook

### Example: Add `/analyze` endpoint

```tsx
// client.ts
async analyze(data: InputData): Promise<AnalysisResponse> {
  return this.post<AnalysisResponse>("/analyze", data);
}

// hooks.ts
export function useAnalysis(options?: UseApiOptions<AnalysisResponse>) {
  return useApi(
    () => getAPIClient().analyze(data),
    options
  );
}

// DemoPage.tsx
const { data, loading, execute } = useAnalysis();
await execute();
```

---

## Support & Documentation

- **[API_INTEGRATION.md](./API_INTEGRATION.md)** - Full API reference
- **[RELEASE_NOTES.md](./RELEASE_NOTES.md)** - v1.1 changes
- **[README.md](./README.md)** - Project overview
- **Console**: `const eng = window.overhaulEngine;` for debugging

---

## Key Takeaways

✅ **Real AI** - LDRAGo v2 with 7 agents + multi-model reasoning  
✅ **Real Simulations** - Traffic models + environmental engines  
✅ **Real Data** - Live AQI + OSRM + TomTom feeds  
✅ **Live Results** - KPI cards update in real-time  
✅ **Fallback Support** - Demo mode if API unavailable  

🚀 **Ready to deploy** - npm run build && deploy dist/

---

**Happy building!** 🎉
