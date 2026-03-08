# OVERHAUL

**Urban Mobility Analysis Platform for Delhi NCR**

OVERHAUL is a simulation + AI analysis platform for exploring urban mobility, air quality, energy, and infrastructure tradeoffs across Delhi, Noida, Ghaziabad, and Gurugram. It combines 7 domain simulation engines, a multi-model LLM orchestrator (LDRAGo), a unified data adapter layer, and an interactive React frontend.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Frontend (React/Vite)         Three.js 3D Flyover             │
│  landing-react/                ai-flyover-sim/frontend/         │
├─────────────────────────────────────────────────────────────────┤
│                        FastAPI Backend (app.py)                 │
│  ┌──────────────┐  ┌────────────────┐  ┌─────────────────────┐ │
│  │   LDRAGo AI  │  │  7 Simulation  │  │    Data Adapters    │ │
│  │  Orchestrator │  │    Engines     │  │   (9 adapters)      │ │
│  └──────────────┘  └────────────────┘  └─────────────────────┘ │
│  ┌──────────────┐  ┌────────────────┐  ┌─────────────────────┐ │
│  │ Traffic God  │  │  Imagen Overlay│  │  Knowledge RAG      │ │
│  │  Perception  │  │  (AI overlays) │  │  (embeddings)       │ │
│  └──────────────┘  └────────────────┘  └─────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  Microservices (services/)  —  6 services for local dev        │
│  Gateway:8000  Sim:8001  LLM:8002  Data:8003  Val:8004  TG:8005│
└─────────────────────────────────────────────────────────────────┘
```

## LLM Stack

| Provider | Model | Role |
|---|---|---|
| OpenRouter | Qwen 3 4B | Primary fast LLM (chat, JSON) |
| Google | Gemini 3 Pro Preview | Deep reasoning, agents |
| Google | Gemini 2.5 Flash | LDRAGo orchestrator |

No Azure dependencies — fully removed.

## Simulation Engines

Seven domain engines in `engines/`, executed in a two-phase pipeline:

| Phase | Engine | What it models |
|---|---|---|
| 1 | **TransportEngine** | BPR congestion, speed, travel time |
| 1 | **InfrastructureEngine** | Capacity, ROI, construction timeline |
| 1 | **EnvironmentEngine** | PM2.5, AQI, emission reduction |
| 2 | **EnergyEngine** | EV grid load, renewable share |
| 2 | **EconomicEngine** | BCR, NPV, job creation |
| 2 | **PopulationEngine** | 7 demographic segments, mode shift |
| 2 | **LogisticsEngine** | Freight efficiency, last-mile delivery |

Phase 2 engines can consume Phase 1 outputs for cross-engine feedback loops.

## Data Adapter System

Nine adapters in `data_integration/adapters/` provide unified data access:

| Adapter | Source | Domain |
|---|---|---|
| NCRAqiCSVAdapter | Local CSV (2024-2026) | AQI |
| NCRTrafficCSVAdapter | Local CSV + city files | Traffic |
| HistoricalJSONAdapter | historical_metrics.json | Historical |
| OpenMeteoAqiAdapter | Open-Meteo API (free) | AQI |
| OpenMeteoWeatherAdapter | Open-Meteo API (free) | Weather |
| OSRMRoutingAdapter | OSRM API (free) | Routing |
| TomTomFlowAdapter | TomTom API (key-gated) | Traffic |
| NominatimAdapter | OSM Nominatim (free) | Geocoding |
| ValidationDBAdapter | SQLite / Supabase | Validation |

All adapters share: TTL caching, stale-while-revalidate fallback, DataResult envelope.

## Quickstart

### 1) Backend

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Fill in OPENROUTER_API_KEY and GEMINI_API_KEY at minimum

python app.py
```

Backend runs at `http://localhost:8000`.

### 2) Frontend

```bash
cd landing-react
npm install
cp .env.example .env
npm run dev
```

### 3) Microservices (optional, local dev)

```bash
docker-compose up --build
```

Starts all 6 services. Or run individually:
```bash
uvicorn services.simulation.app:app --port 8001
```

## Environment Variables

See [.env.example](.env.example) for the full list. Key variables:

| Variable | Required | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` | Yes | Qwen 3 4B via OpenRouter |
| `GEMINI_API_KEY` | Yes | Gemini models for orchestration |
| `VALIDATION_ADMIN_TOKEN` | Yes | Moderation endpoint auth |
| `TOMTOM_API_KEY` | No | Real-time traffic flow |
| `IMAGEN_API_KEY` | No | AI-generated map overlays |
| `SUPABASE_URL` / `SUPABASE_KEY` | No | Cloud DB (falls back to SQLite) |

## Project Structure

```
app.py                    # FastAPI monolith (production)
agents/                   # LDRAGo orchestrator, Brain, Gemini agents
  ldrago_orchestrator.py  #   3 modes: fast, orchestrate, quick
  ldrago_brain.py         #   Intelligent think/plan/command/synthesize
  ncr_data_loader.py      #   CSV data loaders for Delhi NCR
engines/                  # 7 simulation engines
  base.py                 #   Abstract SimulationEngine
  registry.py             #   Two-phase execution, caching
  scenarios.py            #   6 scenario templates
  geospatial.py           #   GeoJSON output layer
data_integration/         # Data pipeline
  bridge.py               #   Keyword extraction, engine formatting
  adapters/               #   9 unified data adapters
llm/                      # LLM provider abstraction
  chat.py                 #   qwen_chat, gemini_chat, unified llm_chat
  config.py               #   LLMConfig dataclass
services/                 # Microservice layer (local dev)
  gateway/                #   API gateway (port 8000)
  simulation/             #   Engine service (port 8001)
  llm/                    #   LLM service (port 8002)
  data/                   #   Data service (port 8003)
  validation/             #   Validation service (port 8004)
  traffic_god/            #   Perception service (port 8005)
  shared/                 #   Contracts + service client
traffic-god/              # SUMO traffic simulation + RL
landing-react/            # React frontend (Vite)
ai-flyover-sim/           # Three.js 3D city flyover
imagen_overlay/           # Imagen 3 map overlay generation
knowledge/                # RAG knowledge base documents
```

## Deployment

**Current**: Monolith deployed on Render (`app.py` via uvicorn).
See [render.yaml](render.yaml) for Render Blueprint config.

## License

See [LICENSE](LICENSE).