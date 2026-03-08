# OVERHAUL — System Architecture

> AI-Powered Decision Intelligence Platform for Urban Simulation  
> Current scope: Delhi NCR (Delhi, Noida, Ghaziabad, Gurugram, Faridabad, Greater Noida)

---

## Table of Contents

1. [System Overview](#system-overview)
2. [LDRAGO v2 Cognitive Pipeline](#ldrago-v2-cognitive-pipeline)
3. [Model Routing Strategy](#model-routing-strategy)
4. [Simulation Engine Framework](#simulation-engine-framework)
5. [Cross-Engine Feedback System](#cross-engine-feedback-system)
6. [Data Integration Architecture](#data-integration-architecture)
7. [API Surface](#api-surface)
8. [Project Structure](#project-structure)
9. [Development Roadmap](#development-roadmap)

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         OVERHAUL PLATFORM                          │
│                                                                     │
│  ┌──────────┐   ┌──────────────┐   ┌──────────────────────────┐   │
│  │ Frontend │   │   FastAPI     │   │   LLM Multi-Model Stack  │   │
│  │ React +  │──▶│   Gateway     │──▶│   Qwen / Llama / GPT-OSS│   │
│  │ Three.js │   │   (app.py)    │   │   / Gemini               │   │
│  └──────────┘   └──────┬───────┘   └──────────────────────────┘   │
│                         │                                           │
│         ┌───────────────┼───────────────┐                          │
│         │               │               │                          │
│  ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐                 │
│  │  LDRAGO v2  │ │  7 Engines  │ │  Data Layer │                 │
│  │  6-Agent    │ │  Transport  │ │  9 Adapters │                 │
│  │  Cognitive  │ │  Environ.   │ │  Schema Norm│                 │
│  │  Pipeline   │ │  Infrastr.  │ │  NCR CSVs   │                 │
│  └─────────────┘ │  Energy     │ └─────────────┘                 │
│                   │  Economic   │                                  │
│                   │  Population │                                  │
│                   │  Logistics  │                                  │
│                   └─────────────┘                                  │
└─────────────────────────────────────────────────────────────────────┘
```

### Core Principles

1. **Physics-first reasoning** — Every prediction is bounded by real-world constraints (e.g., EVs can reduce PM2.5 by max 35%, signal optimization max 20% congestion reduction)
2. **Multi-model consensus** — No single LLM is trusted alone; cross-validation via different model architectures
3. **Data-grounded** — Trained on 10,965 AQI records, 1,512 traffic records, 452,750 trajectory points from Delhi NCR
4. **Temporal causality** — Feedback loops model how transport changes cascade through environment, economy, and population

---

## LDRAGO v2 Cognitive Pipeline

**Logic Driven Reasoning and Adaptive Governance Orchestrator**

Six specialized agents form a cognitive pipeline. Each agent has a clear responsibility, a designated LLM model, and a fallback strategy.

### Pipeline Architecture

```
USER QUERY
    │
    ▼
┌──────────┐     ┌──────────┐     ┌────────────────────────────────┐
│  PARSER  │────▶│ PLANNER  │────▶│        PARALLEL PHASE          │
│  Qwen 3  │     │Heuristic │     │  ┌────────────┐ ┌───────────┐ │
│  4B      │     │  + LLM   │     │  │ RESEARCHER │ │  ENGINES  │ │
│  <1s     │     │  <0.5s   │     │  │ (Data APIs)│ │ (7 Sims)  │ │
└──────────┘     └──────────┘     │  └────────────┘ └───────────┘ │
                                   └───────────────┬──────────────┘
                                                   │
                                   ┌───────────────▼──────────────┐
                                   │        PARALLEL PHASE         │
                                   │  ┌──────────┐ ┌───────────┐  │
                                   │  │ REASONER │ │  CRITIC   │  │
                                   │  │ Llama 70B│ │ GPT-OSS   │  │
                                   │  │  ~5s     │ │ 120B ~5s  │  │
                                   │  └──────────┘ └───────────┘  │
                                   └───────────────┬──────────────┘
                                                   │
                                           ┌───────▼───────┐
                                           │ SYNTHESIZER   │
                                           │ Gemini 3.1    │
                                           │ Pro  ~5s      │
                                           └───────────────┘
                                                   │
                                                   ▼
                                             RESPONSE
```

### Agent Specifications

| Agent | Model | Latency | Responsibility | Fallback |
|-------|-------|---------|----------------|----------|
| **Parser** | Qwen 3 4B | <1s | Extract intent, interventions, entities, complexity | Keyword heuristic (25+ domain terms) |
| **Planner** | Heuristic + LLM | <0.5s | Generate execution plan, select engines/models | Rule-based plan from complexity score |
| **Researcher** | Data APIs | ~1s | Parallel fetch: NCR CSV data, live AQI, traffic | Cached defaults with city calibration |
| **Reasoner** | Llama 3.3 70B | ~5s | Causal chains, cross-domain impact matrix, second-order effects | Qwen fallback |
| **Critic** | GPT-OSS 120B | ~5s | Physics validation, consistency scoring, bound checking | Heuristic physics checks |
| **Synthesizer** | Gemini 3.1 Pro | ~5s | Executive summary, recommendations, risk assessment | Llama fallback |

### Execution Modes

| Mode | Pipeline | Latency | When |
|------|----------|---------|------|
| `full` | All 6 agents | 8-12s | Complex policy analysis, multi-scenario |
| `fast` | Skip Critic | 5-8s | Real-time chat, simple queries |

### Agent Context Flow

All agents share a mutable `AgentContext` dataclass:

```python
@dataclass
class AgentContext:
    query: str                    # User's original query
    city: str = "delhi"           # Target city
    parsed_intent: dict           # Parser output: type, interventions, entities
    execution_plan: dict          # Planner output: engines, models, strategy
    research_data: dict           # Researcher output: NCR data, AQI, traffic
    reasoning_output: dict        # Reasoner output: causal chains, impacts
    critique: dict                # Critic output: validation, adjustments
    synthesis: str                # Synthesizer output: final response
    engine_results: dict          # Simulation engine outputs
    agent_logs: list              # Trace of each agent's execution
    errors: list                  # Non-fatal errors for debugging
```

---

## Model Routing Strategy

### Model Profiles

| Model | Provider | Parameters | Context | Cost | Latency | Strengths |
|-------|----------|-----------|---------|------|---------|-----------|
| Qwen 3 4B | OpenRouter | 4B | 32K | Free | <1s | Fast JSON, parsing |
| Llama 3.3 70B | OpenRouter | 70B | 128K | Free | ~3s | Deep analysis, reasoning |
| GPT-OSS 120B | OpenRouter | 120B | 128K | Free | ~5s | Cross-validation, fact-checking |
| Gemini 3.1 Pro | Google AI | ~200B* | 1M | $0.00125/1K | ~3s | Synthesis, grounding, long context |

### Task-Based Routing Table

| Task | Speed Priority | Quality Priority | Cost Priority | Balanced |
|------|---------------|-----------------|--------------|----------|
| Parse | Qwen (fast) | Qwen (fast) | Qwen (fast) | Qwen (fast) |
| Plan | Qwen (fast) | Llama (analysis) | Qwen (fast) | Qwen (fast) |
| Analyze | Llama (analysis) | Llama (analysis) | Llama (analysis) | Llama (analysis) |
| Validate | Qwen (fast) | GPT-OSS (validate) | Qwen (fast) | GPT-OSS (validate) |
| Synthesize | Llama (analysis) | Gemini (reason) | Llama (analysis) | Gemini (reason) |
| Chat | Qwen (fast) | Llama (analysis) | Qwen (fast) | Qwen (fast) |
| Forecast | Llama (analysis) | Gemini (reason) | Llama (analysis) | Gemini (reason) |

### Fallback Chains

```
Qwen → Llama → GPT-OSS → Gemini (fast tasks)
Llama → GPT-OSS → Gemini → Qwen (analysis tasks)
GPT-OSS → Llama → Gemini (validation tasks)
Gemini → Llama → GPT-OSS (synthesis tasks)
```

Every LLM call attempts the primary model, then falls through the chain. The system never fails silently — it degrades gracefully.

---

## Simulation Engine Framework

### 7 Domain Engines

Each engine extends `SimulationEngine` (abstract base) and implements `run(scenario, data) → dict`.

| Engine | Key Models | Outputs |
|--------|-----------|---------|
| **Transport** | BPR link model, Dijkstra routing, 12 NCR nodes | congestion, travel time, mode split, emissions |
| **Environment** | AQI regression, emission factors, PM2.5 dispersion | air quality projections, health impact |
| **Infrastructure** | Capacity modeling, construction phasing | network utilization, bottleneck identification |
| **Energy** | Grid capacity, EV charging demand curves | peak load, renewable potential |
| **Economic** | Cost-benefit analysis, productivity impacts | GDP impact, employment effects |
| **Population** | 7 demographic segments, behavior adaptation | mode shift, EV adoption curves |
| **Logistics** | Freight routing, last-mile optimization | delivery efficiency, freight emissions |

### Two-Phase Execution

```
Phase 1 (parallel):  Transport + Infrastructure + Environment
         ↓ merge metrics
Phase 2 (parallel):  Energy + Economic + Population + Logistics
```

Phase 2 engines receive merged outputs from Phase 1, enabling cross-domain awareness.

### Population Segments

| Segment | Share | Key Attributes |
|---------|-------|---------------|
| Office Workers | 30% | WFH capable, medium EV readiness |
| Gig Workers | 12% | High flexibility, low EV readiness |
| Students | 15% | Metro dependent, no EV ownership |
| Service Sector | 18% | Fixed schedule, bus dependent |
| Industrial Workers | 10% | Long commute, car dependent |
| Senior Citizens | 8% | Low mobility, off-peak travel |
| High Income | 7% | High EV readiness, car dominant |

---

## Cross-Engine Feedback System

### Temporal Simulation

`run_temporal()` executes multi-step simulations over time, modeling how interventions propagate through the city system.

```
Step 0 (Day 0)     → Baseline state
Step 1 (Day 90)    → 27% implementation (S-curve)
Step 2 (Day 180)   → 73% implementation (S-curve)
Step 3 (Day 270)   → 95% implementation (S-curve)
Step 4 (Day 360)   → 99% implementation (S-curve)
```

### S-Curve Progress Scaling

Implementation progress follows a sigmoid curve, not linear:

```
progress(t) = 1 / (1 + exp(-10 * (t - 0.5)))
```

This models the real-world pattern: slow start → rapid middle → plateau.

### Feedback Loop Diagram

```
Transport ──────────────────────────────────────▶ Environment
  avg_speed → emission_factor                      PM2.5, CO2
  ev_share → emission_reduction                        │
  VKT → total_emissions                               │
                                                       ▼
Population ◀──────── Economic ◀──────────────── Environment
  GDP impact                    PM2.5 → health_cost
  productivity                  AQI → productivity
      │
      │ mode_share, EV adopters
      ▼
Transport (next step)
```

Each feedback loop updates the input data for the next time step:
- **Transport → Environment**: Speed, CO2, EV share, vehicle-km-traveled
- **Environment → Economic**: PM2.5 health costs, AQI productivity impact
- **Economic → Population**: GDP impact, productivity gains
- **Population → Transport**: Car mode share, metro share, EV adoption rate

---

## Data Integration Architecture

### 9 Data Adapters

| Adapter | Source | Type | Schedule |
|---------|--------|------|----------|
| NCRAqiCSV | Local CSV | AQI data | Static (2024-2026) |
| NCRTrafficCSV | Local CSV | Traffic data | Static (2024-2026) |
| HistoricalJSON | Local JSON | Historical metrics | Static |
| OpenMeteoAqi | Open-Meteo API | Live AQI | Real-time |
| OpenMeteoWeather | Open-Meteo API | Weather data | Real-time |
| OSRMRouting | OSRM API | Route calculation | On-demand |
| TomTomFlow | TomTom API | Traffic flow | Real-time |
| Nominatim | OSM Nominatim | Geocoding | On-demand |
| ValidationDB | Supabase | Predictions DB | Real-time |

### Schema Normalization

The `data_integration/schema.py` module normalizes heterogeneous city data into a canonical schema.

**Domains**: traffic, aqi, population, infrastructure, energy

**Features**:
- **Alias resolution**: e.g., `speed`, `avg_speed`, `average_speed` → `avg_speed_kmh`
- **Missing data estimation**: City-calibrated defaults (e.g., Ghaziabad PM2.5 = Delhi × 1.10)
- **Completeness scoring**: Validates what percentage of required fields are present
- **Type coercion**: Ensures all fields match expected types

### Data Flow

```
Raw Data (CSV, API, JSON)
    │
    ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│  Adapters   │───▶│  Normalize   │───▶│  Estimate    │
│  (9 types)  │    │  (schema.py) │    │  Missing     │
└─────────────┘    └──────────────┘    └──────┬──────┘
                                              │
                                       ┌──────▼──────┐
                                       │  Engine     │
                                       │  Input Dict │
                                       └─────────────┘
```

---

## API Surface

### Core Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/chat` | LDRAGO v1 — Legacy chat pipeline |
| `POST` | `/chat/v2` | LDRAGO v2 — 6-agent cognitive pipeline |
| `POST` | `/simulate` | Single-step engine simulation |
| `POST` | `/simulate/temporal` | Multi-step temporal simulation with feedback loops |
| `POST` | `/scenarios/compare` | Compare multiple scenarios side-by-side |
| `GET` | `/scenarios/templates` | List available scenario templates |
| `POST` | `/scenarios/templates/{id}` | Run a template scenario |

### Data Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/engines` | List available simulation engines |
| `GET` | `/live/aqi` | Live AQI data from Open-Meteo |
| `GET` | `/live/route` | Route calculation via OSRM |
| `GET` | `/geocode` | Forward geocoding |
| `GET` | `/reverse-geocode` | Reverse geocoding |
| `GET` | `/schema/info` | Canonical schema documentation |

### System Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | System health check |
| `GET` | `/health/llm` | LLM provider status |
| `GET` | `/integrations/status` | All integration statuses |
| `POST` | `/config/reload` | Hot-reload configuration |

### `/chat/v2` Request/Response

**Request:**
```json
{
  "prompt": "What if Delhi implements congestion pricing?",
  "city": "delhi",
  "mode": "full"
}
```

**Response:**
```json
{
  "summary": "Executive summary text...",
  "outputs": {
    "tldr": "Detailed analysis...",
    "confidenceLevel": "high",
    "impactCards": [...],
    "domains": { "transport": {...}, "environment": {...} },
    "critique": { "validation_pass": true, "consistency_score": 0.87 },
    "brainInsights": {
      "orchestrator": "LDRAGo v2 (full)",
      "models_used": { "parser": "qwen", "reasoner": "llama", ... },
      "agent_trace": [...],
      "duration_seconds": 9.4
    }
  },
  "manifest": { "run_id": "uuid", "mode": "ldrago_v2", ... }
}
```

---

## Project Structure

```
OVERHAUL-main/
├── app.py                          # FastAPI gateway (39 routes)
├── config.yaml                     # System configuration
├── .env                            # API keys (gitignored)
│
├── agents/
│   ├── cognitive/                  # LDRAGO v2 cognitive pipeline
│   │   ├── orchestrator.py         # LDRAGOv2 class (run, run_fast)
│   │   ├── parser_agent.py         # Intent extraction (Qwen)
│   │   ├── planner_agent.py        # Execution planning
│   │   ├── researcher_agent.py     # Parallel data gathering
│   │   ├── reasoner_agent.py       # Deep analysis (Llama 70B)
│   │   ├── critic_agent.py         # Physics validation (GPT-OSS)
│   │   ├── synthesizer_agent.py    # Final synthesis (Gemini)
│   │   └── roles.py                # AgentRole, AgentContext, AgentOutput
│   ├── ldrago_orchestrator.py      # LDRAGO v1 (legacy, still used by /chat)
│   ├── master_brain.py             # Physics-based validation brain
│   ├── unified_brain.py            # Output correction brain
│   ├── ncr_data_loader.py          # NCR CSV data loading
│   └── ...
│
├── engines/
│   ├── base.py                     # SimulationEngine ABC
│   ├── registry.py                 # 2-phase executor + run_temporal
│   ├── transport/engine.py         # BPR model, 12 NCR nodes
│   ├── environment/engine.py       # AQI regression
│   ├── infrastructure/engine.py    # Capacity modeling
│   ├── energy/engine.py            # Grid + EV demand
│   ├── economic/engine.py          # Cost-benefit analysis
│   ├── population/engine.py        # 7 demographic segments
│   └── logistics/engine.py         # Freight routing
│
├── llm/
│   ├── chat.py                     # LLM router (llm_chat_text, llm_ensemble)
│   ├── config.py                   # LLMConfig (5 models)
│   └── routing.py                  # Model routing strategy
│
├── data_integration/
│   ├── bridge.py                   # Prompt→Scenario, Results→Frontend
│   ├── schema.py                   # Canonical schema normalization
│   └── adapters/                   # 9 data adapters
│
├── services/                       # Microservice definitions
│   ├── gateway/app.py              # Port 8000
│   ├── simulation/app.py           # Port 8001
│   ├── llm/app.py                  # Port 8002
│   ├── data/app.py                 # Port 8003
│   ├── validation/app.py           # Port 8004
│   └── traffic_god/app.py          # Port 8005
│
├── data/
│   ├── ncr_aqi_data.csv            # 10,965 AQI records (2024-2026)
│   ├── ncr_traffic_data.csv        # 1,512 traffic records (2024-2026)
│   ├── trajectories_full.csv       # 452,750 trajectory points
│   └── historical_metrics.json     # Historical baseline data
│
└── traffic-god/                    # Custom ML models (no external APIs)
    ├── src/perception/infer.py     # Video → trajectory extraction
    ├── src/control/train_rl.py     # RL signal optimization
    └── src/analysis/metrics.py     # Traffic metrics computation
```

---

## Development Roadmap

### Phase 1 — Current (MVP) ✅
- 7 simulation engines with 2-phase execution
- LDRAGO v1 + v2 cognitive pipelines
- 4-model LLM stack with fallback chains
- 9 data adapters + schema normalization
- Delhi NCR focused with 3 cities
- FastAPI monolith deployment

### Phase 2 — Regional Scale
- Expand to 10+ Indian cities
- Real-time SUMO integration via traffic-god
- WebSocket streaming for live analysis updates
- Redis caching layer for engine results
- User authentication + rate limiting

### Phase 3 — National Scale
- Microservice deployment (Docker Compose / K8s)
- Event-driven architecture (message queue between engines)
- Historical prediction validation pipeline
- A/B testing framework for model routing
- Multi-language support

### Phase 4 — Planetary Scale
- City-agnostic engine framework (plug any city's data)
- Federated simulation across city clusters
- Real-time satellite/IoT data ingestion
- GPU-accelerated engine computation
- Multi-region deployment with edge inference
