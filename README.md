# 🚀 OVERHAUL: AI-Powered Civic Simulation System

> A groundbreaking agentic-AI prototype that thinks like a government, reacts like a strategist, and simulates like a supercomputer — all in one interactive platform.

![OVERHAUL Banner](https://raw.githubusercontent.com/yourusername/overhaul/main/assets/cover.png)

---

## 🎯 What is OVERHAUL?

**OVERHAUL** is a city-scale simulation engine powered by AI agents that lets users test civic ideas, policy interventions, and real-time infrastructure changes. Ask it anything — traffic bottlenecks, pollution drops, rerouting strategies — and OVERHAUL's models respond like an expert task force.

### 🧠 Core Models

* **🧠 LDRAGO** — Master controller agent that interprets user queries & orchestrates model collaboration
* **🚦 TrafficSim** — Simulates traffic flow, bottlenecks, rerouting, and infrastructure logic
* **🌫 AQISim** — Projects pollution levels, health impacts, and air quality deltas

---

## 🧪 Key Features

* ✅ Chat-like user interface powered by natural language understanding
* 🗺 Real-time dynamic map updates with congestion routes, alt-paths, pollution zones
* 🔁 Agentic reasoning loop — models talk to each other before finalizing any decision
* 💬 Interactive simulations with just a typed prompt
* 📈 Output includes real-world stats, projections, and action plans

---

## 📦 Tech Stack

| Layer    | Tech                                                      |
| -------- | --------------------------------------------------------- |
| UI       | HTML, CSS, JS (Vanilla or React)                          |
| Mapping  | Leaflet.js / MapLibre                                     |
| Backend  | FastAPI + Python                                          |
| AI Logic | Rule-based + ML (XGBoost / Light models)                  |
| Hosting  | Render / Vercel (Frontend), Localhost / Railway (Backend) |

---

## 🚦 Sample Use Case

```bash
User: "How to reduce traffic between Sector 78 and Vasundhara during rush hour?"

OVERHAUL:
- Simulates congestion zones in Sector 62/63
- Suggests bypass construction + alternate route via Indirapuram
- Calculates AQI drop: -28 PM2.5 points
- Visualizes suggested new route on map
```

---

## 🛠️ Setup Instructions

### 🔹 Clone the repo

```bash
git clone https://github.com/yourusername/overhaul
cd overhaul
```

### 🔹 Start Backend (FastAPI)

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### 🔹 Launch Frontend

* Open `index.html` in VS Code
* Install Live Server extension
* Right click > Open with Live Server

---

## ✨ Demo Preview

![Demo](https://raw.githubusercontent.com/yourusername/overhaul/main/assets/demo.gif)

---

## 🌍 Future Roadmap

* Add 3D simulation & flood models
* Connect with real-time APIs (AQI, weather, traffic)
* Train LLM-based LDRAGO agent for smarter reasoning
* Deploy fully on cloud or HuggingFace Spaces

---

## 🙌 Credits

* Built with ❤️ by [Your Name](https://github.com/aayush110410)
* Conceptualized as a smart city solution for traffic, pollution & civic policy

---

## 📬 Contact / Collaboration

* 📧 Email: [yourmail@gmail.com](mailto:)
* 🤝 Open to feedback, forks, and collabs!

---

## 📄 License

MIT License — feel free to use, build, and remix.

---

> “If cities are the engines of growth — OVERHAUL is the AI that keeps them running.”




<<<<<<< HEAD
# OVERHAUL Multi-Agent Scaffold

Production-grade agentic framework for analyzing EV adoption, AQI, noise, economic impacts, and SUMO traffic simulations for the Sector-78 -> Vasundhara corridor (Delhi NCR). Follows the Agentic Design Patterns brief: prompt chaining, routing, RAG, planning, and a critic pass.

## Repository layout
```
agents/
   coordinator.py          # Router + planner (prompt chaining + routing)
   data_collector.py       # Fetch APIs/web search (OpenAQ, WHO, World Bank, etc.)
   cleaner.py              # Schema normalization
   embeddings_rag.py       # Sentence-transformer embeddings + vector store
   behavior_model.py       # Behavioral segmentation + modal shift heuristics
   simulator_adapter.py    # SUMO + surrogate bridge
   impact_estimator.py     # AQI/noise/economic deltas + report writer
   critic.py               # Safety, counterfactual, red-team review
copilot_instructions.md   # How to iterate each agent
config.yaml               # Region bbox, API endpoints, SUMO hyperparams
llm_client.py             # Provider-agnostic LLM wrapper
privacy_guard.py          # Consent/ethics guardrails
notebooks/demo.ipynb      # Fetch -> ingest -> simulate -> summarize POC
sumo_net/build_from_osm.py# OSM -> SUMO pipeline
storage/                  # Raw/cleaned/vector/report outputs
run_poc.sh                # End-to-end demo runner
requirements.txt          # Python deps
```

## Quickstart (VS Code / Windows; adapt paths for macOS/Linux)
```
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
set SUMO_HOME=C:\tools\sumo  # update to your SUMO install
python -m pytest tests -q      # optional regression pass
bash run_poc.sh                # fetch -> RAG -> SUMO baseline -> scenario A
```
`run_poc.sh` orchestrates:
1. `python agents/data_collector.py` (script mode) to pull OpenAQ + placeholder searches.
2. `python sumo_net/build_from_osm.py` if `storage/sumo/corridor.net.xml` is missing.
3. `python -m agents.coordinator` to execute the multi-agent chain.
4. Writes `storage/reports/latest_report.json` and prints a text summary.

## Agentic patterns
- **Prompt chaining**: `coordinator.orchestrate()` constructs the execution plan; every agent emits structured JSON consumed by the next stage.
- **Routing**: `simple_router()` picks which specialist agent handles a user request (swap for classifier later).
- **RAG**: `embeddings_rag.py` ingests cleaned docs, builds a sentence-transformer store, and serves `query()` for planner/context.
- **Planning**: Coordinator logs steps and artifacts for reproducibility.
- **Critic + safety**: `critic.py` plus `privacy_guard.py` apply rule-based checks before releasing reports.
- **Collaboration**: DataCollector -> Cleaner -> Embeddings -> BehaviorModel -> SimulatorAdapter -> ImpactEstimator -> Critic.

## DataCollector search seeds
```
[
   "EV adoption AQI reduction study China 2018 2019 2020",
   "electric vehicle noise impact study urban",
   "Noida vehicle count dataset",
   "Delhi NCR vehicular emissions 2019 dataset",
   "e-rickshaw adoption India study",
   "electric two-wheeler impacts India study",
   "study 'EV penetration' 'AQI' 'China' 'health' 'case study'"
]
```
Add more queries or API clients inside `DataCollector` once keys are available.

## Coordinator -> ImpactEstimator payload template
```
{
   "task": "estimate_impacts",
   "region": "Sector-78 to Vasundhara",
   "time_horizon_years": 3,
   "ev_share_change": {"cars": 0.3, "two_wheelers": 0.5, "e_rickshaw": 0.4},
   "infrastructure_changes": [
      {"type": "left_turn_lane", "location": "edge_XXXXX"},
      {"type": "signal_retime", "tls": "tls_YYYY"}
   ],
   "objectives": ["minimize_travel_time", "minimize_aqi", "minimize_noise", "maximize_economic_output"],
   "constraints": {"budget_usd": 1000000}
}
```

## Data/API guidance
| Source | Endpoint | Notes |
| --- | --- | --- |
| OpenAQ | https://api.openaq.org/v3/measurements | Use bbox from `config.yaml`.
| WHO GHO | https://ghoapi.azureedge.net/api/ | Filter PM2.5, noise, DALY indicators.
| World Bank | https://api.worldbank.org/v2/country/IND/indicator/EN.ATM.PM10.MC.M3 | Paged JSON; use `per_page=200`.
| NITI Aayog / MoRTH | Manual download | Add as document ingestion (TODO).
| China EV health studies | PDF/HTML scraping from search results (TODO).
| Local media feeds | Add RSS/HTML parser to `web_search_and_scrape` (TODO).

## SUMO workflow
1. Install SUMO and export `SUMO_HOME`; ensure `sumo`, `netconvert`, `randomTrips.py` are on PATH.
2. Run `python sumo_net/build_from_osm.py` to download the bounding box and build `corridor.net.xml` plus default routes.
3. `agents/simulator_adapter.py` runs baseline + scenarios (A/B/C) and aggregates KPIs (travel time, delay, emissions placeholders).
4. TODO: integrate TraCI for per-edge metrics, queue lengths, noise proxies.

## Behavioral modeling
- `behavior_model.segment()` creates seeded traveler cohorts (income, commute length, vehicle preference) with modal shift probabilities.
- Configure driver aggressiveness, autorickshaw share, and EV adoption priors in `config.yaml`.
- TODO: plug socio-economic datasets + consent workflow via `privacy_guard`.

## Explainability & outputs
- `impact_estimator.estimate()` merges SUMO outputs and RAG snippets, returning deltas for AQI, travel time, idling, noise, economic value.
- Reports saved to `storage/reports/latest_report.json`; `llm_client.summarize()` turns them into narrative text (currently mocked).
- `critic.review()` adds warnings, counterfactuals, and safety notes before final delivery.

## Safety & ethics checklist
- [ ] Obtain user consent before ingesting socio-economic or caste proxies (use `privacy_guard.require_consent`).
- [ ] Strip PII from raw documents; log provenance IDs.
- [ ] Validate recommended interventions against municipal design manuals.
- [ ] Publish assumptions (value of time, fuel costs, EV incentives) inside reports.
- [ ] Require human reviewer sign-off on critic findings before external release.

## Tests
```
pytest tests/test_data_ingestion.py -q
pytest tests/test_rag.py -q
pytest tests/test_sumo_pipeline.py -q
pytest tests/test_end_to_end.py -q
```

## Developer checklist
1. Populate `.env` with API keys (OpenAQ optional, WHO auth headers, Bing/SerpAPI, municipal feeds) and reference them in `config.yaml`.
2. Replace placeholder search/scrape code with real clients plus caching.
3. Wire SUMO metrics via TraCI, calibrate against observed travel times/AQI.
4. Train behavior and impact models (PyTorch/XGBoost) and drop into respective agents.
5. Launch VS Code terminal, activate `.venv`, run `bash run_poc.sh`, then open `notebooks/demo.ipynb` to inspect artifacts.
=======
# OVERHAUL
>>>>>>> 8c2a93c7ebc34889483b26a56249ecab1746737e
