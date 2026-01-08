# OVERHAUL

OVERHAUL is an education-first simulation platform for teaching and exploring urban mobility, air quality, and infrastructure tradeoffs using interactive demos and agentic analysis.

## Microsoft AI powered

This repo integrates Microsoft services via environment variables:

- **Azure OpenAI Service** (primary LDRAGo provider): structured reasoning for think/plan/synthesize.
- **Azure Maps**: geocoding and reverse-geocoding used by the UI and by the Customer Validation feature.

## Quickstart

### 1) Backend (FastAPI)

```bash
cd /Users/aayushsharma/Desktop/Overhaul/OVERHAUL-main
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# copy env template and fill keys as needed
cp .env.example .env

python app.py
```

Backend defaults to `http://localhost:8000`.

### 2) Frontend (landing-react)

```bash
cd /Users/aayushsharma/Desktop/Overhaul/OVERHAUL-main/landing-react
npm install

# copy env template and fill tokens as needed
cp .env.example .env

npm run dev
```

## Customer Validation

The `landing-react` site includes a `/validation` page that lets users submit feedback and displays approved entries.

Backend endpoints:

- `GET /validation/entries?page=1&page_size=12` — list approved entries
- `POST /validation/entries` — create entry (best-effort Azure Maps enrichment when configured)
- `POST /validation/entries/{id}/approve` — approve an entry (requires `VALIDATION_ADMIN_TOKEN` and `X-Admin-Token` header)

## Azure Maps endpoints

- `GET /azure/maps/geocode?query=...&limit=1`
- `GET /azure/maps/reverse?lat=...&lon=...`

## Repo structure (high level)

- `app.py` — FastAPI backend (demo + agentic analysis)
- `agents/` — agent modules (Master Brain, LDRAGo, etc.)
- `landing-react/` — live demo site (React + Vite)

