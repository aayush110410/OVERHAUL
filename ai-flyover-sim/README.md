# AI Flyover Simulation Platform

Constraint-driven urban flyover planner with grounded data and hard-coded engineering rules.

## Stack
- Frontend: React + Vite, Mapbox GL JS (terrain + 3D), Three.js custom layer, Zustand state.
- Backend: FastAPI, Overpass (OSM), Gemini (planning decisions only), hard-coded IRC-style heuristics.

## Run

Backend:
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Frontend:
```bash
cd frontend
npm install
VITE_MAPBOX_TOKEN=your_token npm run dev
```

## API
- `POST /plan` `{ "prompt": "flyover from A to B" }` → returns geometry, pillars, widths, heights, traffic metrics.
- `GET /health` health check.

## Design Rules
- Lane width 3.5m; default lanes by highway type.
- Road width = lanes * 3.5 + 1.5.
- Flyover lanes = min(2, ground lanes); flyover width = lanes * 3.5 + 2.0.
- Clearance: 8.5m (12m at major junctions).
- Offset = road_width/2 + flyover_width/2 + 1.0.
- Pillar spacing from Gemini (fallback 30m).

## Pipeline
1) Parse prompt → geocode start/end (Nominatim).
2) Overpass fetch → highway type, lanes, centerline.
3) Rules → widths, offset, clearance.
4) Gemini → lane choice, pillar spacing, rationale (strict JSON contract).
5) Build offset flyover path + pillars; compute traffic metrics.
6) Frontend renders Mapbox map with Three.js TubeGeometry following path.

## Notes
- Gemini optional: missing key triggers deterministic fallback.
- Geodesic math is approximate (lat/lon scale); production should project to local meters.
- Three.js mesh rendered via Mapbox custom layer; no Mapbox extrusion for flyover.
