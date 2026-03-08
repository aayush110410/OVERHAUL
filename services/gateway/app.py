"""
Gateway Microservice
=====================

Central entry-point that routes client requests to backend
microservices: Simulation (8001), LLM (8002), Data (8003),
Validation (8004), TrafficGod (8005).

Owns all cross-cutting concerns:
  - CORS
  - Rate limiting (per-IP, 60 req/min)
  - Request body size guard (50 KB)
  - Chat orchestration (Data → Simulation → LLM → merge)

Port: 8000 (default)

Endpoints:
  GET  /health            — aggregated health from all services
  POST /chat              — full orchestrated chat flow
  GET  /engines           — proxy → Simulation
  POST /simulate          — proxy → Simulation
  POST /scenarios/compare — proxy → Simulation
  GET  /scenarios/templates      — proxy → Simulation
  POST /scenarios/templates/{id} — proxy → Simulation
  GET  /ncr/summary       — proxy → Data
  GET  /live/aqi          — proxy → Data
  GET  /live/route        — proxy → Data
  GET  /geocode           — proxy → Data
  GET  /reverse-geocode   — proxy → Data
  GET  /validation/entries — proxy → Validation
  POST /validation/entries — proxy → Validation
  POST /validation/entries/{id}/approve — proxy → Validation
  GET  /validation/stats   — proxy → Validation
  POST /traffic-god-llm   — proxy → TrafficGod
  POST /traffic-god/perception — proxy → TrafficGod
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from services.shared.client import get_client, ServiceError, ServiceUnavailableError  # noqa: E402
from services.shared.contracts import ServiceName  # noqa: E402

app = FastAPI(title="OVERHAUL Gateway", version="1.0.0")

# ── CORS ──────────────────────────────────────────────────────────

_ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# ── Rate limiter (per-IP, 60 req/min) ────────────────────────────

_RATE_STORE: Dict[str, list] = {}
_RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))
_RATE_WINDOW = 60


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    hits = _RATE_STORE.setdefault(client_ip, [])
    hits[:] = [t for t in hits if now - t < _RATE_WINDOW]
    if len(hits) >= _RATE_LIMIT:
        return Response(
            content=json.dumps({"detail": "Rate limit exceeded. Try again shortly."}),
            status_code=429,
            media_type="application/json",
        )
    hits.append(now)
    return await call_next(request)


# ── Body size guard (50 KB) ──────────────────────────────────────

_MAX_BODY_BYTES = 50_000


@app.middleware("http")
async def body_size_middleware(request: Request, call_next):
    cl = request.headers.get("content-length")
    if cl and int(cl) > _MAX_BODY_BYTES:
        return Response(
            content=json.dumps({"detail": "Request body too large."}),
            status_code=413,
            media_type="application/json",
        )
    return await call_next(request)


# ══════════════════════════════════════════════════════════════════
#  Health
# ══════════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    client = get_client()
    checks: Dict[str, Any] = {}
    services = [
        ServiceName.SIMULATION,
        ServiceName.LLM,
        ServiceName.DATA,
        ServiceName.VALIDATION,
        ServiceName.TRAFFIC_GOD,
    ]

    async def _probe(svc: ServiceName):
        try:
            return svc.value, await client.get(svc, "/health")
        except Exception as e:
            return svc.value, {"status": "unreachable", "error": str(e)[:100]}

    results = await asyncio.gather(*[_probe(s) for s in services])
    for name, status in results:
        checks[name] = status

    all_ok = all(
        isinstance(v, dict) and v.get("status") == "ok" for v in checks.values()
    )
    return {
        "service": "gateway",
        "status": "ok" if all_ok else "degraded",
        "backends": checks,
    }


# ══════════════════════════════════════════════════════════════════
#  Chat orchestration  (Data → Simulation → LLM → merge)
# ══════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=5000)
    mode: str = "fast"   # fast | deep | agents
    scenario: Optional[Dict[str, Any]] = None


@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    """
    Full orchestrated chat flow:
      1. Parallel: fetch NCR data + live context from Data service
      2. Run simulation engines from Simulation service
      3. Send everything to LLM service for analysis
      4. Merge results into final response
    """
    start = time.time()
    started_at = datetime.utcnow().isoformat() + "Z"
    logs: List[str] = []
    client = get_client()

    # ── Step 1: Gather data (parallel) ────────────────────────────

    logs.append("📡 Fetching NCR data and live context…")

    async def _get_ncr_formatted():
        try:
            return await client.get(ServiceName.DATA, "/ncr/formatted")
        except Exception:
            return {}

    async def _get_live_aqi():
        try:
            return await client.get(
                ServiceName.DATA, "/live/aqi",
                params={"lat": 28.62, "lon": 77.22},
            )
        except Exception:
            return {}

    async def _get_live_route():
        try:
            return await client.get(
                ServiceName.DATA, "/live/route",
                params={
                    "origin_lat": 28.62, "origin_lon": 77.22,
                    "dest_lat": 28.57, "dest_lon": 77.32,
                },
            )
        except Exception:
            return {}

    ncr_fmt, live_aqi, live_route = await asyncio.gather(
        _get_ncr_formatted(), _get_live_aqi(), _get_live_route(),
    )

    ncr_text = ncr_fmt.get("formatted", "")
    live_context = {
        "aqi": live_aqi,
        "traffic": live_route,
    }
    logs.append("✅ Data gathered")

    # ── Step 2: Run simulation engines ────────────────────────────

    engine_results: Dict[str, Any] = {}
    try:
        logs.append("⚙️ Running simulation engines…")
        sim_resp = await client.post(
            ServiceName.SIMULATION,
            "/simulate",
            data={"prompt": req.prompt, "city": "delhi"},
        )
        engine_results = sim_resp
        logs.append(f"✅ Engines complete ({len(sim_resp.get('results', {}))} domains)")
    except (ServiceError, ServiceUnavailableError) as e:
        logs.append(f"⚠️ Simulation service unavailable: {str(e)[:100]}")

    # ── Step 3: LLM analysis ─────────────────────────────────────

    llm_response: Dict[str, Any] = {}
    try:
        logs.append(f"🧠 LLM analysis (mode={req.mode})…")
        llm_resp = await client.post(
            ServiceName.LLM,
            "/chat",
            data={
                "prompt": req.prompt,
                "mode": req.mode,
                "context": {
                    "location": "Delhi NCR",
                    "live_aqi": live_aqi,
                    "live_traffic": live_route,
                },
            },
        )
        llm_response = llm_resp
        logs.append(f"✅ LLM complete ({llm_resp.get('duration_seconds', '?')}s)")
    except (ServiceError, ServiceUnavailableError) as e:
        logs.append(f"⚠️ LLM service unavailable: {str(e)[:100]}")

    # ── Step 4: Merge final response ──────────────────────────────

    summary = llm_response.get("response", "Analysis complete — engine results available.")
    completed_at = datetime.utcnow().isoformat() + "Z"

    impact_cards = engine_results.get("results", {}).get("impactCards", [])
    domains = engine_results.get("results", {}).get("domains", {})
    recommendations = engine_results.get("results", {}).get("recommendations", [])
    warnings = engine_results.get("results", {}).get("warnings", [])
    geojson = engine_results.get("geojson", {"type": "FeatureCollection", "features": []})

    outputs = {
        "tldr": summary,
        "confidenceLevel": "high",
        "impactCards": impact_cards,
        "domains": domains,
        "engineRecommendations": recommendations,
        "engineWarnings": warnings,
        "narrative": [],
        "explanation": [],
        "mapOverlays": {},
        "logs": logs,
        "started_at": started_at,
        "completed_at": completed_at,
        "liveContext": live_context,
        "brainInsights": {
            "orchestrator": f"Gateway → Microservices (mode={req.mode})",
            "models_used": llm_response.get("models_used", {}),
            "engines_run": list(domains.keys()),
        },
    }

    return {
        "summary": summary,
        "baseline": {},
        "ranked": [],
        "edges_geojson": geojson,
        "infrastructure": {"type": "FeatureCollection", "features": []},
        "pollution_hotspots": {"type": "FeatureCollection", "features": []},
        "live": live_context,
        "manifest": {
            "run_id": str(uuid.uuid4()),
            "mode": req.mode,
            "prompt": req.prompt,
            "started_at": started_at,
            "completed_at": completed_at,
            "models": llm_response.get("models_used", {}),
            "runtime_s": round(time.time() - start, 2),
        },
        "outputs": outputs,
    }


# ══════════════════════════════════════════════════════════════════
#  Proxy helpers
# ══════════════════════════════════════════════════════════════════

async def _proxy_get(service: ServiceName, path: str, request: Request):
    """Forward a GET request to a downstream service."""
    client = get_client()
    params = dict(request.query_params)
    try:
        return await client.get(service, path, params=params)
    except ServiceUnavailableError as e:
        raise HTTPException(503, str(e)[:200])
    except ServiceError as e:
        raise HTTPException(502, str(e)[:200])


async def _proxy_post(service: ServiceName, path: str, request: Request):
    """Forward a POST request to a downstream service."""
    client = get_client()
    try:
        data = await request.json()
    except Exception:
        data = {}
    try:
        return await client.post(service, path, data=data)
    except ServiceUnavailableError as e:
        raise HTTPException(503, str(e)[:200])
    except ServiceError as e:
        raise HTTPException(502, str(e)[:200])


# ══════════════════════════════════════════════════════════════════
#  Simulation proxies
# ══════════════════════════════════════════════════════════════════

@app.get("/engines")
async def engines(request: Request):
    return await _proxy_get(ServiceName.SIMULATION, "/engines", request)


@app.post("/simulate")
async def simulate(request: Request):
    return await _proxy_post(ServiceName.SIMULATION, "/simulate", request)


@app.post("/scenarios/compare")
async def scenarios_compare(request: Request):
    return await _proxy_post(ServiceName.SIMULATION, "/scenarios/compare", request)


@app.get("/scenarios/templates")
async def scenarios_templates(request: Request):
    return await _proxy_get(ServiceName.SIMULATION, "/scenarios/templates", request)


@app.post("/scenarios/templates/{template_id}")
async def scenarios_template_run(template_id: str, request: Request):
    return await _proxy_post(
        ServiceName.SIMULATION, f"/scenarios/templates/{template_id}", request,
    )


# ══════════════════════════════════════════════════════════════════
#  Data proxies
# ══════════════════════════════════════════════════════════════════

@app.get("/ncr/summary")
async def ncr_summary(request: Request):
    return await _proxy_get(ServiceName.DATA, "/ncr/summary", request)


@app.get("/live/aqi")
async def live_aqi(request: Request):
    return await _proxy_get(ServiceName.DATA, "/live/aqi", request)


@app.get("/live/route")
async def live_route(request: Request):
    return await _proxy_get(ServiceName.DATA, "/live/route", request)


@app.get("/geocode")
async def geocode(request: Request):
    return await _proxy_get(ServiceName.DATA, "/geocode", request)


@app.get("/reverse-geocode")
async def reverse_geocode(request: Request):
    return await _proxy_get(ServiceName.DATA, "/reverse-geocode", request)


# ══════════════════════════════════════════════════════════════════
#  Validation proxies
# ══════════════════════════════════════════════════════════════════

@app.get("/validation/stats")
async def validation_stats(request: Request):
    return await _proxy_get(ServiceName.VALIDATION, "/stats", request)


@app.get("/validation/entries")
async def validation_list(request: Request):
    return await _proxy_get(ServiceName.VALIDATION, "/entries", request)


@app.post("/validation/entries")
async def validation_create(request: Request):
    return await _proxy_post(ServiceName.VALIDATION, "/entries", request)


@app.post("/validation/entries/{entry_id}/approve")
async def validation_approve(entry_id: str, request: Request):
    return await _proxy_post(
        ServiceName.VALIDATION, f"/entries/{entry_id}/approve", request,
    )


# ══════════════════════════════════════════════════════════════════
#  Traffic God proxies
# ══════════════════════════════════════════════════════════════════

@app.post("/traffic-god-llm")
async def traffic_god_llm(request: Request):
    return await _proxy_post(ServiceName.TRAFFIC_GOD, "/llm", request)


@app.post("/traffic-god/perception")
async def traffic_god_perception(request: Request):
    return await _proxy_post(ServiceName.TRAFFIC_GOD, "/perception", request)


# ── Run standalone ────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("GATEWAY_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
