"""
Data Microservice
==================

Standalone FastAPI service owning all data operations:
  - NCR CSV data (AQI, traffic) — via adapter pipeline
  - Live external data (AQI APIs, OSRM routing, TomTom) — via adapters
  - Geocoding (Nominatim) — via adapter
  - Full NCR context aggregation — via adapter manager

Port: 8003 (default)

Endpoints:
  GET  /health
  GET  /ncr/summary
  GET  /ncr/formatted
  GET  /ncr/engine-input
  GET  /ncr/context         (NEW — full aggregated context)
  GET  /live/aqi
  GET  /live/route
  GET  /geocode
  GET  /reverse-geocode
  GET  /adapters            (NEW — list all adapters + status)
  GET  /adapters/health     (NEW — health check all adapters)
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agents.ncr_data_loader import (  # noqa: E402
    get_ncr_summary,
    format_ncr_data_for_prompt,
    get_latest_aqi,
    get_traffic_summary,
    get_aqi_category,
)
from data_integration.bridge import ncr_data_to_engine_input  # noqa: E402
from data_integration.adapters.manager import (  # noqa: E402
    init_adapters,
    get_ncr_context,
    get_engine_input,
    format_for_prompt,
    adapter_health,
)
from data_integration.adapters.base import get_adapter_registry  # noqa: E402

app = FastAPI(title="OVERHAUL Data Service", version="2.0.0")


@app.on_event("startup")
async def _startup():
    await init_adapters()


# ── Health ────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    try:
        summary = get_ncr_summary()
        cities = list(summary.get("aqi", {}).keys())
        registry = get_adapter_registry()
        return {
            "service": "data",
            "status": "ok",
            "cities_loaded": cities,
            "adapters_registered": len(registry.list_adapters()),
        }
    except Exception as e:
        return {"service": "data", "status": "degraded", "error": str(e)[:100]}


# ── NCR CSV data ──────────────────────────────────────────────────

@app.get("/ncr/summary")
async def ncr_summary(city: str = Query("all")):
    """Full NCR summary (AQI + traffic for all cities)."""
    summary = get_ncr_summary()
    if city.lower() != "all":
        title = city.title()
        return {
            "aqi": {title: summary.get("aqi", {}).get(title, {})},
            "traffic": {title: summary.get("traffic", {}).get(title, {})},
        }
    return summary


@app.get("/ncr/formatted")
async def ncr_formatted():
    """NCR data pre-formatted as text for LLM prompt injection."""
    return {"formatted": format_ncr_data_for_prompt()}


@app.get("/ncr/engine-input")
async def ncr_engine_input(city: str = Query("delhi")):
    """NCR data converted to engine-compatible input dict."""
    summary = get_ncr_summary()
    data = ncr_data_to_engine_input(summary, city=city)
    return {"city": city, "engine_input": data}


# ── Live external data ────────────────────────────────────────────

@app.get("/live/aqi")
async def live_aqi(
    lat: float = Query(28.62),
    lon: float = Query(77.35),
):
    """Live AQI from Open-Meteo / OpenAQ."""
    try:
        # Import the fetch functions from the monolith
        # These are async functions using httpx internally
        import httpx
        async with httpx.AsyncClient(timeout=15.0) as client:
            # Open-Meteo Air Quality API
            resp = await client.get(
                "https://air-quality-api.open-meteo.com/v1/air-quality",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "hourly": "pm2_5,pm10,us_aqi",
                    "past_days": 1,
                    "forecast_days": 1,
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                hourly = data.get("hourly", {})
                pm25_vals = [v for v in (hourly.get("pm2_5") or []) if v is not None]
                aqi_vals = [v for v in (hourly.get("us_aqi") or []) if v is not None]
                latest_pm25 = pm25_vals[-1] if pm25_vals else None
                latest_aqi = aqi_vals[-1] if aqi_vals else None
                return {
                    "lat": lat, "lon": lon,
                    "pm25": latest_pm25,
                    "aqi": latest_aqi,
                    "category": get_aqi_category(latest_aqi) if latest_aqi else "Unknown",
                    "source": "open-meteo",
                }
            raise HTTPException(502, "AQI provider returned error")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"AQI fetch error: {str(e)[:100]}")


@app.get("/live/route")
async def live_route(
    from_lat: float = Query(28.5700),
    from_lon: float = Query(77.3219),
    to_lat: float = Query(28.6315),
    to_lon: float = Query(77.2167),
):
    """Live route metrics from OSRM."""
    try:
        import httpx
        coords = f"{from_lon},{from_lat};{to_lon},{to_lat}"
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"https://router.project-osrm.org/route/v1/driving/{coords}",
                params={"overview": "false", "alternatives": "false"},
            )
            if resp.status_code == 200:
                data = resp.json()
                routes = data.get("routes", [])
                if routes:
                    r = routes[0]
                    dist_km = r["distance"] / 1000
                    duration_min = r["duration"] / 60
                    speed_kmh = dist_km / (duration_min / 60) if duration_min > 0 else 0
                    return {
                        "distance_km": round(dist_km, 2),
                        "duration_min": round(duration_min, 1),
                        "speed_kmh": round(speed_kmh, 1),
                        "source": "osrm",
                    }
            raise HTTPException(502, "OSRM returned error")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Route fetch error: {str(e)[:100]}")


# ── Geocoding ─────────────────────────────────────────────────────

@app.get("/geocode")
async def geocode(q: str = Query(..., min_length=2, max_length=200)):
    """Forward geocode via Nominatim."""
    import httpx
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": q, "format": "json", "limit": 5},
            headers={"User-Agent": "OVERHAUL/1.0"},
        )
        if resp.status_code == 200:
            return {"results": resp.json()}
        raise HTTPException(502, "Geocoding provider error")


@app.get("/reverse-geocode")
async def reverse_geocode(
    lat: float = Query(...),
    lon: float = Query(...),
):
    """Reverse geocode via Nominatim."""
    import httpx
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lon, "format": "json"},
            headers={"User-Agent": "OVERHAUL/1.0"},
        )
        if resp.status_code == 200:
            return resp.json()
        raise HTTPException(502, "Reverse geocoding provider error")


# ══════════════════════════════════════════════════════════════════
#  Adapter-powered endpoints
# ══════════════════════════════════════════════════════════════════

@app.get("/ncr/context")
async def ncr_context(
    city: str = Query("delhi"),
    include_live: bool = Query(True),
    lat: float = Query(28.62),
    lon: float = Query(77.22),
):
    """Full NCR context via adapter pipeline (CSV + live + historical)."""
    ctx = await get_ncr_context(city=city, include_live=include_live, lat=lat, lon=lon)
    return ctx


@app.get("/adapters")
async def list_adapters():
    """List all registered data adapters and their status."""
    registry = get_adapter_registry()
    return {"adapters": registry.list_adapters()}


@app.get("/adapters/health")
async def adapters_health():
    """Health check across all adapters."""
    return await adapter_health()


# ── Run standalone ────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("DATA_PORT", "8003"))
    uvicorn.run(app, host="0.0.0.0", port=port)
