"""
Data Adapter Manager
=====================

Configures and wires all adapters into the central registry.
Provides high-level query methods that aggregate results across
adapters with fallback chains.

Usage::

    from data_integration.adapters.manager import init_adapters, get_ncr_context

    await init_adapters()

    # High-level: get full NCR context for the chat pipeline
    ctx = await get_ncr_context(city="delhi")
    # → { "aqi": {...}, "traffic": {...}, "live_aqi": {...},
    #     "route": {...}, "historical": {...} }
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from data_integration.adapters.base import (
    AdapterRegistry,
    DataDomain,
    DataResult,
    get_adapter_registry,
)
from data_integration.adapters.csv_adapter import (
    NCRAqiCSVAdapter,
    NCRTrafficCSVAdapter,
    HistoricalJSONAdapter,
)
from data_integration.adapters.api_adapter import (
    OpenMeteoAqiAdapter,
    OpenMeteoWeatherAdapter,
    OSRMRoutingAdapter,
    TomTomFlowAdapter,
    NominatimAdapter,
)
from data_integration.adapters.db_adapter import ValidationDBAdapter


# ── Bootstrap ─────────────────────────────────────────────────────

_initialised = False


async def init_adapters() -> AdapterRegistry:
    """
    Register all adapters.  Safe to call multiple times (idempotent).
    """
    global _initialised
    registry = get_adapter_registry()

    if _initialised:
        return registry

    # Static file adapters
    registry.register(NCRAqiCSVAdapter())
    registry.register(NCRTrafficCSVAdapter())
    registry.register(HistoricalJSONAdapter())

    # Live API adapters
    registry.register(OpenMeteoAqiAdapter(ttl_seconds=300))     # 5-min cache
    registry.register(OpenMeteoWeatherAdapter(ttl_seconds=600)) # 10-min cache
    registry.register(OSRMRoutingAdapter(ttl_seconds=120))      # 2-min cache
    registry.register(TomTomFlowAdapter(ttl_seconds=120))       # 2-min cache
    registry.register(NominatimAdapter(ttl_seconds=600))        # 10-min cache

    # Database adapter
    registry.register(ValidationDBAdapter())

    _initialised = True
    return registry


# ══════════════════════════════════════════════════════════════════
#  High-level query helpers
# ══════════════════════════════════════════════════════════════════

async def get_ncr_context(
    city: str = "delhi",
    include_live: bool = True,
    lat: float = 28.62,
    lon: float = 77.22,
) -> Dict[str, Any]:
    """
    Gather a complete NCR data context for the chat pipeline.

    Returns a dict with keys:
      aqi          — CSV AQI summary (per-city)
      traffic      — CSV traffic summary (per-city)
      live_aqi     — Open-Meteo live PM2.5/AQI (if include_live)
      route        — OSRM route distance/speed (if include_live)
      tomtom       — TomTom flow data (if include_live and key set)
      historical   — yearly historical metrics
      sources      — list of source labels used
    """
    registry = get_adapter_registry()
    sources: List[str] = []

    # ── Parallel: CSV data (always) + live data (optional) ────────

    tasks: Dict[str, Any] = {}
    tasks["csv_aqi"] = registry.query_one("ncr_aqi_csv", city=city)
    tasks["csv_traffic"] = registry.query_one("ncr_traffic_csv", city=city)
    tasks["historical"] = registry.query_one("historical_json")

    if include_live:
        tasks["live_aqi"] = registry.query_one("open_meteo_aqi", lat=lat, lon=lon)
        tasks["weather"] = registry.query_one("open_meteo_weather", lat=lat, lon=lon)
        tasks["route"] = registry.query_one("osrm_routing")
        tasks["tomtom"] = registry.query_one("tomtom_flow", lat=lat, lon=lon)

    keys = list(tasks.keys())
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    result_map: Dict[str, DataResult] = {}
    for k, r in zip(keys, results):
        if isinstance(r, Exception):
            result_map[k] = DataResult(data=None, error=str(r)[:200])
        else:
            result_map[k] = r

    # ── Assemble context ──────────────────────────────────────────

    ctx: Dict[str, Any] = {}

    # CSV AQI
    csv_aqi = result_map["csv_aqi"]
    if csv_aqi.ok and csv_aqi.data:
        ctx["aqi"] = csv_aqi.data
        sources.append(csv_aqi.source)

    # CSV Traffic
    csv_traffic = result_map["csv_traffic"]
    if csv_traffic.ok and csv_traffic.data:
        ctx["traffic"] = csv_traffic.data
        sources.append(csv_traffic.source)

    # Historical
    hist = result_map["historical"]
    if hist.ok and hist.data:
        ctx["historical"] = hist.data
        sources.append(hist.source)

    # Live AQI
    if include_live:
        live_aqi = result_map.get("live_aqi")
        if live_aqi and live_aqi.ok and live_aqi.data:
            ctx["live_aqi"] = live_aqi.data
            sources.append(live_aqi.source)

        route = result_map.get("route")
        if route and route.ok and route.data:
            ctx["route"] = route.data
            sources.append(route.source)

        weather = result_map.get("weather")
        if weather and weather.ok and weather.data:
            ctx["weather"] = weather.data
            sources.append(weather.source)

        tomtom = result_map.get("tomtom")
        if tomtom and tomtom.ok and tomtom.data:
            ctx["tomtom"] = tomtom.data
            sources.append(tomtom.source)

    ctx["sources"] = sources
    return ctx


async def get_engine_input(city: str = "delhi") -> Dict[str, Any]:
    """
    Build the engine-compatible input dict from adapter data.
    Mirrors data_integration.bridge.ncr_data_to_engine_input()
    but sourced through the adapter pipeline.
    """
    registry = get_adapter_registry()

    aqi_result, traffic_result = await asyncio.gather(
        registry.query_one("ncr_aqi_csv", city=city),
        registry.query_one("ncr_traffic_csv", city=city),
    )

    data: Dict[str, Any] = {"city": city}

    if aqi_result.ok and aqi_result.data:
        city_aqi = aqi_result.data.get(city.title(), {})
        if city_aqi:
            data["baseline_pm25"] = city_aqi.get("pm25", 285)

    if traffic_result.ok and traffic_result.data:
        city_traffic = traffic_result.data.get(city.title(), {})
        if city_traffic:
            data["baseline_speed_kmh"] = city_traffic.get("avg_speed_kmph", 25)
            data["baseline_congestion_pct"] = city_traffic.get("congestion_pct", 55)
            ev_pct = city_traffic.get("ev_percentage", 3)
            data["ev_count"] = int(12_000_000 * ev_pct / 100)

    return data


async def format_for_prompt(city: Optional[str] = None) -> str:
    """
    Format adapter data as a text block for LLM prompt grounding.
    Replaces the monolithic format_ncr_data_for_prompt().
    """
    registry = get_adapter_registry()

    aqi_result, traffic_result = await asyncio.gather(
        registry.query_one("ncr_aqi_csv", city=city) if city else registry.query_one("ncr_aqi_csv"),
        registry.query_one("ncr_traffic_csv", city=city) if city else registry.query_one("ncr_traffic_csv"),
    )

    lines = ["=== LIVE NCR DATA (from trained CSV) ===", ""]

    lines.append("📊 AIR QUALITY (AQI):")
    if aqi_result.ok and aqi_result.data:
        for cname, d in aqi_result.data.items():
            lines.append(
                f"  • {cname}: PM2.5={d.get('pm25', 'N/A')} µg/m³, "
                f"AQI={d.get('aqi', 'N/A')} ({d.get('category', '?')})"
            )
    else:
        lines.append("  • Data unavailable")

    lines.append("")
    lines.append("🚗 TRAFFIC:")
    if traffic_result.ok and traffic_result.data:
        for cname, d in traffic_result.data.items():
            lines.append(
                f"  • {cname}: Speed={d.get('avg_speed_kmph', 'N/A')} km/h, "
                f"Congestion={d.get('congestion_pct', 'N/A')}%, "
                f"EV={d.get('ev_percentage', 'N/A')}%"
            )
    else:
        lines.append("  • Data unavailable")

    return "\n".join(lines)


async def adapter_health() -> Dict[str, Any]:
    """Return health status for all adapters."""
    return await get_adapter_registry().health()
