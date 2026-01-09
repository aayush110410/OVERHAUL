from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx

from azure_ai.config import AzureConfig, azure_openai_deployment_for, azure_openai_enabled, load_azure_config
from azure_ai.openai.chat import azure_openai_chat_json


@dataclass(frozen=True)
class WeatherSnapshot:
    source: str
    lat: float
    lon: float
    summary: Dict[str, Any]


async def fetch_weather_snapshot(*, lat: float, lon: float) -> WeatherSnapshot:
    """Fetch a compact weather snapshot via Open-Meteo (no key).

    This is intentionally small to control token/cost.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": str(lat),
        "longitude": str(lon),
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max",
        "forecast_days": "3",
        "timezone": "auto",
    }

    async with httpx.AsyncClient(timeout=12.0) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

    compact = {
        "current": data.get("current", {}),
        "daily": {
            k: data.get("daily", {}).get(k)
            for k in [
                "time",
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
                "wind_speed_10m_max",
            ]
        },
    }

    return WeatherSnapshot(source="open-meteo.com", lat=lat, lon=lon, summary=compact)


async def weather_model_assess(*, prompt: str, lat: float, lon: float, cfg: Optional[AzureConfig] = None) -> Dict[str, Any]:
    cfg = cfg or load_azure_config()
    snap = await fetch_weather_snapshot(lat=lat, lon=lon)

    # Deterministic baseline
    base = {
        "domain": "weather",
        "source": snap.source,
        "lat": lat,
        "lon": lon,
        "snapshot": snap.summary,
        "impact_notes": [
            "Rain typically increases congestion and can temporarily suppress dust-related PM.",
            "High wind can disperse pollutants but may also resuspend road dust.",
        ],
    }

    if not azure_openai_enabled(cfg):
        base["llm_used"] = False
        return base

    deployment = azure_openai_deployment_for("weather", cfg)

    system = (
        "You are the Weather Model. You translate near-term weather into traffic and air-quality impacts. "
        "Return ONLY JSON. Be conservative and cite the fields you used from the snapshot."
    )

    user = {
        "user_prompt": prompt,
        "weather_snapshot": snap.summary,
        "task": "Summarize weather impacts for traffic, AQI, and confidence/limitations.",
        "output_schema": {
            "summary": "string",
            "traffic_impact": "string",
            "aqi_impact": "string",
            "risk_flags": ["string"],
            "confidence": "low|medium|high",
            "fields_used": ["string"],
        },
    }

    enriched = await azure_openai_chat_json(
        prompt=str(user),
        system=system,
        cfg=cfg,
        deployment=deployment,
        max_output_tokens=3000,
    )

    return {**base, "llm_used": True, "assessment": enriched}
