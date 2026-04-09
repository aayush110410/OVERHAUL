from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(slots=True)
class Demo4Settings:
    service_name: str = "OVERHAUL Demo 4 Codex"
    database_url: str | None = os.getenv("DEMO4_DATABASE_URL") or os.getenv("DATABASE_URL")
    stream_interval_seconds: float = float(os.getenv("DEMO4_STREAM_INTERVAL_SECONDS", "8"))
    traffic_particle_budget: int = int(os.getenv("DEMO4_TRAFFIC_PARTICLE_BUDGET", "24000"))
    satellite_limit: int = int(os.getenv("DEMO4_SATELLITE_LIMIT", "600"))
    flight_limit: int = int(os.getenv("DEMO4_FLIGHT_LIMIT", "700"))
    building_limit: int = int(os.getenv("DEMO4_BUILDING_LIMIT", "900"))
    weather_city_limit: int = int(os.getenv("DEMO4_WEATHER_CITY_LIMIT", "12"))
    transport_radius_m: int = int(os.getenv("DEMO4_TRANSPORT_RADIUS_M", "9000"))
    enable_external_llms: bool = os.getenv("DEMO4_ENABLE_EXTERNAL_LLMS", "false").lower() == "true"
    openrouter_api_key: str | None = os.getenv("OPENROUTER_API_KEY")
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")
    user_agent: str = os.getenv("DEMO4_USER_AGENT", "OVERHAUL-Demo4-Codex/1.0")


def get_settings() -> Demo4Settings:
    return Demo4Settings()
