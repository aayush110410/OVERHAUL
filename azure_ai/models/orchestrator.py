from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional, Tuple

from azure_ai.config import AzureConfig, load_azure_config

from .behavior import behavior_model_assess
from .policy_econ import policy_econ_assess
from .weather import weather_model_assess


async def run_domain_models(
    *,
    prompt: str,
    signals: Dict[str, Any],
    live_context: Dict[str, Any],
    baseline_metrics: Dict[str, Any],
    candidate_metrics: Dict[str, Any],
    travel_delta_pct: float,
    pm_delta_pct: float,
    cfg: Optional[AzureConfig] = None,
    default_latlon: Tuple[float, float] = (28.62, 77.35),
) -> Dict[str, Any]:
    """Run non-traffic/non-AQI domain models.

    Traffic + AQI are handled elsewhere (Traffic God + AQI agent); these models
    add additional intelligence layers for the master response.
    """
    cfg = cfg or load_azure_config()

    lat, lon = default_latlon
    # If live_context has travel geometry, we could derive a midpoint later.

    weather_task = asyncio.create_task(weather_model_assess(prompt=prompt, lat=lat, lon=lon, cfg=cfg))
    behavior_task = asyncio.create_task(behavior_model_assess(prompt=prompt, signals=signals, cfg=cfg))
    policy_task = asyncio.create_task(
        policy_econ_assess(
            prompt=prompt,
            travel_delta_pct=travel_delta_pct,
            pm_delta_pct=pm_delta_pct,
            baseline_metrics=baseline_metrics,
            candidate_metrics=candidate_metrics,
            cfg=cfg,
        )
    )

    weather, behavior, policy = await asyncio.gather(weather_task, behavior_task, policy_task, return_exceptions=True)

    def _ok(x: Any) -> Dict[str, Any]:
        if isinstance(x, Exception):
            return {"error": str(x)[:200]}
        return x

    return {
        "weather": _ok(weather),
        "behavior": _ok(behavior),
        "policy_econ": _ok(policy),
    }
