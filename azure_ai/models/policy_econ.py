from __future__ import annotations

from typing import Any, Dict, Optional

from azure_ai.config import AzureConfig, azure_openai_deployment_for, azure_openai_enabled, load_azure_config
from azure_ai.openai.chat import azure_openai_chat_json


def _econ_calcs(*, travel_delta_pct: float, pm_delta_pct: float, baseline_travel_time_min: float) -> Dict[str, Any]:
    """Deterministic economics back-of-envelope.

    Keep this stable and explainable; the LLM should narrate, not invent numbers.
    """
    # Assume a corridor with 120k daily trips impacted (placeholder). You can replace with real counts.
    trips_per_day = 120_000

    # Travel time savings: if travel_delta_pct is negative => improvement.
    tt_change_min = baseline_travel_time_min * (travel_delta_pct / 100.0)
    minutes_saved_per_trip = max(0.0, -tt_change_min)

    # Value of time (INR/min). Conservative placeholder.
    inr_per_min = 2.5

    daily_time_value = minutes_saved_per_trip * trips_per_day * inr_per_min
    annual_time_value_crore = (daily_time_value * 365.0) / 1e7

    # Health/economic benefit from PM2.5 reduction: if pm_delta_pct negative => improvement.
    pm_improvement = max(0.0, -pm_delta_pct)
    annual_health_benefit_crore = 20.0 * (pm_improvement / 10.0)  # placeholder scaling

    total_benefit_crore = annual_time_value_crore + annual_health_benefit_crore

    return {
        "assumptions": {
            "trips_per_day": trips_per_day,
            "value_of_time_inr_per_min": inr_per_min,
            "health_benefit_scaling": "20 Cr per 10% PM2.5 improvement (placeholder)",
        },
        "time_savings": {
            "minutes_saved_per_trip": round(minutes_saved_per_trip, 3),
            "annual_value_crore": round(annual_time_value_crore, 2),
        },
        "health_benefit": {
            "pm_improvement_pct": round(pm_improvement, 2),
            "annual_value_crore": round(annual_health_benefit_crore, 2),
        },
        "total_annual_benefit_crore": round(total_benefit_crore, 2),
    }


async def policy_econ_assess(
    *,
    prompt: str,
    travel_delta_pct: float,
    pm_delta_pct: float,
    baseline_metrics: Dict[str, Any],
    candidate_metrics: Dict[str, Any],
    cfg: Optional[AzureConfig] = None,
) -> Dict[str, Any]:
    cfg = cfg or load_azure_config()

    baseline_tt = float(baseline_metrics.get("avg_travel_time_min", 35.0) or 35.0)

    calcs = _econ_calcs(
        travel_delta_pct=float(travel_delta_pct or 0.0),
        pm_delta_pct=float(pm_delta_pct or 0.0),
        baseline_travel_time_min=baseline_tt,
    )

    base = {
        "domain": "policy_econ",
        "calcs": calcs,
    }

    if not azure_openai_enabled(cfg):
        base["llm_used"] = False
        return base

    deployment = azure_openai_deployment_for("policy_econ", cfg)

    system = (
        "You are the Policy & Economics Model. You must ONLY use provided calculations and metrics. "
        "Return ONLY JSON. If something is missing, state it as a limitation."
    )

    user = {
        "user_prompt": prompt,
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "travel_delta_pct": travel_delta_pct,
        "pm_delta_pct": pm_delta_pct,
        "calcs": calcs,
        "task": "Turn the deterministic calcs into a policy/econ briefing with risks + data needs.",
        "output_schema": {
            "summary": "string",
            "cost_considerations": ["string"],
            "benefits": ["string"],
            "risks": ["string"],
            "data_needed_to_improve": ["string"],
            "confidence": "low|medium|high",
        },
    }

    assessment = await azure_openai_chat_json(
        prompt=str(user),
        system=system,
        cfg=cfg,
        deployment=deployment,
        max_output_tokens=3000,
    )

    return {**base, "llm_used": True, "assessment": assessment}
