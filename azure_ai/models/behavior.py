from __future__ import annotations

import math
from typing import Any, Dict, Optional

from azure_ai.config import AzureConfig, azure_openai_deployment_for, azure_openai_enabled, load_azure_config
from azure_ai.openai.chat import azure_openai_chat_json


def _logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _heuristic_adoption(ev_share_pct: float) -> Dict[str, Any]:
    # Very simple behavior heuristics to keep the system honest when LLM is off.
    # Interpreting EV share as intervention intensity proxy.
    intensity = max(0.0, min(1.0, ev_share_pct / 100.0))

    compliance = 0.55 + 0.25 * intensity  # 0.55 → 0.80
    mode_shift = 0.06 + 0.10 * intensity  # 6% → 16%
    rebound = 0.03 + 0.05 * intensity  # induced demand risk

    return {
        "estimated_compliance": round(compliance, 2),
        "estimated_mode_shift": round(mode_shift, 2),
        "rebound_risk": round(rebound, 2),
        "notes": [
            "Heuristics only; calibrate with surveys / mobility data.",
            "Behavior varies by corridor, income group, and trip purpose.",
        ],
    }


async def behavior_model_assess(
    *,
    prompt: str,
    signals: Dict[str, Any],
    cfg: Optional[AzureConfig] = None,
) -> Dict[str, Any]:
    cfg = cfg or load_azure_config()
    ev_share_pct = float(signals.get("ev_share_pct", 0.0) or 0.0)

    base = {
        "domain": "behavior",
        "heuristics": _heuristic_adoption(ev_share_pct),
    }

    if not azure_openai_enabled(cfg):
        base["llm_used"] = False
        return base

    deployment = azure_openai_deployment_for("behavior", cfg)

    system = (
        "You are the Behavioral Model. You reason about adoption, compliance, and human responses to policies. "
        "Return ONLY JSON and invent demographics; if unknown, state assumptions."
    )

    user = {
        "user_prompt": prompt,
        "signals": signals,
        "heuristics": base["heuristics"],
        "task": "Provide a behavior impact assessment for the intervention and note risks/assumptions.",
        "output_schema": {
            "summary": "string",
            "adoption_drivers": ["string"],
            "equity_considerations": ["string"],
            "risks": ["string"],
            "assumptions": ["string"],
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
