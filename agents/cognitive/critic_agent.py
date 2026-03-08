"""Critic Agent — Validation, challenge, and consistency checking.

Uses GPT-OSS-120B as an independent cross-validator to:
  - Check simulation results against physical constraints
  - Identify logical inconsistencies between engine outputs
  - Challenge over-optimistic or unrealistic predictions
  - Flag data quality concerns
  - Verify causal reasoning

The Critic is deliberately adversarial — its job is to find problems.
"""

from __future__ import annotations

import time
from typing import Any, Dict

from agents.cognitive.roles import AgentContext, AgentOutput, AgentRole


_CRITIC_SYSTEM = """You are the validation engine of OVERHAUL. Your role is to CHALLENGE and VERIFY.

You receive simulation results and analysis. Your job:

1. PHYSICS CHECK — Are the numbers physically possible?
   - EV adoption cannot reduce PM2.5 more than 35% (brake/tire particles remain)
   - Signal optimization max congestion reduction: ~20%
   - New metro line max mode shift: ~25% in 5-year horizon
   - AQI cannot improve more than 40% from transport-only interventions

2. CONSISTENCY CHECK — Do the engine outputs agree with each other?
   - If transport says congestion drops 30%, does economic engine show matching productivity gain?
   - If EV adoption is 50%, does energy engine show grid load increase?

3. DATA QUALITY — Are claims grounded in the data provided?
   - Flag claims not supported by engine output
   - Flag suspiciously precise numbers with no basis

4. OPTIMISM CHECK — Are predictions realistic?
   - Implementation timelines are often 2-3x longer than planned in India
   - Behavioral adoption curves are S-shaped, not linear
   - Infrastructure projects face land acquisition delays

Output format (JSON):
{
  "validation_pass": true/false,
  "issues": [
    {"severity": "critical|warning|info", "domain": "string", "description": "string", "suggestion": "string"}
  ],
  "consistency_score": 0.0-1.0,
  "adjusted_predictions": {"metric": {"original": x, "adjusted": y, "reason": "string"}},
  "overall_confidence": 0.0-1.0
}"""


async def critique_results(ctx: AgentContext) -> AgentOutput:
    """Cross-validate analysis using GPT-OSS-120B."""
    t0 = time.time()

    # If no reasoning output to critique, skip
    if not ctx.reasoning_output and not ctx.engine_results:
        dur = (time.time() - t0) * 1000
        result = {"validation_pass": True, "issues": [], "consistency_score": 1.0, "overall_confidence": 0.5}
        ctx.critique = result
        ctx.log("critic", "Nothing to critique", dur)
        return AgentOutput(role=AgentRole.CRITIC, result=result, duration_ms=dur)

    import json

    prompt_parts = [f"# Original Query\n{ctx.query}\n"]

    if ctx.engine_results:
        prompt_parts.append("# Engine Simulation Results\n")
        for name, res in ctx.engine_results.items():
            if name == "raw":
                continue
            if isinstance(res, dict):
                prompt_parts.append(f"## {name}\n{json.dumps(res, indent=1, default=str)[:2000]}\n")

    if ctx.reasoning_output.get("analysis"):
        analysis = ctx.reasoning_output["analysis"]
        if len(analysis) > 3000:
            analysis = analysis[:3000] + "\n... (truncated)"
        prompt_parts.append(f"# Analysis to Validate\n{analysis}\n")

    prompt = "\n".join(prompt_parts)
    prompt += "\n\nValidate these results. Output ONLY the JSON format specified."

    try:
        from llm.chat import llm_chat_json
        result = await llm_chat_json(
            prompt=prompt,
            system=_CRITIC_SYSTEM,
            prefer="validate",  # Routes to GPT-OSS-120B
            max_output_tokens=3000,
        )

        # Ensure required fields
        result.setdefault("validation_pass", True)
        result.setdefault("issues", [])
        result.setdefault("consistency_score", 0.8)
        result.setdefault("overall_confidence", 0.7)

        ctx.critique = result
        dur = (time.time() - t0) * 1000
        n_issues = len(result.get("issues", []))
        ctx.log("critic", f"Found {n_issues} issues, confidence={result['overall_confidence']}", dur)
        return AgentOutput(role=AgentRole.CRITIC, result=result, duration_ms=dur, model_used="gpt_oss")

    except Exception as e:
        dur = (time.time() - t0) * 1000
        result = _heuristic_critique(ctx)
        ctx.critique = result
        ctx.errors.append(f"Critic LLM failed, used heuristic: {e}")
        ctx.log("critic", f"Heuristic critique: {e}", dur)
        return AgentOutput(role=AgentRole.CRITIC, result=result, duration_ms=dur, error=str(e))


def _heuristic_critique(ctx: AgentContext) -> Dict[str, Any]:
    """Rule-based validation when LLM is unavailable."""
    issues = []

    # Check engine results for physics violations
    for name, result in ctx.engine_results.items():
        if not isinstance(result, dict):
            continue
        metrics = result.get("metrics", {})

        # AQI improvement > 40% from transport alone is suspicious
        aqi_change = metrics.get("aqi_change_pct", 0)
        if abs(aqi_change) > 40:
            issues.append({
                "severity": "warning",
                "domain": name,
                "description": f"AQI change of {aqi_change}% exceeds physical limit for transport interventions",
                "suggestion": "Cap at 35-40% maximum",
            })

        # Congestion improvement > 50% is unrealistic
        congestion_change = metrics.get("congestion_reduction_pct", 0)
        if congestion_change > 50:
            issues.append({
                "severity": "warning",
                "domain": name,
                "description": f"Congestion reduction of {congestion_change}% is unrealistic",
                "suggestion": "Real-world max is typically 25-35%",
            })

    return {
        "validation_pass": len([i for i in issues if i["severity"] == "critical"]) == 0,
        "issues": issues,
        "consistency_score": 0.7 if issues else 0.9,
        "overall_confidence": 0.6,
    }
