"""Reasoner Agent — Deep analysis and causal inference.

The heavyweight agent. Uses Llama 3.3 70B for deep analytical reasoning
over simulation engine results + research data.

Produces:
  - Causal analysis: what causes what
  - Impact chains: how interventions cascade through systems
  - Quantified predictions with confidence intervals
  - Counter-intuitive findings
  - Cross-domain interactions
"""

from __future__ import annotations

import time
from typing import Any, Dict

from agents.cognitive.roles import AgentContext, AgentOutput, AgentRole


_REASONER_SYSTEM = """You are the deep analysis engine of OVERHAUL, an AI-powered urban simulation platform for Delhi NCR.

You have access to simulation engine results and real data. Your job is to:
1. Identify CAUSAL CHAINS — how does each intervention propagate through interconnected systems?
2. Quantify CROSS-DOMAIN IMPACTS — e.g., how does a metro extension affect AQI, GDP, energy grid?
3. Find COUNTER-INTUITIVE effects — where does common intuition fail?
4. Assess CONFIDENCE — where are the predictions reliable vs uncertain?
5. Identify SECOND-ORDER EFFECTS — what happens 6-12 months after implementation?

Output format (markdown):
## Executive Analysis
[2-3 sentence bottom line]

## Causal Chain Analysis
[For each intervention, trace the chain: Action → Direct Effect → Secondary Effect → Tertiary Effect]

## Cross-Domain Impact Matrix
| Domain | Direct Impact | Indirect Impact | Confidence |
[table]

## Counter-Intuitive Findings
[Things that surprised the model]

## Confidence Assessment
[Where predictions are strong vs weak, and why]

## Second-Order Effects (6-12 month horizon)
[Delayed consequences]

Ground every claim in the data provided. Cite specific numbers from engine outputs.
Delhi NCR context: population ~20M commuters, 7M daily commuters, PM2.5 typically 150-300 μg/m³ in winter."""


async def deep_reasoning(ctx: AgentContext) -> AgentOutput:
    """Run deep causal analysis using Llama 3.3 70B."""
    t0 = time.time()

    # Build the analysis prompt with all available context
    prompt_parts = [f"# User Query\n{ctx.query}\n"]

    # Parsed intent
    if ctx.parsed_intent:
        prompt_parts.append(f"# Parsed Intent\n```json\n{_compact_json(ctx.parsed_intent)}\n```\n")

    # Engine results
    if ctx.engine_results:
        prompt_parts.append("# Simulation Engine Results\n")
        for engine_name, result in ctx.engine_results.items():
            if engine_name == "raw":
                continue
            if isinstance(result, dict):
                prompt_parts.append(f"## {engine_name}\n```json\n{_compact_json(result)}\n```\n")
            else:
                prompt_parts.append(f"## {engine_name}\n{result}\n")

    # Research data
    if ctx.research_data:
        ncr = ctx.research_data.get("ncr_data", {}).get("data", {})
        if ncr:
            prompt_parts.append(f"# NCR Ground Truth Data\n```json\n{_compact_json(ncr)}\n```\n")

    prompt = "\n".join(prompt_parts)
    prompt += "\n\nProvide deep causal analysis following the format specified in your instructions."

    try:
        from llm.chat import llm_chat_text
        response = await llm_chat_text(
            prompt=prompt,
            system=_REASONER_SYSTEM,
            prefer="analysis",  # Routes to Llama 3.3 70B
            max_output_tokens=8000,
        )
        result = {"analysis": response, "model": "llama-3.3-70b"}
        ctx.reasoning_output = result
        dur = (time.time() - t0) * 1000
        ctx.log("reasoner", f"Deep analysis complete ({len(response)} chars)", dur)
        return AgentOutput(role=AgentRole.REASONER, result=result, duration_ms=dur, model_used="llama")

    except Exception as e:
        dur = (time.time() - t0) * 1000
        ctx.errors.append(f"Reasoner failed: {e}")
        ctx.log("reasoner", f"Failed: {e}", dur)
        return AgentOutput(role=AgentRole.REASONER, result={}, duration_ms=dur, error=str(e))


def _compact_json(obj: Any) -> str:
    """JSON dump truncated to reasonable size for prompts."""
    import json
    try:
        s = json.dumps(obj, indent=1, default=str)
        if len(s) > 4000:
            s = s[:4000] + "\n... (truncated)"
        return s
    except Exception:
        return str(obj)[:4000]
