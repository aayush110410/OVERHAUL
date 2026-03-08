"""Synthesizer Agent — Final output generation.

Merges all agent outputs into a coherent, user-facing response:
  - Reasoner's deep analysis
  - Critic's validation and adjustments
  - Engine simulation metrics
  - Research data grounding

Uses Gemini 3.1 Pro for its superior multi-source synthesis capability.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict

from agents.cognitive.roles import AgentContext, AgentOutput, AgentRole


_SYNTHESIZER_SYSTEM = """You are the final synthesis engine of OVERHAUL, an AI-powered decision intelligence platform for Delhi NCR.

You receive outputs from multiple analysis agents and simulation engines. Your job is to produce
the FINAL user-facing response that is:

1. COHERENT — Merge multiple sources into a single narrative
2. GROUNDED — Every claim backed by engine data or research
3. ADJUSTED — Incorporate the Critic's corrections
4. ACTIONABLE — Clear recommendations with priorities
5. HONEST — State confidence levels and uncertainties

Response format:

## Executive Summary
[3-5 sentences. The bottom line.]

## Key Findings
| Metric | Current | Projected | Change | Confidence |
[table of quantified impacts]

## Detailed Analysis
[Structured by domain, referencing engine outputs]

## Cross-System Interactions
[How changes in one domain cascade to others]

## Recommendations
1. [Priority 1: action + expected impact + timeline]
2. [Priority 2: ...]
3. [Priority 3: ...]

## Risks & Uncertainties
[What could go wrong, data gaps, model limitations]

## Data Sources
[List what data was used and its recency]

Rules:
- If the Critic flagged issues, acknowledge and adjust
- Never claim precision beyond what the models can deliver
- Cite specific numbers from engine outputs
- Include implementation difficulty (Easy/Medium/Hard/Very Hard)
- Use ₹ for costs, km for distances, μg/m³ for PM2.5"""


async def synthesize_response(ctx: AgentContext) -> AgentOutput:
    """Produce final user-facing response from all agent outputs."""
    t0 = time.time()

    prompt_parts = [f"# User Query\n{ctx.query}\n"]

    # Reasoner output
    if ctx.reasoning_output.get("analysis"):
        analysis = ctx.reasoning_output["analysis"]
        if len(analysis) > 5000:
            analysis = analysis[:5000] + "\n... (truncated)"
        prompt_parts.append(f"# Deep Analysis (Llama 3.3 70B)\n{analysis}\n")

    # Engine results
    if ctx.engine_results:
        prompt_parts.append("# Simulation Engine Results\n")
        for name, res in ctx.engine_results.items():
            if name == "raw":
                continue
            if isinstance(res, dict):
                prompt_parts.append(f"## {name}\n{json.dumps(res, indent=1, default=str)[:2000]}\n")

    # Critique
    if ctx.critique:
        prompt_parts.append(f"# Validation (GPT-OSS-120B Critic)\n{json.dumps(ctx.critique, indent=1, default=str)[:2000]}\n")

    # Research data
    ncr = ctx.research_data.get("ncr_data", {}).get("data", {})
    if ncr:
        prompt_parts.append(f"# NCR Ground Truth\n{json.dumps(ncr, indent=1, default=str)[:1500]}\n")

    prompt = "\n".join(prompt_parts)
    prompt += "\n\nSynthesize all of the above into a comprehensive, user-facing response."

    try:
        from llm.chat import llm_chat_text
        response = await llm_chat_text(
            prompt=prompt,
            system=_SYNTHESIZER_SYSTEM,
            prefer="reason",  # Routes to Gemini 3.1 Pro
            max_output_tokens=8000,
        )
        ctx.synthesis = response
        dur = (time.time() - t0) * 1000
        ctx.log("synthesizer", f"Final response: {len(response)} chars", dur)
        return AgentOutput(
            role=AgentRole.SYNTHESIZER, result={"response": response},
            duration_ms=dur, model_used="gemini",
        )
    except Exception as e:
        # Fallback: use reasoner output directly
        dur = (time.time() - t0) * 1000
        fallback = ctx.reasoning_output.get("analysis", "Analysis could not be completed.")
        ctx.synthesis = fallback
        ctx.errors.append(f"Synthesizer failed, using reasoner output: {e}")
        ctx.log("synthesizer", f"Fallback to reasoner output: {e}", dur)
        return AgentOutput(
            role=AgentRole.SYNTHESIZER, result={"response": fallback},
            duration_ms=dur, model_used="fallback", error=str(e),
        )
