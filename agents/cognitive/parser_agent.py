"""Parser Agent — Intent decomposition and entity extraction.

Takes raw user query and produces structured intent:
  - query_type: simulation | analysis | comparison | forecast | factual
  - interventions: list of policy/infrastructure changes mentioned
  - entities: cities, corridors, metrics referenced
  - time_horizon: implied temporal scope
  - complexity: simple | compound | multi-scenario

Uses Qwen 3 4B (fastest model) for sub-second parsing.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict

from agents.cognitive.roles import AgentContext, AgentOutput, AgentRole


_PARSER_SYSTEM = """You are a query parser for OVERHAUL, an urban simulation platform for Delhi NCR.

Extract structured intent from user queries. Output ONLY valid JSON:
{
  "query_type": "simulation|analysis|comparison|forecast|factual",
  "interventions": [
    {"name": "string", "domain": "transport|environment|infrastructure|energy|economic|population|logistics", "parameters": {}}
  ],
  "entities": {
    "cities": ["delhi", "noida", "ghaziabad", "gurugram"],
    "corridors": [],
    "metrics": []
  },
  "time_horizon_days": 365,
  "complexity": "simple|compound|multi_scenario",
  "sub_questions": []
}

Rules:
- Always identify at least one domain
- Extract numeric parameters (percentages, distances, counts)
- If query implies comparison, set query_type to "comparison" and list sub-questions
- For compound queries, decompose into sub_questions
- Default time_horizon is 365 days unless stated otherwise"""


async def parse_intent(ctx: AgentContext) -> AgentOutput:
    """Parse user query into structured intent using Qwen (fast)."""
    t0 = time.time()
    try:
        from llm.chat import llm_chat_json
        result = await llm_chat_json(
            prompt=f"Parse this query:\n\n{ctx.query}",
            system=_PARSER_SYSTEM,
            prefer="fast",
            max_output_tokens=2000,
        )

        # Validate required fields
        if "query_type" not in result or result.get("error"):
            result = _fallback_parse(ctx.query)

        # Normalize
        result.setdefault("interventions", [])
        result.setdefault("entities", {"cities": [ctx.city], "corridors": [], "metrics": []})
        result.setdefault("time_horizon_days", 365)
        result.setdefault("complexity", "simple")
        result.setdefault("sub_questions", [])

        if not result["entities"].get("cities"):
            result["entities"]["cities"] = [ctx.city]

        ctx.parsed_intent = result
        dur = (time.time() - t0) * 1000
        ctx.log("parser", f"Parsed as {result['query_type']} with {len(result['interventions'])} interventions", dur)
        return AgentOutput(role=AgentRole.PARSER, result=result, duration_ms=dur, model_used="qwen")
    except Exception as e:
        dur = (time.time() - t0) * 1000
        result = _fallback_parse(ctx.query)
        ctx.parsed_intent = result
        ctx.errors.append(f"Parser LLM failed, used heuristic: {e}")
        ctx.log("parser", f"Fallback parse: {e}", dur)
        return AgentOutput(role=AgentRole.PARSER, result=result, duration_ms=dur, error=str(e))


def _fallback_parse(query: str) -> Dict[str, Any]:
    """Rule-based fallback when LLM is unavailable."""
    q = query.lower()

    # Detect query type
    query_type = "analysis"
    if any(w in q for w in ["compare", "vs", "versus", "difference"]):
        query_type = "comparison"
    elif any(w in q for w in ["what if", "simulate", "scenario", "implement"]):
        query_type = "simulation"
    elif any(w in q for w in ["forecast", "predict", "will", "future"]):
        query_type = "forecast"

    # Detect domains
    interventions = []
    _DOMAIN_KEYWORDS = {
        "transport": ["traffic", "road", "highway", "signal", "congestion", "metro", "bus", "route"],
        "environment": ["aqi", "pollution", "air quality", "pm2.5", "emission", "co2"],
        "infrastructure": ["flyover", "bridge", "cycling", "construction"],
        "energy": ["ev", "electric vehicle", "solar", "grid", "energy"],
        "economic": ["gdp", "cost", "economy", "productivity", "revenue"],
        "population": ["commuter", "wfh", "work from home", "mode shift"],
    }
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            interventions.append({"name": domain + "_analysis", "domain": domain, "parameters": {}})

    if not interventions:
        interventions.append({"name": "general_analysis", "domain": "transport", "parameters": {}})

    # Detect cities
    cities = []
    for city in ["delhi", "noida", "ghaziabad", "gurugram", "greater noida", "faridabad"]:
        if city in q:
            cities.append(city)
    if not cities:
        cities = ["delhi"]

    return {
        "query_type": query_type,
        "interventions": interventions,
        "entities": {"cities": cities, "corridors": [], "metrics": []},
        "time_horizon_days": 365,
        "complexity": "compound" if len(interventions) > 1 else "simple",
        "sub_questions": [],
    }
