"""Planner Agent — Decomposes intent into an execution plan.

Takes parsed intent and produces:
  - Which engines to run and in what order
  - What data to gather
  - Which LLM models to use for which subtasks
  - Parallel vs sequential execution strategy
  - Expected output structure

Uses Gemini 3.1 Pro (best reasoning) for complex planning,
falls back to heuristic for simple queries.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from agents.cognitive.roles import AgentContext, AgentOutput, AgentRole


def _needs_llm_planning(intent: Dict[str, Any]) -> bool:
    """Only use LLM for genuinely complex planning."""
    return (
        intent.get("complexity") == "multi_scenario"
        or len(intent.get("sub_questions", [])) > 2
        or len(intent.get("interventions", [])) > 3
    )


async def plan_execution(ctx: AgentContext) -> AgentOutput:
    """Generate execution plan from parsed intent."""
    t0 = time.time()
    intent = ctx.parsed_intent

    if _needs_llm_planning(intent):
        plan = await _llm_plan(ctx)
    else:
        plan = _heuristic_plan(intent)

    ctx.execution_plan = plan
    dur = (time.time() - t0) * 1000
    ctx.log("planner", f"Plan: {len(plan['engine_tasks'])} engines, "
            f"{len(plan['research_tasks'])} research, "
            f"strategy={plan['strategy']}", dur)
    return AgentOutput(role=AgentRole.PLANNER, result=plan, duration_ms=dur, model_used="heuristic")


def _heuristic_plan(intent: Dict[str, Any]) -> Dict[str, Any]:
    """Rule-based planning for simple/compound queries."""

    # Map interventions to required engines
    _DOMAIN_TO_ENGINE = {
        "transport": "TransportEngine",
        "environment": "EnvironmentEngine",
        "infrastructure": "InfrastructureEngine",
        "energy": "EnergyEngine",
        "economic": "EconomicEngine",
        "population": "PopulationEngine",
        "logistics": "LogisticsEngine",
    }

    engine_tasks = []
    domains_needed = set()
    for intervention in intent.get("interventions", []):
        domain = intervention.get("domain", "transport")
        domains_needed.add(domain)

    # Always include transport (primary) and add downstream engines
    domains_needed.add("transport")
    if "environment" not in domains_needed and any(
        d in domains_needed for d in ["transport", "energy"]
    ):
        domains_needed.add("environment")
    if "economic" not in domains_needed:
        domains_needed.add("economic")

    for domain in domains_needed:
        engine_name = _DOMAIN_TO_ENGINE.get(domain)
        if engine_name:
            engine_tasks.append({
                "engine": engine_name,
                "domain": domain,
                "priority": 1 if domain in ("transport", "infrastructure", "environment") else 2,
            })

    engine_tasks.sort(key=lambda x: x["priority"])

    # Determine research requirements
    research_tasks = _determine_research(intent)

    # Model assignment based on task type
    model_assignments = {
        "parsing": {"model": "qwen", "reason": "Fast, sub-second intent extraction"},
        "planning": {"model": "heuristic", "reason": "Rule-based for simple queries"},
        "research": {"model": "gemini", "reason": "Google Search grounding for real-time data"},
        "analysis": {"model": "llama", "reason": "Deep 70B parameter analysis"},
        "validation": {"model": "gpt_oss", "reason": "Cross-validation with independent model"},
        "synthesis": {"model": "gemini", "reason": "Complex multi-source reasoning"},
    }

    # Execution strategy
    query_type = intent.get("query_type", "analysis")
    if query_type == "comparison":
        strategy = "parallel_scenarios"
    elif query_type == "simulation" and len(engine_tasks) > 2:
        strategy = "phased_parallel"
    else:
        strategy = "standard_pipeline"

    return {
        "engine_tasks": engine_tasks,
        "research_tasks": research_tasks,
        "model_assignments": model_assignments,
        "strategy": strategy,
        "phases": _build_phases(engine_tasks, strategy),
        "expected_outputs": _expected_outputs(intent),
    }


def _determine_research(intent: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Identify what data needs to be gathered."""
    tasks = []
    cities = intent.get("entities", {}).get("cities", ["delhi"])

    # Always fetch NCR data
    tasks.append({
        "type": "ncr_data",
        "source": "local_csv",
        "cities": cities,
        "priority": 0,
    })

    # AQI data if environment domain involved
    domains = {i.get("domain") for i in intent.get("interventions", [])}
    if "environment" in domains or intent.get("query_type") == "forecast":
        tasks.append({
            "type": "live_aqi",
            "source": "open_meteo",
            "cities": cities,
            "priority": 1,
        })

    # Traffic data if transport domain
    if "transport" in domains:
        tasks.append({
            "type": "traffic_data",
            "source": "local_csv",
            "cities": cities,
            "priority": 0,
        })

    return tasks


def _build_phases(engine_tasks: List[Dict], strategy: str) -> List[Dict[str, Any]]:
    """Organize engine tasks into execution phases."""
    phase1 = [t for t in engine_tasks if t["priority"] == 1]
    phase2 = [t for t in engine_tasks if t["priority"] == 2]

    phases = []
    if phase1:
        phases.append({
            "phase": 1,
            "label": "Primary simulation",
            "engines": [t["engine"] for t in phase1],
            "parallel": True,
        })
    if phase2:
        phases.append({
            "phase": 2,
            "label": "Downstream impact",
            "engines": [t["engine"] for t in phase2],
            "parallel": True,
            "depends_on": 1,
        })
    return phases


def _expected_outputs(intent: Dict[str, Any]) -> List[str]:
    """What the user expects to see."""
    outputs = ["executive_summary", "impact_metrics"]
    qt = intent.get("query_type", "analysis")
    if qt == "comparison":
        outputs.append("scenario_comparison_table")
    if qt == "simulation":
        outputs.extend(["simulation_results", "recommendations"])
    if qt == "forecast":
        outputs.extend(["trend_projection", "confidence_intervals"])
    if any(i.get("domain") == "environment" for i in intent.get("interventions", [])):
        outputs.append("aqi_impact")
    return outputs


async def _llm_plan(ctx: AgentContext) -> Dict[str, Any]:
    """Use Gemini for complex multi-scenario planning."""
    base = _heuristic_plan(ctx.parsed_intent)
    # For multi-scenario queries, duplicate engine tasks per sub-question
    sub_qs = ctx.parsed_intent.get("sub_questions", [])
    if sub_qs:
        base["sub_scenarios"] = [
            {"question": q, "engines": [t["engine"] for t in base["engine_tasks"]]}
            for q in sub_qs
        ]
        base["strategy"] = "parallel_scenarios"
    return base
