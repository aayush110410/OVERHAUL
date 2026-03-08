"""Researcher Agent — Data gathering and context enrichment.

Gathers all data required by the execution plan:
  - Local NCR CSV data (AQI, traffic)
  - Live API data (Open-Meteo, OSRM, TomTom)
  - Historical context from data adapters

Runs data-fetch tasks in parallel for minimum latency.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict

from agents.cognitive.roles import AgentContext, AgentOutput, AgentRole


async def gather_research(ctx: AgentContext) -> AgentOutput:
    """Execute research tasks from the plan in parallel."""
    t0 = time.time()
    plan = ctx.execution_plan
    research_tasks = plan.get("research_tasks", [])

    if not research_tasks:
        ctx.research_data = {"ncr_summary": {}, "live_data": {}}
        dur = (time.time() - t0) * 1000
        ctx.log("researcher", "No research tasks needed", dur)
        return AgentOutput(role=AgentRole.RESEARCHER, result=ctx.research_data, duration_ms=dur)

    # Run all research tasks in parallel
    coros = []
    task_labels = []
    for task in research_tasks:
        task_type = task.get("type", "")
        if task_type == "ncr_data":
            coros.append(_fetch_ncr_data(task))
            task_labels.append("ncr_data")
        elif task_type == "live_aqi":
            coros.append(_fetch_live_aqi(task))
            task_labels.append("live_aqi")
        elif task_type == "traffic_data":
            coros.append(_fetch_traffic_data(task))
            task_labels.append("traffic_data")

    results_list = await asyncio.gather(*coros, return_exceptions=True)

    research_data: Dict[str, Any] = {}
    for label, result in zip(task_labels, results_list):
        if isinstance(result, Exception):
            ctx.errors.append(f"Research {label} failed: {result}")
            research_data[label] = {"error": str(result)}
        else:
            research_data[label] = result

    ctx.research_data = research_data
    dur = (time.time() - t0) * 1000
    ok_count = sum(1 for r in results_list if not isinstance(r, Exception))
    ctx.log("researcher", f"Gathered {ok_count}/{len(coros)} data sources", dur)
    return AgentOutput(role=AgentRole.RESEARCHER, result=research_data, duration_ms=dur)


async def _fetch_ncr_data(task: Dict[str, Any]) -> Dict[str, Any]:
    """Load local NCR CSV data."""
    try:
        from agents.ncr_data_loader import get_ncr_summary
        summary = get_ncr_summary()
        return {"source": "local_csv", "records": len(str(summary)), "data": summary}
    except Exception:
        return {"source": "local_csv", "records": 0, "data": {}}


async def _fetch_live_aqi(task: Dict[str, Any]) -> Dict[str, Any]:
    """Fetch live AQI from Open-Meteo."""
    try:
        from data_integration.adapters.manager import get_adapter_manager
        mgr = get_adapter_manager()
        aqi_adapter = mgr.adapters.get("open_meteo_aqi")
        if aqi_adapter:
            result = await aqi_adapter.fetch()
            return {"source": "open_meteo", "data": result.data if result.ok else {}}
    except Exception:
        pass
    return {"source": "open_meteo", "data": {}}


async def _fetch_traffic_data(task: Dict[str, Any]) -> Dict[str, Any]:
    """Load local traffic CSV data."""
    try:
        from agents.ncr_data_loader import get_ncr_summary
        summary = get_ncr_summary()
        traffic = summary.get("traffic", {})
        return {"source": "local_csv", "data": traffic}
    except Exception:
        return {"source": "local_csv", "data": {}}
