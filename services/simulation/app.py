"""
Simulation Microservice
========================

Standalone FastAPI service owning the 7 simulation engines,
scenario templates, geospatial output, and the engine registry.

Port: 8001 (default)

Endpoints:
  GET  /health
  GET  /engines
  POST /simulate
  POST /scenarios/compare
  GET  /scenarios/templates
  POST /scenarios/templates/{template_id}
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Ensure project root is on sys.path so engine imports resolve
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from engines import get_registry, Intervention, Scenario  # noqa: E402
from engines.scenarios import list_templates, build_scenario_from_template  # noqa: E402
from engines.geospatial import generate_geojson  # noqa: E402
from engines.transport.prediction import run_prediction  # noqa: E402
from engines.transport.network_model import NCRTrafficNetwork  # noqa: E402
from data_integration.bridge import (  # noqa: E402
    ncr_data_to_engine_input,
    build_scenario_from_prompt,
    format_engine_results_for_chat,
)

app = FastAPI(title="OVERHAUL Simulation Service", version="1.0.0")

_VALID_CITIES = {"delhi", "noida", "gurugram", "ghaziabad", "faridabad", "ncr"}


# ── Health ────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    registry = get_registry()
    engines = registry.list_engines()
    return {
        "service": "simulation",
        "status": "ok",
        "engines_loaded": len(engines),
        "engine_names": [e["name"] for e in engines],
    }


# ── Engine listing ────────────────────────────────────────────────

@app.get("/engines")
async def list_engines_endpoint():
    registry = get_registry()
    return {"engines": registry.list_engines()}


# ── Simulation ────────────────────────────────────────────────────

class SimulateRequest(BaseModel):
    prompt: str = Field("", max_length=2000)
    city: str = Field("delhi")
    interventions: Optional[List[Dict[str, Any]]] = None
    engines: Optional[List[str]] = None
    time_horizon_days: int = Field(365, ge=1, le=3650)
    data: Optional[Dict[str, Any]] = None  # pre-built engine input data


@app.post("/simulate")
async def simulate(req: SimulateRequest):
    if req.city.lower() not in _VALID_CITIES:
        raise HTTPException(400, f"Unsupported city. Use: {', '.join(_VALID_CITIES)}")

    registry = get_registry()
    data = dict(req.data or {})

    if req.interventions:
        intv_objs = [
            Intervention(
                name=i["name"],
                domain=i.get("domain", "transport"),
                parameters=i.get("parameters", {}),
                description=i.get("description", ""),
            )
            for i in req.interventions
        ]
        scenario = Scenario(
            name=f"api_scenario_{req.city}",
            description=req.prompt or "API simulation",
            city=req.city,
            interventions=intv_objs,
            time_horizon_days=req.time_horizon_days,
        )
    else:
        scenario = build_scenario_from_prompt(req.prompt, city=req.city)

    raw = await registry.run_scenario(scenario, data, engines=req.engines)
    formatted = format_engine_results_for_chat(raw, scenario.name)
    geojson = generate_geojson(raw)

    return {
        "scenario": scenario.name,
        "city": req.city,
        "interventions": [
            {"name": i.name, "domain": i.domain, "parameters": i.parameters}
            for i in scenario.interventions
        ],
        "results": formatted,
        "geojson": geojson,
    }


# ── Scenario comparison ──────────────────────────────────────────

class CompareRequest(BaseModel):
    scenarios: List[Dict[str, Any]]
    city: str = "delhi"
    data: Optional[Dict[str, Any]] = None


@app.post("/scenarios/compare")
async def compare_scenarios(req: CompareRequest):
    registry = get_registry()
    data = dict(req.data or {})

    scenario_objs = []
    for s in req.scenarios:
        intv_objs = [
            Intervention(
                name=i["name"],
                domain=i.get("domain", "transport"),
                parameters=i.get("parameters", {}),
            )
            for i in s.get("interventions", [])
        ]
        scenario_objs.append(Scenario(
            name=s.get("name", f"scenario_{len(scenario_objs)}"),
            description=s.get("description", ""),
            city=req.city,
            interventions=intv_objs,
        ))

    comparison = await registry.compare_scenarios(scenario_objs, data)

    return {
        "city": req.city,
        "comparison": {
            name: format_engine_results_for_chat(results, name)
            for name, results in comparison.items()
        },
    }


# ── Scenario templates ────────────────────────────────────────────

@app.get("/scenarios/templates")
async def templates_list():
    return {"templates": list_templates()}


@app.post("/scenarios/templates/{template_id}")
async def run_template(template_id: str, data: Optional[Dict[str, Any]] = None):
    try:
        scenario = build_scenario_from_template(template_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc))

    registry = get_registry()
    raw = await registry.run_scenario(scenario, data or {})
    geojson = generate_geojson(raw)

    return {
        "template": template_id,
        "scenario": scenario.name,
        "description": scenario.description,
        "results": format_engine_results_for_chat(raw, scenario.name),
        "geojson": geojson,
    }


# ── Predictive simulation ────────────────────────────────────────

class PredictRequest(BaseModel):
    interventions: List[Dict[str, Any]] = Field(default_factory=list)
    flow_ratio: float = Field(0.6, ge=0.1, le=1.0)
    ev_share: float = Field(0.05, ge=0.0, le=1.0)
    steps: Optional[List[Dict[str, Any]]] = None  # [{"label":"Current","years":0},...]


@app.post("/simulate/predict")
async def predict(req: PredictRequest):
    """Run a predictive temporal simulation.

    Returns time-stepped results (current → 1yr → 5yr) with
    infrastructure interventions, traffic redistribution, and
        pollution impact at each step.

        Each timeline step also includes agent-based behavioral outputs,
        showing how population segments respond differently to the same
        infrastructure or policy changes.

    Intervention types:
      - new_road: {from, to, dist_km, free_speed, capacity, lanes}
      - lane_expansion: {from, to, additional_lanes}
      - flyover: {from, to, speed_gain_pct}
      - road_closure: {from, to}
      - congestion_pricing: {demand_reduction_pct}
      - signal_optimization: {speed_gain_pct}
            - remote_work_policy: {wfh_adoption_pct}
            - vehicle_rationing: {}
    """
    result = run_prediction(
        interventions=req.interventions,
        flow_ratio=req.flow_ratio,
        ev_share=req.ev_share,
        steps=req.steps,
    )
    return result


@app.get("/network/state")
async def network_state():
    """Get current network state as GeoJSON (no interventions)."""
    network = NCRTrafficNetwork(flow_ratio=0.6)
    network.assign_traffic(iterations=10)
    return {
        "metrics": network.get_summary_metrics(),
        "bottlenecks": network.get_bottlenecks(),
        "geojson": network.to_geojson(),
    }


@app.get("/network/nodes")
async def network_nodes():
    """List all network nodes."""
    network = NCRTrafficNetwork()
    return {
        "nodes": [
            {"id": nid, **attrs}
            for nid, attrs in network.G.nodes(data=True)
        ]
    }


# ── Run standalone ────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("SIMULATION_PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)
