"""
Scenario Template System
========================

Pre-built intervention combinations for common Delhi NCR planning goals.
Each template can be used directly via the /simulate endpoint or as
starting points that the LLM refines.
"""

from __future__ import annotations

from typing import Dict, List

from engines.base import Intervention, Scenario

# ── Template definitions ──────────────────────────────────────────

TEMPLATES: Dict[str, Dict] = {
    "green_delhi": {
        "description": "Aggressive air quality improvement through green infra, EV adoption, and emission controls",
        "interventions": [
            Intervention("ev_fleet_expansion", "transport", {"ev_share_increase": 0.20}),
            Intervention("green_infrastructure", "environment", {"green_cover_increase_pct": 10}),
            Intervention("stubble_burning_solution", "environment", {"reduction_pct": 80}),
            Intervention("construction_dust_control", "environment", {"reduction_pct": 60}),
            Intervention("industrial_relocation", "environment", {"reduction_pct": 40}),
            Intervention("renewable_energy", "energy", {"solar_mw": 1000}),
        ],
        "time_horizon_days": 730,
    },
    "metro_first": {
        "description": "Mass transit expansion with metro, BRT, and last-mile connectivity",
        "interventions": [
            Intervention("metro_expansion", "infrastructure", {"km": 60}),
            Intervention("bus_rapid_transit", "transport", {"km": 30}),
            Intervention("pedestrian_cycling_infra", "infrastructure", {"cycle_km": 100, "footpath_km": 200}),
            Intervention("signal_optimization", "transport", {"junctions": 500}),
            Intervention("micro_mobility", "transport", {"fleet_size": 15000}),
        ],
        "time_horizon_days": 1095,
    },
    "ev_revolution": {
        "description": "Rapid electrification of transport with grid upgrades",
        "interventions": [
            Intervention("ev_fleet_expansion", "transport", {"ev_share_increase": 0.30}),
            Intervention("renewable_energy", "energy", {"solar_mw": 2000}),
            Intervention("last_mile_optimization", "logistics", {"micro_hubs": 100}),
        ],
        "time_horizon_days": 1825,
    },
    "freight_optimization": {
        "description": "Optimize commercial vehicle operations and last-mile logistics",
        "interventions": [
            Intervention("freight_optimization", "logistics", {"off_peak_shift_pct": 40}),
            Intervention("consolidation_centers", "logistics", {"centers": 12}),
            Intervention("last_mile_optimization", "logistics", {"micro_hubs": 80}),
            Intervention("congestion_pricing", "transport", {"demand_reduction_pct": 10}),
        ],
        "time_horizon_days": 365,
    },
    "congestion_buster": {
        "description": "Immediate congestion relief via pricing, signals, and demand management",
        "interventions": [
            Intervention("congestion_pricing", "transport", {"demand_reduction_pct": 15}),
            Intervention("signal_optimization", "transport", {"junctions": 800}),
            Intervention("vehicle_rationing", "transport", {"demand_reduction_pct": 15}),
            Intervention("remote_work_policy", "transport", {"wfh_adoption_pct": 25}),
            Intervention("bus_rapid_transit", "transport", {"km": 20}),
        ],
        "time_horizon_days": 365,
    },
    "sustainable_ncr": {
        "description": "Comprehensive sustainable development combining transport, environment, and energy",
        "interventions": [
            Intervention("ev_fleet_expansion", "transport", {"ev_share_increase": 0.15}),
            Intervention("metro_expansion", "infrastructure", {"km": 40}),
            Intervention("bus_rapid_transit", "transport", {"km": 25}),
            Intervention("green_infrastructure", "environment", {"green_cover_increase_pct": 8}),
            Intervention("renewable_energy", "energy", {"solar_mw": 800}),
            Intervention("freight_optimization", "logistics", {"off_peak_shift_pct": 30}),
            Intervention("pedestrian_cycling_infra", "infrastructure", {"cycle_km": 80, "footpath_km": 150}),
            Intervention("congestion_pricing", "transport", {"demand_reduction_pct": 10}),
        ],
        "time_horizon_days": 1095,
    },
}


def list_templates() -> List[Dict]:
    """Return summary of all available templates."""
    return [
        {
            "id": tid,
            "description": t["description"],
            "interventions": len(t["interventions"]),
            "time_horizon_days": t["time_horizon_days"],
            "domains": sorted({i.domain for i in t["interventions"]}),
        }
        for tid, t in TEMPLATES.items()
    ]


def build_scenario_from_template(
    template_id: str,
    city: str = "delhi",
    overrides: Dict | None = None,
) -> Scenario:
    """Build a Scenario from a named template with optional parameter overrides."""
    t = TEMPLATES.get(template_id)
    if not t:
        available = ", ".join(TEMPLATES.keys())
        raise ValueError(f"Unknown template '{template_id}'. Available: {available}")

    interventions = list(t["interventions"])

    # Apply parameter overrides if provided
    if overrides:
        for i, intv in enumerate(interventions):
            if intv.name in overrides:
                merged = {**intv.parameters, **overrides[intv.name]}
                interventions[i] = Intervention(
                    intv.name, intv.domain, merged, intv.description,
                )

    return Scenario(
        name=f"template_{template_id}_{city}",
        description=t["description"],
        city=city,
        interventions=interventions,
        time_horizon_days=overrides.get("time_horizon_days", t["time_horizon_days"])
        if overrides else t["time_horizon_days"],
    )
