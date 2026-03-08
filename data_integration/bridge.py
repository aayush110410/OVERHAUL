"""
Data Integration Layer
=======================

Bridges between existing data sources (NCR CSV, live APIs)
and the simulation engine framework.

Provides:
  - NCR data → engine input conversion
  - Engine result → chat response formatting
  - Cross-engine result aggregation
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from engines.base import EngineCapability, Intervention, Scenario


def ncr_data_to_engine_input(
    ncr_summary: Dict[str, Any],
    city: str = "delhi",
) -> Dict[str, Any]:
    """Convert NCR CSV summary data into engine-compatible input dict."""
    data: Dict[str, Any] = {"city": city}

    # Extract AQI data
    aqi_data = ncr_summary.get("aqi", {}).get(city.title(), {})
    if aqi_data:
        data["baseline_pm25"] = aqi_data.get("pm25") or aqi_data.get("avg_pm25", 285)

    # Extract traffic data
    traffic_data = ncr_summary.get("traffic", {}).get(city.title(), {})
    if traffic_data:
        data["baseline_speed_kmh"] = traffic_data.get("avg_speed_kmph", 25)
        data["baseline_congestion_pct"] = traffic_data.get("congestion_pct", 55)
        ev_pct = traffic_data.get("ev_percentage", 3)
        data["ev_count"] = int(12_000_000 * ev_pct / 100)

    return data


def prompt_to_interventions(prompt: str) -> List[Intervention]:
    """
    Extract plausible interventions from a natural-language prompt.
    This is a lightweight keyword-based extractor — the LLM refines later.
    """
    interventions: List[Intervention] = []
    lower = prompt.lower()

    keyword_map = [
        (["ev", "electric vehicle", "electric car", "ev adoption"],
         Intervention("ev_fleet_expansion", "transport", {"ev_share_increase": 0.10})),
        (["metro", "dmrc", "metro expansion"],
         Intervention("metro_expansion", "infrastructure", {"km": 25})),
        (["brt", "bus rapid", "bus lane"],
         Intervention("bus_rapid_transit", "transport", {"km": 15})),
        (["congestion pricing", "road pricing", "toll"],
         Intervention("congestion_pricing", "transport", {"demand_reduction_pct": 12})),
        (["signal", "traffic signal", "smart signal"],
         Intervention("signal_optimization", "transport", {"junctions": 200})),
        (["flyover", "overpass", "grade separation"],
         Intervention("flyover_construction", "infrastructure", {"km": 5})),
        (["road widen", "lane expansion", "road capacity"],
         Intervention("road_capacity_expansion", "infrastructure", {"km": 20})),
        (["solar", "renewable", "wind energy", "clean energy"],
         Intervention("renewable_energy", "energy", {"solar_mw": 500})),
        (["green", "tree", "plantation", "urban forest"],
         Intervention("green_infrastructure", "environment", {"green_cover_increase_pct": 5})),
        (["pedestrian", "cycling", "bicycle", "walkway"],
         Intervention("pedestrian_cycling_infra", "infrastructure", {"cycle_km": 50, "footpath_km": 100})),
        (["parking", "multi-level parking"],
         Intervention("parking_infrastructure", "infrastructure", {"spots": 5000})),
        (["industrial relocation", "factory shift"],
         Intervention("industrial_relocation", "environment", {"reduction_pct": 30})),
        (["stubble", "crop burning", "parali"],
         Intervention("stubble_burning_solution", "environment", {"reduction_pct": 70})),
        (["construction dust", "dust control"],
         Intervention("construction_dust_control", "environment", {"reduction_pct": 50})),
        # Logistics & freight
        (["freight", "truck", "goods vehicle", "commercial vehicle"],
         Intervention("freight_optimization", "logistics", {"off_peak_shift_pct": 30})),
        (["last mile", "last-mile", "delivery", "micro-hub"],
         Intervention("last_mile_optimization", "logistics", {"micro_hubs": 50})),
        (["consolidation center", "logistics hub", "freight hub"],
         Intervention("consolidation_centers", "logistics", {"centers": 8})),
        # Demand management
        (["work from home", "wfh", "remote work", "telecommut"],
         Intervention("remote_work_policy", "transport", {"wfh_adoption_pct": 20})),
        (["odd even", "odd-even", "vehicle restriction", "rationing"],
         Intervention("vehicle_rationing", "transport", {"demand_reduction_pct": 15})),
        # Micro-mobility
        (["e-scooter", "scooter sharing", "bike sharing", "micro-mobility"],
         Intervention("micro_mobility", "transport", {"fleet_size": 10000})),
    ]

    for keywords, intervention in keyword_map:
        if any(kw in lower for kw in keywords):
            interventions.append(intervention)

    return interventions


def build_scenario_from_prompt(
    prompt: str,
    city: str = "delhi",
    interventions: Optional[List[Intervention]] = None,
) -> Scenario:
    """Build a Scenario from a user prompt with auto-detected interventions."""
    if interventions is None:
        interventions = prompt_to_interventions(prompt)

    # If no interventions were detected, create a baseline scenario
    if not interventions:
        interventions = [
            Intervention("signal_optimization", "transport", {"junctions": 200},
                          "Default: smart traffic signal optimization"),
        ]

    return Scenario(
        name=f"user_scenario_{city}",
        description=prompt[:200],
        city=city,
        interventions=interventions,
        time_horizon_days=365,
    )


def format_engine_results_for_chat(
    results: Dict[str, Any],
    scenario_name: str = "",
) -> Dict[str, Any]:
    """
    Convert engine simulation results into the chat response format
    expected by the frontend (impact cards, metrics, recommendations).
    """
    impact_cards: List[Dict[str, Any]] = []
    all_recommendations: List[str] = []
    all_warnings: List[str] = []
    all_metrics: Dict[str, Any] = {}
    domains: Dict[str, Any] = {}

    for engine_name, result in results.items():
        if hasattr(result, "metrics"):
            metrics = result.metrics
            domain = result.domain.value if hasattr(result.domain, "value") else str(result.domain)
            domains[domain] = {
                "metrics": metrics,
                "confidence": result.confidence,
                "impacts": result.impacts,
            }
            all_metrics.update(metrics)
            all_recommendations.extend(result.recommendations or [])
            all_warnings.extend(result.warnings or [])

            # Generate impact cards from key metrics
            if domain == "transport":
                if "avg_speed_kmh" in metrics:
                    impact_cards.append({
                        "metric": "Average Speed",
                        "value": f"{metrics['avg_speed_kmh']:.1f} km/h",
                        "delta": f"Congestion: {metrics.get('congestion_pct', 0):.0f}%",
                        "direction": "positive" if metrics.get("congestion_pct", 50) < 50 else "negative",
                    })
                if "co2_tonnes" in metrics:
                    impact_cards.append({
                        "metric": "CO₂ Emissions",
                        "value": f"{metrics['co2_tonnes']:,.0f} t/day",
                        "delta": "transport sector",
                        "direction": "neutral",
                    })

            elif domain == "environment":
                if "projected_aqi" in metrics:
                    impact_cards.append({
                        "metric": "Projected AQI",
                        "value": str(metrics["projected_aqi"]),
                        "delta": f"PM2.5: {metrics.get('projected_pm25', 'N/A')} µg/m³",
                        "direction": "positive" if metrics["projected_aqi"] < 200 else "negative",
                        "category": metrics.get("aqi_category", ""),
                    })
                if "lives_saved_annual" in metrics:
                    impact_cards.append({
                        "metric": "Lives Saved",
                        "value": f"{metrics['lives_saved_annual']:,}/yr",
                        "delta": "from PM2.5 reduction",
                        "direction": "positive",
                    })

            elif domain == "infrastructure":
                if "total_cost_inr_crore" in metrics:
                    impact_cards.append({
                        "metric": "Infrastructure Cost",
                        "value": f"₹{metrics['total_cost_inr_crore']:,.0f} Cr",
                        "delta": f"ROI: {metrics.get('roi_years', 'N/A')} years",
                        "direction": "neutral",
                    })

            elif domain == "energy":
                if "ev_share_pct" in metrics:
                    impact_cards.append({
                        "metric": "EV Share",
                        "value": f"{metrics['ev_share_pct']:.1f}%",
                        "delta": f"Grid: {metrics.get('grid_stress', 'N/A')}",
                        "direction": "positive" if metrics["ev_share_pct"] > 10 else "neutral",
                    })

            elif domain == "economic":
                if "benefit_cost_ratio" in metrics:
                    impact_cards.append({
                        "metric": "Benefit-Cost Ratio",
                        "value": f"{metrics['benefit_cost_ratio']:.1f}x",
                        "delta": f"Annual benefits: ₹{metrics.get('total_annual_benefits_cr', 0):,.0f} Cr",
                        "direction": "positive" if metrics["benefit_cost_ratio"] > 1 else "negative",
                    })

            elif domain == "population":
                if "vkt_reduction_pct" in metrics:
                    impact_cards.append({
                        "metric": "VKT Reduction",
                        "value": f"{metrics['vkt_reduction_pct']:.1f}%",
                        "delta": f"EV adopters: {metrics.get('ev_adopters_projected', 0):,}",
                        "direction": "positive" if metrics["vkt_reduction_pct"] > 5 else "neutral",
                    })

            elif domain == "logistics":
                if "projected_freight_vkt" in metrics:
                    impact_cards.append({
                        "metric": "Freight VKT",
                        "value": f"{metrics['projected_freight_vkt']:,.0f} km/day",
                        "delta": f"CO₂: {metrics.get('projected_co2_tonnes', 0):,.0f} t/day",
                        "direction": "positive" if metrics.get('total_vkt_reduction_pct', 0) > 0 else "neutral",
                    })
                if "parcels_daily" in metrics:
                    impact_cards.append({
                        "metric": "Last-Mile Efficiency",
                        "value": f"Load: {metrics.get('load_factor_pct', 0):.0f}%",
                        "delta": f"{metrics['parcels_daily']:,.0f} parcels/day",
                        "direction": "positive" if metrics.get('total_vkt_reduction_pct', 0) > 5 else "neutral",
                    })

    return {
        "impactCards": impact_cards,
        "recommendations": all_recommendations,
        "warnings": all_warnings,
        "domains": domains,
        "metrics": all_metrics,
        "scenario": scenario_name,
    }
