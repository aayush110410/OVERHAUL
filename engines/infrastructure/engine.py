"""
Infrastructure Engine
======================

Models road network capacity, flyover/bridge utilisation, metro expansion,
parking infrastructure, and construction disruption impacts.

Uses simplified capacity models calibrated for Delhi NCR.
"""

from __future__ import annotations

from typing import Any, Dict, List

from engines.base import (
    EngineCapability,
    Scenario,
    SimulationEngine,
    SimulationResult,
)

# ── Delhi NCR Infra Baselines ──
_ROAD_NETWORK = {
    "total_road_km": 33_198,
    "national_highway_km": 1_750,
    "metro_km": 392,            # DMRC Phase IV target
    "metro_stations": 286,
    "flyover_count": 78,
    "underpass_count": 42,
    "parking_capacity_vehicles": 185_000,
    "daily_vehicle_registrations": 1_400,
}

# Cost benchmarks (INR crore per km or unit)
_COST_BENCHMARKS = {
    "road_widening_per_km_cr": 18.0,
    "flyover_per_km_cr": 120.0,
    "metro_per_km_cr": 350.0,
    "bus_corridor_per_km_cr": 8.5,
    "parking_per_spot_lakh": 3.5,
    "footpath_per_km_cr": 1.2,
    "cycle_track_per_km_cr": 2.0,
}

# Capacity multipliers
_CAPACITY_MULTIPLIERS = {
    "road_widening": 1.4,         # 4→6 lane widening
    "flyover": 1.6,               # grade separation
    "signal_optimization": 1.15,
    "roundabout_to_signal": 1.25,
    "bus_rapid_transit": 0.9,     # Reduces car capacity but adds bus throughput
    "metro_expansion": 1.0,       # Doesn't change road capacity but shifts demand
    "pedestrianization": 0.3,     # Removes car access
}

_INR_TO_USD = 83.0


class InfrastructureEngine(SimulationEngine):
    """Infrastructure capacity and cost-benefit simulation."""

    @property
    def name(self) -> str:
        return "InfrastructureEngine"

    @property
    def capabilities(self) -> List[EngineCapability]:
        return [EngineCapability.INFRASTRUCTURE]

    @property
    def optional_data(self) -> List[str]:
        return ["city", "existing_capacity_vph", "daily_demand"]

    def estimate_missing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if "existing_capacity_vph" not in data:
            data["existing_capacity_vph"] = 180_000  # Rough Delhi arterial capacity
        if "daily_demand" not in data:
            data["daily_demand"] = 12_000_000  # Registered vehicles
        return data

    async def _run(self, scenario: Scenario, data: Dict[str, Any]) -> SimulationResult:
        capacity = data.get("existing_capacity_vph", 180_000)
        daily_demand = data.get("daily_demand", 12_000_000)

        total_cost_cr = 0.0
        capacity_added_pct = 0.0
        demand_shifted = 0.0
        construction_disruption_months = 0
        new_metro_km = 0.0
        new_road_km = 0.0
        new_parking_spots = 0

        for intv in scenario.interventions:
            p = intv.parameters
            if intv.name == "road_capacity_expansion":
                km = p.get("km", 20)
                new_road_km += km
                capacity_added_pct += km / _ROAD_NETWORK["total_road_km"] * 100 * _CAPACITY_MULTIPLIERS["road_widening"]
                total_cost_cr += km * _COST_BENCHMARKS["road_widening_per_km_cr"]
                construction_disruption_months += max(12, int(km * 0.8))

            elif intv.name == "flyover_construction":
                km = p.get("km", 5)
                capacity_added_pct += km * 1200 / capacity * 100  # ~1200 vph per km flyover
                total_cost_cr += km * _COST_BENCHMARKS["flyover_per_km_cr"]
                construction_disruption_months += max(18, int(km * 4))

            elif intv.name == "metro_expansion":
                km = p.get("km", 25)
                new_metro_km += km
                # Metro shifts ~2000 passengers/km/hour off road
                demand_shifted += km * 2000 * 16 / daily_demand * 100  # 16 operating hours
                total_cost_cr += km * _COST_BENCHMARKS["metro_per_km_cr"]
                construction_disruption_months += max(36, int(km * 2))

            elif intv.name == "bus_rapid_transit":
                km = p.get("km", 15)
                # BRT carries ~4000 pphpd, reduces car demand
                demand_shifted += km * 4000 * 16 / daily_demand * 100
                total_cost_cr += km * _COST_BENCHMARKS["bus_corridor_per_km_cr"]
                construction_disruption_months += max(6, int(km * 0.5))

            elif intv.name == "parking_infrastructure":
                spots = p.get("spots", 5000)
                new_parking_spots += spots
                total_cost_cr += spots * _COST_BENCHMARKS["parking_per_spot_lakh"] / 100
                construction_disruption_months += 8

            elif intv.name == "signal_optimization":
                # Low-cost, high-impact
                junctions = p.get("junctions", 200)
                capacity_added_pct += junctions * 0.01  # ~1% per 100 junctions
                total_cost_cr += junctions * 0.02  # 2 lakh per junction

            elif intv.name == "pedestrian_cycling_infra":
                cycle_km = p.get("cycle_km", 50)
                footpath_km = p.get("footpath_km", 100)
                total_cost_cr += (
                    cycle_km * _COST_BENCHMARKS["cycle_track_per_km_cr"] +
                    footpath_km * _COST_BENCHMARKS["footpath_per_km_cr"]
                )
                demand_shifted += 0.5  # Marginal mode shift

        # Effective capacity change
        effective_capacity_change = capacity_added_pct + demand_shifted
        new_congestion_ratio = daily_demand / (capacity * (1 + effective_capacity_change / 100))
        base_congestion_ratio = daily_demand / capacity

        # Time savings
        avg_commute_min = 52  # Delhi NCR average
        time_saved_min = avg_commute_min * (1 - new_congestion_ratio / base_congestion_ratio)
        time_saved_min = max(0, min(time_saved_min, 25))

        # Economic value of time saved (₹250/hour per commuter, 5M daily commuters)
        daily_commuters = 5_000_000
        annual_time_value_cr = (
            time_saved_min / 60 * 250 * daily_commuters * 300 / 1e7
        )

        roi_years = total_cost_cr / max(annual_time_value_cr, 1) if total_cost_cr > 0 else 0

        metrics = {
            "total_cost_inr_crore": round(total_cost_cr, 1),
            "total_cost_usd_million": round(total_cost_cr * 10 / _INR_TO_USD, 1),
            "capacity_increase_pct": round(capacity_added_pct, 2),
            "demand_shifted_pct": round(demand_shifted, 2),
            "effective_improvement_pct": round(effective_capacity_change, 2),
            "congestion_ratio_before": round(base_congestion_ratio, 3),
            "congestion_ratio_after": round(new_congestion_ratio, 3),
            "avg_commute_time_saved_min": round(time_saved_min, 1),
            "construction_disruption_months": construction_disruption_months,
            "new_metro_km": new_metro_km,
            "new_road_km": new_road_km,
            "new_parking_spots": new_parking_spots,
            "roi_years": round(roi_years, 1),
            "annual_time_value_saved_cr": round(annual_time_value_cr, 1),
        }

        impacts = {
            "cost_effectiveness": "high" if roi_years < 8 else ("medium" if roi_years < 15 else "low"),
            "disruption_severity": "high" if construction_disruption_months > 24 else ("medium" if construction_disruption_months > 12 else "low"),
            "capacity_score": min(100, round(effective_capacity_change * 5)),
        }

        recommendations = []
        if roi_years > 15:
            recommendations.append(f"ROI of {roi_years:.0f} years is poor — consider phased implementation")
        if construction_disruption_months > 24:
            recommendations.append("Long construction window — recommend night construction + traffic management plan")
        if demand_shifted > capacity_added_pct:
            recommendations.append("Demand-side solutions (metro/BRT) more cost-effective than road expansion")
        if total_cost_cr > 10000:
            recommendations.append(f"Total cost ₹{total_cost_cr:,.0f} Cr — recommend PPP model for financing")
        if new_metro_km > 0:
            recommendations.append(f"Metro expansion of {new_metro_km:.0f} km would shift {demand_shifted:.1f}% demand off roads")

        warnings = []
        if capacity_added_pct > 10:
            warnings.append("Induced demand may offset capacity gains within 3-5 years")

        return SimulationResult(
            engine=self.name,
            domain=EngineCapability.INFRASTRUCTURE,
            scenario_name=scenario.name,
            metrics=metrics,
            impacts=impacts,
            recommendations=recommendations,
            confidence=0.60,
            warnings=warnings,
        )
