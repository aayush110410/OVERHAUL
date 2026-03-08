"""
Logistics & Supply Chain Engine
================================

Models freight movement, last-mile delivery, supply-chain corridors,
and commercial vehicle impact on urban networks.

Covers:
  - Freight corridor capacity and VKT
  - Last-mile delivery density & micro-hub placement
  - Off-peak freight shifting
  - Commercial vehicle emission impact
  - E-commerce delivery load projection
  - Freight consolidation center ROI

Calibrated for Delhi NCR freight/logistics data.
"""

from __future__ import annotations

from typing import Any, Dict, List

from engines.base import (
    EngineCapability,
    Scenario,
    SimulationEngine,
    SimulationResult,
)

# ── Delhi NCR Freight Baselines ──
_BASELINES = {
    "daily_freight_trips": 420_000,         # Trucks + LCVs entering/within NCR
    "avg_freight_trip_km": 38,
    "freight_vkt_daily": 420_000 * 38,      # ~16M VKT
    "freight_share_of_road_vkt_pct": 12,
    "commercial_vehicles_registered": 950_000,
    "ev_freight_share_pct": 0.5,
    "avg_load_factor_pct": 62,              # % of truck capacity utilised
    "empty_return_pct": 28,                 # Empty-run percentage
    "peak_hour_freight_pct": 35,            # Freight moving in peak hours
    "e_commerce_parcels_daily": 2_800_000,  # NCR daily parcels
    "avg_last_mile_km": 6.5,
    "last_mile_mode_split": {
        "two_wheeler": 0.55,
        "three_wheeler": 0.20,
        "van": 0.20,
        "cycle": 0.05,
    },
}

# Emission factors for freight
_EMISSION_FACTORS = {
    "diesel_truck_co2_g_per_vkm": 650,
    "diesel_lcv_co2_g_per_vkm": 280,
    "ev_truck_co2_g_per_vkm": 210,     # Grid-dependent
    "ev_lcv_co2_g_per_vkm": 85,
    "pm25_diesel_truck_g_per_vkm": 0.18,
    "pm25_diesel_lcv_g_per_vkm": 0.08,
    "nox_diesel_truck_g_per_vkm": 5.2,
}

# Cost benchmarks
_COSTS = {
    "freight_consolidation_center_cr": 45,   # Per center
    "micro_hub_cr": 3.5,                      # Last-mile hub
    "ev_truck_premium_lakh": 25,              # Over diesel truck
    "fuel_cost_per_km_diesel": 12.5,          # INR
    "fuel_cost_per_km_ev": 3.2,
}


class LogisticsEngine(SimulationEngine):
    """Freight, supply-chain, and last-mile delivery simulation."""

    @property
    def name(self) -> str:
        return "LogisticsEngine"

    @property
    def capabilities(self) -> List[EngineCapability]:
        return [EngineCapability.LOGISTICS]

    @property
    def optional_data(self) -> List[str]:
        return ["city", "daily_freight_trips", "e_commerce_parcels"]

    def estimate_missing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if "daily_freight_trips" not in data:
            data["daily_freight_trips"] = _BASELINES["daily_freight_trips"]
        if "e_commerce_parcels" not in data:
            data["e_commerce_parcels"] = _BASELINES["e_commerce_parcels_daily"]
        return data

    async def _run(self, scenario: Scenario, data: Dict[str, Any]) -> SimulationResult:
        freight_trips = data.get("daily_freight_trips", _BASELINES["daily_freight_trips"])
        parcels = data.get("e_commerce_parcels", _BASELINES["e_commerce_parcels_daily"])

        # Baseline metrics
        baseline_vkt = freight_trips * _BASELINES["avg_freight_trip_km"]
        baseline_empty_vkt = baseline_vkt * _BASELINES["empty_return_pct"] / 100
        last_mile_vkt = parcels * _BASELINES["avg_last_mile_km"]

        # Intervention effects
        off_peak_shift_pct = 0.0
        consolidation_centers = 0
        micro_hubs = 0
        ev_freight_increase_pct = 0.0
        load_factor_improvement = 0.0
        total_cost_cr = 0.0

        for intv in scenario.interventions:
            p = intv.parameters
            if intv.name == "off_peak_freight":
                off_peak_shift_pct += p.get("shift_pct", 40)
            elif intv.name == "freight_consolidation":
                consolidation_centers += p.get("centers", 5)
                total_cost_cr += consolidation_centers * _COSTS["freight_consolidation_center_cr"]
            elif intv.name == "last_mile_hubs":
                micro_hubs += p.get("hubs", 20)
                total_cost_cr += micro_hubs * _COSTS["micro_hub_cr"]
            elif intv.name == "ev_fleet_expansion":
                ev_freight_increase_pct += p.get("freight_ev_pct", 5)
            elif intv.name == "congestion_pricing":
                # Congestion pricing pushes freight off-peak
                off_peak_shift_pct += 15
            elif intv.name == "metro_expansion":
                # Metro freight integration (rare but emerging)
                load_factor_improvement += 2

        # Compute effects
        # Consolidation reduces empty runs and improves load factor
        empty_run_reduction = min(
            _BASELINES["empty_return_pct"],
            consolidation_centers * 3.5  # Each center reduces empty runs by ~3.5%
        )
        new_empty_pct = _BASELINES["empty_return_pct"] - empty_run_reduction
        new_load_factor = min(95, _BASELINES["avg_load_factor_pct"] + consolidation_centers * 2.5 + load_factor_improvement)

        # Higher load factor → fewer trips needed
        trip_reduction_from_load = (new_load_factor - _BASELINES["avg_load_factor_pct"]) / 100
        new_freight_trips = int(freight_trips * (1 - trip_reduction_from_load * 0.5))
        new_freight_vkt = new_freight_trips * _BASELINES["avg_freight_trip_km"]

        # Last-mile optimization from micro hubs
        last_mile_reduction = min(40, micro_hubs * 1.5)  # Each hub reduces last-mile ~1.5%
        new_last_mile_vkt = last_mile_vkt * (1 - last_mile_reduction / 100)

        # Peak-hour relief
        peak_freight_pct = max(5, _BASELINES["peak_hour_freight_pct"] - off_peak_shift_pct * 0.7)

        # Emissions
        ev_freight_share = _BASELINES["ev_freight_share_pct"] + ev_freight_increase_pct
        diesel_vkt = new_freight_vkt * (1 - ev_freight_share / 100)
        ev_vkt = new_freight_vkt * (ev_freight_share / 100)

        baseline_co2_tonnes = (
            baseline_vkt * _EMISSION_FACTORS["diesel_truck_co2_g_per_vkm"] * 0.6 +  # 60% trucks
            baseline_vkt * _EMISSION_FACTORS["diesel_lcv_co2_g_per_vkm"] * 0.4      # 40% LCVs
        ) / 1e6
        new_co2_tonnes = (
            diesel_vkt * _EMISSION_FACTORS["diesel_truck_co2_g_per_vkm"] * 0.6 +
            diesel_vkt * _EMISSION_FACTORS["diesel_lcv_co2_g_per_vkm"] * 0.4 +
            ev_vkt * _EMISSION_FACTORS["ev_truck_co2_g_per_vkm"] * 0.6 +
            ev_vkt * _EMISSION_FACTORS["ev_lcv_co2_g_per_vkm"] * 0.4
        ) / 1e6

        # Fuel savings
        diesel_fuel_savings_cr = (
            (baseline_vkt - diesel_vkt) * _COSTS["fuel_cost_per_km_diesel"] / 1e7
        )
        ev_fuel_cost_cr = ev_vkt * _COSTS["fuel_cost_per_km_ev"] / 1e7

        vkt_reduction_pct = (1 - (new_freight_vkt + new_last_mile_vkt) / (baseline_vkt + last_mile_vkt)) * 100

        metrics = {
            "baseline_freight_vkt": baseline_vkt,
            "projected_freight_vkt": new_freight_vkt,
            "baseline_last_mile_vkt": round(last_mile_vkt),
            "projected_last_mile_vkt": round(new_last_mile_vkt),
            "total_vkt_reduction_pct": round(vkt_reduction_pct, 1),
            "freight_trips_daily": new_freight_trips,
            "load_factor_pct": round(new_load_factor, 1),
            "empty_run_pct": round(new_empty_pct, 1),
            "peak_hour_freight_pct": round(peak_freight_pct, 1),
            "ev_freight_share_pct": round(ev_freight_share, 1),
            "baseline_co2_tonnes": round(baseline_co2_tonnes, 1),
            "projected_co2_tonnes": round(new_co2_tonnes, 1),
            "co2_reduction_pct": round((1 - new_co2_tonnes / max(baseline_co2_tonnes, 1)) * 100, 1),
            "consolidation_centers": consolidation_centers,
            "micro_hubs": micro_hubs,
            "infrastructure_cost_cr": round(total_cost_cr, 1),
            "daily_fuel_savings_cr": round(diesel_fuel_savings_cr - ev_fuel_cost_cr, 2),
            "parcels_daily": parcels,
        }

        impacts = {
            "freight_efficiency_gain": round(new_load_factor - _BASELINES["avg_load_factor_pct"], 1),
            "peak_hour_congestion_relief_pct": round(_BASELINES["peak_hour_freight_pct"] - peak_freight_pct, 1),
            "emission_reduction_score": min(100, round((baseline_co2_tonnes - new_co2_tonnes) / max(baseline_co2_tonnes, 1) * 200)),
        }

        recommendations = []
        if new_empty_pct > 15:
            recommendations.append(f"Empty-run rate still {new_empty_pct:.0f}% — digital freight platforms can match return loads")
        if consolidation_centers == 0 and freight_trips > 300_000:
            recommendations.append("No consolidation centers — 3-5 centers on city periphery would reduce urban truck entry by 15-25%")
        if ev_freight_share < 5:
            recommendations.append("EV freight share below 5% — incentivize last-mile EV vans (subsidy + charging depots)")
        if peak_freight_pct > 20:
            recommendations.append(f"Peak-hour freight at {peak_freight_pct:.0f}% — night delivery corridors and off-peak pricing recommended")
        if micro_hubs > 0:
            recommendations.append(f"{micro_hubs} micro-hubs projected to reduce last-mile VKT by {last_mile_reduction:.0f}%")

        warnings = []
        if parcels > 3_000_000:
            warnings.append("E-commerce delivery growth will increase urban last-mile congestion 15-20% annually without intervention")

        return SimulationResult(
            engine=self.name,
            domain=EngineCapability.LOGISTICS,
            scenario_name=scenario.name,
            metrics=metrics,
            impacts=impacts,
            recommendations=recommendations,
            confidence=0.55,
            warnings=warnings,
        )
