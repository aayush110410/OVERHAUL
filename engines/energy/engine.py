"""
Energy Engine
==============

Models energy consumption from transport, EV charging infrastructure
load, grid impact, fuel savings, and renewable energy integration.
Calibrated for Indian energy mix and Delhi NCR electricity grid.
"""

from __future__ import annotations

from typing import Any, Dict, List

from engines.base import (
    EngineCapability,
    Scenario,
    SimulationEngine,
    SimulationResult,
)

# ── Delhi NCR Energy Baselines ──
_BASELINES = {
    "daily_fuel_consumption_ml": 12.5,     # Million litres (petrol+diesel)
    "petrol_pct": 0.55,
    "diesel_pct": 0.40,
    "cng_pct": 0.05,
    "ev_share": 0.03,                       # 3% as of 2025
    "grid_capacity_mw": 8_200,
    "peak_demand_mw": 7_400,
    "renewable_share": 0.12,                # Solar + wind in Delhi grid mix
    "coal_share": 0.65,
    "gas_share": 0.15,
    "hydro_share": 0.08,
    "electricity_rate_per_kwh": 8.0,        # INR
    "petrol_price_per_litre": 96.7,
    "diesel_price_per_litre": 89.6,
}

# EV energy parameters
_EV_PARAMS = {
    "avg_kwh_per_km": 0.15,               # Weighted avg for cars+2W+buses
    "avg_daily_km": 35,
    "charger_kw_slow": 3.3,
    "charger_kw_fast": 50,
    "charger_kw_ultra": 150,
    "charging_hours_slow": 8,
    "charging_hours_fast": 0.7,
    "grid_load_per_1000_evs_kw": 165,     # Avg concurrent draw
}

# Emission factors per unit fuel
_EMISSION_FACTORS = {
    "petrol_co2_kg_per_litre": 2.31,
    "diesel_co2_kg_per_litre": 2.68,
    "cng_co2_kg_per_kg": 2.75,
    "grid_co2_kg_per_kwh": 0.82,          # Indian grid average
}


class EnergyEngine(SimulationEngine):
    """Energy consumption, EV infrastructure, and grid impact simulation."""

    @property
    def name(self) -> str:
        return "EnergyEngine"

    @property
    def capabilities(self) -> List[EngineCapability]:
        return [EngineCapability.ENERGY]

    @property
    def optional_data(self) -> List[str]:
        return ["registered_vehicles", "ev_count", "daily_vkt"]

    def estimate_missing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if "registered_vehicles" not in data:
            data["registered_vehicles"] = 12_000_000
        if "ev_count" not in data:
            data["ev_count"] = int(data["registered_vehicles"] * _BASELINES["ev_share"])
        if "daily_vkt" not in data:
            data["daily_vkt"] = data["registered_vehicles"] * 12  # Avg 12 km/day
        return data

    async def _run(self, scenario: Scenario, data: Dict[str, Any]) -> SimulationResult:
        registered = data.get("registered_vehicles", 12_000_000)
        ev_count = data.get("ev_count", 360_000)
        daily_vkt = data.get("daily_vkt", 144_000_000)

        ev_increase = 0
        new_chargers_slow = 0
        new_chargers_fast = 0
        fuel_reduction_pct = 0.0
        renewable_addition_mw = 0
        congestion_pricing_active = False

        for intv in scenario.interventions:
            p = intv.parameters
            if intv.name == "ev_fleet_expansion":
                pct = p.get("ev_share_increase", 0.10)
                ev_increase += int(registered * pct)
                new_chargers_slow += p.get("chargers_slow", int(ev_increase * 0.1))
                new_chargers_fast += p.get("chargers_fast", int(ev_increase * 0.02))

            elif intv.name == "renewable_energy":
                renewable_addition_mw += p.get("solar_mw", 500) + p.get("wind_mw", 0)

            elif intv.name == "congestion_pricing":
                congestion_pricing_active = True
                fuel_reduction_pct += p.get("demand_reduction_pct", 12) * 0.8

            elif intv.name == "bus_rapid_transit":
                # BRT shifts demand from private ICE to electric/CNG buses
                fuel_reduction_pct += 3.0

            elif intv.name == "metro_expansion":
                km = p.get("km", 25)
                # Each metro km replaces ~800,000 vehicle-km/day
                shifted_vkt = km * 800_000
                fuel_reduction_pct += shifted_vkt / daily_vkt * 100

        # New totals
        total_ev = ev_count + ev_increase
        new_ev_share = total_ev / registered

        # Daily electricity demand from EVs
        ev_daily_kwh = total_ev * _EV_PARAMS["avg_daily_km"] * _EV_PARAMS["avg_kwh_per_km"]
        ev_peak_load_mw = total_ev / 1000 * _EV_PARAMS["grid_load_per_1000_evs_kw"] / 1000  # kW→MW

        # Grid headroom
        existing_headroom = _BASELINES["grid_capacity_mw"] - _BASELINES["peak_demand_mw"]
        new_headroom = existing_headroom + renewable_addition_mw - ev_peak_load_mw

        # Fuel savings
        baseline_fuel_ml = _BASELINES["daily_fuel_consumption_ml"]
        ev_fuel_displaced_ml = (ev_increase * _EV_PARAMS["avg_daily_km"] * 0.07) / 1_000_000  # ~0.07 L/km avg
        total_fuel_reduction_ml = baseline_fuel_ml * (fuel_reduction_pct / 100) + ev_fuel_displaced_ml
        new_fuel_ml = max(0, baseline_fuel_ml - total_fuel_reduction_ml)

        # CO2 from fuel
        baseline_co2_tonnes = (
            baseline_fuel_ml * 1e6 * _BASELINES["petrol_pct"] * _EMISSION_FACTORS["petrol_co2_kg_per_litre"] +
            baseline_fuel_ml * 1e6 * _BASELINES["diesel_pct"] * _EMISSION_FACTORS["diesel_co2_kg_per_litre"]
        ) / 1000
        new_co2_fuel = baseline_co2_tonnes * (new_fuel_ml / baseline_fuel_ml)
        ev_co2 = ev_daily_kwh * _EMISSION_FACTORS["grid_co2_kg_per_kwh"] / 1000
        # Renewable offset
        renewable_co2_offset = renewable_addition_mw * 24 * _EMISSION_FACTORS["grid_co2_kg_per_kwh"] * 0.3 / 1000  # 30% capacity factor

        net_co2_change = (new_co2_fuel + ev_co2 - baseline_co2_tonnes - renewable_co2_offset)

        # Financial
        daily_fuel_savings_cr = total_fuel_reduction_ml * 1e6 * (
            _BASELINES["petrol_pct"] * _BASELINES["petrol_price_per_litre"] +
            _BASELINES["diesel_pct"] * _BASELINES["diesel_price_per_litre"]
        ) / 1e7
        ev_electricity_cost_cr = ev_daily_kwh * _BASELINES["electricity_rate_per_kwh"] / 1e7
        charger_infra_cost_cr = (
            new_chargers_slow * 1.5 +   # ₹1.5 lakh per slow charger
            new_chargers_fast * 25       # ₹25 lakh per fast charger
        ) / 100

        metrics = {
            "current_ev_count": ev_count,
            "projected_ev_count": total_ev,
            "ev_share_pct": round(new_ev_share * 100, 1),
            "ev_daily_electricity_mwh": round(ev_daily_kwh / 1000, 1),
            "ev_peak_load_mw": round(ev_peak_load_mw, 1),
            "grid_headroom_mw": round(new_headroom, 1),
            "grid_stress": "critical" if new_headroom < 0 else ("high" if new_headroom < 200 else "manageable"),
            "fuel_reduction_ml_per_day": round(total_fuel_reduction_ml, 2),
            "fuel_reduction_pct": round(total_fuel_reduction_ml / baseline_fuel_ml * 100, 1),
            "co2_change_tonnes_daily": round(net_co2_change, 1),
            "daily_fuel_savings_cr": round(daily_fuel_savings_cr, 2),
            "charger_infra_cost_cr": round(charger_infra_cost_cr, 1),
            "new_slow_chargers": new_chargers_slow,
            "new_fast_chargers": new_chargers_fast,
            "renewable_addition_mw": renewable_addition_mw,
        }

        impacts = {
            "energy_security_improvement": round(total_fuel_reduction_ml / baseline_fuel_ml * 100, 1),
            "grid_readiness": "ready" if new_headroom > 500 else ("marginal" if new_headroom > 0 else "insufficient"),
            "annual_fuel_savings_cr": round(daily_fuel_savings_cr * 365, 0),
        }

        recommendations = []
        if new_headroom < 0:
            recommendations.append(f"Grid capacity insufficient — need {abs(new_headroom):.0f} MW additional generation before EV scale-up")
        if new_ev_share > 0.15 and new_chargers_fast < ev_increase * 0.01:
            recommendations.append("Fast charger density too low for projected EV fleet — target 1 fast charger per 100 EVs")
        if renewable_addition_mw == 0 and ev_increase > 100_000:
            recommendations.append("Large EV fleet without renewable additions increases coal dependency — add solar/wind capacity")
        if daily_fuel_savings_cr > 10:
            recommendations.append(f"Annual fuel import savings of ₹{daily_fuel_savings_cr * 365:,.0f} Cr — significant energy security benefit")

        warnings = []
        if new_headroom < 200:
            warnings.append("Grid stress during summer peak + EV charging overlap — implement smart charging schedules")

        return SimulationResult(
            engine=self.name,
            domain=EngineCapability.ENERGY,
            scenario_name=scenario.name,
            metrics=metrics,
            impacts=impacts,
            recommendations=recommendations,
            confidence=0.60,
            warnings=warnings,
        )
