"""
Economic Engine
================

Models cost-benefit analysis of urban interventions including:
  - Infrastructure investment ROI
  - Commuter time-value savings
  - Healthcare cost reductions from lower pollution
  - Property value impacts
  - GDP impact of congestion
  - Employment generation from projects

Calibrated for Indian economic parameters (INR, Indian health costs, etc.).
"""

from __future__ import annotations

from typing import Any, Dict, List

from engines.base import (
    EngineCapability,
    Scenario,
    SimulationEngine,
    SimulationResult,
)

# ── Economic Constants (India 2025) ──
_PARAMS = {
    "avg_hourly_wage_inr": 250,           # Metro city average
    "daily_commuters": 5_000_000,          # Delhi NCR
    "avg_commute_minutes": 52,
    "working_days_year": 300,
    "healthcare_cost_per_respiratory_case_inr": 45_000,
    "premature_death_vsl_cr": 3.5,         # Value of Statistical Life in crore INR
    "property_premium_per_pct_aqi_improvement": 0.15,  # % increase per 1% AQI drop
    "avg_property_value_cr": 1.2,          # Delhi avg residential
    "total_residential_properties": 4_500_000,
    "gdp_ncr_lakh_cr": 12.5,              # Delhi NCR GDP ~₹12.5 lakh crore
    "congestion_gdp_loss_pct": 3.5,        # % GDP lost to congestion
    "discount_rate": 0.08,
    "inflation_rate": 0.05,
}


class EconomicEngine(SimulationEngine):
    """Economic impact and cost-benefit analysis engine."""

    @property
    def name(self) -> str:
        return "EconomicEngine"

    @property
    def capabilities(self) -> List[EngineCapability]:
        return [EngineCapability.ECONOMIC]

    @property
    def optional_data(self) -> List[str]:
        return ["transport_result", "environment_result", "infrastructure_result", "energy_result"]

    def estimate_missing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Economic engine works best with results from other engines,
        # but can estimate independently
        if "time_saved_min" not in data:
            data["time_saved_min"] = 0
        if "pm25_reduction" not in data:
            data["pm25_reduction"] = 0
        if "infrastructure_cost_cr" not in data:
            data["infrastructure_cost_cr"] = 0
        if "fuel_savings_cr_daily" not in data:
            data["fuel_savings_cr_daily"] = 0
        return data

    async def _run(self, scenario: Scenario, data: Dict[str, Any]) -> SimulationResult:
        horizon_years = max(1, scenario.time_horizon_days // 365) if scenario.time_horizon_days > 365 else 5

        # Extract cross-engine results if available
        time_saved_min = data.get("time_saved_min", 0)
        pm25_reduction = data.get("pm25_reduction", 0)
        infra_cost_cr = data.get("infrastructure_cost_cr", 0)
        fuel_savings_daily_cr = data.get("fuel_savings_cr_daily", 0)

        # Estimate from interventions if cross-engine data missing
        for intv in scenario.interventions:
            p = intv.parameters
            if intv.name == "signal_optimization":
                if time_saved_min == 0:
                    time_saved_min += 3
                infra_cost_cr += p.get("junctions", 200) * 0.02
            elif intv.name == "metro_expansion":
                km = p.get("km", 25)
                if time_saved_min == 0:
                    time_saved_min += km * 0.3
                infra_cost_cr += km * 350
            elif intv.name == "congestion_pricing":
                if time_saved_min == 0:
                    time_saved_min += 5
            elif intv.name == "bus_rapid_transit":
                km = p.get("km", 15)
                if time_saved_min == 0:
                    time_saved_min += 4
                infra_cost_cr += km * 8.5
            elif intv.name == "road_capacity_expansion":
                km = p.get("km", 20)
                if time_saved_min == 0:
                    time_saved_min += km * 0.1
                infra_cost_cr += km * 18
            elif intv.name == "flyover_construction":
                km = p.get("km", 5)
                infra_cost_cr += km * 120
            elif intv.name == "ev_fleet_expansion":
                ev_pct = p.get("ev_share_increase", 0.10)
                pm25_reduction += ev_pct * 10  # Rough estimate
            elif intv.name == "green_infrastructure":
                pm25_reduction += 5

        # ── Time Value Savings ──
        daily_time_value_cr = (
            time_saved_min / 60 *
            _PARAMS["avg_hourly_wage_inr"] *
            _PARAMS["daily_commuters"]
        ) / 1e7
        annual_time_value_cr = daily_time_value_cr * _PARAMS["working_days_year"]

        # ── Healthcare Savings ──
        population = 20_000_000
        pop_millions = population / 1e6
        lives_saved = pm25_reduction * 12 * pop_millions  # 12 per µg per million
        respiratory_cases_avoided = pm25_reduction * 85 * pop_millions
        annual_healthcare_savings_cr = (
            respiratory_cases_avoided * _PARAMS["healthcare_cost_per_respiratory_case_inr"] +
            lives_saved * _PARAMS["premature_death_vsl_cr"] * 1e7
        ) / 1e7

        # ── Property Value Impact ──
        if pm25_reduction > 0:
            aqi_improvement_pct = min(30, pm25_reduction / 285 * 100)
            property_premium_pct = aqi_improvement_pct * _PARAMS["property_premium_per_pct_aqi_improvement"]
            total_property_appreciation_cr = (
                _PARAMS["total_residential_properties"] *
                _PARAMS["avg_property_value_cr"] *
                property_premium_pct / 100
            )
        else:
            property_premium_pct = 0
            total_property_appreciation_cr = 0

        # ── GDP Impact ──
        congestion_loss_cr = _PARAMS["gdp_ncr_lakh_cr"] * 1e5 * _PARAMS["congestion_gdp_loss_pct"] / 100
        if time_saved_min > 0:
            congestion_reduction_pct = min(40, time_saved_min / _PARAMS["avg_commute_minutes"] * 100)
            gdp_recovered_cr = congestion_loss_cr * congestion_reduction_pct / 100
        else:
            congestion_reduction_pct = 0
            gdp_recovered_cr = 0

        # ── Fuel Import Savings ──
        annual_fuel_savings_cr = fuel_savings_daily_cr * 365

        # ── Employment Generation ──
        # Rough: ₹1 Cr infra spend → 15 direct jobs + 10 indirect
        jobs_direct = int(infra_cost_cr * 15)
        jobs_indirect = int(infra_cost_cr * 10)

        # ── NPV Calculation ──
        total_annual_benefits = (
            annual_time_value_cr +
            annual_healthcare_savings_cr +
            annual_fuel_savings_cr +
            gdp_recovered_cr
        )

        npv_benefits = sum(
            total_annual_benefits / (1 + _PARAMS["discount_rate"]) ** y
            for y in range(1, horizon_years + 1)
        )
        npv_costs = infra_cost_cr  # Upfront cost

        bcr = npv_benefits / max(npv_costs, 1) if npv_costs > 0 else 999.99

        metrics = {
            "annual_time_value_saved_cr": round(annual_time_value_cr, 1),
            "annual_healthcare_savings_cr": round(annual_healthcare_savings_cr, 1),
            "annual_fuel_savings_cr": round(annual_fuel_savings_cr, 1),
            "gdp_recovered_annually_cr": round(gdp_recovered_cr, 1),
            "total_annual_benefits_cr": round(total_annual_benefits, 1),
            "infrastructure_cost_cr": round(infra_cost_cr, 1),
            "npv_benefits_cr": round(npv_benefits, 1),
            "benefit_cost_ratio": round(bcr, 2),
            "property_appreciation_cr": round(total_property_appreciation_cr, 1),
            "property_premium_pct": round(property_premium_pct, 2),
            "jobs_direct": jobs_direct,
            "jobs_indirect": jobs_indirect,
            "jobs_total": jobs_direct + jobs_indirect,
            "lives_saved_annually": round(lives_saved),
            "congestion_gdp_recovery_pct": round(congestion_reduction_pct, 1),
            "analysis_horizon_years": horizon_years,
        }

        impacts = {
            "economic_viability": "excellent" if bcr > 3 else ("good" if bcr > 1.5 else ("marginal" if bcr > 1 else "negative")),
            "social_benefit_score": min(100, round(lives_saved + jobs_direct / 100 + annual_healthcare_savings_cr / 10)),
            "fiscal_sustainability": "self_sustaining" if total_annual_benefits > infra_cost_cr / 5 else "requires_subsidy",
        }

        recommendations = []
        if bcr < 1:
            recommendations.append(f"Benefit-cost ratio {bcr:.1f} < 1.0 — project economically unviable without additional revenue streams")
        elif bcr > 3:
            recommendations.append(f"BCR of {bcr:.1f} indicates strong economic case — prioritize implementation")
        if annual_healthcare_savings_cr > annual_time_value_cr:
            recommendations.append("Health benefits exceed mobility benefits — pollution reduction should be primary policy lever")
        if jobs_direct > 10000:
            recommendations.append(f"Project generates {jobs_direct:,} direct + {jobs_indirect:,} indirect jobs — significant employment impact")
        if property_premium_pct > 1:
            recommendations.append(f"Property values projected to rise {property_premium_pct:.1f}% — land value capture can fund infrastructure")

        return SimulationResult(
            engine=self.name,
            domain=EngineCapability.ECONOMIC,
            scenario_name=scenario.name,
            metrics=metrics,
            impacts=impacts,
            recommendations=recommendations,
            confidence=0.55,
            warnings=["Economic projections use simplified models — actual ROI depends on implementation quality and policy stability"],
        )
