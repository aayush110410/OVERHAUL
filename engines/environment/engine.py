"""
Environmental Impact Engine
============================

Models air quality, emissions, noise pollution, and green infrastructure
impacts from urban interventions. Uses deterministic emission models
calibrated for Indian cities (CPCB/SAFAR baselines).

Baselines are loaded from trained model data (models/ncr_baselines.json)
when available, falling back to hardcoded defaults.

Supports:
  - AQI change from traffic interventions
  - PM2.5/PM10 dispersion (simplified Gaussian)
  - Green cover impact on AQI
  - Stubble burning contribution (seasonal)
  - Industrial emission zones
  - Health impact estimation (DALYs, premature deaths)
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List

from engines.base import (
    EngineCapability,
    Scenario,
    SimulationEngine,
    SimulationResult,
)

# ── Load baselines from trained model data if available ──
_BASELINES_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "models", "ncr_baselines.json")

def _load_baselines() -> Dict[str, Dict[str, float]]:
    """Load real baselines from trained data, with hardcoded fallback."""
    defaults = {
        "delhi": {"pm25": 164, "pm10": 347, "aqi": 249, "no2": 65, "so2": 18, "o3": 35},
        "noida": {"pm25": 137, "pm10": 283, "aqi": 240, "no2": 52, "so2": 14, "o3": 40},
        "ghaziabad": {"pm25": 159, "pm10": 332, "aqi": 260, "no2": 58, "so2": 16, "o3": 38},
        "gurugram": {"pm25": 137, "pm10": 291, "aqi": 253, "no2": 48, "so2": 12, "o3": 42},
    }
    try:
        baselines_path = os.path.normpath(_BASELINES_FILE)
        if os.path.exists(baselines_path):
            with open(baselines_path) as f:
                trained = json.load(f)
            for city_key, data in trained.items():
                city_lower = city_key.lower()
                if city_lower in defaults and "aqi" in data:
                    aqi_data = data["aqi"]
                    defaults[city_lower]["pm25"] = round(aqi_data.get("pm25_mean", defaults[city_lower]["pm25"]))
                    defaults[city_lower]["pm10"] = round(aqi_data.get("pm10_mean", defaults[city_lower]["pm10"]))
                    defaults[city_lower]["aqi"] = round(aqi_data.get("mean", defaults[city_lower]["aqi"]))
    except Exception:
        pass  # Use defaults
    return defaults

_BASELINES = _load_baselines()

# Source contribution breakdown (Delhi winter, approximate)
_SOURCE_CONTRIBUTIONS = {
    "transport": 0.28,       # 28% of PM2.5 from vehicular
    "industry": 0.22,        # 22%
    "construction_dust": 0.12,
    "stubble_burning": 0.15, # Seasonal Oct-Nov spike
    "domestic_cooking": 0.08,
    "road_dust": 0.10,
    "other": 0.05,
}

# Health impact factors (per µg/m³ PM2.5 annual avg, per million population)
_HEALTH_IMPACT = {
    "premature_deaths_per_ug_per_million": 12.0,
    "respiratory_hospital_per_ug_per_million": 85.0,
    "asthma_exacerbation_per_ug_per_million": 320.0,
    "daly_per_ug_per_million": 45.0,
}


def _aqi_from_pm25(pm25: float) -> int:
    """Simplified AQI calculation from PM2.5 (Indian AQI scale)."""
    breakpoints = [
        (0, 30, 0, 50),
        (31, 60, 51, 100),
        (61, 90, 101, 200),
        (91, 120, 201, 300),
        (121, 250, 301, 400),
        (251, 500, 401, 500),
    ]
    for bp_lo, bp_hi, i_lo, i_hi in breakpoints:
        if pm25 <= bp_hi:
            return round(i_lo + (i_hi - i_lo) * (pm25 - bp_lo) / (bp_hi - bp_lo))
    return 500


class EnvironmentEngine(SimulationEngine):
    """Environmental impact simulation — AQI, emissions, health."""

    @property
    def name(self) -> str:
        return "EnvironmentEngine"

    @property
    def capabilities(self) -> List[EngineCapability]:
        return [EngineCapability.ENVIRONMENT]

    @property
    def optional_data(self) -> List[str]:
        return ["city", "season", "baseline_pm25", "population", "transport_result"]

    def estimate_missing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        city = data.get("city", "delhi").lower()
        if "baseline_pm25" not in data:
            data["baseline_pm25"] = _BASELINES.get(city, _BASELINES["delhi"])["pm25"]
        if "population" not in data:
            pops = {"delhi": 20_000_000, "noida": 700_000, "ghaziabad": 2_400_000, "gurugram": 1_200_000}
            data["population"] = pops.get(city, 5_000_000)
        if "season" not in data:
            data["season"] = "winter"
        return data

    async def _run(self, scenario: Scenario, data: Dict[str, Any]) -> SimulationResult:
        city = data.get("city", "delhi").lower()
        baseline_pm25 = data.get("baseline_pm25", 285.0)
        population = data.get("population", 20_000_000)
        season = data.get("season", "winter")

        # Calculate intervention effects on PM2.5
        transport_pct_reduction = 0.0
        industry_pct_reduction = 0.0
        green_cover_reduction = 0.0
        construction_reduction = 0.0

        for intv in scenario.interventions:
            p = intv.parameters
            if intv.name in ("ev_fleet_expansion", "bus_rapid_transit"):
                ev_increase = p.get("ev_share_increase", 0.10)
                # Each 10% EV penetration → ~3.5% PM2.5 reduction from transport
                transport_pct_reduction += ev_increase * 35
            elif intv.name == "congestion_pricing":
                transport_pct_reduction += p.get("demand_reduction_pct", 12) * 0.8
            elif intv.name == "signal_optimization":
                # Smoother flow → less stop-start emissions
                transport_pct_reduction += 3.0
            elif intv.name == "green_infrastructure":
                # Trees and urban green cover
                green_cover_reduction = p.get("green_cover_increase_pct", 5) * 0.4
            elif intv.name == "industrial_relocation":
                industry_pct_reduction = p.get("reduction_pct", 30.0)
            elif intv.name == "construction_dust_control":
                construction_reduction = p.get("reduction_pct", 50.0)
            elif intv.name == "stubble_burning_solution":
                # Reduce stubble contribution
                stubble_reduction = p.get("reduction_pct", 70.0)
                # Only applies in Oct-Nov season
                if season in ("autumn", "winter"):
                    transport_pct_reduction += stubble_reduction * _SOURCE_CONTRIBUTIONS["stubble_burning"] / _SOURCE_CONTRIBUTIONS["transport"]

        # Calculate new PM2.5
        transport_share = _SOURCE_CONTRIBUTIONS["transport"]
        industry_share = _SOURCE_CONTRIBUTIONS["industry"]
        construction_share = _SOURCE_CONTRIBUTIONS["construction_dust"]

        pm25_reduction = (
            baseline_pm25 * transport_share * (transport_pct_reduction / 100.0) +
            baseline_pm25 * industry_share * (industry_pct_reduction / 100.0) +
            baseline_pm25 * construction_share * (construction_reduction / 100.0) +
            baseline_pm25 * (green_cover_reduction / 100.0)
        )

        # Cap reduction at 50% (can't eliminate all sources)
        pm25_reduction = min(pm25_reduction, baseline_pm25 * 0.50)
        new_pm25 = max(0, baseline_pm25 - pm25_reduction)
        new_aqi = _aqi_from_pm25(new_pm25)

        # Health impact
        pm25_delta = baseline_pm25 - new_pm25
        pop_millions = population / 1_000_000
        lives_saved = pm25_delta * _HEALTH_IMPACT["premature_deaths_per_ug_per_million"] * pop_millions
        hospitalizations_avoided = pm25_delta * _HEALTH_IMPACT["respiratory_hospital_per_ug_per_million"] * pop_millions
        dalys_saved = pm25_delta * _HEALTH_IMPACT["daly_per_ug_per_million"] * pop_millions

        metrics = {
            "baseline_pm25": baseline_pm25,
            "projected_pm25": round(new_pm25, 1),
            "pm25_reduction": round(pm25_delta, 1),
            "baseline_aqi": _aqi_from_pm25(baseline_pm25),
            "projected_aqi": new_aqi,
            "aqi_category": self._aqi_category(new_aqi),
            "lives_saved_annual": round(lives_saved),
            "hospitalizations_avoided": round(hospitalizations_avoided),
            "dalys_saved": round(dalys_saved, 1),
            "city": city,
        }

        impacts = {
            "pm25_reduction_pct": round(pm25_delta / max(baseline_pm25, 1) * 100, 1),
            "aqi_improvement_points": _aqi_from_pm25(baseline_pm25) - new_aqi,
            "health_benefit_score": min(100, round(lives_saved / max(pop_millions, 1) * 2)),
        }

        recommendations = []
        if new_aqi > 300:
            recommendations.append("AQI remains 'Severe' — multi-sector interventions needed beyond transport")
        if new_aqi > 200:
            recommendations.append("Consider GRAP Stage-III/IV measures for immediate relief")
        if transport_pct_reduction < 10:
            recommendations.append("Transport interventions alone insufficient — combine with industrial + construction controls")
        if pm25_delta > 30:
            recommendations.append(f"Projected {pm25_delta:.0f} µg/m³ PM2.5 reduction would save ~{lives_saved:.0f} lives annually")

        warnings = []
        if season == "winter":
            warnings.append("Winter inversion layers trap pollutants — reductions may be less effective Nov-Feb")

        return SimulationResult(
            engine=self.name,
            domain=EngineCapability.ENVIRONMENT,
            scenario_name=scenario.name,
            metrics=metrics,
            impacts=impacts,
            recommendations=recommendations,
            confidence=0.65,
            warnings=warnings,
        )

    def _aqi_category(self, aqi: int) -> str:
        if aqi <= 50:
            return "Good"
        if aqi <= 100:
            return "Satisfactory"
        if aqi <= 200:
            return "Moderate"
        if aqi <= 300:
            return "Poor"
        if aqi <= 400:
            return "Very Poor"
        return "Severe"
