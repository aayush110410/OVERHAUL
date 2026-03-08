"""
Population Behavior Engine
============================

Models how different population segments respond to urban interventions.
Segments by occupation, income, mobility pattern, and mode preference.
Simulates mode shift, route change, schedule adaptation, and EV adoption.

Calibrated for Delhi NCR commuter demographics.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

from engines.base import (
    EngineCapability,
    Scenario,
    SimulationEngine,
    SimulationResult,
)

# ── Population Segments ──
_SEGMENTS = {
    "office_workers": {
        "share": 0.30,
        "avg_income_monthly": 55_000,
        "primary_mode": "car",
        "mode_split": {"car": 0.50, "metro": 0.25, "bus": 0.10, "two_wheeler": 0.12, "other": 0.03},
        "flexibility": 0.4,       # 0-1, willingness to change
        "ev_readiness": 0.35,
        "wfh_capable": 0.60,
        "avg_commute_km": 18,
    },
    "gig_workers": {
        "share": 0.12,
        "avg_income_monthly": 20_000,
        "primary_mode": "two_wheeler",
        "mode_split": {"two_wheeler": 0.65, "bus": 0.20, "metro": 0.05, "car": 0.05, "other": 0.05},
        "flexibility": 0.2,
        "ev_readiness": 0.15,
        "wfh_capable": 0.0,
        "avg_commute_km": 25,
    },
    "students": {
        "share": 0.15,
        "avg_income_monthly": 0,
        "primary_mode": "bus",
        "mode_split": {"bus": 0.35, "metro": 0.30, "two_wheeler": 0.15, "car": 0.10, "other": 0.10},
        "flexibility": 0.6,
        "ev_readiness": 0.10,
        "wfh_capable": 0.40,
        "avg_commute_km": 12,
    },
    "service_sector": {
        "share": 0.18,
        "avg_income_monthly": 15_000,
        "primary_mode": "bus",
        "mode_split": {"bus": 0.40, "two_wheeler": 0.25, "metro": 0.15, "auto": 0.10, "other": 0.10},
        "flexibility": 0.15,
        "ev_readiness": 0.05,
        "wfh_capable": 0.0,
        "avg_commute_km": 15,
    },
    "industrial_workers": {
        "share": 0.10,
        "avg_income_monthly": 18_000,
        "primary_mode": "bus",
        "mode_split": {"bus": 0.45, "two_wheeler": 0.30, "car": 0.05, "company_bus": 0.15, "other": 0.05},
        "flexibility": 0.1,
        "ev_readiness": 0.05,
        "wfh_capable": 0.0,
        "avg_commute_km": 22,
    },
    "senior_citizens": {
        "share": 0.08,
        "avg_income_monthly": 25_000,
        "primary_mode": "car",
        "mode_split": {"car": 0.40, "auto": 0.25, "bus": 0.15, "metro": 0.10, "other": 0.10},
        "flexibility": 0.3,
        "ev_readiness": 0.20,
        "wfh_capable": 0.0,
        "avg_commute_km": 6,
    },
    "high_income": {
        "share": 0.07,
        "avg_income_monthly": 250_000,
        "primary_mode": "car",
        "mode_split": {"car": 0.80, "metro": 0.10, "cab": 0.08, "other": 0.02},
        "flexibility": 0.5,
        "ev_readiness": 0.60,
        "wfh_capable": 0.70,
        "avg_commute_km": 15,
    },
}


class PopulationEngine(SimulationEngine):
    """Population behavior and mode-shift simulation."""

    @property
    def name(self) -> str:
        return "PopulationEngine"

    @property
    def capabilities(self) -> List[EngineCapability]:
        return [EngineCapability.POPULATION]

    @property
    def optional_data(self) -> List[str]:
        return ["total_population", "commuter_population"]

    def estimate_missing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if "total_population" not in data:
            data["total_population"] = 20_000_000
        if "commuter_population" not in data:
            data["commuter_population"] = int(data["total_population"] * 0.35)
        return data

    async def _run(self, scenario: Scenario, data: Dict[str, Any]) -> SimulationResult:
        total_pop = data.get("total_population", 20_000_000)
        commuters = data.get("commuter_population", 7_000_000)

        # Determine intervention effects on each mode
        mode_push = self._compute_mode_push(scenario)
        ev_incentive = self._compute_ev_incentive(scenario)
        wfh_push = self._compute_wfh_push(scenario)

        segment_results = {}
        total_mode_shift = {"car": 0, "metro": 0, "bus": 0, "two_wheeler": 0, "ev": 0, "wfh": 0, "cycle": 0}
        total_ev_adopters = 0
        total_wfh_adopters = 0
        total_vkt_before = 0
        total_vkt_after = 0

        for seg_name, seg in _SEGMENTS.items():
            seg_pop = int(commuters * seg["share"])

            # Mode shift computation
            new_mode_split = dict(seg["mode_split"])
            car_shift_out = 0

            # Metro/bus improvements pull from car and two_wheeler
            if mode_push.get("metro", 0) > 0:
                pull = mode_push["metro"] * seg["flexibility"] * 0.5
                shift_from_car = min(new_mode_split.get("car", 0), pull * 0.6)
                shift_from_2w = min(new_mode_split.get("two_wheeler", 0), pull * 0.3)
                new_mode_split["car"] = new_mode_split.get("car", 0) - shift_from_car
                new_mode_split["two_wheeler"] = new_mode_split.get("two_wheeler", 0) - shift_from_2w
                new_mode_split["metro"] = new_mode_split.get("metro", 0) + shift_from_car + shift_from_2w
                car_shift_out += shift_from_car

            if mode_push.get("bus", 0) > 0:
                pull = mode_push["bus"] * seg["flexibility"] * 0.3
                shift = min(new_mode_split.get("two_wheeler", 0), pull)
                new_mode_split["two_wheeler"] = new_mode_split.get("two_wheeler", 0) - shift
                new_mode_split["bus"] = new_mode_split.get("bus", 0) + shift

            if mode_push.get("congestion_pricing", 0) > 0:
                price_effect = mode_push["congestion_pricing"] * seg["flexibility"]
                shift = min(new_mode_split.get("car", 0), price_effect * 0.15)
                new_mode_split["car"] = new_mode_split.get("car", 0) - shift
                new_mode_split["metro"] = new_mode_split.get("metro", 0) + shift * 0.6
                new_mode_split["bus"] = new_mode_split.get("bus", 0) + shift * 0.4
                car_shift_out += shift

            # WFH adoption
            wfh_adopters = int(seg_pop * seg["wfh_capable"] * wfh_push * seg["flexibility"])

            # EV adoption (S-curve)
            ev_propensity = seg["ev_readiness"] * (1 + ev_incentive)
            ev_propensity = min(0.8, ev_propensity)
            ev_adopters = int(seg_pop * new_mode_split.get("car", 0) * ev_propensity * 0.3)  # 30% of car users adopt over horizon

            # VKT
            seg_vkt_before = seg_pop * seg["avg_commute_km"] * 2  # Round trip
            vkt_reduction = car_shift_out * seg_pop * seg["avg_commute_km"] * 2
            wfh_vkt_saved = wfh_adopters * seg["avg_commute_km"] * 2 * 0.6  # Assume 3 WFH days/week
            seg_vkt_after = seg_vkt_before - vkt_reduction - wfh_vkt_saved

            total_vkt_before += seg_vkt_before
            total_vkt_after += max(0, seg_vkt_after)
            total_ev_adopters += ev_adopters
            total_wfh_adopters += wfh_adopters

            segment_results[seg_name] = {
                "population": seg_pop,
                "car_mode_shift_pct": round(car_shift_out * 100, 1),
                "ev_adopters": ev_adopters,
                "wfh_adopters": wfh_adopters,
                "vkt_reduction_pct": round((1 - seg_vkt_after / max(seg_vkt_before, 1)) * 100, 1),
            }

        vkt_reduction_pct = (1 - total_vkt_after / max(total_vkt_before, 1)) * 100

        metrics = {
            "total_commuters_modeled": commuters,
            "vkt_before_daily": total_vkt_before,
            "vkt_after_daily": round(total_vkt_after),
            "vkt_reduction_pct": round(vkt_reduction_pct, 1),
            "ev_adopters_projected": total_ev_adopters,
            "ev_adoption_pct": round(total_ev_adopters / max(commuters, 1) * 100, 1),
            "wfh_adopters": total_wfh_adopters,
            "wfh_pct": round(total_wfh_adopters / max(commuters, 1) * 100, 1),
            "segments_modeled": len(_SEGMENTS),
        }

        impacts = {
            "mode_shift_effectiveness": "high" if vkt_reduction_pct > 10 else ("medium" if vkt_reduction_pct > 5 else "low"),
            "equity_score": self._equity_score(segment_results),
            "segment_details": segment_results,
        }

        recommendations = []
        # Check equity across segments
        low_income_shift = segment_results.get("service_sector", {}).get("car_mode_shift_pct", 0)
        high_income_shift = segment_results.get("high_income", {}).get("car_mode_shift_pct", 0)
        if high_income_shift < low_income_shift * 0.5:
            recommendations.append("High-income segments resist mode shift — stronger congestion pricing or parking restrictions needed")
        if total_ev_adopters > 500_000:
            recommendations.append(f"Projected {total_ev_adopters:,} EV adopters — ensure adequate charging infrastructure")
        if total_wfh_adopters > 1_000_000:
            recommendations.append(f"{total_wfh_adopters:,} potential WFH workers — digital infrastructure investment needed")
        if vkt_reduction_pct < 5:
            recommendations.append("Minimal VKT reduction — interventions need stronger demand-side measures")

        return SimulationResult(
            engine=self.name,
            domain=EngineCapability.POPULATION,
            scenario_name=scenario.name,
            metrics=metrics,
            impacts=impacts,
            recommendations=recommendations,
            confidence=0.55,
            warnings=["Behavioral models use stated-preference assumptions — actual adoption may differ by ±30%"],
        )

    def _compute_mode_push(self, scenario: Scenario) -> Dict[str, float]:
        push: Dict[str, float] = {}
        for intv in scenario.interventions:
            if intv.name == "metro_expansion":
                push["metro"] = push.get("metro", 0) + intv.parameters.get("km", 25) / 50
            elif intv.name == "bus_rapid_transit":
                push["bus"] = push.get("bus", 0) + intv.parameters.get("km", 15) / 30
            elif intv.name == "congestion_pricing":
                push["congestion_pricing"] = push.get("congestion_pricing", 0) + 1.0
            elif intv.name == "pedestrian_cycling_infra":
                push["cycle"] = push.get("cycle", 0) + 0.3
        return push

    def _compute_ev_incentive(self, scenario: Scenario) -> float:
        incentive = 0.0
        for intv in scenario.interventions:
            if intv.name == "ev_fleet_expansion":
                incentive += intv.parameters.get("subsidy_factor", 0.5)
        return incentive

    def _compute_wfh_push(self, scenario: Scenario) -> float:
        push = 0.0
        for intv in scenario.interventions:
            if intv.name == "congestion_pricing":
                push += 0.3  # Pricing pushes WFH-capable workers home
            if intv.name == "wfh_incentive":
                push += intv.parameters.get("factor", 0.5)
        return min(1.0, push)

    def _equity_score(self, results: Dict[str, Dict]) -> int:
        """Score 0-100 for how equitably benefits distribute across segments."""
        reductions = [r.get("vkt_reduction_pct", 0) for r in results.values()]
        if not reductions:
            return 50
        avg = sum(reductions) / len(reductions)
        variance = sum((r - avg) ** 2 for r in reductions) / len(reductions)
        # Lower variance = more equitable
        cv = math.sqrt(variance) / max(avg, 1)
        return max(0, min(100, round(100 - cv * 50)))
