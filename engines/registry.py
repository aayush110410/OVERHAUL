"""
Engine Registry
================

Discovers, registers, and coordinates all simulation engines.
Engines self-register on import — the registry auto-discovers
all engines in the engines/ subpackages.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Any, Dict, List, Optional

from engines.base import EngineCapability, Scenario, SimulationEngine, SimulationResult

# ── Simple in-memory TTL cache ──────────────────────────────────

_CACHE: Dict[str, tuple] = {}  # key → (result, expire_ts)
_CACHE_TTL = 300  # 5 minutes


def _cache_key(scenario: Scenario, data: Dict, engines: Optional[List[str]]) -> str:
    """Deterministic hash from scenario + data + engine selection."""
    raw = json.dumps({
        "s": scenario.name,
        "c": scenario.city,
        "i": [(i.name, i.domain, sorted(i.parameters.items())) for i in scenario.interventions],
        "t": scenario.time_horizon_days,
        "d": sorted((k, v) for k, v in data.items() if isinstance(v, (int, float, str, bool))),
        "e": sorted(engines) if engines else None,
    }, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _cache_get(key: str):
    entry = _CACHE.get(key)
    if entry and entry[1] > time.time():
        return entry[0]
    _CACHE.pop(key, None)
    return None


def _cache_set(key: str, value, ttl: int = _CACHE_TTL):
    _CACHE[key] = (value, time.time() + ttl)


class EngineRegistry:
    """Central registry for all simulation engines."""

    def __init__(self):
        self._engines: Dict[str, SimulationEngine] = {}

    def register(self, engine: SimulationEngine) -> None:
        self._engines[engine.name] = engine

    def get(self, name: str) -> Optional[SimulationEngine]:
        return self._engines.get(name)

    def list_engines(self) -> List[Dict[str, Any]]:
        return [e.info() for e in self._engines.values()]

    def find_by_capability(self, cap: EngineCapability) -> List[SimulationEngine]:
        return [e for e in self._engines.values() if cap in e.capabilities]

    # ── Phase execution ordering ──
    # Phase-1 engines generate primary metrics; Phase-2 engines consume them.
    _PHASE_1 = {EngineCapability.TRANSPORT, EngineCapability.INFRASTRUCTURE, EngineCapability.ENVIRONMENT}
    _PHASE_2 = {EngineCapability.ENERGY, EngineCapability.ECONOMIC, EngineCapability.POPULATION, EngineCapability.LOGISTICS}

    async def run_scenario(
        self,
        scenario: Scenario,
        data: Dict[str, Any] | None = None,
        engines: List[str] | None = None,
    ) -> Dict[str, SimulationResult]:
        """Run a scenario across relevant engines using two-phase execution.

        Phase 1: Transport, Infrastructure, Environment run in parallel.
        Phase 2: Energy, Economic, Population, Logistics run in parallel
                 with Phase-1 metrics merged into the shared data dict.

        Results are cached by scenario signature for 5 minutes.
        If `engines` is provided, only those engines run (single phase).
        """
        data = dict(data or {})

        # Check cache
        ck = _cache_key(scenario, data, engines)
        cached = _cache_get(ck)
        if cached is not None:
            return cached

        # Determine which engines to run
        if engines:
            targets = [self._engines[n] for n in engines if n in self._engines]
        else:
            needed_caps = {i.domain for i in scenario.interventions}
            always_run = {
                EngineCapability.ENVIRONMENT,
                EngineCapability.ENERGY,
                EngineCapability.ECONOMIC,
                EngineCapability.POPULATION,
            }
            targets = []
            for eng in self._engines.values():
                caps = set(eng.capabilities)
                if (needed_caps & caps) or (caps & always_run):
                    targets.append(eng)

        if not targets:
            return {}

        # Fill in missing data with engine estimates
        for eng in targets:
            data = eng.estimate_missing(data)

        # If explicit engine list, run single-phase (all parallel)
        if engines:
            result = await self._run_batch(targets, scenario, data)
            _cache_set(ck, result)
            return result

        # Two-phase execution for cross-engine data flow
        phase1 = [e for e in targets if set(e.capabilities) & self._PHASE_1]
        phase2 = [e for e in targets if set(e.capabilities) & self._PHASE_2]

        results: Dict[str, SimulationResult] = {}

        # Phase 1 — primary metric generators
        if phase1:
            p1 = await self._run_batch(phase1, scenario, data)
            results.update(p1)
            # Inject Phase-1 outputs into shared data for Phase-2
            data = self._merge_phase1_results(data, p1)

        # Phase 2 — downstream impact engines
        if phase2:
            p2 = await self._run_batch(phase2, scenario, data)
            results.update(p2)

        _cache_set(ck, results)
        return results

    @staticmethod
    def _merge_phase1_results(
        data: Dict[str, Any],
        phase1: Dict[str, SimulationResult],
    ) -> Dict[str, Any]:
        """Extract key metrics from Phase-1 results and inject into data dict."""
        for name, r in phase1.items():
            if not hasattr(r, "metrics") or not r.metrics:
                continue
            m = r.metrics
            # Transport → downstream
            if "avg_speed_kmh" in m:
                data.setdefault("baseline_speed_kmh", m["avg_speed_kmh"])
            if "co2_tonnes" in m:
                data.setdefault("transport_co2_tonnes", m["co2_tonnes"])
            if "total_vkt" in m:
                data.setdefault("daily_vkt", m["total_vkt"])
            if "ev_share" in m:
                data.setdefault("ev_share", m["ev_share"])
            # Infrastructure → downstream
            if "avg_commute_time_saved_min" in m:
                data["time_saved_min"] = data.get("time_saved_min", 0) + m["avg_commute_time_saved_min"]
            if "total_cost_inr_crore" in m:
                data["infrastructure_cost_cr"] = data.get("infrastructure_cost_cr", 0) + m["total_cost_inr_crore"]
            # Environment → downstream
            if "pm25_reduction" in m:
                data["pm25_reduction"] = data.get("pm25_reduction", 0) + m["pm25_reduction"]
            if "projected_pm25" in m:
                data.setdefault("projected_pm25", m["projected_pm25"])
        return data

    async def _run_batch(
        self,
        batch: List[SimulationEngine],
        scenario: Scenario,
        data: Dict[str, Any],
    ) -> Dict[str, SimulationResult]:
        """Run a batch of engines in parallel and collect results."""
        tasks = {eng.name: eng.simulate(scenario, data) for eng in batch}
        results: Dict[str, SimulationResult] = {}
        for name, coro in tasks.items():
            try:
                results[name] = await coro
            except Exception as exc:
                results[name] = SimulationResult(
                    engine=name,
                    domain=EngineCapability.TRANSPORT,
                    scenario_name=scenario.name,
                    warnings=[f"Engine error: {exc}"],
                    confidence=0.0,
                )
        return results

    async def compare_scenarios(
        self,
        scenarios: List[Scenario],
        data: Dict[str, Any] | None = None,
    ) -> Dict[str, Dict[str, SimulationResult]]:
        """Run multiple scenarios and return results for comparison."""
        comparison = {}
        for scenario in scenarios:
            comparison[scenario.name] = await self.run_scenario(scenario, data)
        return comparison

    async def run_temporal(
        self,
        scenario: Scenario,
        data: Dict[str, Any] | None = None,
        steps: int = 4,
        step_days: int = 90,
    ) -> Dict[str, Any]:
        """Multi-step temporal simulation with feedback loops.

        Runs the scenario repeatedly over `steps` time periods.
        Each step's output feeds back as input to the next step,
        modeling how interventions evolve over time.

        Example: 4 steps × 90 days = 1-year evolution
          Step 1 (Q1): Initial metro construction begins
          Step 2 (Q2): 25% metro complete, partial mode shift
          Step 3 (Q3): 70% metro complete, growing ridership
          Step 4 (Q4): Full operation, equilibrium mode shift

        Returns:
          {"steps": [...], "timeline_days": int, "trend": {...}}
        """
        data = dict(data or {})
        timeline = []
        accumulated_days = 0

        for step_idx in range(steps):
            accumulated_days += step_days

            # Scale intervention parameters by time progress
            step_scenario = self._scale_scenario(
                scenario, progress=accumulated_days / scenario.time_horizon_days
            )

            # Run full two-phase simulation
            results = await self.run_scenario(step_scenario, data)

            # Collect step metrics
            step_metrics = {
                "step": step_idx + 1,
                "day": accumulated_days,
                "label": f"Q{step_idx + 1}" if step_days >= 80 else f"Month {step_idx + 1}",
                "engines": {},
            }
            for ename, result in results.items():
                step_metrics["engines"][ename] = {
                    "metrics": result.metrics if hasattr(result, "metrics") else {},
                    "confidence": result.confidence if hasattr(result, "confidence") else 0,
                }

            timeline.append(step_metrics)

            # Feed forward: inject this step's results as data for next step
            data = self._feedback_loop(data, results, step_idx, steps)

        # Compute trends across steps
        trends = self._compute_trends(timeline)

        return {
            "scenario": scenario.name,
            "steps": timeline,
            "timeline_days": accumulated_days,
            "step_size_days": step_days,
            "trend": trends,
        }

    def _scale_scenario(self, scenario: Scenario, progress: float) -> Scenario:
        """Scale intervention parameters by implementation progress.

        progress: 0.0 → 1.0 (fraction of time_horizon completed)
        Uses S-curve for realistic adoption modeling.
        """
        import math

        # S-curve: slow start, rapid middle, plateau at end
        # sigmoid centered at 0.5 with steepness k=10
        s = 1.0 / (1.0 + math.exp(-10 * (progress - 0.5)))
        s = min(s, 1.0)

        scaled_interventions = []
        for intervention in scenario.interventions:
            scaled_params = {}
            for k, v in intervention.parameters.items():
                if isinstance(v, (int, float)):
                    scaled_params[k] = v * s
                else:
                    scaled_params[k] = v
            from engines.base import Intervention
            scaled_interventions.append(Intervention(
                name=intervention.name,
                domain=intervention.domain,
                parameters=scaled_params,
                description=intervention.description,
            ))
        return Scenario(
            name=f"{scenario.name}_t{int(progress * 100)}",
            description=scenario.description,
            city=scenario.city,
            interventions=scaled_interventions,
            time_horizon_days=scenario.time_horizon_days,
            metadata={**scenario.metadata, "progress": round(progress, 2)},
        )

    @staticmethod
    def _feedback_loop(
        data: Dict[str, Any],
        results: Dict[str, SimulationResult],
        step: int,
        total_steps: int,
    ) -> Dict[str, Any]:
        """Feed simulation results back as input for the next time step.

        This creates the cross-engine feedback loop:
          Transport → Environment: more cars → worse AQI
          Environment → Economic: worse AQI → lower productivity
          Economic → Population: lower productivity → mode shift
          Population → Transport: mode shift → less congestion
        """
        for name, r in results.items():
            if not hasattr(r, "metrics"):
                continue
            m = r.metrics

            # Transport outputs become Environment inputs
            if "avg_speed_kmh" in m:
                data["baseline_speed_kmh"] = m["avg_speed_kmh"]
            if "co2_tonnes" in m:
                data["transport_co2_tonnes"] = m["co2_tonnes"]
            if "ev_share" in m:
                data["ev_share"] = m["ev_share"]
            if "total_vkt" in m:
                data["daily_vkt"] = m["total_vkt"]

            # Environment outputs become Economic inputs
            if "projected_pm25" in m:
                data["current_pm25"] = m["projected_pm25"]
            if "aqi_improvement_pct" in m:
                data["aqi_trend"] = m["aqi_improvement_pct"]

            # Economic outputs become Population inputs
            if "gdp_impact_pct" in m:
                data["economic_growth"] = m["gdp_impact_pct"]
            if "productivity_gain_pct" in m:
                data["productivity_signal"] = m["productivity_gain_pct"]

            # Population outputs become Transport inputs
            if "car_mode_share_after" in m:
                data["car_share"] = m["car_mode_share_after"]
            if "metro_mode_share_after" in m:
                data["metro_share"] = m["metro_mode_share_after"]
            if "total_ev_adopters" in m:
                data["ev_adopters_baseline"] = m["total_ev_adopters"]

        return data

    @staticmethod
    def _compute_trends(timeline: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract metric trends across time steps."""
        trends: Dict[str, list] = {}
        for step in timeline:
            for ename, edata in step.get("engines", {}).items():
                for metric, value in edata.get("metrics", {}).items():
                    if isinstance(value, (int, float)):
                        key = f"{ename}.{metric}"
                        trends.setdefault(key, []).append(value)

        # Compute direction and magnitude for each trend
        summary = {}
        for key, values in trends.items():
            if len(values) < 2:
                continue
            first, last = values[0], values[-1]
            if first == 0:
                continue
            change_pct = ((last - first) / abs(first)) * 100
            direction = "improving" if change_pct > 2 else "declining" if change_pct < -2 else "stable"
            summary[key] = {
                "values": [round(v, 2) if isinstance(v, float) else v for v in values],
                "change_pct": round(change_pct, 1),
                "direction": direction,
            }
        return summary


# ── Singleton ──
_REGISTRY: Optional[EngineRegistry] = None


def get_registry() -> EngineRegistry:
    """Get the global engine registry, auto-loading all engines on first call."""
    global _REGISTRY
    if _REGISTRY is not None:
        return _REGISTRY

    _REGISTRY = EngineRegistry()

    # Auto-discover and register all engines
    try:
        from engines.transport.engine import TransportEngine
        _REGISTRY.register(TransportEngine())
    except Exception as e:
        print(f"[EngineRegistry] ⚠ TransportEngine: {e}")

    try:
        from engines.environment.engine import EnvironmentEngine
        _REGISTRY.register(EnvironmentEngine())
    except Exception as e:
        print(f"[EngineRegistry] ⚠ EnvironmentEngine: {e}")

    try:
        from engines.infrastructure.engine import InfrastructureEngine
        _REGISTRY.register(InfrastructureEngine())
    except Exception as e:
        print(f"[EngineRegistry] ⚠ InfrastructureEngine: {e}")

    try:
        from engines.energy.engine import EnergyEngine
        _REGISTRY.register(EnergyEngine())
    except Exception as e:
        print(f"[EngineRegistry] ⚠ EnergyEngine: {e}")

    try:
        from engines.economic.engine import EconomicEngine
        _REGISTRY.register(EconomicEngine())
    except Exception as e:
        print(f"[EngineRegistry] ⚠ EconomicEngine: {e}")

    try:
        from engines.population.engine import PopulationEngine
        _REGISTRY.register(PopulationEngine())
    except Exception as e:
        print(f"[EngineRegistry] ⚠ PopulationEngine: {e}")

    try:
        from engines.logistics.engine import LogisticsEngine
        _REGISTRY.register(LogisticsEngine())
    except Exception as e:
        print(f"[EngineRegistry] ⚠ LogisticsEngine: {e}")

    # Agent-based simulation engine (swarm intelligence)
    try:
        from engines.agent_simulation.engine import AgentSimulationEngine
        _REGISTRY.register(AgentSimulationEngine())
    except Exception as e:
        print(f"[EngineRegistry] ⚠ AgentSimulationEngine: {e}")

    loaded = list(_REGISTRY._engines.keys())
    print(f"[EngineRegistry] ✓ Loaded {len(loaded)} engines: {loaded}")
    return _REGISTRY
