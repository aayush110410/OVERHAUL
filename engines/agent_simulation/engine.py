"""
Agent-Based Simulation Engine
===============================

SimulationEngine implementation that runs a multi-agent swarm simulation
on the NCR road network. Produces output in the EXACT same format as
TransportEngine so all downstream engines work without changes.

This engine:
  1. Generates agent population from population segments
  2. Builds knowledge graph from NCR road network
  3. Applies scenario interventions to agent environment
  4. Runs N-timestep swarm simulation
  5. Calibrates results against ML model predictions
  6. Returns standard SimulationResult

Downstream engines (Environment, Economic, Energy, Population, Logistics)
consume this output identically to TransportEngine output.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from engines.base import (
    EngineCapability,
    Scenario,
    SimulationEngine,
    SimulationResult,
)
from engines.agent_simulation.swarm import UrbanSwarm, SwarmResult
from engines.agent_simulation.config import get_agent_sim_config


class AgentSimulationEngine(SimulationEngine):
    """Multi-agent urban simulation engine.

    Runs thousands of individual commuter and freight agents on the
    NCR road graph. Agents make route decisions, adapt to congestion,
    share information within segments (swarm intelligence), and produce
    emergent traffic patterns.

    Output format is identical to TransportEngine for seamless integration
    with all downstream engines.
    """

    @property
    def name(self) -> str:
        return "AgentSimulationEngine"

    @property
    def capabilities(self) -> List[EngineCapability]:
        return [EngineCapability.TRANSPORT]

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def required_data(self) -> List[str]:
        return []  # Has built-in NCR network

    @property
    def optional_data(self) -> List[str]:
        return [
            "nodes", "edges", "flow_map", "ev_share",
            "agent_count", "timesteps", "use_persistent_memory",
        ]

    def estimate_missing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate missing data using NCR defaults and ML baselines."""
        if "ev_share" not in data:
            ev_count = data.get("ev_count", 0)
            total_vehicles = 12_000_000
            data["ev_share"] = max(ev_count / total_vehicles, 0.05) if ev_count else 0.05

        # Load ML model baselines for calibration
        if "baseline_speed_kmh" not in data:
            baselines = self._load_baselines()
            city = data.get("city", "delhi").lower()
            city_data = baselines.get(city.capitalize(), baselines.get("Delhi", {}))
            traffic = city_data.get("traffic", {})
            data["baseline_speed_kmh"] = traffic.get("avg_speed_kmph", 40.0)
            data["baseline_congestion_pct"] = traffic.get("congestion_pct_mean", 25.0)

        return data

    async def _run(self, scenario: Scenario, data: Dict[str, Any]) -> SimulationResult:
        """Run multi-agent simulation and return standard SimulationResult."""
        config = get_agent_sim_config()

        # Override config from data if provided
        swarm_config = config.swarm
        if "agent_count" in data:
            swarm_config.commuter_count = int(data["agent_count"])
        if "timesteps" in data:
            swarm_config.timesteps = int(data["timesteps"])

        # Build and initialize swarm
        swarm = UrbanSwarm(
            nodes=data.get("nodes"),
            edges=data.get("edges"),
            config=swarm_config,
        )
        swarm.initialize(zep_api_key=config.zep.api_key if config.zep.enabled else None)

        # Apply scenario interventions
        interventions = [
            {"name": intv.name, "parameters": intv.parameters}
            for intv in scenario.interventions
        ]
        swarm.apply_interventions(interventions)

        # Run simulation
        swarm_result = await swarm.run()

        # Calibrate against baselines
        swarm_result = self._calibrate(swarm_result, data)

        # Convert to standard SimulationResult
        return self._to_simulation_result(scenario, swarm_result, data)

    def _calibrate(self, result: SwarmResult, data: Dict[str, Any]) -> SwarmResult:
        """Calibrate agent sim results against ML model predictions.

        If deviation exceeds tolerance, adjust metrics to stay grounded
        in real-world data while preserving relative patterns from agents.
        """
        baseline_speed = data.get("baseline_speed_kmh", 40.0)
        baseline_congestion = data.get("baseline_congestion_pct", 25.0)
        tolerance = get_agent_sim_config().swarm.calibration_tolerance

        # Speed calibration
        if result.avg_speed_kmh > 0:
            speed_deviation = abs(result.avg_speed_kmh - baseline_speed) / baseline_speed
            if speed_deviation > tolerance:
                # Blend agent result with baseline
                blend = 0.6  # 60% agent, 40% baseline
                result.avg_speed_kmh = round(
                    result.avg_speed_kmh * blend + baseline_speed * (1 - blend), 1
                )

        return result

    def _to_simulation_result(
        self,
        scenario: Scenario,
        swarm: SwarmResult,
        data: Dict[str, Any],
    ) -> SimulationResult:
        """Convert SwarmResult to standard SimulationResult.

        Output format matches TransportEngine EXACTLY so downstream
        engines (Environment, Economic, etc.) work without changes.
        """
        ev_share = data.get("ev_share", 0.05)

        # Compute CO2 in tonnes (matching TransportEngine format)
        co2_tonnes = round(swarm.total_co2_kg / 1000, 2)

        # Edge-level results for metadata
        top_congested = sorted(
            [
                {
                    "edge": h["edge"],
                    "v_over_c": h["congestion_ratio"],
                    "avg_speed_kmh": round(
                        swarm.avg_speed_kmh * (1.0 - h["congestion_ratio"] * 0.5), 1
                    ),
                }
                for h in swarm.hotspots
            ],
            key=lambda x: -x["v_over_c"],
        )[:5]

        metrics = {
            # Core transport metrics (same keys as TransportEngine)
            "avg_speed_kmh": round(swarm.avg_speed_kmh, 1),
            "total_vkt": round(swarm.total_vkt),
            "co2_tonnes": co2_tonnes,
            "pm25_kg": round(swarm.total_pm25_g / 1000, 2),
            "congestion_pct": swarm.congestion_pct,
            "ev_share": round(swarm.ev_share if swarm.ev_share > 0 else ev_share, 3),
            "edge_count": len(swarm.final_edge_flows) // 2,
            "node_count": len(self._get_nodes(data)),
            # Agent-specific metrics (bonus data for LDRAGO)
            "simulation_mode": "agent_based",
            "agents_simulated": swarm.agents_total,
            "agents_arrived": swarm.agents_arrived,
            "mode_shifts_observed": swarm.total_mode_shifts,
            "timesteps_completed": len(swarm.timesteps),
            "hotspots_detected": len(swarm.hotspots),
            "runtime_seconds": round(swarm.runtime_seconds, 2),
        }

        # Impact deltas (same keys as TransportEngine)
        baseline_co2 = swarm.total_vkt * 170.0 / 1e6  # Baseline all-ICE
        baseline_speed = float(data.get("baseline_speed_kmh", 40.0))
        impacts = {
            "co2_reduction_pct": float(round(
                (1.0 - co2_tonnes / max(baseline_co2, 0.001)) * 100, 1
            )),
            "speed_change_pct": float(round(
                (swarm.avg_speed_kmh - baseline_speed)
                / max(baseline_speed, 1) * 100, 1
            )),
            "congestion_change_pct": float(round(-swarm.congestion_pct, 1)),
        }

        # Recommendations based on agent behavior
        recommendations = []
        if swarm.congestion_pct > 50:
            recommendations.append(
                f"High congestion ({swarm.congestion_pct:.0f}%) — "
                "agents are discovering bottlenecks. Consider congestion pricing or metro expansion."
            )
        if swarm.total_mode_shifts > swarm.agents_total * 0.05:
            recommendations.append(
                f"{swarm.total_mode_shifts} agents shifted transport modes — "
                "emergent mode shift detected. Ensure metro/bus capacity can absorb demand."
            )
        if swarm.hotspots:
            worst = swarm.hotspots[0]
            recommendations.append(
                f"Traffic hotspot at {worst['edge']} "
                f"(congestion: {worst['congestion_ratio']:.0%}) — "
                "agents are avoiding this corridor, causing cascade effects."
            )
        if swarm.ev_share < 0.15:
            recommendations.append(
                f"EV adoption at {swarm.ev_share*100:.0f}% among agents — "
                "accelerating EV fleet would significantly reduce PM2.5."
            )

        # Sanitize any framework references from metadata
        metadata = {
            "top_congested": top_congested,
            "segment_breakdown": swarm.segment_breakdown,
            "graph_entities": swarm.graph_info.get("entity_count", 0),
            "emergent_patterns": {
                "mode_shifts": swarm.total_mode_shifts,
                "hotspots": len(swarm.hotspots),
                "swarm_learning": swarm.graph_info.get("persistent_memory", False),
            },
        }

        return SimulationResult(
            engine=self.name,
            domain=EngineCapability.TRANSPORT,
            scenario_name=scenario.name,
            metrics=metrics,
            impacts=impacts,
            recommendations=recommendations,
            confidence=0.70,  # Agent sim has higher confidence than pure BPR
            metadata=metadata,
        )

    @staticmethod
    def _get_nodes(data: Dict[str, Any]) -> Dict[str, Any]:
        from engines.transport.engine import _DEFAULT_NODES
        return data.get("nodes", _DEFAULT_NODES)

    @staticmethod
    def _load_baselines() -> Dict[str, Any]:
        """Load trained model baselines for calibration."""
        path = os.path.join(
            os.path.dirname(__file__), "..", "..", "models", "ncr_baselines.json"
        )
        try:
            with open(os.path.normpath(path)) as f:
                return json.load(f)
        except Exception:
            return {
                "Delhi": {"traffic": {"avg_speed_kmph": 40.4, "congestion_pct_mean": 23.1}},
            }
