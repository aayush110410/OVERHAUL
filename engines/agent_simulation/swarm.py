"""
Urban Swarm Orchestrator
==========================

Runs the multi-agent timestep simulation loop:
  1. Agents choose routes (Dijkstra + memory + peer influence)
  2. Edge flows updated from agent movements
  3. BPR travel times recomputed with new flows
  4. Emissions computed per-agent
  5. Agents adapt (memory update, satisfaction, mode shift)
  6. Swarm intelligence: segment peers share congestion info
  7. Repeat for N timesteps

Wraps the OASIS simulation lifecycle when available,
falls back to pure Python loop otherwise.

Output: Aggregated metrics in the same format as TransportEngine,
ready for consumption by downstream engines.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from engines.transport.engine import (
    _DEFAULT_NODES,
    _DEFAULT_EDGES,
    _EMISSION_FACTORS,
    _bpr_time,
)
from engines.agent_simulation.agents import (
    AgentProfile,
    AgentType,
    TransportMode,
    agent_choose_route,
    agent_update_after_trip,
    share_congestion_within_segment,
    compute_agent_emissions,
    generate_commuter_agents,
    generate_freight_agents,
)
from engines.agent_simulation.graph_bridge import (
    build_ncr_knowledge_graph,
    LocalKnowledgeGraph,
    ZepKnowledgeGraph,
)
from engines.agent_simulation.config import SwarmConfig, get_agent_sim_config


@dataclass
class TimestepResult:
    """Metrics from a single simulation timestep."""
    step: int
    edge_flows: Dict[str, float] = field(default_factory=dict)
    edge_travel_times: Dict[str, float] = field(default_factory=dict)
    edge_congestion: Dict[str, float] = field(default_factory=dict)
    total_vkt: float = 0.0
    total_co2_kg: float = 0.0
    total_pm25_g: float = 0.0
    agents_arrived: int = 0
    agents_rerouted: int = 0
    mode_shifts: int = 0
    avg_speed_kmh: float = 0.0
    avg_travel_time_min: float = 0.0


@dataclass
class SwarmResult:
    """Complete simulation output from the swarm."""
    timesteps: List[TimestepResult] = field(default_factory=list)
    final_edge_flows: Dict[str, float] = field(default_factory=dict)
    final_edge_congestion: Dict[str, float] = field(default_factory=dict)
    total_vkt: float = 0.0
    total_co2_kg: float = 0.0
    total_pm25_g: float = 0.0
    avg_speed_kmh: float = 0.0
    congestion_pct: float = 0.0
    agents_total: int = 0
    agents_arrived: int = 0
    total_mode_shifts: int = 0
    hotspots: List[Dict[str, Any]] = field(default_factory=list)
    segment_breakdown: Dict[str, Any] = field(default_factory=dict)
    ev_share: float = 0.0
    runtime_seconds: float = 0.0
    graph_info: Dict[str, Any] = field(default_factory=dict)


class UrbanSwarm:
    """Multi-agent urban simulation swarm orchestrator.

    Manages agent population, knowledge graph, simulation loop,
    and result aggregation.
    """

    def __init__(
        self,
        nodes: Optional[Dict[str, Any]] = None,
        edges: Optional[List[Dict[str, Any]]] = None,
        config: Optional[SwarmConfig] = None,
    ):
        self.nodes = nodes or dict(_DEFAULT_NODES)
        # Only use one direction for edges (dedup bidirectional)
        raw_edges = edges or list(_DEFAULT_EDGES)
        self.edges = self._dedup_edges(raw_edges)
        self.config = config or get_agent_sim_config().swarm

        self.agents: List[AgentProfile] = []
        self.graph = None
        self._flow_map: Dict[str, float] = {}  # Current edge flows

    @staticmethod
    def _dedup_edges(edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keep bidirectional edges — needed for full routing."""
        return edges

    def initialize(self, zep_api_key: Optional[str] = None):
        """Set up knowledge graph and generate agent population."""
        # Build knowledge graph
        self.graph = build_ncr_knowledge_graph(
            nodes=self.nodes,
            edges=self.edges,
            zep_api_key=zep_api_key,
        )

        # Generate agent populations
        commuters = generate_commuter_agents(
            count=self.config.commuter_count,
            nodes=self.nodes,
        )
        freight = generate_freight_agents(
            count=self.config.freight_count,
            nodes=self.nodes,
        )
        self.agents = commuters + freight

        # Initialize flow map from edge capacities (baseline ~60% utilization)
        for edge in self.edges:
            eid = f"{edge['u']}->{edge['v']}"
            self._flow_map[eid] = edge["capacity"] * 0.6

    def apply_interventions(self, interventions: List[Dict[str, Any]]):
        """Apply scenario interventions to the simulation state.

        Modifies edge properties, agent behavior, and flow assumptions
        based on the user's what-if scenario.
        """
        for intv in interventions:
            name = intv.get("name", "")
            params = intv.get("parameters", {})

            if name == "congestion_pricing":
                # Reduce demand across all edges
                reduction = params.get("demand_reduction_pct", 12) / 100.0
                for eid in self._flow_map:
                    self._flow_map[eid] *= (1.0 - reduction)

            elif name == "metro_expansion":
                # Shift some car agents to metro
                shift_pct = params.get("mode_shift_pct", 8) / 100.0
                for agent in self.agents:
                    if (
                        agent.agent_type == AgentType.COMMUTER
                        and agent.mode == TransportMode.CAR
                        and agent.flexibility > 0.3
                    ):
                        import random
                        if random.random() < shift_pct * agent.flexibility:
                            agent.mode = TransportMode.METRO

            elif name == "signal_optimization":
                # Increase free speed on all edges
                speed_gain = params.get("speed_gain_pct", 10) / 100.0
                for edge in self.edges:
                    edge["free_speed"] *= (1.0 + speed_gain)

            elif name == "bus_rapid_transit":
                # Reduce car capacity, shift some agents to bus
                car_cap_reduction = params.get("car_capacity_reduction_pct", 15) / 100.0
                for edge in self.edges:
                    edge["capacity"] *= (1.0 - car_cap_reduction)

            elif name == "ev_fleet_expansion":
                # Increase EV readiness of agents
                ev_increase = params.get("ev_share_increase", 0.10)
                for agent in self.agents:
                    agent.ev_readiness = min(1.0, agent.ev_readiness + ev_increase)

            elif name == "road_capacity_expansion":
                # Increase capacity on all edges
                cap_increase = params.get("capacity_increase_pct", 20) / 100.0
                for edge in self.edges:
                    edge["capacity"] *= (1.0 + cap_increase)

    async def run(self) -> SwarmResult:
        """Execute the multi-agent simulation loop.

        For each timestep:
          1. Active agents (departed, not arrived) choose routes
          2. Edge flows updated from agent routes
          3. BPR travel times computed
          4. Emissions calculated per agent
          5. Agents update memory and adapt
          6. Swarm intelligence: peers share info
        """
        t0 = time.perf_counter()
        result = SwarmResult(agents_total=len(self.agents))
        config_weights = {
            "congestion_memory_weight": self.config.congestion_memory_weight,
            "peer_influence_weight": self.config.peer_influence_weight,
        }

        total_mode_shifts = 0

        for step in range(self.config.timesteps):
            step_result = await self._run_timestep(
                step=step,
                config_weights=config_weights,
            )
            result.timesteps.append(step_result)
            total_mode_shifts += step_result.mode_shifts

            # Swarm intelligence: share congestion info within segments
            if self.config.enable_swarm_intelligence:
                commuters = [a for a in self.agents if a.agent_type == AgentType.COMMUTER]
                share_congestion_within_segment(
                    commuters, self.config.peer_influence_weight
                )

            # Update knowledge graph with current congestion
            if self.graph:
                for eid, congestion in step_result.edge_congestion.items():
                    road_entity_id = eid.replace("->", "__")
                    self.graph.update_entity_property(
                        f"road_{road_entity_id}", "current_congestion", congestion
                    )

        # Aggregate final results
        last_step = result.timesteps[-1] if result.timesteps else TimestepResult(step=0)
        result.final_edge_flows = last_step.edge_flows
        result.final_edge_congestion = last_step.edge_congestion
        result.total_vkt = sum(s.total_vkt for s in result.timesteps)
        result.total_co2_kg = sum(s.total_co2_kg for s in result.timesteps)
        result.total_pm25_g = sum(s.total_pm25_g for s in result.timesteps)
        # Average speed across all timesteps that had arrivals
        speeds_all = [s.avg_speed_kmh for s in result.timesteps if s.avg_speed_kmh > 0]
        result.avg_speed_kmh = round(sum(speeds_all) / max(len(speeds_all), 1), 1) if speeds_all else 0.0
        result.agents_arrived = sum(s.agents_arrived for s in result.timesteps)
        result.total_mode_shifts = total_mode_shifts

        # Congestion percentage
        congested_edges = sum(
            1 for c in last_step.edge_congestion.values() if c > 0.8
        )
        total_edges = max(len(last_step.edge_congestion), 1)
        result.congestion_pct = round(congested_edges / total_edges * 100, 1)

        # Hotspot detection (top 5 most congested edges)
        sorted_congestion = sorted(
            last_step.edge_congestion.items(), key=lambda x: -x[1]
        )
        result.hotspots = [
            {"edge": eid, "congestion_ratio": round(c, 3)}
            for eid, c in sorted_congestion[:5]
        ]

        # Segment breakdown
        result.segment_breakdown = self._compute_segment_breakdown()

        # EV share
        ev_agents = sum(
            1 for a in self.agents
            if a.ev_readiness > 0.5 and a.mode == TransportMode.CAR
        )
        car_agents = sum(1 for a in self.agents if a.mode == TransportMode.CAR)
        result.ev_share = round(ev_agents / max(car_agents, 1), 3)

        # Graph info
        if self.graph:
            result.graph_info = self.graph.to_dict()

        result.runtime_seconds = time.perf_counter() - t0
        return result

    async def _run_timestep(
        self,
        step: int,
        config_weights: Dict[str, float],
    ) -> TimestepResult:
        """Run a single simulation timestep."""
        ts = TimestepResult(step=step)
        step_time_min = step * self.config.step_duration_minutes

        # Reset edge flows for this timestep
        step_flows: Dict[str, float] = {eid: 0.0 for eid in self._flow_map}
        travel_times = []
        speeds = []

        for agent in self.agents:
            if agent.arrived:
                continue

            # Check if agent has departed yet
            if agent.departure_offset_min > step_time_min:
                continue

            # Non-road agents don't participate in routing
            if agent.mode in (TransportMode.METRO, TransportMode.WFH, TransportMode.CYCLE):
                agent.arrived = True
                ts.agents_arrived += 1
                continue

            # Choose route
            route, travel_time = agent_choose_route(
                agent=agent,
                nodes=self.nodes,
                edges=self.edges,
                flow_map=self._flow_map,
                config_weights=config_weights,
            )

            if not route:
                continue

            # Add agent to edge flows
            for edge in route:
                eid = f"{edge['u']}->{edge['v']}"
                step_flows[eid] = step_flows.get(eid, 0) + 1

            # Compute emissions
            emissions = compute_agent_emissions(agent, route)
            ts.total_co2_kg += emissions["co2_kg"]
            ts.total_pm25_g += emissions["pm25_g"]

            # Compute VKT
            trip_km = sum(e["dist_km"] for e in route)
            ts.total_vkt += trip_km

            travel_times.append(travel_time)
            if travel_time > 0:
                speeds.append(trip_km / (travel_time / 60))  # km/h

            # Agent's trip state
            old_mode = agent.mode
            agent.travel_time_current = travel_time
            agent.current_route = route
            agent.arrived = True
            ts.agents_arrived += 1

            # Compute per-edge congestion for this agent's route
            edge_congestion = {}
            for edge in route:
                eid = f"{edge['u']}->{edge['v']}"
                flow = self._flow_map.get(eid, 0) + step_flows.get(eid, 0)
                edge_congestion[eid] = flow / max(edge["capacity"], 1)

            # Update agent memory and check for adaptation
            agent_update_after_trip(
                agent=agent,
                route=route,
                travel_time=travel_time,
                edge_congestion=edge_congestion,
                adaptation_threshold=self.config.adaptation_threshold,
                mode_shift_threshold=self.config.mode_shift_threshold,
            )

            if agent.mode != old_mode:
                ts.mode_shifts += 1

        # Update global flow map with this timestep's flows
        # Each simulated agent represents ~commuter_population/agent_count real commuters
        # Spread across all edges, so per-edge scale uses total_edges as denominator
        scale_factor = 7_000_000 / max(len(self.agents), 1)
        total_edges = max(len(self.edges), 1)
        for eid, flow in step_flows.items():
            # Scale: agent flow → real-world flow for this edge
            # Divide by total_edges to avoid concentrating all scaled flow
            real_flow = flow * scale_factor / total_edges
            old_flow = self._flow_map.get(eid, 0)
            # Blend: 80% existing (preserves baseline), 20% agent-observed
            self._flow_map[eid] = old_flow * 0.8 + real_flow * 0.2

        # Compute edge-level metrics
        for edge in self.edges:
            eid = f"{edge['u']}->{edge['v']}"
            flow = self._flow_map.get(eid, edge["capacity"] * 0.6)
            tt = _bpr_time(edge["dist_km"], edge["free_speed"], edge["capacity"], flow)
            congestion = flow / max(edge["capacity"], 1)

            ts.edge_flows[eid] = round(flow)
            ts.edge_travel_times[eid] = round(tt, 2)
            ts.edge_congestion[eid] = round(congestion, 3)

        ts.avg_travel_time_min = round(sum(travel_times) / max(len(travel_times), 1), 2)
        ts.avg_speed_kmh = round(sum(speeds) / max(len(speeds), 1), 1) if speeds else 0.0

        return ts

    def _compute_segment_breakdown(self) -> Dict[str, Any]:
        """Compute per-segment metrics for population analysis."""
        segments: Dict[str, Dict] = {}

        for agent in self.agents:
            if agent.agent_type != AgentType.COMMUTER:
                continue
            seg = agent.segment
            if seg not in segments:
                segments[seg] = {
                    "count": 0,
                    "mode_shifts": 0,
                    "car_count": 0,
                    "metro_count": 0,
                    "bus_count": 0,
                    "avg_satisfaction": 0.0,
                    "avg_travel_time": 0.0,
                }

            segments[seg]["count"] += 1
            segments[seg]["avg_satisfaction"] += agent.memory.satisfaction_score
            segments[seg]["avg_travel_time"] += agent.memory.avg_travel_time
            segments[seg]["mode_shifts"] += len(agent.memory.mode_shift_events)

            if agent.mode == TransportMode.CAR:
                segments[seg]["car_count"] += 1
            elif agent.mode == TransportMode.METRO:
                segments[seg]["metro_count"] += 1
            elif agent.mode == TransportMode.BUS:
                segments[seg]["bus_count"] += 1

        # Average out
        for seg, data in segments.items():
            n = max(data["count"], 1)
            data["avg_satisfaction"] = round(data["avg_satisfaction"] / n, 3)
            data["avg_travel_time"] = round(data["avg_travel_time"] / n, 2)
            data["car_mode_share"] = round(data["car_count"] / n, 3)
            data["metro_mode_share"] = round(data["metro_count"] / n, 3)

        return segments

    async def interview_agent(self, agent_id: int, question: str) -> Dict[str, Any]:
        """Interview a specific agent about its decisions.

        Uses the IPC system for mid-simulation queries.
        Returns agent's state, memory, and reasoning.
        """
        agent = next((a for a in self.agents if a.agent_id == agent_id), None)
        if not agent:
            return {"error": f"Agent {agent_id} not found"}

        return {
            "agent_id": agent_id,
            "type": agent.agent_type.value,
            "segment": agent.segment,
            "mode": agent.mode.value,
            "origin": agent.origin,
            "destination": agent.destination,
            "arrived": agent.arrived,
            "travel_time": agent.travel_time_current,
            "satisfaction": agent.memory.satisfaction_score,
            "known_congested_edges": len(agent.memory.congestion_memory),
            "mode_shift_events": agent.memory.mode_shift_events,
            "total_trips": agent.memory.total_trips,
            "question": question,
            "reasoning": self._generate_agent_reasoning(agent, question),
        }

    @staticmethod
    def _generate_agent_reasoning(agent: AgentProfile, question: str) -> str:
        """Generate rule-based reasoning for agent interview response."""
        lines = []
        lines.append(f"I am a {agent.segment} commuter from {agent.origin} to {agent.destination}.")
        lines.append(f"I currently use {agent.mode.value} for my commute.")

        if agent.memory.satisfaction_score < 0.5:
            lines.append("I am dissatisfied with my current commute.")
        elif agent.memory.satisfaction_score > 0.8:
            lines.append("My commute experience has been good recently.")

        if agent.memory.mode_shift_events:
            last_shift = agent.memory.mode_shift_events[-1]
            lines.append(
                f"I recently switched from {last_shift['from_mode']} due to "
                f"{last_shift['trigger']} (congestion: {last_shift['avg_congestion']:.0%})."
            )

        if agent.memory.congestion_memory:
            worst_edges = sorted(
                agent.memory.congestion_memory.items(), key=lambda x: -x[1]
            )[:3]
            congested = [f"{eid} ({c:.0%})" for eid, c in worst_edges]
            lines.append(f"I know these routes are congested: {', '.join(congested)}.")

        return " ".join(lines)
