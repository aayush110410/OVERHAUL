"""
Agent Type Definitions
========================

Four agent types for urban simulation, each using the full agent lifecycle
(creation → profile generation → simulation → memory persistence → interview).

Agent types:
  CommuterAgent     — Individual commuter with route choice, mode preference, adaptation
  FreightAgent      — Commercial vehicle with load optimization, off-peak preference
  PolicyAgent       — LLM-enhanced policy evaluator (only agent type using LLM)
  EnvironmentAgent  — Tracks cumulative emissions and AQI per timestep

Decision logic is rule-based using OVERHAUL's existing physics models
(BPR travel time, Dijkstra routing, emission factors, population segment curves).
"""

from __future__ import annotations

import math
import random
import heapq
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# Reuse OVERHAUL's existing transport physics
from engines.transport.engine import (
    _BPR_ALPHA,
    _BPR_BETA,
    _DEFAULT_NODES,
    _DEFAULT_EDGES,
    _EMISSION_FACTORS,
    _bpr_time,
)

# Reuse population segment definitions
from engines.population.engine import _SEGMENTS


class AgentType(str, Enum):
    COMMUTER = "commuter"
    FREIGHT = "freight"
    POLICY = "policy"
    ENVIRONMENT = "environment"


class TransportMode(str, Enum):
    CAR = "car"
    METRO = "metro"
    BUS = "bus"
    TWO_WHEELER = "two_wheeler"
    AUTO = "auto"
    CYCLE = "cycle"
    WFH = "wfh"


@dataclass
class AgentMemory:
    """Persistent memory for an agent — mirrors Zep memory structure."""
    route_history: List[Dict[str, Any]] = field(default_factory=list)
    congestion_memory: Dict[str, float] = field(default_factory=dict)  # edge_id → avg congestion
    mode_shift_events: List[Dict[str, Any]] = field(default_factory=list)
    peer_shared_info: Dict[str, Any] = field(default_factory=dict)
    total_trips: int = 0
    avg_travel_time: float = 0.0
    satisfaction_score: float = 0.7  # 0-1, affects adaptation willingness

    def update_congestion(self, edge_id: str, congestion_ratio: float, weight: float = 0.3):
        """Update congestion memory with exponential moving average."""
        current = self.congestion_memory.get(edge_id, congestion_ratio)
        self.congestion_memory[edge_id] = current * (1 - weight) + congestion_ratio * weight

    def record_trip(self, travel_time: float, route_edges: List[str]):
        """Record a completed trip in memory."""
        self.total_trips += 1
        self.avg_travel_time = (
            (self.avg_travel_time * (self.total_trips - 1) + travel_time)
            / self.total_trips
        )
        self.route_history.append({
            "edges": route_edges,
            "travel_time": travel_time,
            "trip_number": self.total_trips,
        })
        # Keep last 20 trips
        if len(self.route_history) > 20:
            self.route_history = self.route_history[-20:]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "route_history_count": len(self.route_history),
            "known_congested_edges": len(self.congestion_memory),
            "mode_shift_events": len(self.mode_shift_events),
            "total_trips": self.total_trips,
            "avg_travel_time": round(self.avg_travel_time, 2),
            "satisfaction_score": round(self.satisfaction_score, 3),
        }


@dataclass
class AgentProfile:
    """Full agent profile — generated from population segments."""
    agent_id: int
    agent_type: AgentType
    segment: str                          # Population segment name
    origin: str                           # Node ID
    destination: str                      # Node ID
    mode: TransportMode
    departure_offset_min: float = 0.0     # Minutes after rush hour start
    flexibility: float = 0.4             # 0-1, willingness to adapt
    ev_readiness: float = 0.1
    income_monthly: int = 30_000
    commute_km: float = 15.0
    wfh_capable: bool = False
    memory: AgentMemory = field(default_factory=AgentMemory)
    # Current state
    current_node: str = ""
    current_route: List[Dict[str, Any]] = field(default_factory=list)
    travel_time_current: float = 0.0
    arrived: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "type": self.agent_type.value,
            "segment": self.segment,
            "origin": self.origin,
            "destination": self.destination,
            "mode": self.mode.value,
            "flexibility": self.flexibility,
            "memory": self.memory.to_dict(),
            "arrived": self.arrived,
        }


# ── Agent Factory ──

def generate_commuter_agents(
    count: int,
    nodes: Dict[str, Any],
    segments: Optional[Dict[str, Any]] = None,
    seed: int = 42,
) -> List[AgentProfile]:
    """Generate commuter agent population from population segments.

    Each agent is assigned a segment proportional to segment share,
    then given origin/destination from available nodes, mode from
    segment mode_split, and behavioral parameters.
    """
    rng = random.Random(seed)
    segments = segments or _SEGMENTS
    node_ids = list(nodes.keys())
    agents = []

    # Build segment distribution
    seg_names = list(segments.keys())
    seg_weights = [segments[s]["share"] for s in seg_names]

    for i in range(count):
        # Pick segment proportional to share
        seg_name = rng.choices(seg_names, weights=seg_weights, k=1)[0]
        seg = segments[seg_name]

        # Pick origin and destination (different nodes)
        origin = rng.choice(node_ids)
        destination = rng.choice([n for n in node_ids if n != origin])

        # Pick mode from segment mode_split
        modes = list(seg["mode_split"].keys())
        mode_weights = list(seg["mode_split"].values())
        mode_str = rng.choices(modes, weights=mode_weights, k=1)[0]
        # Map to TransportMode enum
        mode_map = {
            "car": TransportMode.CAR,
            "metro": TransportMode.METRO,
            "bus": TransportMode.BUS,
            "two_wheeler": TransportMode.TWO_WHEELER,
            "auto": TransportMode.AUTO,
            "cycle": TransportMode.CYCLE,
            "cab": TransportMode.CAR,
            "company_bus": TransportMode.BUS,
            "other": TransportMode.BUS,
        }
        mode = mode_map.get(mode_str, TransportMode.CAR)

        # Departure offset — spread across rush hour (0-90 min)
        departure = rng.gauss(45, 20)
        departure = max(0, min(90, departure))

        agents.append(AgentProfile(
            agent_id=i,
            agent_type=AgentType.COMMUTER,
            segment=seg_name,
            origin=origin,
            destination=destination,
            mode=mode,
            departure_offset_min=round(departure, 1),
            flexibility=seg["flexibility"] * rng.uniform(0.7, 1.3),  # Individual variation
            ev_readiness=seg["ev_readiness"] * rng.uniform(0.5, 1.5),
            income_monthly=seg["avg_income_monthly"],
            commute_km=seg["avg_commute_km"] * rng.uniform(0.6, 1.4),
            wfh_capable=rng.random() < seg["wfh_capable"],
            current_node=origin,
        ))

    return agents


def generate_freight_agents(
    count: int,
    nodes: Dict[str, Any],
    seed: int = 43,
) -> List[AgentProfile]:
    """Generate freight agent population."""
    rng = random.Random(seed)
    node_ids = list(nodes.keys())
    agents = []

    for i in range(count):
        origin = rng.choice(node_ids)
        destination = rng.choice([n for n in node_ids if n != origin])

        # Freight agents prefer off-peak but some forced to peak
        departure = rng.gauss(30, 40)  # More spread than commuters
        departure = max(-60, min(120, departure))  # Can depart before rush

        agents.append(AgentProfile(
            agent_id=10000 + i,  # Offset to avoid ID collision
            agent_type=AgentType.FREIGHT,
            segment="freight",
            origin=origin,
            destination=destination,
            mode=TransportMode.CAR,  # Trucks use road network
            departure_offset_min=round(departure, 1),
            flexibility=0.25,  # Freight less flexible than commuters
            commute_km=38.0 * rng.uniform(0.5, 1.5),  # NCR avg freight trip
            current_node=origin,
        ))

    return agents


# ── Agent Decision Logic ──

def agent_choose_route(
    agent: AgentProfile,
    nodes: Dict[str, Any],
    edges: List[Dict[str, Any]],
    flow_map: Dict[str, float],
    config_weights: Optional[Dict[str, float]] = None,
) -> Tuple[List[Dict[str, Any]], float]:
    """Agent chooses optimal route using Dijkstra + congestion memory + peer info.

    This is the core decision function. Uses OVERHAUL's existing BPR travel
    time model and Dijkstra shortest path, enhanced with agent memory.

    Returns: (route_edges, travel_time_minutes)
    """
    weights = config_weights or {}
    memory_weight = weights.get("congestion_memory_weight", 0.3)
    peer_weight = weights.get("peer_influence_weight", 0.15)

    origin = agent.origin if not agent.current_node else agent.current_node
    destination = agent.destination

    if origin == destination:
        return [], 0.0

    # Build adjacency with BPR + memory-adjusted weights
    adj: Dict[str, List] = {n: [] for n in nodes}
    for e in edges:
        eid = f"{e['u']}->{e['v']}"
        base_flow = flow_map.get(eid, e["capacity"] * 0.6)

        # Standard BPR travel time
        tt = _bpr_time(e["dist_km"], e["free_speed"], e["capacity"], base_flow)

        # Memory adjustment: if agent remembers this edge as congested, penalize
        if eid in agent.memory.congestion_memory:
            remembered_congestion = agent.memory.congestion_memory[eid]
            # Higher remembered congestion → higher travel time penalty
            memory_penalty = 1.0 + memory_weight * max(0, remembered_congestion - 0.7)
            tt *= memory_penalty

        # Peer influence: if segment peers reported congestion here
        if eid in agent.memory.peer_shared_info:
            peer_congestion = agent.memory.peer_shared_info[eid]
            tt *= (1.0 + peer_weight * max(0, peer_congestion - 0.6))

        adj[e["u"]].append((e["v"], tt, e))

    # Dijkstra shortest path
    dist = {n: float("inf") for n in nodes}
    prev: Dict[str, Any] = {n: None for n in nodes}
    dist[origin] = 0.0
    pq = [(0.0, origin)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        if u == destination:
            break
        for v, w, edge in adj.get(u, []):
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = (u, edge)
                heapq.heappush(pq, (nd, v))

    # Reconstruct path
    if dist[destination] == float("inf"):
        return [], float("inf")

    path = []
    node = destination
    while prev[node]:
        u, edge = prev[node]
        path.append(edge)
        node = u
    path.reverse()

    return path, round(dist[destination], 2)


def agent_update_after_trip(
    agent: AgentProfile,
    route: List[Dict[str, Any]],
    travel_time: float,
    edge_congestion: Dict[str, float],
    adaptation_threshold: float = 0.2,
    mode_shift_threshold: float = 0.4,
):
    """Update agent state after completing a trip.

    Updates memory, checks if agent should switch routes or modes.
    This is where emergent behavior comes from — agents learn and adapt.
    """
    edge_ids = [f"{e['u']}->{e['v']}" for e in route]

    # Update congestion memory for all traversed edges
    for eid in edge_ids:
        congestion = edge_congestion.get(eid, 0.5)
        agent.memory.update_congestion(eid, congestion)

    # Record trip
    agent.memory.record_trip(travel_time, edge_ids)

    # Update satisfaction based on travel time vs expectation
    if agent.memory.avg_travel_time > 0:
        time_ratio = travel_time / agent.memory.avg_travel_time
        if time_ratio > 1.3:  # 30% worse than average
            agent.memory.satisfaction_score = max(0.1, agent.memory.satisfaction_score - 0.05)
        elif time_ratio < 0.9:  # 10% better than average
            agent.memory.satisfaction_score = min(1.0, agent.memory.satisfaction_score + 0.02)

    # Check for mode shift trigger
    avg_congestion = sum(edge_congestion.get(eid, 0.5) for eid in edge_ids) / max(len(edge_ids), 1)
    if (
        avg_congestion > mode_shift_threshold
        and agent.flexibility > 0.3
        and agent.memory.satisfaction_score < 0.5
        and agent.mode == TransportMode.CAR
    ):
        # Dissatisfied car driver in high congestion → consider metro/bus
        agent.memory.mode_shift_events.append({
            "from_mode": agent.mode.value,
            "trigger": "high_congestion_low_satisfaction",
            "avg_congestion": round(avg_congestion, 3),
            "satisfaction": round(agent.memory.satisfaction_score, 3),
        })
        # Actually shift mode based on flexibility
        if random.random() < agent.flexibility:
            agent.mode = TransportMode.METRO


def share_congestion_within_segment(
    agents: List[AgentProfile],
    peer_influence_weight: float = 0.15,
):
    """Swarm intelligence: agents in the same segment share congestion info.

    This is the emergent behavior mechanism — office workers in Noida
    collectively discover that a route is congested and shift together.
    """
    # Group agents by segment
    by_segment: Dict[str, List[AgentProfile]] = {}
    for agent in agents:
        by_segment.setdefault(agent.segment, []).append(agent)

    for seg_name, seg_agents in by_segment.items():
        if len(seg_agents) < 2:
            continue

        # Aggregate congestion knowledge across segment
        combined_congestion: Dict[str, List[float]] = {}
        for agent in seg_agents:
            for eid, congestion in agent.memory.congestion_memory.items():
                combined_congestion.setdefault(eid, []).append(congestion)

        # Share averaged congestion back to all agents in segment
        shared_info: Dict[str, float] = {}
        for eid, values in combined_congestion.items():
            if len(values) >= 2:  # Need multiple data points
                shared_info[eid] = sum(values) / len(values)

        for agent in seg_agents:
            agent.memory.peer_shared_info = shared_info


# ── Emission Computation Per Agent ──

def compute_agent_emissions(
    agent: AgentProfile,
    route: List[Dict[str, Any]],
    ev_share: float = 0.05,
) -> Dict[str, float]:
    """Compute emissions for a single agent's trip using OVERHAUL emission factors."""
    total_km = sum(e["dist_km"] for e in route)

    if agent.mode in (TransportMode.METRO, TransportMode.BUS, TransportMode.CYCLE):
        # Public transit / active transport — minimal per-passenger emissions
        return {"co2_kg": total_km * 0.025, "pm25_g": total_km * 0.001}

    # Check if this agent is an EV user
    is_ev = random.random() < agent.ev_readiness
    if is_ev:
        co2_factor = _EMISSION_FACTORS["co2_g_per_vkm"] * _EMISSION_FACTORS["ev_co2_factor"]
        pm25_factor = _EMISSION_FACTORS["pm25_g_per_vkm"] * _EMISSION_FACTORS["ev_pm25_factor"]
    else:
        co2_factor = _EMISSION_FACTORS["co2_g_per_vkm"]
        pm25_factor = _EMISSION_FACTORS["pm25_g_per_vkm"]

    # Freight vehicles have higher emissions
    if agent.agent_type == AgentType.FREIGHT:
        co2_factor *= 3.8  # Trucks emit ~3.8x more than cars
        pm25_factor *= 4.0

    return {
        "co2_kg": total_km * co2_factor / 1000,
        "pm25_g": total_km * pm25_factor,
    }
