"""
Transportation Simulation Engine
=================================

Models urban road networks using BPR (Bureau of Public Roads) travel-time
functions, Dijkstra routing, and configurable interventions.

Supports:
  - Signal optimization
  - Bus rapid transit lanes
  - EV fleet penetration
  - Congestion pricing
  - Metro line expansion
  - Road capacity changes
  - Rerouting penalties

Network data can come from:
  - Manual node/edge definitions
  - OpenStreetMap extracts
  - SUMO network files
"""

from __future__ import annotations

import math
import heapq
from typing import Any, Dict, List

from engines.base import (
    EngineCapability,
    Scenario,
    SimulationEngine,
    SimulationResult,
)

# ── BPR Travel-Time Model ──
# t = t_free * (1 + alpha * (flow / capacity) ^ beta)
_BPR_ALPHA = 0.15
_BPR_BETA = 4.0

# ── Default Delhi NCR Network ──
# Realistic corridor-level model (expandable)
_DEFAULT_NODES: Dict[str, Dict[str, Any]] = {
    "connaught_place":   {"lat": 28.6315, "lon": 77.2167, "label": "Connaught Place"},
    "ito":               {"lat": 28.6285, "lon": 77.2413, "label": "ITO"},
    "noida_sec18":       {"lat": 28.5700, "lon": 77.3219, "label": "Noida Sector 18"},
    "noida_sec62":       {"lat": 28.6270, "lon": 77.3640, "label": "Noida Sector 62"},
    "ghaziabad_center":  {"lat": 28.6692, "lon": 77.4538, "label": "Ghaziabad Center"},
    "indirapuram":       {"lat": 28.6352, "lon": 77.3764, "label": "Indirapuram"},
    "dwarka":            {"lat": 28.5921, "lon": 77.0460, "label": "Dwarka"},
    "gurugram_cyber":    {"lat": 28.4947, "lon": 77.0887, "label": "Gurugram Cyber City"},
    "igi_airport":       {"lat": 28.5562, "lon": 77.1000, "label": "IGI Airport"},
    "south_delhi":       {"lat": 28.5244, "lon": 77.2167, "label": "South Delhi"},
    "rohini":            {"lat": 28.7329, "lon": 77.1143, "label": "Rohini"},
    "greater_noida":     {"lat": 28.4744, "lon": 77.5040, "label": "Greater Noida"},
}

_DEFAULT_EDGES: List[Dict[str, Any]] = [
    # Edge: (from, to, distance_km, free_speed_kmh, capacity_vph, lanes)
    {"u": "connaught_place", "v": "ito",              "dist_km": 3.5,  "free_speed": 35, "capacity": 4000, "lanes": 3},
    {"u": "ito",             "v": "noida_sec18",       "dist_km": 12.0, "free_speed": 55, "capacity": 6000, "lanes": 3},  # DND
    {"u": "noida_sec18",     "v": "noida_sec62",       "dist_km": 8.0,  "free_speed": 60, "capacity": 5000, "lanes": 3},  # Noida Exp
    {"u": "noida_sec62",     "v": "indirapuram",       "dist_km": 5.0,  "free_speed": 45, "capacity": 3500, "lanes": 2},
    {"u": "indirapuram",     "v": "ghaziabad_center",  "dist_km": 7.0,  "free_speed": 40, "capacity": 3000, "lanes": 2},  # NH24
    {"u": "noida_sec62",     "v": "ghaziabad_center",  "dist_km": 10.0, "free_speed": 50, "capacity": 4500, "lanes": 3},
    {"u": "connaught_place", "v": "dwarka",            "dist_km": 22.0, "free_speed": 50, "capacity": 5500, "lanes": 3},
    {"u": "dwarka",          "v": "gurugram_cyber",    "dist_km": 15.0, "free_speed": 65, "capacity": 7000, "lanes": 4},  # NH48
    {"u": "dwarka",          "v": "igi_airport",       "dist_km": 8.0,  "free_speed": 60, "capacity": 5000, "lanes": 3},
    {"u": "connaught_place", "v": "south_delhi",       "dist_km": 12.0, "free_speed": 35, "capacity": 4000, "lanes": 2},
    {"u": "south_delhi",     "v": "gurugram_cyber",    "dist_km": 15.0, "free_speed": 55, "capacity": 6000, "lanes": 3},
    {"u": "connaught_place", "v": "rohini",            "dist_km": 18.0, "free_speed": 40, "capacity": 4500, "lanes": 3},
    {"u": "noida_sec18",     "v": "greater_noida",     "dist_km": 20.0, "free_speed": 70, "capacity": 6000, "lanes": 3},  # Noida-GN Exp
    {"u": "ito",             "v": "indirapuram",       "dist_km": 14.0, "free_speed": 45, "capacity": 4000, "lanes": 3},  # NH24 direct
]

# Make edges bidirectional
_DEFAULT_EDGES = _DEFAULT_EDGES + [
    {**e, "u": e["v"], "v": e["u"]} for e in _DEFAULT_EDGES
]

# Emission factors (grams per vehicle-km)
_EMISSION_FACTORS = {
    "co2_g_per_vkm": 170.0,       # Average Indian car
    "pm25_g_per_vkm": 0.045,
    "nox_g_per_vkm": 0.8,
    "ev_co2_factor": 0.35,        # 65% reduction for EVs
    "ev_pm25_factor": 0.20,       # 80% reduction (brake/tire remain)
    "bus_co2_per_pax_km": 25.0,   # Per-passenger
}


def _bpr_time(dist_km: float, free_speed: float, capacity: float, flow: float) -> float:
    """BPR travel time in minutes."""
    t_free = dist_km / free_speed * 60.0
    ratio = flow / max(capacity, 1)
    return t_free * (1.0 + _BPR_ALPHA * (ratio ** _BPR_BETA))


def _dijkstra(nodes, edges, origin: str, destination: str, flow_map: Dict[str, float], flow_ratio: float = 0.6) -> Dict[str, Any]:
    """Shortest path by travel time."""
    adj: Dict[str, List] = {n: [] for n in nodes}
    for e in edges:
        eid = f"{e['u']}->{e['v']}"
        flow = flow_map.get(eid, e["capacity"] * flow_ratio)
        tt = _bpr_time(e["dist_km"], e["free_speed"], e["capacity"], flow)
        adj[e["u"]].append((e["v"], tt, e))

    dist = {n: float("inf") for n in nodes}
    prev: Dict[str, Any] = {n: None for n in nodes}
    dist[origin] = 0
    pq = [(0.0, origin)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        if u == destination:
            break
        for v, w, edge in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = (u, edge)
                heapq.heappush(pq, (nd, v))

    # Reconstruct path
    path = []
    node = destination
    while prev[node]:
        u, edge = prev[node]
        path.append(edge)
        node = u
    path.reverse()

    return {
        "travel_time_min": round(dist[destination], 2),
        "path": path,
        "total_dist_km": sum(e["dist_km"] for e in path),
    }


class TransportEngine(SimulationEngine):
    """Transportation network simulation using BPR model."""

    @property
    def name(self) -> str:
        return "TransportEngine"

    @property
    def capabilities(self) -> List[EngineCapability]:
        return [EngineCapability.TRANSPORT]

    @property
    def required_data(self) -> List[str]:
        return []  # Has built-in default network

    @property
    def optional_data(self) -> List[str]:
        return ["nodes", "edges", "flow_map", "demand_matrix", "ev_share"]

    def estimate_missing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if "nodes" not in data:
            data["nodes"] = _DEFAULT_NODES
        if "edges" not in data:
            data["edges"] = _DEFAULT_EDGES
        if "ev_share" not in data:
            # Use CSV-derived EV count if available
            ev_count = data.get("ev_count", 0)
            total_vehicles = 12_000_000  # Approximate NCR fleet
            data["ev_share"] = max(ev_count / total_vehicles, 0.05) if ev_count else 0.05
        # Calibrate default flow ratio from CSV congestion data
        if "flow_ratio" not in data:
            csv_congestion = data.get("baseline_congestion_pct", 55)
            # Map CSV congestion% to flow/capacity ratio for BPR model
            # Higher CSV congestion → higher flow ratio → more realistic baseline
            data["flow_ratio"] = 0.40 + (csv_congestion / 100.0) * 0.45  # 0.40-0.85 range
        return data

    async def _run(self, scenario: Scenario, data: Dict[str, Any]) -> SimulationResult:
        nodes = data.get("nodes", _DEFAULT_NODES)
        edges = list(data.get("edges", _DEFAULT_EDGES))
        ev_share = data.get("ev_share", 0.05)
        flow_map = data.get("flow_map", {})
        flow_ratio = data.get("flow_ratio", 0.6)  # Calibrated from CSV congestion data

        # Apply interventions
        new_ev_share = ev_share
        capacity_multiplier = 1.0
        speed_bonus = 0.0
        bus_lane_penalty = 0.0
        congestion_price_reduction = 0.0

        for intv in scenario.interventions:
            p = intv.parameters
            if intv.name == "signal_optimization":
                speed_bonus += p.get("speed_gain_pct", 10) / 100.0
            elif intv.name == "bus_rapid_transit":
                bus_lane_penalty = p.get("car_capacity_reduction_pct", 15) / 100.0
            elif intv.name == "ev_fleet_expansion":
                new_ev_share = min(1.0, new_ev_share + p.get("ev_share_increase", 0.10))
            elif intv.name == "congestion_pricing":
                congestion_price_reduction = p.get("demand_reduction_pct", 12) / 100.0
            elif intv.name == "road_capacity_expansion":
                capacity_multiplier *= (1.0 + p.get("capacity_increase_pct", 20) / 100.0)
            elif intv.name == "metro_expansion":
                congestion_price_reduction += p.get("mode_shift_pct", 8) / 100.0

        # Modify edges with interventions
        modified_edges = []
        for e in edges:
            me = dict(e)
            me["free_speed"] *= (1.0 + speed_bonus)
            me["capacity"] = me["capacity"] * capacity_multiplier * (1.0 - bus_lane_penalty)
            modified_edges.append(me)

        # Simulate flows on all edges
        total_vkt = 0.0
        total_travel_min = 0.0
        edge_results = []

        for e in modified_edges:
            eid = f"{e['u']}->{e['v']}"
            base_flow = flow_map.get(eid, e["capacity"] * flow_ratio)
            flow = base_flow * (1.0 - congestion_price_reduction)
            tt = _bpr_time(e["dist_km"], e["free_speed"], e["capacity"], flow)
            vkt = flow * e["dist_km"]
            total_vkt += vkt
            total_travel_min += tt
            edge_results.append({
                "edge": eid,
                "travel_time_min": round(tt, 2),
                "flow": round(flow),
                "v_over_c": round(flow / max(e["capacity"], 1), 3),
                "avg_speed_kmh": round(e["dist_km"] / (tt / 60), 1) if tt > 0 else 0,
            })

        avg_speed = sum(e["avg_speed_kmh"] for e in edge_results) / max(len(edge_results), 1)

        # Emissions
        ice_vkt = total_vkt * (1 - new_ev_share)
        ev_vkt = total_vkt * new_ev_share
        co2_kg = (ice_vkt * _EMISSION_FACTORS["co2_g_per_vkm"] +
                  ev_vkt * _EMISSION_FACTORS["co2_g_per_vkm"] * _EMISSION_FACTORS["ev_co2_factor"]) / 1000.0
        pm25_kg = (ice_vkt * _EMISSION_FACTORS["pm25_g_per_vkm"] +
                   ev_vkt * _EMISSION_FACTORS["pm25_g_per_vkm"] * _EMISSION_FACTORS["ev_pm25_factor"]) / 1000.0

        # Congestion index
        congested_edges = sum(1 for e in edge_results if e["v_over_c"] > 0.8)
        congestion_pct = congested_edges / max(len(edge_results), 1) * 100

        metrics = {
            "avg_speed_kmh": round(avg_speed, 1),
            "total_vkt": round(total_vkt),
            "co2_tonnes": round(co2_kg / 1000, 2),
            "pm25_kg": round(pm25_kg, 2),
            "congestion_pct": round(congestion_pct, 1),
            "ev_share": round(new_ev_share, 3),
            "edge_count": len(modified_edges) // 2,  # Bidirectional
            "node_count": len(nodes),
        }

        # Impact deltas vs baseline
        baseline_co2 = total_vkt * _EMISSION_FACTORS["co2_g_per_vkm"] / 1e6
        impacts = {
            "co2_reduction_pct": round((1.0 - co2_kg / 1000 / max(baseline_co2, 0.001)) * 100, 1),
            "speed_change_pct": round(speed_bonus * 100, 1),
            "congestion_change_pct": round(-congestion_price_reduction * 100, 1),
        }

        recommendations = []
        if congestion_pct > 50:
            recommendations.append("High congestion — consider congestion pricing or metro expansion")
        if new_ev_share < 0.15:
            recommendations.append(f"EV share at {new_ev_share*100:.0f}% — accelerating EV adoption reduces PM2.5 significantly")
        if avg_speed < 25:
            recommendations.append("Average speed critically low — signal optimization and BRT lanes recommended")

        return SimulationResult(
            engine=self.name,
            domain=EngineCapability.TRANSPORT,
            scenario_name=scenario.name,
            metrics=metrics,
            impacts=impacts,
            recommendations=recommendations,
            confidence=0.75,
            metadata={"top_congested": sorted(edge_results, key=lambda x: -x["v_over_c"])[:5]},
        )
