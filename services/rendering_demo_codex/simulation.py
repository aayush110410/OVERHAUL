from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from .schemas import (
    DatasetRecord,
    GeoLine,
    GeoPoint,
    LayerSnapshot,
    SimulationIntervention,
    SimulationReport,
    SimulationRequest,
    TimelineState,
)

try:  # pragma: no cover - dependency presence varies by environment
    import networkx as nx
except Exception:  # pragma: no cover - dependency presence varies by environment
    nx = None


@dataclass
class SimulationInputs:
    traffic: LayerSnapshot
    focus: Dict[str, float]


class NetworkTrafficSimulation:
    def __init__(self) -> None:
        self._networkx_missing_message = (
            "NetworkX is not installed. Install it with `pip install networkx` to run Demo 4 simulations."
        )

    def simulate(self, request: SimulationRequest, traffic: LayerSnapshot, focus: Dict[str, float]) -> SimulationReport:
        if nx is None:
            return self._fallback_report(request)

        graph = self._build_graph(traffic)
        if graph.number_of_edges() == 0:
            return self._fallback_report(request)

        timeline = []
        years = request.years or [0, 1, 5]
        for year in years:
            state_graph = graph.copy()
            self._apply_interventions(state_graph, request.interventions, year)
            timeline.append(self._run_state(state_graph, year, request.interventions))

        baseline = timeline[0]
        final = timeline[-1]
        traffic_improvement_pct = self._delta_percent(
            baseline.metrics.get("avg_speed_kph", 1.0),
            final.metrics.get("avg_speed_kph", 1.0),
        )
        travel_time_delta_pct = self._delta_percent(
            baseline.metrics.get("travel_time_minutes", 1.0),
            final.metrics.get("travel_time_minutes", 1.0),
            invert=True,
        )
        congestion_delta_pct = self._delta_percent(
            baseline.metrics.get("congestion_pct", 1.0),
            final.metrics.get("congestion_pct", 1.0),
            invert=True,
        )
        pollution_delta_pct = self._delta_percent(
            baseline.metrics.get("pollution_index", 1.0),
            final.metrics.get("pollution_index", 1.0),
            invert=True,
        )

        recommendations = self._build_recommendations(request.interventions, final)
        summary = (
            f"Traffic speed improves by {traffic_improvement_pct:.1f}% while travel time drops by "
            f"{travel_time_delta_pct:.1f}% across the modeled corridor."
        )

        return SimulationReport(
            summary=summary,
            traffic_improvement_pct=traffic_improvement_pct,
            travel_time_delta_pct=travel_time_delta_pct,
            congestion_delta_pct=congestion_delta_pct,
            pollution_delta_pct=pollution_delta_pct,
            active_models=["traffic_flow", "pollution_impact", "infrastructure_change"],
            timeline=timeline,
            recommendations=recommendations,
        )

    def _build_graph(self, traffic: LayerSnapshot):
        graph = nx.DiGraph()
        for item in traffic.items:
            if not item.path or len(item.path.coordinates) < 2:
                continue
            coords = item.path.coordinates
            for start, end in zip(coords[:-1], coords[1:]):
                start_id = self._node_id(start)
                end_id = self._node_id(end)
                distance_m = self._distance_m(start, end)
                speed_kph = max(8.0, float(item.properties.get("speed_kph") or 40.0))
                capacity = max(400.0, float(item.properties.get("lane_capacity") or 1000.0))
                congestion = float(item.properties.get("congestion") or 0.0)
                travel_time_s = distance_m / max(1.0, speed_kph / 3.6)
                edge_payload = {
                    "segment_id": item.id,
                    "distance_m": distance_m,
                    "base_speed_kph": speed_kph,
                    "capacity": capacity,
                    "travel_time_s": travel_time_s * (1.0 + congestion * 0.35),
                    "baseline_congestion": congestion,
                    "lanes": float(item.properties.get("lanes") or 2),
                    "path": [start, end],
                }
                graph.add_node(start_id, longitude=start.longitude, latitude=start.latitude)
                graph.add_node(end_id, longitude=end.longitude, latitude=end.latitude)
                graph.add_edge(start_id, end_id, **edge_payload)
                graph.add_edge(end_id, start_id, **edge_payload)
        return graph

    def _apply_interventions(
        self,
        graph,
        interventions: List[SimulationIntervention],
        year_offset: int,
    ) -> None:
        maturity = 0.45 if year_offset == 0 else 0.75 if year_offset == 1 else 1.0
        for intervention in interventions:
            if intervention.kind == "new_road":
                self._apply_new_road(graph, intervention, maturity)
            elif intervention.kind == "lane_expansion":
                self._apply_lane_expansion(graph, intervention, maturity)
            elif intervention.kind == "flyover":
                self._apply_flyover(graph, intervention, maturity)
            elif intervention.kind == "road_closure":
                self._apply_road_closure(graph, intervention, maturity)
            elif intervention.kind == "signal_optimization":
                self._apply_signal_optimization(graph, intervention, maturity)

    def _apply_new_road(self, graph, intervention: SimulationIntervention, maturity: float) -> None:
        if len(intervention.coordinates) < 2:
            return
        for start, end in zip(intervention.coordinates[:-1], intervention.coordinates[1:]):
            start_id = self._node_id(start)
            end_id = self._node_id(end)
            distance_m = self._distance_m(start, end)
            speed_kph = max(40.0, 70.0 + intervention.speed_delta) * maturity
            capacity = 1200.0 + max(0.0, intervention.capacity_delta) + intervention.lane_delta * 450.0
            payload = {
                "segment_id": f"new-road-{intervention.name}-{start_id}-{end_id}",
                "distance_m": distance_m,
                "base_speed_kph": speed_kph,
                "capacity": max(900.0, capacity),
                "travel_time_s": distance_m / max(1.0, speed_kph / 3.6),
                "baseline_congestion": 0.12,
                "lanes": max(2.0, float(intervention.lane_delta or 2)),
                "path": [start, end],
            }
            graph.add_node(start_id, longitude=start.longitude, latitude=start.latitude)
            graph.add_node(end_id, longitude=end.longitude, latitude=end.latitude)
            graph.add_edge(start_id, end_id, **payload)
            graph.add_edge(end_id, start_id, **payload)

    def _apply_lane_expansion(self, graph, intervention: SimulationIntervention, maturity: float) -> None:
        for _, _, data in self._target_edges(graph, intervention):
            data["capacity"] *= 1.0 + max(0.12, intervention.lane_delta * 0.16) * maturity
            data["travel_time_s"] *= 1.0 - 0.1 * maturity
            data["lanes"] += max(1.0, intervention.lane_delta * maturity)

    def _apply_flyover(self, graph, intervention: SimulationIntervention, maturity: float) -> None:
        targeted = list(self._target_edges(graph, intervention))
        if not targeted and len(intervention.coordinates) >= 2:
            self._apply_new_road(graph, intervention, maturity)
            return

        for _, _, data in targeted:
            start, end = data["path"]
            elevated_speed = data["base_speed_kph"] * (1.25 + 0.25 * maturity)
            payload = {
                **data,
                "segment_id": f"flyover-{intervention.name}-{data['segment_id']}",
                "base_speed_kph": elevated_speed,
                "capacity": data["capacity"] * (1.18 + 0.12 * maturity),
                "travel_time_s": data["distance_m"] / max(1.0, elevated_speed / 3.6),
                "baseline_congestion": data["baseline_congestion"] * (1.0 - 0.35 * maturity),
                "path": [
                    GeoPoint(longitude=start.longitude, latitude=start.latitude, altitude=18),
                    GeoPoint(longitude=end.longitude, latitude=end.latitude, altitude=18),
                ],
            }
            graph.add_edge(self._node_id(start), self._node_id(end), **payload)

    def _apply_road_closure(self, graph, intervention: SimulationIntervention, maturity: float) -> None:
        for source, target, _ in list(self._target_edges(graph, intervention)):
            if maturity >= 0.45 and graph.has_edge(source, target):
                graph.remove_edge(source, target)

    def _apply_signal_optimization(self, graph, intervention: SimulationIntervention, maturity: float) -> None:
        for _, _, data in self._target_edges(graph, intervention):
            data["travel_time_s"] *= 1.0 - 0.15 * maturity
            data["capacity"] *= 1.0 + 0.08 * maturity

    def _run_state(
        self,
        graph,
        year_offset: int,
        interventions: List[SimulationIntervention],
    ) -> TimelineState:
        od_pairs = self._sample_od_pairs(graph)
        if not od_pairs:
            return TimelineState(label=self._label_for_year(year_offset), year_offset=year_offset)

        edge_loads: Dict[Tuple[str, str], float] = {}
        total_distance = 0.0
        total_time = 0.0

        demand_scale = 1.0 + 0.06 * year_offset
        for origin, destination in od_pairs:
            try:
                path = nx.shortest_path(graph, origin, destination, weight="travel_time_s")
            except Exception:
                continue
            for source, target in zip(path[:-1], path[1:]):
                data = graph[source][target]
                key = (source, target)
                edge_loads[key] = edge_loads.get(key, 0.0) + demand_scale * 210.0
                total_distance += data["distance_m"]
                total_time += data["travel_time_s"]

        enriched_edges = []
        for source, target, data in graph.edges(data=True):
            load = edge_loads.get((source, target), 0.0)
            capacity = max(1.0, float(data["capacity"]))
            congestion = min(0.99, load / capacity)
            data["dynamic_congestion"] = congestion
            data["travel_time_s"] = data["travel_time_s"] * (1.0 + congestion * congestion * 0.6)
            enriched_edges.append((source, target, data))

        avg_speed = (total_distance / 1000.0) / max(total_time / 3600.0, 0.001)
        avg_travel_time_min = (total_time / max(len(od_pairs), 1)) / 60.0
        congestion_pct = (
            sum(float(data.get("dynamic_congestion") or 0.0) for _, _, data in enriched_edges)
            / max(len(enriched_edges), 1)
            * 100.0
        )
        pollution_index = max(18.0, 82.0 - avg_speed * 0.9 + congestion_pct * 0.22)

        ranked_edges = sorted(
            enriched_edges,
            key=lambda edge: float(edge[2].get("dynamic_congestion") or 0.0),
            reverse=True,
        )[: 140]

        heatmap = []
        flows = []
        for idx, (_, _, data) in enumerate(ranked_edges):
            start, end = data["path"]
            midpoint = GeoPoint(
                longitude=(start.longitude + end.longitude) / 2,
                latitude=(start.latitude + end.latitude) / 2,
                altitude=max(start.altitude, end.altitude, 160),
            )
            intensity = float(data.get("dynamic_congestion") or 0.0)
            heatmap.append(
                DatasetRecord(
                    id=f"heat-{year_offset}-{idx}",
                    position=midpoint,
                    properties={"kind": "heat", "intensity": intensity},
                )
            )
            flows.append(
                DatasetRecord(
                    id=f"flow-{year_offset}-{idx}",
                    path=GeoLine(coordinates=[start, end]),
                    properties={
                        "kind": "flow",
                        "intensity": intensity,
                        "travel_time_s": float(data["travel_time_s"]),
                    },
                )
            )

        infrastructure = self._intervention_geometry(interventions, year_offset)
        return TimelineState(
            label=self._label_for_year(year_offset),
            year_offset=year_offset,
            metrics={
                "avg_speed_kph": round(avg_speed, 2),
                "travel_time_minutes": round(avg_travel_time_min, 2),
                "congestion_pct": round(congestion_pct, 2),
                "pollution_index": round(pollution_index, 2),
            },
            heatmap=heatmap,
            flows=flows,
            infrastructure=infrastructure,
        )

    def _sample_od_pairs(self, graph) -> List[Tuple[str, str]]:
        nodes = list(graph.nodes)
        if len(nodes) < 4:
            return []
        step = max(1, len(nodes) // 18)
        sampled = nodes[::step][:18]
        pairs = []
        for origin, destination in itertools.combinations(sampled, 2):
            pairs.append((origin, destination))
            if len(pairs) >= 24:
                break
        return pairs

    def _target_edges(self, graph, intervention: SimulationIntervention):
        if intervention.target_segment_id:
            for source, target, data in graph.edges(data=True):
                if data.get("segment_id") == intervention.target_segment_id:
                    yield source, target, data
            return

        if intervention.coordinates:
            targets = {self._node_id(coord) for coord in intervention.coordinates}
            for source, target, data in graph.edges(data=True):
                if source in targets or target in targets:
                    yield source, target, data
            return

        for edge in list(graph.edges(data=True))[:24]:
            yield edge

    def _intervention_geometry(
        self,
        interventions: List[SimulationIntervention],
        year_offset: int,
    ) -> List[DatasetRecord]:
        maturity = 0.35 if year_offset == 0 else 0.7 if year_offset == 1 else 1.0
        records = []
        for idx, intervention in enumerate(interventions):
            if len(intervention.coordinates) >= 2:
                coords = [
                    GeoPoint(
                        longitude=point.longitude,
                        latitude=point.latitude,
                        altitude=15.0 * maturity,
                    )
                    for point in intervention.coordinates
                ]
                records.append(
                    DatasetRecord(
                        id=f"infra-{year_offset}-{idx}",
                        path=GeoLine(coordinates=coords),
                        properties={
                            "kind": "infrastructure",
                            "name": intervention.name,
                            "type": intervention.kind,
                            "progress": maturity,
                        },
                    )
                )
        return records

    def _fallback_report(self, request: SimulationRequest) -> SimulationReport:
        timeline = [
            TimelineState(
                label="Current state",
                year_offset=0,
                metrics={
                    "avg_speed_kph": 28.0,
                    "travel_time_minutes": 22.4,
                    "congestion_pct": 61.0,
                    "pollution_index": 71.0,
                },
            ),
            TimelineState(
                label="1 year later",
                year_offset=1,
                metrics={
                    "avg_speed_kph": 32.3,
                    "travel_time_minutes": 19.8,
                    "congestion_pct": 49.0,
                    "pollution_index": 65.4,
                },
            ),
            TimelineState(
                label="5 years later",
                year_offset=5,
                metrics={
                    "avg_speed_kph": 37.4,
                    "travel_time_minutes": 16.5,
                    "congestion_pct": 38.0,
                    "pollution_index": 58.2,
                },
            ),
        ]
        return SimulationReport(
            summary=f"Fallback simulation generated for prompt '{request.prompt}'. {self._networkx_missing_message}",
            traffic_improvement_pct=21.2,
            travel_time_delta_pct=26.3,
            congestion_delta_pct=37.7,
            pollution_delta_pct=18.0,
            active_models=["traffic_flow", "pollution_impact", "infrastructure_change"],
            timeline=timeline,
            recommendations=[
                "Install NetworkX in the backend environment for graph-backed simulation updates.",
                "Keep infrastructure interventions tied to segment ids for more precise rerouting.",
            ],
        )

    def _build_recommendations(
        self,
        interventions: List[SimulationIntervention],
        final: TimelineState,
    ) -> List[str]:
        items = []
        if interventions:
            items.append("Continue animating interventions gradually so users can see rerouting occur over time.")
        if final.metrics.get("congestion_pct", 0.0) > 45.0:
            items.append("Add another lane-balancing or signal-optimization pass on the most saturated corridors.")
        if final.metrics.get("avg_speed_kph", 0.0) < 30.0:
            items.append("Introduce a new high-capacity connector or flyover to create a faster bypass path.")
        if not items:
            items.append("Lock the improved network state into the simulation layer and monitor the live feed for drift.")
        return items

    def _delta_percent(self, baseline: float, final: float, invert: bool = False) -> float:
        if baseline <= 0:
            return 0.0
        value = ((final - baseline) / baseline) * 100.0
        return -value if invert else value

    def _label_for_year(self, year: int) -> str:
        if year == 0:
            return "Current state"
        if year == 1:
            return "1 year later"
        return f"{year} years later"

    def _node_id(self, point: GeoPoint) -> str:
        return f"{point.longitude:.5f}:{point.latitude:.5f}"

    def _distance_m(self, start: GeoPoint, end: GeoPoint) -> float:
        radius = 6_371_000.0
        lat1 = math.radians(start.latitude)
        lat2 = math.radians(end.latitude)
        dlat = lat2 - lat1
        dlon = math.radians(end.longitude - start.longitude)
        hav = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        )
        return 2 * radius * math.asin(min(1.0, math.sqrt(hav)))
