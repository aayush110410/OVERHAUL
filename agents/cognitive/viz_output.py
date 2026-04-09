"""Visualization output generator - converts results to globe-ready GeoJSON."""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime


def build_viz_output(
    agent_context: Dict[str, Any],
    engine_results: Dict[str, Any],
    locations: List[Dict[str, Any]],
    center: Dict[str, float]
) -> Dict[str, Any]:
    """Build visualization data from agent context and engine results.

    Returns GeoJSON and graph data for globe rendering.
    """
    # Extract key metrics from engine results
    metrics = _extract_metrics(engine_results)

    # Generate GeoJSON features for locations
    geo_features = _build_geojson(locations, metrics)

    # Generate node/edge graph for relationships
    graph_data = _build_graph(agent_context, engine_results, locations)

    # Generate heatmap data
    heatmap_data = _build_heatmap(locations, metrics)

    return {
        "geojson": {
            "type": "FeatureCollection",
            "features": geo_features
        },
        "graph": graph_data,
        "heatmap": heatmap_data,
        "center": center,
        "timestamp": datetime.now().isoformat()
    }


def build_temporal_viz(
    temporal_result: Dict[str, Any],
    locations: List[Dict[str, Any]],
    center: Dict[str, float]
) -> Dict[str, Any]:
    """Build visualization data for temporal simulations."""
    timeline_steps = temporal_result.get("steps", [])

    # Convert timeline steps to animated GeoJSON
    animated_features = []
    for step in timeline_steps:
        step_time = step.get("time_index", 0)
        features = _build_temporal_geojson(step, locations, step_time)
        animated_features.extend(features)

    return {
        "animated_geojson": {
            "type": "FeatureCollection",
            "features": animated_features
        },
        "timeline": [step.get("time_label", f"Step {i}") for i, step in enumerate(timeline_steps)],
        "center": center,
        "timestamp": datetime.now().isoformat()
    }


def _extract_metrics(engine_results: Dict[str, Any]) -> Dict[str, float]:
    """Extract key metrics from engine results."""
    metrics = {}

    # Try to extract from impact cards
    impact_cards = engine_results.get("impactCards", [])
    for card in impact_cards:
        metric_name = card.get("metric", "").lower()
        if metric_name:
            metrics[metric_name] = card.get("value", 0)

    # Try to extract from raw results
    raw_results = engine_results.get("raw", {})
    if isinstance(raw_results, dict):
        for engine_name, result in raw_results.items():
            if isinstance(result, dict) and "metrics" in result:
                metrics.update(result["metrics"])

    return metrics


def _build_geojson(
    locations: List[Dict[str, Any]],
    metrics: Dict[str, float]
) -> List[Dict[str, Any]]:
    """Build GeoJSON features from locations and metrics."""
    features = []

    for loc in locations:
        # Create point feature for each location
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [loc["lon"], loc["lat"]]
            },
            "properties": {
                "name": loc["name"],
                "metrics": metrics,
                "category": "location"
            }
        }
        features.append(feature)

    return features


def _build_graph(
    agent_context: Dict[str, Any],
    engine_results: Dict[str, Any],
    locations: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Build node/edge graph showing relationships."""
    nodes = []
    edges = []

    # Add location nodes
    for i, loc in enumerate(locations):
        nodes.append({
            "id": f"loc_{i}",
            "name": loc["name"],
            "type": "location",
            "position": {"x": loc["lon"], "y": loc["lat"], "z": 0}
        })

    # Add engine result nodes
    raw_results = engine_results.get("raw", {})
    if isinstance(raw_results, dict):
        for engine_name, result in raw_results.items():
            nodes.append({
                "id": f"engine_{engine_name}",
                "name": engine_name,
                "type": "engine",
                "metrics": _extract_simple_metrics(result)
            })

            # Connect to locations
            for i in range(len(locations)):
                edges.append({
                    "source": f"loc_{i}",
                    "target": f"engine_{engine_name}",
                    "type": "affects"
                })

    return {"nodes": nodes, "edges": edges}


def _build_heatmap(
    locations: List[Dict[str, Any]],
    metrics: Dict[str, float]
) -> List[Dict[str, Any]]:
    """Build heatmap data from locations and metrics."""
    heatmap_points = []

    # Use a key metric for heatmap intensity (default to first available)
    intensity_key = next(iter(metrics), "value")
    max_value = max(metrics.values()) if metrics else 1

    for loc in locations:
        intensity = metrics.get(intensity_key, 0) / max_value if max_value > 0 else 0
        heatmap_points.append({
            "lat": loc["lat"],
            "lon": loc["lon"],
            "intensity": intensity
        })

    return heatmap_points


def _build_temporal_geojson(
    step: Dict[str, Any],
    locations: List[Dict[str, Any]],
    time_index: int
) -> List[Dict[str, Any]]:
    """Build GeoJSON features for a temporal step."""
    features = []

    # Add timestamp to each location feature
    for loc in locations:
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [loc["lon"], loc["lat"]]
            },
            "properties": {
                "name": loc["name"],
                "time_index": time_index,
                "timestamp": step.get("time_label", f"T+{time_index}"),
                "metrics": step.get("metrics", {})
            }
        }
        features.append(feature)

    return features


def _extract_simple_metrics(result: Any) -> Dict[str, float]:
    """Extract simple numeric metrics from a result object."""
    metrics = {}
    if isinstance(result, dict):
        for key, value in result.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[key] = value
    return metrics