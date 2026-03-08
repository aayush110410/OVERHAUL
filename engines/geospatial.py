"""
Geospatial Output Layer
========================

Converts engine simulation results into GeoJSON FeatureCollections
for map-based visualization in the frontend (MapLibre GL).

Generates:
  - Congestion heatmap overlay from TransportEngine
  - AQI zone polygons from EnvironmentEngine
  - Infrastructure markers from InfrastructureEngine
  - Freight corridor lines from LogisticsEngine
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

from engines.base import SimulationResult

# ── Delhi NCR reference points for spatial distribution ──

_NCR_NODES = {
    "connaught_place":  (28.6315, 77.2167),
    "ito":              (28.6285, 77.2413),
    "noida_sec18":      (28.5700, 77.3219),
    "noida_sec62":      (28.6270, 77.3640),
    "ghaziabad_center": (28.6692, 77.4538),
    "indirapuram":      (28.6352, 77.3764),
    "dwarka":           (28.5921, 77.0460),
    "gurugram_cyber":   (28.4947, 77.0887),
    "faridabad":        (28.4089, 77.3178),
    "rohini":           (28.7495, 77.0565),
    "saket":            (28.5244, 77.2066),
    "lajpat_nagar":     (28.5700, 77.2400),
}

_EDGES = [
    ("connaught_place", "ito"),
    ("ito", "noida_sec18"),
    ("noida_sec18", "noida_sec62"),
    ("noida_sec62", "ghaziabad_center"),
    ("noida_sec62", "indirapuram"),
    ("connaught_place", "dwarka"),
    ("dwarka", "gurugram_cyber"),
    ("connaught_place", "rohini"),
    ("connaught_place", "saket"),
    ("saket", "faridabad"),
    ("saket", "gurugram_cyber"),
    ("ito", "lajpat_nagar"),
    ("lajpat_nagar", "faridabad"),
    ("indirapuram", "ghaziabad_center"),
]


def _circle_polygon(lat: float, lon: float, radius_km: float, n: int = 24) -> List[List[float]]:
    """Generate approximate circle polygon coordinates."""
    coords = []
    for i in range(n + 1):
        angle = 2 * math.pi * i / n
        dlat = radius_km / 111.0 * math.cos(angle)
        dlon = radius_km / (111.0 * math.cos(math.radians(lat))) * math.sin(angle)
        coords.append([round(lon + dlon, 6), round(lat + dlat, 6)])
    return coords


def generate_geojson(
    results: Dict[str, SimulationResult],
) -> Dict[str, Any]:
    """Generate a combined GeoJSON FeatureCollection from all engine results."""
    features: List[Dict[str, Any]] = []

    for name, result in results.items():
        if not hasattr(result, "metrics") or not result.metrics:
            continue
        domain = result.domain.value if hasattr(result.domain, "value") else str(result.domain)

        if domain == "transport":
            features.extend(_transport_features(result.metrics))
        elif domain == "environment":
            features.extend(_environment_features(result.metrics))
        elif domain == "infrastructure":
            features.extend(_infrastructure_features(result.metrics))
        elif domain == "logistics":
            features.extend(_logistics_features(result.metrics))

    return {
        "type": "FeatureCollection",
        "features": features,
    }


def _transport_features(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Congestion intensity on road links + node speed markers."""
    features = []
    congestion_pct = metrics.get("congestion_pct", 50)
    avg_speed = metrics.get("avg_speed_kmh", 25)
    link_speeds = metrics.get("link_speeds", {})

    # Road link congestion lines
    for a, b in _EDGES:
        ca, cb = _NCR_NODES.get(a), _NCR_NODES.get(b)
        if not ca or not cb:
            continue
        link_key = f"{a}-{b}"
        speed = link_speeds.get(link_key, avg_speed)
        intensity = max(0, min(1, 1 - speed / 60))  # 0=free, 1=gridlock

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[ca[1], ca[0]], [cb[1], cb[0]]],
            },
            "properties": {
                "layer": "congestion",
                "link": link_key,
                "speed_kmh": round(speed, 1),
                "intensity": round(intensity, 3),
                "color": _heat_color(intensity),
            },
        })

    # Node congestion points
    for node_id, (lat, lon) in _NCR_NODES.items():
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {
                "layer": "congestion_node",
                "name": node_id,
                "avg_speed_kmh": round(avg_speed, 1),
                "congestion_pct": round(congestion_pct, 1),
            },
        })

    return features


def _environment_features(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """AQI zone polygons around monitoring points."""
    features = []
    aqi = metrics.get("projected_aqi", 200)
    pm25 = metrics.get("projected_pm25", 150)

    # Hotspot zones — industrial/traffic-heavy areas get higher AQI
    zone_weights = {
        "ito": 1.15,
        "connaught_place": 1.10,
        "ghaziabad_center": 1.20,
        "faridabad": 1.18,
        "dwarka": 0.95,
        "rohini": 1.08,
        "gurugram_cyber": 1.05,
        "saket": 1.00,
        "noida_sec18": 1.02,
        "noida_sec62": 0.98,
        "indirapuram": 1.05,
        "lajpat_nagar": 1.10,
    }

    for node_id, (lat, lon) in _NCR_NODES.items():
        w = zone_weights.get(node_id, 1.0)
        local_aqi = round(aqi * w)
        local_pm25 = round(pm25 * w, 1)
        category = _aqi_category(local_aqi)

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [_circle_polygon(lat, lon, 2.5)],
            },
            "properties": {
                "layer": "aqi_zone",
                "name": node_id,
                "aqi": local_aqi,
                "pm25": local_pm25,
                "category": category,
                "fill_color": _aqi_color(local_aqi),
                "fill_opacity": 0.25,
            },
        })

    return features


def _infrastructure_features(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Markers for planned infrastructure projects."""
    features = []
    projects = metrics.get("projects", [])

    if not projects:
        # Fallback: create markers based on metrics keys
        if metrics.get("total_cost_inr_crore"):
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [77.2167, 28.6315]},
                "properties": {
                    "layer": "infrastructure",
                    "type": "project_summary",
                    "cost_crore": metrics["total_cost_inr_crore"],
                    "roi_years": metrics.get("roi_years"),
                    "icon": "construction",
                },
            })
        return features

    for proj in projects:
        lat = proj.get("lat", 28.6315)
        lon = proj.get("lon", 77.2167)
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {
                "layer": "infrastructure",
                "type": proj.get("type", "project"),
                "name": proj.get("name", ""),
                "cost_crore": proj.get("cost_crore", 0),
                "icon": proj.get("type", "pin"),
            },
        })

    return features


def _logistics_features(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Freight corridors and micro-hub locations."""
    features = []

    # Major freight corridors
    freight_corridors = [
        ("ghaziabad_center", "connaught_place", "NH-24 Corridor"),
        ("faridabad", "saket", "Faridabad-Delhi Corridor"),
        ("gurugram_cyber", "dwarka", "Gurugram-Dwarka Corridor"),
        ("connaught_place", "rohini", "GT Karnal Road"),
    ]

    freight_vkt = metrics.get("freight_vkt_daily", 0)
    for a, b, label in freight_corridors:
        ca, cb = _NCR_NODES.get(a), _NCR_NODES.get(b)
        if not ca or not cb:
            continue
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[ca[1], ca[0]], [cb[1], cb[0]]],
            },
            "properties": {
                "layer": "freight_corridor",
                "name": label,
                "freight_vkt": round(freight_vkt / len(freight_corridors)),
                "color": "#ff9800",
                "dash": [8, 4],
            },
        })

    # Micro-hub points (distribute near major nodes)
    hub_count = metrics.get("micro_hubs", 0)
    if hub_count > 0:
        hub_nodes = list(_NCR_NODES.keys())[:min(hub_count, len(_NCR_NODES))]
        for node_id in hub_nodes:
            lat, lon = _NCR_NODES[node_id]
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon + 0.005, lat + 0.005]},
                "properties": {
                    "layer": "micro_hub",
                    "name": f"Hub-{node_id}",
                    "icon": "warehouse",
                },
            })

    return features


def _heat_color(intensity: float) -> str:
    """Map 0‑1 intensity to green→yellow→red hex color."""
    if intensity < 0.33:
        return "#4caf50"  # green
    if intensity < 0.66:
        return "#ffeb3b"  # yellow
    return "#f44336"      # red


def _aqi_color(aqi: int) -> str:
    """Return fill color based on Indian AQI categories."""
    if aqi <= 50:
        return "#00e400"
    if aqi <= 100:
        return "#92d050"
    if aqi <= 200:
        return "#ffff00"
    if aqi <= 300:
        return "#ff7e00"
    if aqi <= 400:
        return "#ff0000"
    return "#7e0023"


def _aqi_category(aqi: int) -> str:
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
