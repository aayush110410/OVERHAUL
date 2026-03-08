"""Schema normalizer for heterogeneous city data.

Cities provide data in wildly different formats. This module normalizes
everything into OVERHAUL's canonical schema so engines can consume
data from any city without changes.

Canonical schema domains:
  - traffic: speed, congestion, volume, mode_split
  - aqi: pm25, pm10, aqi_index, pollutants
  - population: total, commuters, segments
  - infrastructure: roads_km, metro_km, bus_routes
  - energy: grid_capacity_mw, ev_chargers, solar_mw
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# ── Canonical Field Names ──
# Each domain has a set of canonical field names that engines expect.
_CANONICAL = {
    "traffic": {
        "avg_speed_kmh": {"aliases": ["speed", "avg_speed", "average_speed", "speed_kmph", "avg_speed_kmph"],
                          "type": float, "default": 25.0, "unit": "km/h"},
        "congestion_pct": {"aliases": ["congestion", "congestion_level", "congestion_percent", "congestion_index"],
                           "type": float, "default": 55.0, "unit": "%"},
        "daily_volume": {"aliases": ["volume", "traffic_volume", "vehicles_per_day", "aadt"],
                         "type": int, "default": 5_000_000, "unit": "vehicles/day"},
        "ev_percentage": {"aliases": ["ev_pct", "ev_share", "electric_vehicle_pct"],
                          "type": float, "default": 3.0, "unit": "%"},
        "mode_split_car": {"aliases": ["car_share", "car_pct", "private_vehicle_share"],
                           "type": float, "default": 0.30, "unit": "fraction"},
        "mode_split_metro": {"aliases": ["metro_share", "metro_pct", "rail_share"],
                             "type": float, "default": 0.12, "unit": "fraction"},
        "mode_split_bus": {"aliases": ["bus_share", "bus_pct", "public_bus_share"],
                           "type": float, "default": 0.15, "unit": "fraction"},
    },
    "aqi": {
        "pm25": {"aliases": ["pm2.5", "pm25_ug", "pm25_ugm3", "particulate_matter_25"],
                 "type": float, "default": 285.0, "unit": "µg/m³"},
        "pm10": {"aliases": ["pm_10", "pm10_ug", "particulate_matter_10"],
                 "type": float, "default": 400.0, "unit": "µg/m³"},
        "aqi_index": {"aliases": ["aqi", "air_quality_index", "aqi_value"],
                      "type": float, "default": 350.0, "unit": "index"},
        "no2": {"aliases": ["nitrogen_dioxide", "no2_ppb"],
                "type": float, "default": 45.0, "unit": "ppb"},
        "co": {"aliases": ["carbon_monoxide", "co_ppm"],
               "type": float, "default": 1.5, "unit": "ppm"},
    },
    "population": {
        "total_population": {"aliases": ["population", "pop", "total_pop"],
                             "type": int, "default": 20_000_000, "unit": "people"},
        "commuter_population": {"aliases": ["commuters", "daily_commuters"],
                                "type": int, "default": 7_000_000, "unit": "people"},
        "area_sq_km": {"aliases": ["area", "city_area", "area_km2"],
                       "type": float, "default": 1484.0, "unit": "km²"},
    },
    "infrastructure": {
        "road_km": {"aliases": ["roads_km", "total_road_length", "road_network_km"],
                    "type": float, "default": 33000.0, "unit": "km"},
        "metro_km": {"aliases": ["metro_length", "rail_km", "metro_network_km"],
                     "type": float, "default": 390.0, "unit": "km"},
        "bus_routes": {"aliases": ["bus_route_count", "dtc_routes"],
                       "type": int, "default": 700, "unit": "routes"},
    },
    "energy": {
        "grid_capacity_mw": {"aliases": ["grid_mw", "power_capacity", "electricity_mw"],
                             "type": float, "default": 8000.0, "unit": "MW"},
        "ev_chargers": {"aliases": ["charging_stations", "ev_charging_points"],
                        "type": int, "default": 2500, "unit": "chargers"},
    },
}


def normalize(raw_data: Dict[str, Any], domain: str = "auto") -> Dict[str, Any]:
    """Normalize raw data dict to canonical field names.

    Args:
        raw_data: Data with potentially non-standard field names
        domain: Which domain schema to normalize against, or "auto" to try all

    Returns:
        Dict with canonical field names (unrecognized fields passed through)
    """
    if domain == "auto":
        # Try all domains
        result = dict(raw_data)
        for dom_name, dom_schema in _CANONICAL.items():
            result = _normalize_domain(result, dom_schema)
        return result

    schema = _CANONICAL.get(domain, {})
    return _normalize_domain(raw_data, schema)


def _normalize_domain(data: Dict[str, Any], schema: Dict[str, Dict]) -> Dict[str, Any]:
    """Apply one domain's schema normalization."""
    result = {}
    used_keys = set()

    for canonical_name, spec in schema.items():
        # Check if canonical name already present
        if canonical_name in data:
            result[canonical_name] = spec["type"](data[canonical_name])
            used_keys.add(canonical_name)
            continue

        # Check aliases
        for alias in spec["aliases"]:
            if alias in data:
                result[canonical_name] = spec["type"](data[alias])
                used_keys.add(alias)
                break

    # Pass through unrecognized fields
    for k, v in data.items():
        if k not in used_keys and k not in result:
            result[k] = v

    return result


def estimate_missing(
    data: Dict[str, Any],
    domain: str = "auto",
    city: str = "delhi",
) -> Dict[str, Any]:
    """Fill in missing fields with city-calibrated defaults.

    Uses Delhi NCR defaults as baseline. These are not guesses —
    they're calibrated from CPCB, DMRC, Delhi Statistical Handbook data.

    Args:
        data: Partially complete data dict (already normalized)
        domain: Domain to fill defaults for, or "auto" for all
        city: City for calibration (affects defaults)

    Returns:
        Data dict with missing fields filled in
    """
    result = dict(data)

    # City-specific adjustments to defaults
    _CITY_FACTORS = {
        "noida": {"pm25": 0.85, "congestion_pct": 0.90, "avg_speed_kmh": 1.15},
        "ghaziabad": {"pm25": 1.10, "congestion_pct": 0.95, "avg_speed_kmh": 0.90},
        "gurugram": {"pm25": 0.80, "congestion_pct": 1.10, "avg_speed_kmh": 1.05},
        "greater noida": {"pm25": 0.70, "congestion_pct": 0.60, "avg_speed_kmh": 1.30},
        "faridabad": {"pm25": 0.95, "congestion_pct": 0.85, "avg_speed_kmh": 1.00},
        "delhi": {},  # Baseline
    }
    city_factors = _CITY_FACTORS.get(city.lower(), {})

    domains_to_fill = list(_CANONICAL.keys()) if domain == "auto" else [domain]

    for dom in domains_to_fill:
        schema = _CANONICAL.get(dom, {})
        for canonical_name, spec in schema.items():
            if canonical_name not in result:
                default = spec["default"]
                # Apply city factor if available
                factor = city_factors.get(canonical_name, 1.0)
                if isinstance(default, (int, float)) and isinstance(factor, (int, float)):
                    default = type(default)(default * factor)
                result[canonical_name] = default

    return result


def get_schema_info(domain: Optional[str] = None) -> Dict[str, Any]:
    """Return schema documentation for API consumers."""
    if domain:
        return {
            "domain": domain,
            "fields": {
                name: {"aliases": spec["aliases"], "type": spec["type"].__name__,
                       "default": spec["default"], "unit": spec["unit"]}
                for name, spec in _CANONICAL.get(domain, {}).items()
            },
        }
    return {
        "domains": list(_CANONICAL.keys()),
        "total_fields": sum(len(s) for s in _CANONICAL.values()),
    }


def validate_completeness(
    data: Dict[str, Any],
    required_domains: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Check how complete a dataset is against the canonical schema.

    Returns:
      {"completeness_pct": float, "missing": {domain: [fields]}, "present": {domain: [fields]}}
    """
    required_domains = required_domains or list(_CANONICAL.keys())
    total = 0
    present_count = 0
    missing: Dict[str, List[str]] = {}
    present: Dict[str, List[str]] = {}

    for dom in required_domains:
        schema = _CANONICAL.get(dom, {})
        m = []
        p = []
        for name in schema:
            total += 1
            if name in data:
                present_count += 1
                p.append(name)
            else:
                m.append(name)
        if m:
            missing[dom] = m
        if p:
            present[dom] = p

    return {
        "completeness_pct": round(present_count / max(total, 1) * 100, 1),
        "total_fields": total,
        "present_fields": present_count,
        "missing": missing,
        "present": present,
    }
