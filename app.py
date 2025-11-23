# app.py
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from statistics import mean
import joblib, os, time, uuid, json, hashlib, re, asyncio
import numpy as np
import random
import math
import httpx

from traffic_god_bridge import TrafficGodService

# --- Simulation / small graph setup (same as in Colab) ---
NODES = [
    ("A","Sector107_North", (77.302,28.590)),
    ("B","Sector107_South", (77.304,28.587)),
    ("C","Midpoint_Junction", (77.307,28.585)),
    ("D","Alternate1", (77.310,28.588)),
    ("E","Vasundhara_Approach", (77.314,28.586)),
    ("F","Vasundhara_Center", (77.316,28.584))
]

EDGES_BASE = [
  {"id":"e1", "u":"A","v":"B", "length":0.8, "capacity":1200, "free_speed":40, "baseline_flow":1000},
  {"id":"e2", "u":"B","v":"C", "length":0.6, "capacity":1000, "free_speed":35, "baseline_flow":1100},
  {"id":"e3", "u":"C","v":"E", "length":1.2, "capacity":900,  "free_speed":30, "baseline_flow":1300},
  {"id":"e4", "u":"E","v":"F", "length":0.6, "capacity":1200, "free_speed":40, "baseline_flow":1100},
  {"id":"e5", "u":"B","v":"D", "length":0.9, "capacity":800,  "free_speed":30, "baseline_flow":200},
  {"id":"e6", "u":"D","v":"E", "length":1.1, "capacity":700,  "free_speed":30, "baseline_flow":150},
  {"id":"e7", "u":"C","v":"D", "length":0.7, "capacity":500,  "free_speed":25, "baseline_flow":100}
]

DATA_DIR = Path(__file__).parent / "data"

# Corridor anchor points and live-data providers
CORRIDOR_ORIGIN = (28.5825, 77.3554)  # Sector-78, Noida (lat, lon)
CORRIDOR_DEST = (28.6663, 77.3649)    # Vasundhara, Ghaziabad (lat, lon)
OSRM_BASE_URL = "https://router.project-osrm.org"


def load_historical_metrics() -> Dict[str, Any]:
    payload = {"city": "Corridor", "records": []}
    data_path = DATA_DIR / "historical_metrics.json"
    if not data_path.exists():
        return payload
    try:
        with open(data_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
            payload["records"] = payload.get("records", [])
    except Exception as exc:
        print("Failed to load historical metrics:", exc)
    return payload


HISTORICAL_DATA = load_historical_metrics()

def historical_stats(metric: str, records: List[Dict[str, Any]], window: int = 5) -> Optional[Dict[str, Any]]:
    if not records:
        return None
    slice_records = records[-window:] if len(records) >= window else records[:]
    values = [rec.get(metric) for rec in slice_records if rec.get(metric) is not None]
    if not values:
        return None
    delta = values[-1] - values[0]
    return {
        "start_year": slice_records[0].get("year"),
        "end_year": slice_records[-1].get("year"),
        "start_value": values[0],
        "end_value": values[-1],
        "mean": round(mean(values), 2),
        "delta": round(delta, 2),
    }

# --- BPR travel time ---
def bpr_travel_time(length_km, free_speed_kmph, flow, capacity, alpha=0.15, beta=4.0):
    t_free = (length_km / free_speed_kmph) * 60.0
    ratio = flow / capacity if capacity>0 else flow / (capacity+1e-6)
    factor = 1.0 + alpha * (ratio ** beta)
    return t_free * factor

# naive shortest path on small graph using Dijkstra-like algorithm
import heapq
def dijkstra_shortest_path_edges(edges, origin, dest, flows_override=None):
    # build adjacency
    nodes = set()
    adj = {}
    edge_map = {}
    for e in edges:
        nodes.add(e['u']); nodes.add(e['v'])
        edge_map[(e['u'], e['v'])] = e
        adj.setdefault(e['u'], []).append(e['v'])
    # compute weight per directed edge
    weight = {}
    for e in edges:
        flow = flows_override.get(e['id'], e.get('baseline_flow',0)) if flows_override else e.get('baseline_flow',0)
        tt = bpr_travel_time(e['length'], e['free_speed'], flow, e['capacity'])
        mult = e.get('cost_mult', 1.0)
        weight[(e['u'], e['v'])] = tt * mult
    # dijkstra
    dist = {n: float('inf') for n in nodes}
    prev = {}
    dist[origin] = 0
    heap = [(0, origin)]
    while heap:
        d,u = heapq.heappop(heap)
        if d>dist[u]: continue
        if u==dest: break
        for v in adj.get(u, []):
            w = weight.get((u,v), 1.0)
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    if dist[dest] == float('inf'):
        return None
    # reconstruct path
    path_nodes = []
    cur = dest
    while cur != origin:
        path_nodes.append(cur)
        cur = prev[cur]
    path_nodes.append(origin)
    path_nodes = list(reversed(path_nodes))
    # edge ids & path travel time
    edge_ids=[]
    path_tt=0.0
    for i in range(len(path_nodes)-1):
        u=path_nodes[i]; v=path_nodes[i+1]
        e = edge_map[(u,v)]
        flow = flows_override.get(e['id'], e.get('baseline_flow',0)) if flows_override else e.get('baseline_flow',0)
        tt = bpr_travel_time(e['length'], e['free_speed'], flow, e['capacity'])
        path_tt += tt
        edge_ids.append(e['id'])
    return {"nodes": path_nodes, "edge_ids": edge_ids, "path_tt": path_tt}

# simulate_once (full sim)
def simulate_once(params, edges_base=EDGES_BASE):
    edges = [dict(e) for e in edges_base]
    # apply interventions
    for e in edges:
        e['cost_mult']=1.0
        if params.get('signal_opt_pct',0) and e['id'] in params.get('signal_edges',[]):
            e['free_speed'] = e['free_speed'] * (1 + params['signal_opt_pct']/100.0)
        if params.get('bus_lane') and e['id'] in params.get('buslane_edges',[]):
            e['capacity'] = e['capacity'] * (1 - params.get('bus_lane_pct',0)/100.0)
        if params.get('reroute_penalty',0) and e['id'] in params.get('penalize_edges',[]):
            e['cost_mult'] = 1.0 + params['reroute_penalty']
        e['flow'] = e.get('baseline_flow',0)
    # demand
    demand = int(params.get('demand', 1200) * (1 - params.get('demand_shift_pct',0)/100.0))
    sp = dijkstra_shortest_path_edges(edges, params['od_from'], params['od_to'], flows_override=None)
    if sp is None:
        return None
    # assign demand to path edges
    for e in edges:
        if e['id'] in sp['edge_ids']:
            e['flow'] += demand
    # compute totals
    total_vkt = 0.0
    path_tt = 0.0
    for e in edges:
        tt = bpr_travel_time(e['length'], e['free_speed'], e['flow'], e['capacity'])
        e['travel_time_min']=tt
        total_vkt += e['flow'] * e['length']
        if e['id'] in sp['edge_ids']:
            path_tt += tt
    co2_kg = total_vkt * params.get('EF_g_per_km', 250.0) / 1000.0
    pm25 = 50 + (total_vkt / 10000.0) * 100 + params.get('weather_factor',0)*10
    return {"edges": edges, "path_edge_ids": sp['edge_ids'], "avg_travel_time_min": path_tt, "total_vkt": total_vkt, "co2_kg": co2_kg, "pm25": pm25}

# --- Candidate generator & infra-suggestion (bypass) ---
def propose_candidates():
    return [
      {"name":"Reroute_25pct","params":{"reroute_penalty":0.6,"penalize_edges":["e1","e2","e3"]},"desc":"Increase cost on main corridor to divert traffic"},
      {"name":"Signal_opt_15","params":{"signal_opt_pct":15,"signal_edges":["e2","e3"]},"desc":"Signal timing optimization"},
      {"name":"Demand_shift_15","params":{"demand_shift_pct":15},"desc":"Demand reduction via incentives"},
      {"name":"Bus_lane_e2_20","params":{"bus_lane":1,"buslane_edges":["e2"],"bus_lane_pct":20},"desc":"Reallocate lane on e2 for buses"}
    ]

def propose_infrastructure_suggestions():
    # generates one "bypass" option with geometry (connect D->F or C->F) as feasible new link
    # choose best candidate nodes (simple heuristic)
    # returns list of dicts: name, description, geojson_linestring, params_to_add
    # We'll propose two options: C->F direct bypass, D->F bypass
    g1 = {"name":"Bypass_C_to_F", "desc":"Construct a connector from C to F (direct bypass)", "params":{"new_edge": {"id":"nb_cf","u":"C","v":"F","length":1.4,"capacity":800,"free_speed":50}}}
    g2 = {"name":"Bypass_D_to_F", "desc":"Construct a connector from D to F (alternate bypass)", "params":{"new_edge":{"id":"nb_df","u":"D","v":"F","length":1.2,"capacity":700,"free_speed":45}}}
    return [g1,g2]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def extract_prompt_signals(prompt_text: str) -> Dict[str, Any]:
    """Pull percentages and targets from free-form text to steer the surrogate outputs."""
    text = prompt_text.lower()
    signals: Dict[str, Any] = {
        "ev_share_pct": 0.0,
        "aqi_before": None,
        "aqi_after": None,
        "noise_before": None,
        "noise_after": None,
        "gdp_focus": bool(re.search(r"gdp|econom", text)),
        "finance_focus": bool(re.search(r"finance|return on investment|roi|capex|opex", text)),
        "health_focus": bool(re.search(r"aqi|pm2|pollution|health", text)),
    }

    ev_match = re.search(r"(\d{1,3})\s*%[^.]{0,20}(ev|electric|battery|vehicle)", text)
    if ev_match:
        signals["ev_share_pct"] = clamp(float(ev_match.group(1)), 0.0, 100.0)
    else:
        generic_pct = re.search(r"(\d{1,3})\s*%", text)
        if generic_pct:
            signals["ev_share_pct"] = clamp(float(generic_pct.group(1)), 0.0, 100.0)

    pm_pair = re.search(r"(\d{2,4})\s*(?:to|-|→)\s*(\d{2,4})\s*(?:aqi|pm|pm2\.?5|µg|ug)", text)
    if pm_pair:
        signals["aqi_before"] = float(pm_pair.group(1))
        signals["aqi_after"] = float(pm_pair.group(2))
    else:
        pm_single = re.search(r"(\d{2,4})\s*(?:aqi|pm|pm2\.?5|µg|ug)", text)
        if pm_single:
            signals["aqi_before"] = float(pm_single.group(1))

    noise_pair = re.search(r"(\d{2,3})\s*(?:to|-|→)\s*(\d{2,3})\s*(?:db|decibel)", text)
    if noise_pair:
        signals["noise_before"] = float(noise_pair.group(1))
        signals["noise_after"] = float(noise_pair.group(2))
    else:
        noise_single = re.search(r"(\d{2,3})\s*(?:db|decibel)", text)
        if noise_single:
            signals["noise_before"] = float(noise_single.group(1))

    signals["economic_focus"] = signals["gdp_focus"] or signals["finance_focus"]
    return signals


def scenario_from_payload(prompt_text: str, scenario_payload: Optional[Dict[str, Any]], signals: Dict[str, Any]) -> Dict[str, Any]:
    scenario = dict(interpret_prompt_to_scenario(prompt_text))
    params = (scenario_payload or {}).get("parameters") or {}
    scenario["demand"] = (scenario_payload or {}).get("demand", scenario.get("demand", 1200))
    scenario["weather_factor"] = (scenario_payload or {}).get("weather_factor", scenario.get("weather_factor", 0.3))
    scenario["demand_shift_pct"] = scenario.get("demand_shift_pct", 0.0) + params.get("ev_shift_pct", 0) * 0.2
    scenario["signal_opt_pct"] = scenario.get("signal_opt_pct", 0.0) + params.get("transit_boost_pct", 0) * 0.35
    if params.get("transit_boost_pct"):
        scenario["signal_edges"] = (scenario_payload or {}).get("signal_edges", ["e2", "e3"])
    if params.get("congestion_pricing_rupees"):
        scenario["reroute_penalty"] = scenario.get("reroute_penalty", 0.0) + params["congestion_pricing_rupees"] / 180.0
    # apply prompt signals
    scenario["ev_share_pct"] = signals.get("ev_share_pct", 0.0)
    scenario["demand_shift_pct"] += scenario["ev_share_pct"] * 0.35
    scenario["aqi_before_override"] = signals.get("aqi_before")
    scenario["aqi_after_target"] = signals.get("aqi_after")
    scenario["noise_before_override"] = signals.get("noise_before")
    scenario["noise_after_target"] = signals.get("noise_after")
    scenario["economic_focus"] = signals.get("economic_focus", False)
    scenario["health_focus"] = signals.get("health_focus", False)
    scenario["intent"] = (scenario_payload or {}).get("intervention", scenario.get("intent", "optimize"))
    scenario.setdefault("EF_g_per_km", 250.0)
    return scenario


def build_edges_geojson(
    baseline_result: Optional[Dict[str, Any]],
    candidate_result: Optional[Dict[str, Any]],
    ev_share_pct: float,
) -> Dict[str, Any]:
    node_map = {n: coord for n, _, coord in NODES}
    baseline_edges = {edge["id"]: edge for edge in (baseline_result or {}).get("edges", [])}
    candidate_edges = {edge["id"]: edge for edge in (candidate_result or {}).get("edges", [])}
    primary_path = (candidate_result or {}).get("path_edge_ids") or (baseline_result or {}).get("path_edge_ids") or []
    max_flow = max(
        [edge.get("flow", edge.get("baseline_flow", 0)) for edge in candidate_edges.values()] or [1.0]
    )
    features = []
    for e in EDGES_BASE:
        coords = [
            [node_map[e["u"]][0], node_map[e["u"]][1]],
            [node_map[e["v"]][0], node_map[e["v"]][1]],
        ]
        cand_edge = candidate_edges.get(e["id"]) or baseline_edges.get(e["id"]) or {}
        flow = cand_edge.get("flow", e.get("baseline_flow", 0))
        flow_ratio = flow / max_flow if max_flow else 0.0
        ev_local = clamp(ev_share_pct * (0.6 + 0.4 * flow_ratio), 0.0, 100.0)
        props = {
            "id": e["id"],
            "baseline_flow": e.get("baseline_flow", 0),
            "primary": e["id"] in primary_path,
            "ev_share": round(ev_local, 1),
            "flow": float(flow),
        }
        features.append({"type": "Feature", "properties": props, "geometry": {"type": "LineString", "coordinates": coords}})
    return {"type": "FeatureCollection", "features": features}


def build_infra_geojson(infra: List[Dict[str, Any]]) -> Dict[str, Any]:
    node_map = {n: coord for n, _, coord in NODES}
    features = []
    for suggestion in infra:
        new_edge = suggestion["params"]["new_edge"]
        coords = [
            [node_map[new_edge["u"]][0], node_map[new_edge["u"]][1]],
            [node_map[new_edge["v"]][0], node_map[new_edge["v"]][1]],
        ]
        features.append(
            {
                "type": "Feature",
                "properties": {"name": suggestion["name"], "desc": suggestion["desc"], "proposed": True},
                "geometry": {"type": "LineString", "coordinates": coords},
            }
        )
    return {"type": "FeatureCollection", "features": features}


def interpolate_series(start: float, end: float, steps: int = 12) -> List[float]:
    series = []
    if steps <= 1:
        return [round(end, 2)]
    for idx in range(steps):
        t = idx / (steps - 1)
        value = start - (start - end) * t * 0.85 + math.sin(idx / 2.2) * 0.2
        series.append(round(value, 2))
    return series


def build_kpis(baseline: Dict[str, Any], candidate: Dict[str, Any]) -> List[Dict[str, str]]:
    def pct_delta(base_value: float, new_value: float) -> float:
        if base_value <= 0:
            return 0.0
        return (base_value - new_value) / base_value * 100.0

    travel_delta = pct_delta(baseline["avg_travel_time_min"], candidate["avg_travel_time_min"])
    vkt_delta = pct_delta(baseline["total_vkt"], candidate["total_vkt"])
    pm_delta = pct_delta(baseline["pm25"], candidate["pm25"])
    co2_delta = pct_delta(baseline["co2_kg"], candidate["co2_kg"])
    return [
        {
            "label": "Avg travel time",
            "value": f"{candidate['avg_travel_time_min']:.1f} min",
            "delta": f"-{max(0.0, travel_delta):.1f}%",
            "confidence": "medium",
        },
        {
            "label": "Vehicle km traveled",
            "value": f"{candidate['total_vkt']:.0f} veh·km",
            "delta": f"-{max(0.0, vkt_delta):.1f}%",
            "confidence": "medium",
        },
        {
            "label": "PM2.5 corridor mean",
            "value": f"{candidate['pm25']:.1f} µg/m³",
            "delta": f"-{max(0.0, pm_delta):.1f}%",
            "confidence": "medium",
        },
        {
            "label": "CO₂ emissions",
            "value": f"{candidate['co2_kg']:.0f} kg",
            "delta": f"-{max(0.0, co2_delta):.1f}%",
            "confidence": "high",
        },
    ]


def pct_delta(base_value: float, new_value: float) -> float:
    if base_value <= 0:
        return 0.0
    return (base_value - new_value) / base_value * 100.0


def format_inr_crore(value: float) -> str:
    return f"₹{value:,.1f} Cr"


def build_impact_cards(
    baseline: Dict[str, Any],
    candidate: Dict[str, Any],
    signals: Dict[str, Any],
) -> List[Dict[str, Any]]:
    travel_delta = pct_delta(baseline["avg_travel_time_min"], candidate["avg_travel_time_min"])
    pm_delta = pct_delta(baseline["pm25"], candidate["pm25"])
    vkt_delta = pct_delta(baseline["total_vkt"], candidate["total_vkt"])
    co2_delta = pct_delta(baseline["co2_kg"], candidate["co2_kg"])
    ev_share = signals.get("ev_share_pct", 0.0)

    aqi_before = signals.get("aqi_before") or baseline["pm25"]
    aqi_after = signals.get("aqi_after") or candidate["pm25"]
    noise_before = signals.get("noise_before") or max(55.0, 72.0 - travel_delta * 0.2)
    noise_after = signals.get("noise_after") or clamp(noise_before - (ev_share * 0.35 + travel_delta * 0.25), 38.0, noise_before)

    fuel_savings_crore = max(0.0, (baseline["total_vkt"] - candidate["total_vkt"]) * 0.0002)
    health_savings_crore = max(0.0, pm_delta * 1.3)
    gdp_uplift_pct = clamp(ev_share * 0.12 + travel_delta * 0.4, -5.0, 12.0)
    gdp_delta_crore = max(0.0, 48000.0 * gdp_uplift_pct / 100.0)
    energy_shift_mwh = max(0.0, ev_share * 3.2)

    cards = [
        {
            "title": "Air quality",
            "metric": f"{aqi_before:.0f} → {aqi_after:.0f} µg/m³",
            "delta": f"-{max(0.0, pm_delta):.1f}%",
            "detail": "EV-heavy fleet slashes corridor particulate load.",
            "theme": "environment",
        },
        {
            "title": "Traffic noise",
            "metric": f"{noise_before:.0f} → {noise_after:.0f} dB",
            "delta": f"-{max(0.0, noise_before - noise_after):.1f} dB",
            "detail": "Battery drivetrains mute hub noise and calm intersections.",
            "theme": "environment",
        },
        {
            "title": "Mobility",
            "metric": f"{baseline['avg_travel_time_min']:.1f} → {candidate['avg_travel_time_min']:.1f} min",
            "delta": f"-{max(0.0, travel_delta):.1f}%",
            "detail": "Faster trunk travel unlocks dwell-time cuts across NH9.",
            "theme": "mobility",
        },
        {
            "title": "CO₂",
            "metric": f"{baseline['co2_kg']:.0f} → {candidate['co2_kg']:.0f} kg/day",
            "delta": f"-{max(0.0, co2_delta):.1f}%",
            "detail": "Cleaner grid mix and fewer VKT trim emissions.",
            "theme": "environment",
        },
        {
            "title": "Fuel + OPEX",
            "metric": format_inr_crore(fuel_savings_crore + health_savings_crore),
            "delta": f"-{max(0.0, vkt_delta):.1f}% VKT",
            "detail": "Lower combustion mileage frees recurring spend & health costs.",
            "theme": "finance",
        },
        {
            "title": "GDP throughput",
            "metric": f"+{gdp_uplift_pct:.1f}% ({format_inr_crore(gdp_delta_crore)})",
            "delta": f"+{gdp_uplift_pct:.1f}%",
            "detail": "Logistics windows widen as travel time compresses.",
            "theme": "economy",
        },
        {
            "title": "Energy mix",
            "metric": f"{energy_shift_mwh:.0f} MWh shifted to EV",
            "delta": f"{ev_share:.0f}% EV share",
            "detail": "Grid absorbs the new EV load across Vaishali/Noida depots.",
            "theme": "energy",
        },
    ]
    return cards


def rollup_domains(cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[str, Dict[str, Any]] = {}
    for card in cards:
        theme = card.get("theme", "other")
        bucket = buckets.setdefault(theme, {"theme": theme, "items": []})
        bucket["items"].append({"title": card["title"], "metric": card["metric"], "delta": card["delta"], "detail": card["detail"]})
    return list(buckets.values())


def build_narratives(signals: Dict[str, Any], travel_delta: float, pm_delta: float, gdp_uplift_pct: float) -> List[str]:
    ev_share = signals.get("ev_share_pct", 0.0)
    stories = []
    if ev_share:
        stories.append(
            f"About {ev_share:.0f}% of Noida's fleet goes electric, replacing roughly {int(ev_share * 550):,} combustion trips each day."
        )
    if pm_delta:
        stories.append(
            f"Particulate exposure along NH9 drops by {max(0.0, pm_delta):.1f}% once the EV surge hits the corridor."
        )
    if travel_delta:
        stories.append(
            f"Average corridor travel time compresses by {max(0.0, travel_delta):.1f}%, freeing delivery buffers for both logistics and transit."
        )
    if gdp_uplift_pct:
        stories.append(
            f"Regional GDP headroom improves by {gdp_uplift_pct:.1f}% as dependable travel times invite higher-value freight and services."
        )
    if not stories:
        stories.append("Scenario simulated without EV cues; add a target to shape narratives.")
    return stories


def build_map_overlays(baseline: Dict[str, Any], candidate: Dict[str, Any], best_name: str) -> Dict[str, Any]:
    travel_ratio = candidate["avg_travel_time_min"] / baseline["avg_travel_time_min"] if baseline["avg_travel_time_min"] else 1.0
    aqi_ratio = candidate["pm25"] / baseline["pm25"] if baseline["pm25"] else 1.0
    return {
        "trafficSpeedFactor": clamp(travel_ratio, 0.35, 1.1),
        "aqiFactor": clamp(aqi_ratio, 0.4, 1.1),
        "floodSeverity": clamp(0.25 + baseline["total_vkt"] / 20000.0, 0.2, 0.65),
        "aqiSummary": f"Applying {best_name} relieves AQI by ~{max(0.0, (1 - aqi_ratio) * 100):.1f}% across NH9.",
        "trafficSummary": f"Corridor travel time improves ~{max(0.0, (1 - travel_ratio) * 100):.1f}% vs baseline.",
    }


def build_explanations(ranked: List[Dict[str, Any]], infra: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    steps = []
    for candidate in ranked[:2]:
        steps.append(
            {
                "title": candidate["name"],
                "detail": f"{candidate['desc']} (ΔTT ≈ {candidate['pred_delta_tt_min']:.1f} min)",
            }
        )
    for suggestion in infra[:2]:
        steps.append(
            {
                "title": suggestion["name"],
                "detail": suggestion["desc"],
            }
        )
    return steps


def hash_prompt(prompt_text: str, scenario_payload: Optional[Dict[str, Any]]) -> str:
    raw = json.dumps({"prompt": prompt_text, "scenario": scenario_payload or {}}, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def scenario_feature_vector(scenario: Dict[str, Any]) -> List[float]:
    return [
        float(scenario.get("demand", 1200)),
        float(scenario.get("reroute_penalty", 0.0)),
        float(scenario.get("signal_opt_pct", 0.0)),
        float(scenario.get("bus_lane", 0)),
        float(scenario.get("bus_lane_pct", 0.0)),
        float(scenario.get("demand_shift_pct", 0.0)),
        float(scenario.get("weather_factor", 0.3)),
    ]


def pollution_feature_vector(vkt: float, scenario: Dict[str, Any]) -> List[float]:
    return [
        float(vkt),
        float(scenario.get("demand", 1200)),
        float(scenario.get("weather_factor", 0.3)),
        float(scenario.get("demand_shift_pct", 0.0)),
        float(scenario.get("signal_opt_pct", 0.0)),
    ]


def surrogate_available() -> bool:
    return all(models.get(key) is not None for key in ("traffic_tt", "traffic_vkt", "pollution"))


def enrich_with_surrogate_metrics(result: Dict[str, Any], scenario: Dict[str, Any]) -> Dict[str, Any]:
    if not surrogate_available():
        return result
    feat = np.array([scenario_feature_vector(scenario)], dtype=float)
    tt_pred = float(models["traffic_tt"].predict(feat)[0])
    vkt_pred = float(models["traffic_vkt"].predict(feat)[0])
    pm_feat = np.array([pollution_feature_vector(vkt_pred, scenario)], dtype=float)
    pm_pred = float(models["pollution"].predict(pm_feat)[0])
    enriched = dict(result)
    enriched["avg_travel_time_min"] = tt_pred
    enriched["total_vkt"] = vkt_pred
    enriched["pm25"] = pm_pred
    enriched["co2_kg"] = vkt_pred * float(scenario.get("EF_g_per_km", 250.0)) / 1000.0
    enriched["source"] = "surrogate"
    return enriched

# --- Model loading (surrogates & meta) ---
MODEL_PATHS = {
    "traffic_tt":"traffic_tt_model.pkl",
    "traffic_vkt":"traffic_vkt_model.pkl",
    "pollution":"pollution_model.pkl",
    "meta":"ldra_go_meta_model.pkl"
}
models = {}
for k,p in MODEL_PATHS.items():
    try:
        models[k] = joblib.load(p)
        print("Loaded model:", p)
    except Exception as e:
        print("Model not found / failed to load:", p, "->", str(e))
        models[k] = None


class LLMAdapter:
    """Lightweight wrapper that can call either heuristic templates or optional open-source LLMs."""

    def __init__(self) -> None:
        self.backend = os.getenv("OVERHAUL_LLM_BACKEND", "rules").lower()
        self.model_id = os.getenv("OVERHAUL_LLM_MODEL", "teknium/OpenHermes-2.5-Mistral")

    def _call_transformers(self, prompt: str) -> Optional[str]:
        try:
            from transformers import pipeline  # type: ignore

            generator = pipeline(
                "text-generation",
                model=self.model_id,
                max_new_tokens=220,
                do_sample=False,
            )
            result = generator(prompt, max_new_tokens=220)
            if result and isinstance(result, list):
                return result[0].get("generated_text", "").strip()
        except Exception as exc:  # pragma: no cover - optional dependency
            print("LLM backend failed, falling back to rules:", exc)
        return None

    def compose_summary(
        self,
        prompt: str,
        signals: Dict[str, Any],
        baseline_metrics: Dict[str, Any],
        candidate_metrics: Dict[str, Any],
        travel_delta: float,
        pm_delta: float,
        deep_bundle: Optional[Dict[str, Any]] = None,
        live_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        ev_phrase = f"{signals.get('ev_share_pct', 0):.0f}% EV adoption" if signals.get("ev_share_pct") else "Scenario"
        aqi_before = signals.get("aqi_before") or baseline_metrics["pm25"]
        aqi_after = signals.get("aqi_after") or candidate_metrics["pm25"]
        base_summary = (
            f"{ev_phrase} trims AQI from {aqi_before:.0f} to {aqi_after:.0f} µg/m³, "
            f"compresses travel times by {max(0.0, travel_delta):.1f}% and pulls particulates down {max(0.0, pm_delta):.1f}%."
        )
        live_snippets: List[str] = []
        if live_context:
            travel_live = live_context.get("travel")
            aqi_live = live_context.get("aqi")
            if travel_live:
                live_snippets.append(
                    f"Live corridor travel is {travel_live['travel_time_min']:.1f} min over {travel_live['distance_km']:.1f} km (OSRM)."
                )
            if aqi_live:
                live_snippets.append(
                    f"OpenAQ latest PM2.5 reads {aqi_live['latest_pm25']:.0f} µg/m³ (Δ{aqi_live['delta_pm25']:+.1f} over window)."
                )
        if live_snippets:
            base_summary += " " + " ".join(live_snippets)
        if deep_bundle and deep_bundle.get("facts"):
            fact = deep_bundle["facts"][0]
            base_summary += f" Historical check: {fact['detail']}"
        if self.backend == "transformers":
            crafted_prompt = (
                "You are LDRAGO, a transport systems orchestrator. Using the context below, craft a terse, factual"
                " executive update under 70 words.\n"
                f"Prompt: {prompt}\n"
                f"Baseline: {baseline_metrics}\nCandidate: {candidate_metrics}\n"
                f"Signals: {signals}\nDeep facts: {deep_bundle}\nSummary skeleton: {base_summary}\n"
            )
            generated = self._call_transformers(crafted_prompt)
            if generated:
                return generated
        return base_summary

    def compose_agent_dialogue(
        self,
        signals: Dict[str, Any],
        travel_delta: float,
        pm_delta: float,
        gdp_uplift_pct: float,
        facts: List[Dict[str, Any]],
    ) -> List[str]:
        ev_share = signals.get("ev_share_pct", 0.0)
        fact_excerpt = facts[0]["detail"] if facts else "Limited archival data available."
        return [
            f"TrafficModel → PollutionModel: Travel time delta {travel_delta:+.1f}% suggests particulate fall of {pm_delta:+.1f}%.",
            f"PollutionModel → FinanceModel: With {ev_share:.0f}% EV share, project operating savings plus GDP lift {gdp_uplift_pct:+.1f}%.",
            f"ArchiveModel → LDRAGO: {fact_excerpt}",
            "LDRAGO → User: Synthesizing surrogate, archive, and finance threads into a corridor briefing.",
        ]


class LDRagoController:
    def __init__(self, llm_adapter: LLMAdapter, historical_payload: Dict[str, Any]) -> None:
        self.llm = llm_adapter
        self.historical_payload = historical_payload
        self.records = historical_payload.get("records", [])
        self.city = historical_payload.get("city", "Corridor")
        self.deep_delay = float(os.getenv("OVERHAUL_DEEP_DELAY", "2.4"))

    async def handle_chat(self, req: "ChatRequest") -> Dict[str, Any]:
        prompt = (req.prompt or "").strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt text is required")

        start = time.time()
        started_at = datetime.utcnow().isoformat() + "Z"
        signals = extract_prompt_signals(prompt)
        scenario = scenario_from_payload(prompt, req.scenario, signals)
        live_context_task = asyncio.create_task(self.collect_live_context())
        baseline_sim = simulate_once(scenario)
        if baseline_sim is None:
            raise HTTPException(status_code=422, detail="Unable to solve baseline route for the supplied prompt")

        use_surrogates = req.mode.lower() != "deep" and surrogate_available()
        baseline_metrics = enrich_with_surrogate_metrics(baseline_sim, scenario) if use_surrogates else dict(baseline_sim)
        baseline_metrics["source"] = baseline_metrics.get("source", "simulation")

        live_context = await live_context_task
        if live_context.get("travel"):
            travel_live = live_context["travel"]
            baseline_metrics["avg_travel_time_min"] = travel_live.get("travel_time_min", baseline_metrics.get("avg_travel_time_min"))
            baseline_metrics["live_travel_distance_km"] = travel_live.get("distance_km")
            baseline_metrics["source"] = "live"
        if live_context.get("aqi"):
            aqi_live = live_context["aqi"]
            baseline_metrics["pm25"] = aqi_live.get("latest_pm25", baseline_metrics.get("pm25"))

        ranked = rank_candidates_meta(scenario)
        best_candidate = ranked[0] if ranked else None
        candidate_result = None
        if best_candidate:
            merged = dict(scenario)
            merged.update(best_candidate.get("params", {}))
            candidate_result = simulate_once(merged)
            if candidate_result and use_surrogates:
                candidate_result = enrich_with_surrogate_metrics(candidate_result, merged)
        candidate_result = candidate_result or dict(baseline_metrics)

        infra = propose_infrastructure_suggestions()       
        edges_geojson = build_edges_geojson(baseline_metrics, candidate_result, scenario.get("ev_share_pct", 0.0))
        infra_geojson = build_infra_geojson(infra)
        pollution_hotspots = build_pollution_hotspots_from_edges(edges_geojson)


        travel_delta = pct_delta(baseline_metrics["avg_travel_time_min"], candidate_result["avg_travel_time_min"])
        pm_delta = pct_delta(baseline_metrics["pm25"], candidate_result["pm25"])
        gdp_uplift_pct = clamp(signals.get("ev_share_pct", 0.0) * 0.12 + travel_delta * 0.4, -5.0, 12.0)
        deep_bundle = None
        if req.mode.lower() == "deep":
            deep_bundle = await self.run_deep_chain(
                prompt,
                signals,
                scenario,
                baseline_metrics,
                candidate_result,
                travel_delta,
                pm_delta,
                gdp_uplift_pct,
            )

        impact_cards = build_impact_cards(baseline_metrics, candidate_result, signals)
        domain_rollups = rollup_domains(impact_cards)
        narratives = build_narratives(signals, travel_delta, pm_delta, gdp_uplift_pct)
        summary = self.llm.compose_summary(
            prompt,
            signals,
            baseline_metrics,
            candidate_result,
            travel_delta,
            pm_delta,
            deep_bundle,
            live_context,
        )

        mode = "deep" if req.mode.lower() == "deep" else "fast"
        completed_at = datetime.utcnow().isoformat() + "Z"
        outputs = {
            "tldr": summary,
            "confidenceLevel": "high" if mode == "deep" else "medium",
            "impactCards": impact_cards,
            "domains": domain_rollups,
            "narrative": narratives,
            "explanation": build_explanations(ranked, infra),
            "mapOverlays": build_map_overlays(baseline_metrics, candidate_result, best_candidate["name"] if best_candidate else "Plan"),
            "logs": [
                "Parsed natural language prompt",
                "Simulated baseline corridor",
                "Ranked candidate controls",
                "Generated overlays & KPIs",
            ],
            "started_at": started_at,
            "completed_at": completed_at,
            "liveContext": live_context,
        }
        if deep_bundle:
            outputs["deepFacts"] = deep_bundle.get("facts", [])
            outputs["multiAgentDigest"] = deep_bundle.get("dialogue", [])
            outputs["sources"] = deep_bundle.get("sources", [])

        manifest = {
            "run_id": str(uuid.uuid4()),
            "mode": mode,
            "prompt": prompt,
            "scenario_hash": hash_prompt(prompt, req.scenario),
            "scenario_snapshot": req.scenario or {},
            "started_at": started_at,
            "completed_at": completed_at,
            "baseline_metrics": {
                "avg_travel_time_min": baseline_metrics["avg_travel_time_min"],
                "total_vkt": baseline_metrics["total_vkt"],
                "pm25": baseline_metrics["pm25"],
                "co2_kg": baseline_metrics["co2_kg"],
            },
            "candidate": best_candidate["name"] if best_candidate else None,
            "impacts": impact_cards,
            "live_sources": live_context.get("sources", []),
            "runtime_s": time.time() - start,
            "timestamp": time.time(),
        }

        return {
            "summary": summary,
            "baseline": baseline_metrics,
            "ranked": ranked,
            "edges_geojson": edges_geojson,
            "infrastructure": infra_geojson,
            "pollution_hotspots": pollution_hotspots,
            "live": live_context,
            "manifest": manifest,
            "outputs": outputs,
        }

    async def run_deep_chain(
        self,
        prompt: str,
        signals: Dict[str, Any],
        scenario: Dict[str, Any],
        baseline_metrics: Dict[str, Any],
        candidate_metrics: Dict[str, Any],
        travel_delta: float,
        pm_delta: float,
        gdp_uplift_pct: float,
    ) -> Dict[str, Any]:
        await asyncio.sleep(self.deep_delay)
        facts = self.build_historical_facts()
        dialogue = self.llm.compose_agent_dialogue(signals, travel_delta, pm_delta, gdp_uplift_pct, facts)
        sources = [
            {
                "name": f"{self.city} transport archives",
                "detail": "Derived from historical_metrics.json (local mirror of open data)",
            }
        ]
        return {"facts": facts, "dialogue": dialogue, "sources": sources}

    def build_historical_facts(self) -> List[Dict[str, Any]]:
        facts: List[Dict[str, Any]] = []
        if not self.records:
            return facts
        pm_stats = historical_stats("pm25", self.records)
        noise_stats = historical_stats("noise_db", self.records)
        gdp_stats = historical_stats("gdp_crore", self.records)
        if pm_stats:
            facts.append(
                {
                    "title": "PM2.5 trend",
                    "stat": f"{pm_stats['start_value']:.0f} → {pm_stats['end_value']:.0f} µg/m³",
                    "detail": f"{self.city} PM2.5 averages shifted {pm_stats['delta']:+.1f} µg/m³ from {pm_stats['start_year']} to {pm_stats['end_year']}.",
                    "source": "City air quality archive",
                }
            )
        if noise_stats:
            facts.append(
                {
                    "title": "Noise corridor",
                    "stat": f"{noise_stats['start_value']:.0f} → {noise_stats['end_value']:.0f} dB",
                    "detail": f"Ambient road noise dropped {noise_stats['delta']:+.1f} dB across {noise_stats['start_year']}-{noise_stats['end_year']} checkposts.",
                    "source": "Noise monitoring grid",
                }
            )
        if gdp_stats:
            facts.append(
                {
                    "title": "Logistics GDP",
                    "stat": f"₹{gdp_stats['start_value']:.0f}Cr → ₹{gdp_stats['end_value']:.0f}Cr",
                    "detail": f"Regional gross output climbed {gdp_stats['delta']:+.0f} Cr in {self.city} over that archive window.",
                    "source": "State economic survey",
                }
            )
        return facts

    async def collect_live_context(self) -> Dict[str, Any]:
        """Collect live travel and AQI data to ground the response."""
        travel_task = asyncio.create_task(fetch_osrm_corridor_metrics())
        aqi_task = asyncio.create_task(fetch_aqi_history(28.62, 77.35))
        travel_raw, aqi_raw = await asyncio.gather(travel_task, aqi_task, return_exceptions=True)

        live_travel = None
        live_aqi = None
        sources: List[Dict[str, str]] = []

        if not isinstance(travel_raw, Exception) and travel_raw:
            live_travel = travel_raw
            sources.append({
                "name": "OSRM",
                "detail": "router.project-osrm.org live routing"
            })

        if not isinstance(aqi_raw, Exception) and aqi_raw:
            aqi_summary = summarize_aqi_results(aqi_raw)
            if aqi_summary:
                live_aqi = aqi_summary
                sources.append({
                    "name": "OpenAQ",
                    "detail": "Latest PM2.5 monitors within 15km"
                })

        return {"travel": live_travel, "aqi": live_aqi, "sources": sources}


llm_adapter = LLMAdapter()
ldrago_controller = LDRagoController(llm_adapter, HISTORICAL_DATA)
traffic_god_service = TrafficGodService()

# --- LDRAGO orchestration ---
def interpret_prompt_to_scenario(prompt_text):
    # Very simple NL->scenario parser: detect demands and locations. For prototype we assume A->F OD.
    scenario = {
        "demand": 1200,
        "reroute_penalty":0.0,
        "signal_opt_pct":0,
        "bus_lane":0,
        "bus_lane_pct":0,
        "demand_shift_pct":0,
        "weather_factor":0.3,
        "od_from":"A","od_to":"F"
    }
    # keywords
    if "reduce" in prompt_text.lower() or "eliminate" in prompt_text.lower():
        scenario['intent']="reduce_congestion"
    # parse numeric demand if provided
    import re
    m = re.search(r'(\d{3,4})\s*veh', prompt_text)
    if m:
        scenario['demand'] = int(m.group(1))
    return scenario

def rank_candidates_meta(scenario):
    # use meta model if available; else fallback to simulated delta ranking
    candidates = propose_candidates()
    ranked=[]
    for c in candidates:
        feat = [scenario['demand'], scenario['reroute_penalty'], scenario['signal_opt_pct'],
                scenario['bus_lane'], scenario['bus_lane_pct'], scenario['demand_shift_pct'],
                scenario['weather_factor'], 0]
        # candidate code index: map name to small int
        cand_names = [x['name'] for x in candidates]
        feat[-1] = cand_names.index(c['name'])
        if models.get('meta') is not None:
            pred = float(models['meta'].predict(np.array(feat).reshape(1,-1))[0])
        else:
            # fallback: run simulate_once baseline and candidate and compute delta
            base = simulate_once(scenario)
            p = dict(scenario); p.update(c['params']); out = simulate_once(p)
            pred = base['avg_travel_time_min'] - out['avg_travel_time_min']
        ranked.append({"name":c['name'], "desc": c['desc'], "pred_delta_tt_min": pred})
    ranked = sorted(ranked, key=lambda x:-x['pred_delta_tt_min'])
    return ranked

# --- Live internet / external data helpers ---

OPENAQ_BASE = "https://api.openaq.org/v2"

async def fetch_aqi_history(lat: float, lon: float, radius_m: int = 15000) -> Dict[str, Any]:
    """
    Example: fetch recent PM2.5 measurements from OpenAQ near a coordinate.
    You can change this to any AQI provider you prefer.
    """
    params = {
        "coordinates": f"{lat},{lon}",
        "radius": radius_m,
        "parameter": "pm25",
        "order_by": "datetime",
        "sort": "asc",
        "limit": 200,
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.get(f"{OPENAQ_BASE}/measurements", params=params)
        resp.raise_for_status()
        return resp.json()


def summarize_aqi_results(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Reduce OpenAQ measurements to a small stats bundle."""
    results = raw.get("results", []) if raw else []
    if not results:
        return None
    series = []
    for entry in results:
        value = entry.get("value")
        stamp = entry.get("date", {}).get("utc")
        if value is None or stamp is None:
            continue
        series.append({"datetime": stamp, "pm25": float(value)})
    if not series:
        return None
    series = sorted(series, key=lambda x: x["datetime"])  # ensure ascending
    latest = series[-1]
    values = [pt["pm25"] for pt in series]
    mean_val = sum(values) / len(values)
    trend = values[-1] - values[0]
    return {
        "latest_pm25": float(latest["pm25"]),
        "latest_timestamp": latest["datetime"],
        "mean_pm25": round(mean_val, 2),
        "delta_pm25": round(trend, 2),
        "series": series[-48:],  # cap to roughly two days of hourly data
    }


async def fetch_osrm_corridor_metrics(
    origin: Tuple[float, float] = CORRIDOR_ORIGIN,
    dest: Tuple[float, float] = CORRIDOR_DEST,
) -> Optional[Dict[str, Any]]:
    """Fetch live travel time/distance and geometry for the corridor via OSRM."""
    origin_lat, origin_lon = origin
    dest_lat, dest_lon = dest
    coords = f"{origin_lon},{origin_lat};{dest_lon},{dest_lat}"
    params = {"overview": "full", "geometries": "geojson"}
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(f"{OSRM_BASE_URL}/route/v1/driving/{coords}", params=params)
        resp.raise_for_status()
        data = resp.json()
    routes = data.get("routes", [])
    if not routes:
        return None
    route = routes[0]
    duration_min = route.get("duration", 0.0) / 60.0
    distance_km = route.get("distance", 0.0) / 1000.0
    geometry = route.get("geometry") or {}
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": "Sector 78 → Vasundhara",
                    "source": "OSRM public routing service",
                },
                "geometry": geometry,
            }
        ],
    }
    return {
        "travel_time_min": round(duration_min, 2),
        "distance_km": round(distance_km, 2),
        "geojson": geojson,
        "source": "router.project-osrm.org",
        "fetched_at": datetime.utcnow().isoformat() + "Z",
    }


async def fetch_demo_route_geojson() -> Dict[str, Any]:
    """
    Stub for a live route between Sector 78 and Vasundhara.

    Replace this with a real routing call (e.g., OSRM, OpenRouteService, or
    Overpass+custom routing) and return their geometry as GeoJSON.
    """
    # Approximate coordinates only (demo)
    coords = [
        [77.302, 28.590],  # Sector 78-ish
        [77.307, 28.585],
        [77.314, 28.586],
        [77.316, 28.584],  # Vasundhara-ish
    ]
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": "Sector 78 → Vasundhara",
                    "source": "demo_stub"
                },
                "geometry": {"type": "LineString", "coordinates": coords},
            }
        ],
    }


def build_pollution_hotspots_from_edges(edges_geojson: Dict[str, Any]) -> Dict[str, Any]:
    """
    Derive translucent 'pollution circles' from current edges by using flow as a proxy.
    Later you can replace this with real AQI exposure values per segment.
    """
    features = []
    for feat in edges_geojson.get("features", []):
        props = feat.get("properties", {})
        geom = feat.get("geometry", {})
        if geom.get("type") != "LineString":
            continue
        coords = geom.get("coordinates") or []
        if not coords:
            continue
        # Use midpoint of the line
        mid_idx = len(coords) // 2
        mid = coords[mid_idx]
        flow = float(props.get("flow", props.get("baseline_flow", 0.0)))
        if flow <= 0:
            continue
        intensity = min(1.0, flow / 2000.0)  # simple scaling
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "intensity": intensity,
                    "edge_id": props.get("id"),
                },
                "geometry": {"type": "Point", "coordinates": mid},
            }
        )
    return {"type": "FeatureCollection", "features": features}


# --- FastAPI app & endpoints ---
app = FastAPI(title="OVERHAUL Prototype API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi import Query

@app.get("/live/aqi")
async def live_aqi(
    lat: float = Query(28.62, description="Latitude near corridor"),
    lon: float = Query(77.35, description="Longitude near corridor"),
):
    """
    Live AQI series near the corridor (example using OpenAQ).
    """
    try:
        raw = await fetch_aqi_history(lat, lon)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"AQI provider error: {exc}") from exc

    results = raw.get("results", [])
    series = [
        {
            "datetime": r.get("date", {}).get("utc"),
            "pm25": r.get("value"),
        }
        for r in results
        if r.get("value") is not None
    ]
    return {"series": series}


@app.get("/live/route")
async def live_route():
    """
    Live route geometry for the corridor.

    Currently returns a demo stub. Replace with a real routing API call
    inside fetch_demo_route_geojson() when you are ready.
    """
    geojson = await fetch_demo_route_geojson()
    return geojson


@app.get("/health")
async def healthcheck():
    return {"status": "ok", "timestamp": time.time()}

class ChatRequest(BaseModel):
    prompt: str
    mode: str = "fast"
    scenario: Optional[Dict[str, Any]] = None


class TrafficGodPerceptionRequest(BaseModel):
    video_path: str
    output_csv: Optional[str] = None
    dry_run: bool = False

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    return await ldrago_controller.handle_chat(req)

@app.post("/run_candidate")
async def run_candidate(req: Request):
    body = await req.json()
    candidate_name = body.get('candidate')
    scenario = body.get('scenario', {"demand":1200,"weather_factor":0.3,"od_from":"A","od_to":"F"})
    # run full sim
    candidates = propose_candidates()
    cand = next((c for c in candidates if c['name']==candidate_name), None)
    if cand is None:
        return {"error":"candidate not found"}
    params = dict(scenario); params.update(cand['params']); params['od_from']='A'; params['od_to']='F'
    out = simulate_once(params)
    manifest = {"run_id":str(uuid.uuid4()), "candidate":candidate_name, "params":cand['params'], "timestamp":time.time()}
    # also compute geojson for chosen path (edges)
    path_edges = out['path_edge_ids'] if out else []
    node_map = {n:coord for n,_,coord in NODES}
    features=[]
    for e in EDGES_BASE:
        coords = [ [node_map[e['u']][0], node_map[e['u']][1]], [node_map[e['v']][0], node_map[e['v']][1]] ]
        feat={"type":"Feature","properties":{"id":e['id'], "in_path": e['id'] in path_edges}, "geometry":{"type":"LineString","coordinates":coords}}
        features.append(feat)
    return {"result": out, "manifest": manifest, "geojson": {"type":"FeatureCollection","features":features}}


@app.post("/traffic-god/perception")
async def traffic_god_perception(req: TrafficGodPerceptionRequest):
    """Trigger the traffic-god perception pipeline from OVERHAUL."""
    try:
        result = traffic_god_service.run_perception(
            video_path=req.video_path,
            output_csv=req.output_csv,
            dry_run=req.dry_run,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"traffic-god failure: {exc}") from exc

    return {"status": "ok", "pipeline": "perception", "result": result}

@app.post("/apply_infra")
async def apply_infra(req: Request):
    body = await req.json()
    infra_name = body.get('infra')  # e.g., "Bypass_C_to_F"
    infra_list = propose_infrastructure_suggestions()
    infra_obj = next((i for i in infra_list if i['name']==infra_name), None)
    if infra_obj is None:
        return {"error":"infra not found"}
    # create a new edges set with the new edge appended and simulate
    new_edge = infra_obj['params']['new_edge']
    edges_new = EDGES_BASE + [new_edge]
    scenario = body.get('scenario', {"demand":1200,"weather_factor":0.3,"od_from":"A","od_to":"F"})
    # run simple simulate with new edge included (re-using simulate_once but passing edges_base param)
    out = simulate_once(scenario, edges_base=edges_new)
    node_map = {n:coord for n,_,coord in NODES}
    # produce geojson with the new edge highlighted
    features=[]
    for e in edges_new:
        coords = [[node_map[e['u']][0], node_map[e['u']][1]], [node_map[e['v']][0], node_map[e['v']][1]]]
        props = {"id":e['id']}
        if e['id'] == new_edge['id']:
            props['proposed']=True
        features.append({"type":"Feature","properties":props,"geometry":{"type":"LineString","coordinates":coords}})
    manifest = {"run_id":str(uuid.uuid4()), "infra":infra_name, "timestamp":time.time()}
    return {"result": out, "manifest": manifest, "geojson": {"type":"FeatureCollection","features":features}}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
