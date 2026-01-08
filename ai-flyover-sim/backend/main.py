from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import math
import requests

import osm, rules, geometry, gemini

app = FastAPI(title="AI Flyover Planner")

# Add CORS middleware to allow frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PlanRequest(BaseModel):
    prompt: str


class PlanResponse(BaseModel):
    flyover_path: List[List[float]]
    pillars: List[List[float]]
    flyover_width_m: float
    flyover_height_m: float
    flyover_lanes: int
    pillar_spacing_m: float
    offset_m: float
    road: Dict[str, Any]
    traffic: Dict[str, Any]
    design_reasoning: str


def simple_geocode(query: str):
    """Geocode a location query to lat/lon using Nominatim."""
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": query, "format": "json", "limit": 1}
    resp = requests.get(url, params=params, headers={"User-Agent": "ai-flyover"}, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        raise HTTPException(status_code=404, detail=f"Location not found: {query}")
    return float(data[0]["lat"]), float(data[0]["lon"])


def extract_location_from_prompt(prompt: str) -> str:
    """Extract location name from a natural language prompt."""
    prompt_lower = prompt.lower()
    
    # Remove common prefixes
    prefixes = [
        "build a flyover at ", "build flyover at ", "build a flyover in ", "build flyover in ",
        "create a flyover at ", "create flyover at ", "plan a flyover at ", "plan flyover at ",
        "construct a flyover at ", "make a flyover at ", "design a flyover at ",
        "build a flyover near ", "flyover at ", "flyover in ", "flyover near ",
        "at ", "in ", "near ",
    ]
    
    location = prompt
    for prefix in prefixes:
        if prompt_lower.startswith(prefix):
            location = prompt[len(prefix):]
            break
    
    return location.strip()


def split_prompt(prompt: str):
    """Split prompt into start and end locations."""
    prompt_lower = prompt.lower()
    
    # Check for "from X to Y" pattern
    if " from " in prompt_lower and " to " in prompt_lower:
        from_idx = prompt_lower.find(" from ")
        to_idx = prompt_lower.find(" to ", from_idx)
        start = prompt[from_idx + 6:to_idx].strip()
        end = prompt[to_idx + 4:].strip()
        return start, end
    
    # Check for simple "X to Y" pattern
    if " to " in prompt_lower:
        parts = prompt.lower().split(" to ", 1)
        # Clean up the start part (remove command prefixes)
        start = extract_location_from_prompt(parts[0])
        end = parts[1].strip()
        return start, end
    
    # Single location - use it for both start and end
    location = extract_location_from_prompt(prompt)
    return location, location


@app.post("/plan", response_model=PlanResponse)
def plan(req: PlanRequest):
    start_text, end_text = split_prompt(req.prompt)
    start_lat, start_lon = simple_geocode(start_text)
    end_lat, end_lon = simple_geocode(end_text)

    start_way = osm.query_nearest_road(start_lat, start_lon)
    if not start_way:
        raise HTTPException(status_code=404, detail="No road found near start point")

    highway_type = start_way.get("tags", {}).get("highway", "secondary")
    lanes = rules.infer_lanes(highway_type, start_way.get("tags", {}))
    road_w = rules.road_width_m(lanes)

    facts = {
        "road_type": highway_type,
        "lanes": lanes,
        "road_width_m": road_w,
        "prompt": req.prompt,
        "area_type": start_way.get("tags", {}).get("area", "urban"),
    }
    gem = gemini.call_gemini(facts)

    flyover_lanes = int(gem.get("flyover_lanes", 2))
    flyover_w = rules.flyover_width_m(flyover_lanes)
    pillar_spacing = float(gem.get("pillar_spacing_m", 30))
    flyover_h = rules.clearance_height()

    road_path = osm.extract_centerline(start_way)
    flyover_path, pillars = geometry.build_flyover_geometry(road_path, road_w, flyover_w, pillar_spacing)
    offset_val = rules.offset_distance_m(road_w, flyover_w)

    traffic = rules.compute_traffic_metrics(
        highway_type,
        gem.get("traffic_assumption", {}).get("congestion_reduction_percent", 20),
    )

    return PlanResponse(
        flyover_path=flyover_path,
        pillars=pillars,
        flyover_width_m=round(flyover_w, 2),
        flyover_height_m=flyover_h,
        flyover_lanes=flyover_lanes,
        pillar_spacing_m=round(pillar_spacing, 1),
        offset_m=round(offset_val, 2),
        road={
            "type": highway_type,
            "lanes": lanes,
            "width_m": round(road_w, 2),
            "tags": start_way.get("tags", {}),
        },
        traffic=traffic,
        design_reasoning=gem.get("design_reasoning", ""),
    )


class PlanCoordsRequest(BaseModel):
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    route_coords: List[List[float]] = []


class InfrastructureAnalysis(BaseModel):
    existing_bridges: List[Dict[str, Any]] = []
    tunnels: List[Dict[str, Any]] = []
    junctions: List[Dict[str, Any]] = []
    railways: List[Dict[str, Any]] = []
    avoidance_zones: List[Dict[str, Any]] = []


class SmartPlanResponse(BaseModel):
    flyover_path: List[List[float]]
    pillars: List[Dict[str, Any]]
    flyover_width_m: float
    flyover_height_m: float
    flyover_lanes: int
    pillar_spacing_m: float
    offset_m: float
    road: Dict[str, Any]
    traffic: Dict[str, Any]
    design_reasoning: str
    infrastructure: InfrastructureAnalysis
    architect_notes: List[str]


@app.post("/plan-coords")
def plan_coords(req: PlanCoordsRequest):
    """
    Smart flyover planning that analyzes existing infrastructure:
    - Existing flyovers/bridges to connect or avoid
    - Underpasses/tunnels
    - Major junctions where pillars can't be placed
    - Railway crossings requiring special clearance
    - Buildings to avoid
    """
    
    architect_notes = []
    architect_notes.append("Initiating terrain and infrastructure analysis...")
    
    # Query road data at the start point
    start_way = osm.query_nearest_road(req.start_lat, req.start_lon)
    if not start_way:
        start_way = {"tags": {"highway": "secondary", "lanes": "2"}}
        architect_notes.append("No OSM road data found - using secondary road defaults")
    
    highway_type = start_way.get("tags", {}).get("highway", "secondary")
    lanes = rules.infer_lanes(highway_type, start_way.get("tags", {}))
    road_w = rules.road_width_m(lanes)
    
    architect_notes.append(f"Road classification: {highway_type} ({lanes} lanes, {road_w}m wide)")
    
    # Calculate distance
    dist_km = math.sqrt(
        (req.end_lat - req.start_lat)**2 + (req.end_lon - req.start_lon)**2
    ) * 111
    
    architect_notes.append(f"Route distance: {dist_km:.2f} km")
    
    # SMART ANALYSIS: Query existing infrastructure along route
    infrastructure_data = {"bridges": [], "tunnels": [], "junctions": [], "railways": [], "buildings": []}
    
    if req.route_coords and len(req.route_coords) > 1:
        architect_notes.append("Scanning route corridor for existing infrastructure...")
        try:
            infrastructure_data = osm.query_infrastructure_along_route(req.route_coords)
            
            if infrastructure_data["bridges"]:
                architect_notes.append(f"⚠️ Found {len(infrastructure_data['bridges'])} existing bridges/flyovers")
                for bridge in infrastructure_data["bridges"][:3]:
                    architect_notes.append(f"  → {bridge.get('name', 'Bridge')} - will design connection/clearance")
            
            if infrastructure_data["tunnels"]:
                architect_notes.append(f"✓ Found {len(infrastructure_data['tunnels'])} underpasses - flyover can span above")
            
            if infrastructure_data["junctions"]:
                architect_notes.append(f"⚠️ Found {len(infrastructure_data['junctions'])} major intersections - avoiding pillar placement")
            
            if infrastructure_data["railways"]:
                architect_notes.append(f"⚠️ Found {len(infrastructure_data['railways'])} railway crossings - requiring 6.5m+ clearance")
            
        except Exception as e:
            architect_notes.append(f"Infrastructure scan partial: {str(e)[:50]}")
    
    # Build facts for AI with infrastructure context
    facts = {
        "road_type": highway_type,
        "lanes": lanes,
        "road_width_m": road_w,
        "prompt": f"Flyover from [{req.start_lat:.4f}, {req.start_lon:.4f}] to [{req.end_lat:.4f}, {req.end_lon:.4f}]",
        "area_type": start_way.get("tags", {}).get("area", "urban"),
        "distance_km": round(dist_km, 2),
        "existing_bridges": len(infrastructure_data.get("bridges", [])),
        "junctions_to_avoid": len(infrastructure_data.get("junctions", [])),
        "railway_crossings": len(infrastructure_data.get("railways", [])),
    }
    
    # Call Gemini AI for intelligent design decisions
    gem = gemini.call_gemini(facts)
    
    flyover_lanes = int(gem.get("flyover_lanes", 2 if dist_km < 2 else 4))
    flyover_w = rules.flyover_width_m(flyover_lanes)
    base_pillar_spacing = float(gem.get("pillar_spacing_m", 30))
    flyover_h = rules.clearance_height()
    
    # Increase height if railway crossings present
    if infrastructure_data.get("railways"):
        flyover_h = max(flyover_h, 7.5)  # IRC requirement for rail crossings
        architect_notes.append(f"Adjusting clearance to {flyover_h}m for railway safety")
    
    architect_notes.append(f"Design: {flyover_lanes}-lane flyover, {flyover_w}m deck, {flyover_h}m clearance")
    
    # SMART PILLAR PLACEMENT
    smart_pillars = []
    avoidance_zones = []
    
    if req.route_coords and len(req.route_coords) > 1:
        architect_notes.append("Computing intelligent pillar positions...")
        
        smart_pillars, avoidance_zones = osm.compute_smart_pillar_positions(
            req.route_coords,
            infrastructure_data,
            target_spacing_m=base_pillar_spacing,
            min_spacing_m=20,
            max_spacing_m=50
        )
        
        # Count different pillar types
        anchor_count = sum(1 for p in smart_pillars if p["type"] == "anchor")
        standard_count = sum(1 for p in smart_pillars if p["type"] == "standard")
        constrained_count = sum(1 for p in smart_pillars if p["type"] == "constrained")
        
        architect_notes.append(f"Pillar plan: {len(smart_pillars)} total ({anchor_count} anchors, {standard_count} standard, {constrained_count} constrained)")
        
        if avoidance_zones:
            architect_notes.append(f"Avoided {len(avoidance_zones)} obstacle zones")
        
        # Convert to simple format for response
        pillars_simple = [p["coords"] for p in smart_pillars]
    else:
        # Fallback to simple path
        road_path = [[req.start_lat, req.start_lon], [req.end_lat, req.end_lon]]
        _, pillars_simple = geometry.build_flyover_geometry(road_path, road_w, flyover_w, base_pillar_spacing)
        smart_pillars = [{"coords": p, "type": "standard", "id": i+1, "reason": "Standard placement"} for i, p in enumerate(pillars_simple)]
    
    offset_val = rules.offset_distance_m(road_w, flyover_w)
    
    traffic = rules.compute_traffic_metrics(
        highway_type,
        gem.get("traffic_assumption", {}).get("congestion_reduction_percent", 20),
    )
    
    # Build flyover path
    if req.route_coords and len(req.route_coords) > 1:
        flyover_path = req.route_coords  # Use the driving route
    else:
        flyover_path = [[req.start_lon, req.start_lat], [req.end_lon, req.end_lat]]
    
    design_reasoning = gem.get("design_reasoning", "")
    if not design_reasoning:
        design_reasoning = f"Designed {flyover_lanes}-lane flyover spanning {dist_km:.1f}km with {len(smart_pillars)} pillars. "
        if infrastructure_data.get("junctions"):
            design_reasoning += f"Avoided {len(infrastructure_data['junctions'])} intersections for pillar placement. "
        if infrastructure_data.get("bridges"):
            design_reasoning += f"Accounted for {len(infrastructure_data['bridges'])} existing bridges. "
    
    architect_notes.append("Design complete - ready for visualization")
    
    return {
        "flyover_path": flyover_path,
        "pillars": smart_pillars,
        "flyover_width_m": round(flyover_w, 2),
        "flyover_height_m": flyover_h,
        "flyover_lanes": flyover_lanes,
        "pillar_spacing_m": round(base_pillar_spacing, 1),
        "offset_m": round(offset_val, 2),
        "road": {
            "type": highway_type,
            "lanes": lanes,
            "width_m": round(road_w, 2),
            "tags": start_way.get("tags", {}),
        },
        "traffic": traffic,
        "design_reasoning": design_reasoning,
        "infrastructure": {
            "existing_bridges": infrastructure_data.get("bridges", [])[:5],
            "tunnels": infrastructure_data.get("tunnels", [])[:5],
            "junctions": infrastructure_data.get("junctions", [])[:10],
            "railways": infrastructure_data.get("railways", [])[:5],
            "avoidance_zones": avoidance_zones[:10],
        },
        "architect_notes": architect_notes,
    }


@app.get("/health")
def health():
    return {"status": "ok"}
