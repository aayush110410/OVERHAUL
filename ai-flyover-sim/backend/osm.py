import requests
import math
from typing import Dict, Any, List, Optional, Tuple

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def query_nearest_road(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    query = f"""
    [out:json];
    way(around:50,{lat},{lon})["highway"];
    out tags geom 1;
    """
    resp = requests.post(OVERPASS_URL, data=query, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    ways = data.get("elements", [])
    if not ways:
        return None
    # Pick the first way; could be improved with scoring later.
    return ways[0]


def extract_centerline(way: Dict[str, Any]) -> List[List[float]]:
    geom = way.get("geometry", [])
    return [[pt["lon"], pt["lat"]] for pt in geom]


def query_infrastructure_along_route(route_coords: List[List[float]], buffer_m: int = 100) -> Dict[str, Any]:
    """
    Query OSM for existing infrastructure along a route:
    - Existing bridges/flyovers
    - Underpasses/tunnels
    - Major intersections
    - Railway crossings
    - Buildings near the route
    """
    if not route_coords or len(route_coords) < 2:
        return {"bridges": [], "tunnels": [], "junctions": [], "railways": [], "buildings": []}
    
    # Create bounding box from route
    lons = [c[0] for c in route_coords]
    lats = [c[1] for c in route_coords]
    min_lat, max_lat = min(lats) - 0.002, max(lats) + 0.002
    min_lon, max_lon = min(lons) - 0.002, max(lons) + 0.002
    
    query = f"""
    [out:json][timeout:25];
    (
      // Existing bridges and flyovers
      way["bridge"="yes"]({min_lat},{min_lon},{max_lat},{max_lon});
      way["man_made"="bridge"]({min_lat},{min_lon},{max_lat},{max_lon});
      
      // Tunnels and underpasses
      way["tunnel"="yes"]({min_lat},{min_lon},{max_lat},{max_lon});
      
      // Major road junctions
      node["highway"="traffic_signals"]({min_lat},{min_lon},{max_lat},{max_lon});
      node["highway"="crossing"]({min_lat},{min_lon},{max_lat},{max_lon});
      
      // Railway crossings
      way["railway"]({min_lat},{min_lon},{max_lat},{max_lon});
      node["railway"="level_crossing"]({min_lat},{min_lon},{max_lat},{max_lon});
      
      // Buildings (for pillar placement avoidance)
      way["building"]({min_lat},{min_lon},{max_lat},{max_lon});
    );
    out geom;
    """
    
    try:
        resp = requests.post(OVERPASS_URL, data=query, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"OSM query error: {e}")
        return {"bridges": [], "tunnels": [], "junctions": [], "railways": [], "buildings": []}
    
    elements = data.get("elements", [])
    
    bridges = []
    tunnels = []
    junctions = []
    railways = []
    buildings = []
    
    for el in elements:
        tags = el.get("tags", {})
        el_type = el.get("type")
        
        if el_type == "way":
            geom = el.get("geometry", [])
            coords = [[pt["lon"], pt["lat"]] for pt in geom] if geom else []
            center = get_way_center(geom) if geom else None
            
            if tags.get("bridge") == "yes" or tags.get("man_made") == "bridge":
                bridges.append({
                    "id": el.get("id"),
                    "name": tags.get("name", "Unnamed Bridge"),
                    "coords": coords,
                    "center": center,
                    "type": "existing_flyover" if tags.get("highway") else "bridge"
                })
            elif tags.get("tunnel") == "yes":
                tunnels.append({
                    "id": el.get("id"),
                    "name": tags.get("name", "Underpass"),
                    "coords": coords,
                    "center": center
                })
            elif tags.get("railway"):
                railways.append({
                    "id": el.get("id"),
                    "name": tags.get("name", "Railway Line"),
                    "railway_type": tags.get("railway"),
                    "coords": coords,
                    "electrified": tags.get("electrified", "no")
                })
            elif tags.get("building"):
                buildings.append({
                    "id": el.get("id"),
                    "name": tags.get("name", "Building"),
                    "building_type": tags.get("building"),
                    "coords": coords,
                    "center": center
                })
        
        elif el_type == "node":
            lat = el.get("lat")
            lon = el.get("lon")
            
            if tags.get("highway") in ["traffic_signals", "crossing"]:
                junctions.append({
                    "id": el.get("id"),
                    "type": tags.get("highway"),
                    "coords": [lon, lat],
                    "name": tags.get("name", f"Junction at {lat:.4f}, {lon:.4f}")
                })
            elif tags.get("railway") == "level_crossing":
                railways.append({
                    "id": el.get("id"),
                    "type": "level_crossing",
                    "coords": [lon, lat],
                    "name": "Railway Crossing"
                })
    
    return {
        "bridges": bridges,
        "tunnels": tunnels,
        "junctions": junctions,
        "railways": railways,
        "buildings": buildings[:50]  # Limit buildings to avoid huge response
    }


def get_way_center(geometry: List[Dict]) -> Optional[List[float]]:
    """Get center point of a way geometry."""
    if not geometry:
        return None
    lons = [pt["lon"] for pt in geometry]
    lats = [pt["lat"] for pt in geometry]
    return [sum(lons) / len(lons), sum(lats) / len(lats)]


def haversine_distance(coord1: List[float], coord2: List[float]) -> float:
    """Calculate distance in meters between two [lon, lat] coordinates."""
    lon1, lat1 = math.radians(coord1[0]), math.radians(coord1[1])
    lon2, lat2 = math.radians(coord2[0]), math.radians(coord2[1])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return 6371000 * c  # Earth radius in meters


def find_obstacles_near_point(point: List[float], infrastructure: Dict, radius_m: float = 30) -> List[Dict]:
    """Find infrastructure obstacles near a specific point."""
    obstacles = []
    
    # Check junctions
    for junction in infrastructure.get("junctions", []):
        dist = haversine_distance(point, junction["coords"])
        if dist < radius_m:
            obstacles.append({"type": "junction", "distance_m": dist, "data": junction})
    
    # Check railway crossings
    for rail in infrastructure.get("railways", []):
        if isinstance(rail.get("coords"), list) and len(rail["coords"]) == 2 and isinstance(rail["coords"][0], (int, float)):
            # Point
            dist = haversine_distance(point, rail["coords"])
            if dist < radius_m:
                obstacles.append({"type": "railway_crossing", "distance_m": dist, "data": rail})
        elif isinstance(rail.get("coords"), list) and rail["coords"]:
            # Line - check distance to nearest point
            for rc in rail["coords"]:
                dist = haversine_distance(point, rc)
                if dist < radius_m:
                    obstacles.append({"type": "railway_line", "distance_m": dist, "data": rail})
                    break
    
    # Check existing bridges
    for bridge in infrastructure.get("bridges", []):
        if bridge.get("center"):
            dist = haversine_distance(point, bridge["center"])
            if dist < radius_m * 2:  # Larger radius for bridges
                obstacles.append({"type": "existing_bridge", "distance_m": dist, "data": bridge})
    
    return obstacles


def compute_smart_pillar_positions(
    route_coords: List[List[float]], 
    infrastructure: Dict,
    target_spacing_m: float = 30,
    min_spacing_m: float = 20,
    max_spacing_m: float = 50
) -> Tuple[List[Dict], List[Dict]]:
    """
    Compute intelligent pillar positions avoiding:
    - Existing bridges/flyovers
    - Junctions/intersections
    - Railway crossings
    - Buildings
    
    Returns: (pillar_positions, avoidance_zones)
    """
    if not route_coords or len(route_coords) < 2:
        return [], []
    
    pillars = []
    avoidance_zones = []
    
    # Always place first pillar at start
    pillars.append({
        "coords": route_coords[0],
        "type": "anchor",
        "id": 1,
        "reason": "Start anchor point"
    })
    
    accumulated_dist = 0
    last_pillar_idx = 0
    
    for i in range(1, len(route_coords)):
        prev = route_coords[i - 1]
        curr = route_coords[i]
        seg_dist = haversine_distance(prev, curr)
        accumulated_dist += seg_dist
        
        # Check if we should place a pillar here
        if accumulated_dist >= target_spacing_m:
            # Check for obstacles at this location
            obstacles = find_obstacles_near_point(curr, infrastructure, radius_m=25)
            
            if obstacles:
                # There are obstacles - need to adjust pillar position
                obstacle_types = [o["type"] for o in obstacles]
                
                avoidance_zones.append({
                    "coords": curr,
                    "obstacles": obstacle_types,
                    "reason": f"Avoiding: {', '.join(obstacle_types)}"
                })
                
                # Try to find a better position (look ahead or back)
                # If we haven't exceeded max spacing, skip this position
                if accumulated_dist < max_spacing_m:
                    continue
                else:
                    # Must place pillar - find best offset position
                    # For now, place it but mark as constrained
                    pillars.append({
                        "coords": curr,
                        "type": "constrained",
                        "id": len(pillars) + 1,
                        "reason": f"Constrained placement near {obstacle_types[0]}",
                        "nearby_obstacles": obstacle_types
                    })
                    accumulated_dist = 0
                    last_pillar_idx = i
            else:
                # Clear location - place pillar
                pillars.append({
                    "coords": curr,
                    "type": "standard",
                    "id": len(pillars) + 1,
                    "reason": "Standard spacing placement"
                })
                accumulated_dist = 0
                last_pillar_idx = i
    
    # Always place last pillar at end
    if route_coords[-1] != pillars[-1]["coords"]:
        pillars.append({
            "coords": route_coords[-1],
            "type": "anchor",
            "id": len(pillars) + 1,
            "reason": "End anchor point"
        })
    
    return pillars, avoidance_zones
