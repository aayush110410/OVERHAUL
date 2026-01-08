LANE_WIDTH_M = 3.5
DEFAULT_LANES = {
    "primary": 4,
    "secondary": 2,
    "tertiary": 2,
    "service": 1,
}

MIN_CLEARANCE_M = 8.5
JUNCTION_CLEARANCE_M = 12.0
SAFETY_GAP_M = 1.0


def infer_lanes(highway_type: str, tags: dict) -> int:
    lanes_tag = tags.get("lanes")
    if lanes_tag and str(lanes_tag).isdigit():
        return int(lanes_tag)
    return DEFAULT_LANES.get(highway_type, 2)


def road_width_m(lanes: int) -> float:
    # Add shoulder allowance (1.5m) as requested.
    return lanes * LANE_WIDTH_M + 1.5


def flyover_lanes(ground_lanes: int) -> int:
    return min(2, max(1, ground_lanes))


def flyover_width_m(flyover_lanes_count: int) -> float:
    return flyover_lanes_count * LANE_WIDTH_M + 2.0


def clearance_height(is_major_junction: bool = False) -> float:
    return JUNCTION_CLEARANCE_M if is_major_junction else MIN_CLEARANCE_M


def offset_distance_m(road_width: float, flyover_width: float) -> float:
    return (road_width / 2.0) + (flyover_width / 2.0) + SAFETY_GAP_M


def speed_assumptions(highway_type: str):
    if highway_type == "primary":
        return 18, 35
    if highway_type == "secondary":
        return 22, 38
    return 20, 32


def compute_traffic_metrics(highway_type: str, congestion_reduction_percent: float):
    speed_before, speed_cap = speed_assumptions(highway_type)
    improved = min(speed_cap, speed_before * (1 + congestion_reduction_percent / 100.0))
    return {
        "speed_before_kmph": round(speed_before, 1),
        "speed_after_kmph": round(improved, 1),
        "congestion_reduction_percent": round(congestion_reduction_percent, 1),
    }
