import math
from typing import List, Tuple

from rules import offset_distance_m

# Utility to compute a simple parallel offset of a polyline in lon/lat (roughly meters via naive scaling).
# For short segments this approximation is adequate; for production, project to a local meter CRS.


def offset_polyline(coords: List[List[float]], offset_m: float) -> List[List[float]]:
    if len(coords) < 2:
        return coords
    # Rough meters per degree at mid-latitude (~111km per deg); better to project but keeping lightweight.
    scale = 1.0 / 111000.0
    result = []
    for i in range(len(coords)):
        lng, lat = coords[i]
        if i == 0:
            n_lng, n_lat = coords[i + 1]
        else:
            n_lng, n_lat = coords[i - 1]
        dx = n_lng - lng
        dy = n_lat - lat
        length = math.hypot(dx, dy) or 1e-9
        nx = -dy / length
        ny = dx / length
        result.append([lng + nx * offset_m * scale, lat + ny * offset_m * scale])
    return result


def sample_pillars(path: List[List[float]], spacing_m: float) -> List[List[float]]:
    if len(path) < 2:
        return path
    scale = 111000.0  # meters per degree
    samples = [path[0]]
    accum = 0.0
    for i in range(1, len(path)):
        p0 = path[i - 1]
        p1 = path[i]
        dist = math.hypot((p1[0] - p0[0]) * scale, (p1[1] - p0[1]) * scale)
        accum += dist
        if accum >= spacing_m:
            samples.append(p1)
            accum = 0.0
    return samples


def build_flyover_geometry(road_coords: List[List[float]], road_width: float, flyover_width: float, pillar_spacing: float):
    offset = offset_distance_m(road_width, flyover_width)
    flyover_path = offset_polyline(road_coords, offset)
    pillars = sample_pillars(flyover_path, pillar_spacing)
    return flyover_path, pillars
