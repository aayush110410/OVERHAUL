"""
Generate synthetic trajectory data that mimics real Delhi NCR traffic.

Creates vehicle trajectories across key corridors in:
  - Delhi (central, South Delhi, East Delhi, North Delhi)
  - Ghaziabad (NH-24 / NH-9 corridor, Indirapuram, Vaishali)
  - Gurugram (MG Road, Golf Course Road, Cyber City, NH-48)
  - Noida (Expressway, Sector 18, Greater Noida Expressway)

Format: track_id, frame, x, y  (matches traffic-god expected schema)

Coordinates are in WGS84 (longitude, latitude) but stored as x, y.
Frame numbers represent 0.5-second intervals (~2 FPS video equivalent).
"""

import csv
import random
import math
import os

random.seed(42)

OUTPUT = os.path.join(os.path.dirname(__file__), "trajectories.csv")

# ── Key corridors with real coordinates ──────────────────────────

CORRIDORS = {
    # Delhi corridors
    "delhi_ring_road_south": {
        "start": (77.1855, 28.5635),   # Dhaula Kuan
        "end":   (77.2785, 28.5305),    # Ashram Chowk
        "lanes": 3, "peak_density": 0.9, "city": "delhi",
    },
    "delhi_vikas_marg": {
        "start": (77.2480, 28.6310),    # ITO
        "end":   (77.3140, 28.6340),    # Laxmi Nagar
        "lanes": 2, "peak_density": 0.85, "city": "delhi",
    },
    "delhi_gt_karnal_road": {
        "start": (77.2167, 28.6690),    # Kashmere Gate
        "end":   (77.1550, 28.7550),    # Alipur
        "lanes": 3, "peak_density": 0.7, "city": "delhi",
    },
    "delhi_outer_ring_road": {
        "start": (77.0880, 28.5890),    # Palam
        "end":   (77.1370, 28.6730),    # Punjabi Bagh
        "lanes": 3, "peak_density": 0.8, "city": "delhi",
    },
    "delhi_mathura_road": {
        "start": (77.2410, 28.6130),    # India Gate area
        "end":   (77.2780, 28.5250),    # Badarpur border
        "lanes": 2, "peak_density": 0.75, "city": "delhi",
    },

    # Ghaziabad corridors
    "ghaziabad_nh24": {
        "start": (77.3155, 28.6340),    # Anand Vihar ISBT
        "end":   (77.4350, 28.6700),    # Ghaziabad city center
        "lanes": 3, "peak_density": 0.95, "city": "ghaziabad",
    },
    "ghaziabad_nh9": {
        "start": (77.3250, 28.6310),    # Kaushambi
        "end":   (77.4540, 28.6350),    # Mohan Nagar
        "lanes": 2, "peak_density": 0.88, "city": "ghaziabad",
    },
    "ghaziabad_indirapuram": {
        "start": (77.3560, 28.6300),    # Vaishali
        "end":   (77.3850, 28.6060),    # Indirapuram Habitat Centre
        "lanes": 2, "peak_density": 0.82, "city": "ghaziabad",
    },

    # Gurugram corridors
    "gurugram_nh48": {
        "start": (77.0690, 28.4870),    # IFFCO Chowk
        "end":   (76.9680, 28.4270),    # Manesar
        "lanes": 4, "peak_density": 0.92, "city": "gurugram",
    },
    "gurugram_mg_road": {
        "start": (77.0820, 28.4790),    # MG Road metro
        "end":   (77.0430, 28.4670),    # Sikanderpur
        "lanes": 2, "peak_density": 0.87, "city": "gurugram",
    },
    "gurugram_golf_course_road": {
        "start": (77.0690, 28.4570),    # Golf Course start
        "end":   (77.0390, 28.4290),    # Sector 54/56
        "lanes": 3, "peak_density": 0.78, "city": "gurugram",
    },
    "gurugram_sohna_road": {
        "start": (77.0530, 28.4390),    # Subhash Chowk
        "end":   (77.0660, 28.3900),    # Badshahpur
        "lanes": 2, "peak_density": 0.73, "city": "gurugram",
    },

    # Noida corridors
    "noida_expressway": {
        "start": (77.3480, 28.5850),    # Sector 18 / Film City
        "end":   (77.4820, 28.4540),    # Greater Noida Pari Chowk
        "lanes": 4, "peak_density": 0.85, "city": "noida",
    },
    "noida_sector18_market": {
        "start": (77.3210, 28.5690),    # Noida Sector 15 metro
        "end":   (77.3560, 28.5700),    # Sector 18 Atta market
        "lanes": 2, "peak_density": 0.90, "city": "noida",
    },
    "noida_dnd_flyway": {
        "start": (77.2790, 28.5595),    # DND Delhi side
        "end":   (77.3230, 28.5660),    # DND Noida side
        "lanes": 3, "peak_density": 0.93, "city": "noida",
    },
}


def _add_noise(val: float, pct: float = 0.003) -> float:
    """Add small random lateral offset to simulate lane position."""
    return val + random.uniform(-pct, pct)


def _speed_factor(frame: int, total_frames: int, peak_density: float) -> float:
    """
    Simulate speed variation along a corridor.
    - Gaussian slowdown near intersections (30%, 60%, 85% of route)
    - Random micro-stops for signals
    """
    progress = frame / total_frames
    factor = 1.0

    # Intersection slowdowns (simulating traffic signals)
    for signal_pos in [0.30, 0.60, 0.85]:
        dist = abs(progress - signal_pos)
        if dist < 0.08:
            factor *= 0.25 + 0.75 * (dist / 0.08)  # slow to 25% at signal

    # Random congestion pockets
    if random.random() < peak_density * 0.15:
        factor *= random.uniform(0.4, 0.7)

    # Base density factor — higher density = slower average
    factor *= (1.0 - 0.35 * peak_density)

    return max(factor, 0.15)  # never fully stop


def generate_trajectory(
    track_id: int,
    corridor: dict,
    departure_offset: int = 0,
) -> list:
    """Generate one vehicle trajectory along a corridor."""
    sx, sy = corridor["start"]
    ex, ey = corridor["end"]

    # Route length in degrees (rough proxy for distance)
    dx = ex - sx
    dy = ey - sy
    route_len = math.sqrt(dx**2 + dy**2)

    # Base speed: ~40 km/h in degrees/frame at 0.5s intervals
    # 1 degree ≈ 111 km, 40 km/h ≈ 0.011 km/s ≈ 0.0001 deg/s
    # At 2 FPS (0.5s intervals): 0.00005 deg/frame
    base_step = 0.00005 + random.uniform(-0.000005, 0.000015)

    # Number of frames to traverse corridor
    total_frames = max(int(route_len / base_step), 50)

    rows = []
    cx, cy = sx, sy
    frame = departure_offset

    for i in range(total_frames):
        sf = _speed_factor(i, total_frames, corridor["peak_density"])
        step = base_step * sf

        # Direction with slight lateral wander
        angle = math.atan2(dy, dx)
        wander = random.gauss(0, 0.02)  # small angle noise
        angle += wander

        cx += step * math.cos(angle)
        cy += step * math.sin(angle)

        # Add lane offset noise
        rx = _add_noise(cx, 0.0003)
        ry = _add_noise(cy, 0.0003)

        rows.append([track_id, frame, round(rx, 6), round(ry, 6)])
        frame += 1

    return rows


def main():
    all_rows = []
    track_id = 1

    for name, corridor in CORRIDORS.items():
        # Number of vehicles per corridor: proportional to lanes and density
        n_vehicles = int(corridor["lanes"] * 8 * corridor["peak_density"])
        n_vehicles = max(n_vehicles, 4)

        for v in range(n_vehicles):
            # Stagger departures to simulate temporal spread
            departure = random.randint(0, 200)
            rows = generate_trajectory(track_id, corridor, departure)
            all_rows.extend(rows)
            track_id += 1

    # Sort by track_id, then frame for nice ordering
    all_rows.sort(key=lambda r: (r[0], r[1]))

    with open(OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["track_id", "frame", "x", "y"])
        writer.writerows(all_rows)

    print(f"Generated {len(all_rows)} trajectory points for {track_id - 1} vehicles across {len(CORRIDORS)} corridors")
    print(f"Corridors: {', '.join(CORRIDORS.keys())}")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
