"""
Noida/NCR Traffic Knowledge Data
================================

Comprehensive data about traffic patterns, road networks,
and urban characteristics of Noida, Indirapuram, and NCR region.
"""

import json
from datetime import datetime, time
from typing import Dict, List, Any


# Road Network Data
ROAD_NETWORK = {
    "highways": [
        {
            "name": "Noida Expressway",
            "id": "noida_exp",
            "length_km": 25.0,
            "lanes": 8,
            "speed_limit_kmh": 100,
            "toll": True,
            "connects": ["Noida", "Greater Noida"],
            "peak_congestion_level": "heavy",
            "typical_flow_vph": 15000
        },
        {
            "name": "DND Flyway",
            "id": "dnd",
            "length_km": 9.2,
            "lanes": 8,
            "speed_limit_kmh": 80,
            "toll": True,
            "connects": ["Delhi", "Noida"],
            "peak_congestion_level": "heavy",
            "typical_flow_vph": 12000
        },
        {
            "name": "NH24",
            "id": "nh24",
            "length_km": 50.0,
            "lanes": 6,
            "speed_limit_kmh": 80,
            "toll": False,
            "connects": ["Delhi", "Ghaziabad", "Lucknow"],
            "peak_congestion_level": "gridlock",
            "typical_flow_vph": 18000
        },
        {
            "name": "Greater Noida Expressway",
            "id": "gn_exp",
            "length_km": 24.0,
            "lanes": 8,
            "speed_limit_kmh": 100,
            "toll": True,
            "connects": ["Greater Noida", "Agra"],
            "peak_congestion_level": "moderate",
            "typical_flow_vph": 8000
        },
        {
            "name": "Yamuna Expressway",
            "id": "yamuna_exp",
            "length_km": 165.0,
            "lanes": 6,
            "speed_limit_kmh": 100,
            "toll": True,
            "connects": ["Greater Noida", "Agra"],
            "peak_congestion_level": "light",
            "typical_flow_vph": 5000
        }
    ],
    "arterials": [
        {
            "name": "Sector 18 Market Road",
            "id": "sec18",
            "length_km": 3.0,
            "lanes": 4,
            "speed_limit_kmh": 40,
            "peak_congestion_level": "gridlock",
            "typical_flow_vph": 4000
        },
        {
            "name": "Film City Road",
            "id": "filmcity",
            "length_km": 5.0,
            "lanes": 4,
            "speed_limit_kmh": 50,
            "peak_congestion_level": "heavy",
            "typical_flow_vph": 3500
        }
    ]
}


# Traffic Patterns by Time
TRAFFIC_PATTERNS = {
    "weekday": {
        "morning_rush": {
            "start_time": "07:30",
            "end_time": "10:30",
            "peak_time": "09:00",
            "congestion_multiplier": 2.5,
            "dominant_direction": "inbound_to_commercial"
        },
        "midday": {
            "start_time": "10:30",
            "end_time": "16:00",
            "peak_time": "13:00",
            "congestion_multiplier": 1.2,
            "dominant_direction": "balanced"
        },
        "evening_rush": {
            "start_time": "17:00",
            "end_time": "21:00",
            "peak_time": "18:30",
            "congestion_multiplier": 2.8,
            "dominant_direction": "outbound_to_residential"
        },
        "night": {
            "start_time": "21:00",
            "end_time": "07:30",
            "peak_time": "23:00",
            "congestion_multiplier": 0.5,
            "dominant_direction": "balanced"
        }
    },
    "weekend": {
        "morning": {
            "start_time": "08:00",
            "end_time": "12:00",
            "congestion_multiplier": 0.8,
            "dominant_direction": "balanced"
        },
        "afternoon": {
            "start_time": "12:00",
            "end_time": "18:00",
            "peak_time": "15:00",
            "congestion_multiplier": 1.8,
            "dominant_direction": "towards_markets_malls"
        },
        "evening": {
            "start_time": "18:00",
            "end_time": "22:00",
            "peak_time": "19:00",
            "congestion_multiplier": 2.0,
            "dominant_direction": "towards_entertainment"
        }
    }
}


# Location Grid for Noida
NOIDA_GRID = {
    "sectors": {
        f"Sector {i}": {
            "id": f"sec_{i}",
            "type": "residential" if i > 30 else "commercial" if i < 20 else "mixed",
            "population_density": "high" if i in range(15, 65) else "medium",
            "metro_station": i in [15, 16, 18, 50, 51, 52, 76, 101, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 153],
            "coordinates": (28.57 + i * 0.002, 77.32 + i * 0.003)
        }
        for i in range(1, 169)
    },
    "landmarks": {
        "DLF Mall of India": {"sector": 18, "type": "mall", "parking_capacity": 5000},
        "Great India Place": {"sector": 38, "type": "mall", "parking_capacity": 4000},
        "Worlds of Wonder": {"sector": 38, "type": "amusement", "parking_capacity": 2000},
        "Botanical Garden": {"sector": 38, "type": "park", "metro": True},
        "Film City": {"sector": 16, "type": "media", "traffic_generator": True},
        "Noida Stadium": {"sector": 21, "type": "sports", "event_capacity": 25000},
        "NSEZ": {"sector": 81, "type": "industrial", "workers": 50000},
        "Tech Park": {"sector": 62, "type": "it_park", "workers": 100000}
    }
}


# Indirapuram Data
INDIRAPURAM_DATA = {
    "khands": [
        "Shakti Khand", "Niti Khand", "Gyan Khand", "Ahinsa Khand",
        "Nyay Khand", "Abhay Khand", "Vaibhav Khand"
    ],
    "major_intersections": [
        {"name": "Shipra Mall Crossing", "congestion_prone": True},
        {"name": "Vaibhav Khand Junction", "congestion_prone": True},
        {"name": "Railway Crossing", "congestion_prone": True, "closure_times": ["09:00", "18:00"]}
    ],
    "malls": [
        {"name": "Shipra Mall", "weekend_traffic_multiplier": 2.5},
        {"name": "Jaipuria Mall", "weekend_traffic_multiplier": 2.0},
        {"name": "Mahagun Mall", "weekend_traffic_multiplier": 2.2}
    ]
}


# AQI Baseline Data (Noida specific)
AQI_PATTERNS = {
    "seasonal": {
        "winter": {  # Nov-Feb
            "baseline_aqi": 250,
            "range": (150, 400),
            "morning_spike": 1.3,
            "evening_spike": 1.4,
            "traffic_contribution_pct": 35
        },
        "summer": {  # Mar-Jun
            "baseline_aqi": 120,
            "range": (80, 200),
            "morning_spike": 1.1,
            "evening_spike": 1.2,
            "traffic_contribution_pct": 40
        },
        "monsoon": {  # Jul-Sep
            "baseline_aqi": 80,
            "range": (40, 150),
            "morning_spike": 1.0,
            "evening_spike": 1.1,
            "traffic_contribution_pct": 45
        },
        "post_monsoon": {  # Oct
            "baseline_aqi": 150,
            "range": (100, 250),
            "morning_spike": 1.2,
            "evening_spike": 1.3,
            "traffic_contribution_pct": 40
        }
    },
    "traffic_impact": {
        "per_1000_vehicles": 0.5,  # AQI increase per 1000 vehicles
        "ev_reduction_factor": 0.7,  # 30% less per EV vs ICE
        "congestion_multiplier": 1.5  # Stop-and-go increases emissions
    }
}


# Metro Network
METRO_NETWORK = {
    "blue_line": {
        "stations_in_noida": ["Noida Sector 15", "Noida Sector 16", "Noida Sector 18", "Botanical Garden", "Golf Course", "Noida City Centre"],
        "frequency_minutes": 4,
        "capacity_per_train": 1800,
        "operating_hours": ("06:00", "23:00")
    },
    "aqua_line": {
        "stations": ["Noida Sector 51", "Noida Sector 50", "Noida Sector 76", "Noida Sector 101", 
                    "Noida Sector 81", "NSEZ", "Noida Sector 83", "Noida Sector 137", 
                    "Noida Sector 142", "Noida Sector 143", "Noida Sector 144", 
                    "Noida Sector 145", "Noida Sector 146", "Noida Sector 147", 
                    "Noida Sector 148", "Pari Chowk", "Alpha 1", "Alpha 2", 
                    "Delta 1", "GNIDA Office", "Knowledge Park II", "Depot"],
        "frequency_minutes": 6,
        "capacity_per_train": 1000,
        "operating_hours": ("06:00", "22:00")
    }
}


# Historical Events Impact (for pattern learning)
EVENT_IMPACTS = {
    "diwali": {
        "traffic_multiplier": 0.7,  # Less office traffic
        "market_multiplier": 3.0,  # More shopping
        "aqi_multiplier": 2.5  # Firecrackers
    },
    "republic_day": {
        "traffic_multiplier": 0.8,
        "route_restrictions": ["DND", "towards_delhi"]
    },
    "match_at_stadium": {
        "affected_sectors": [21, 22, 23],
        "pre_event_hours": 2,
        "post_event_hours": 1.5,
        "traffic_multiplier": 3.0
    },
    "rain": {
        "speed_reduction": 0.3,
        "accident_probability_increase": 2.0,
        "waterlogging_locations": ["Sector 18 underpass", "Noida-Delhi border"]
    },
    "fog": {
        "speed_reduction": 0.5,
        "visibility_km": 0.2,
        "expressway_closure_possible": True
    }
}


def get_current_traffic_pattern() -> Dict[str, Any]:
    """Get the current traffic pattern based on time"""
    now = datetime.now()
    current_time = now.strftime("%H:%M")
    is_weekend = now.weekday() >= 5
    
    patterns = TRAFFIC_PATTERNS["weekend"] if is_weekend else TRAFFIC_PATTERNS["weekday"]
    
    for period_name, period_data in patterns.items():
        start = period_data["start_time"]
        end = period_data["end_time"]
        
        # Handle overnight period
        if start > end:
            if current_time >= start or current_time < end:
                return {"period": period_name, **period_data}
        else:
            if start <= current_time < end:
                return {"period": period_name, **period_data}
    
    return {"period": "unknown", "congestion_multiplier": 1.0}


def get_aqi_estimate(month: int, hour: int, traffic_flow: int) -> float:
    """Estimate AQI based on season, time, and traffic"""
    # Determine season
    if month in [11, 12, 1, 2]:
        season = "winter"
    elif month in [3, 4, 5, 6]:
        season = "summer"
    elif month in [7, 8, 9]:
        season = "monsoon"
    else:
        season = "post_monsoon"
    
    season_data = AQI_PATTERNS["seasonal"][season]
    base_aqi = season_data["baseline_aqi"]
    
    # Time-based adjustment
    if 7 <= hour <= 10:
        base_aqi *= season_data["morning_spike"]
    elif 17 <= hour <= 21:
        base_aqi *= season_data["evening_spike"]
    
    # Traffic contribution
    traffic_aqi = (traffic_flow / 1000) * AQI_PATTERNS["traffic_impact"]["per_1000_vehicles"]
    
    return min(500, base_aqi + traffic_aqi)


def get_route_options(origin: str, destination: str) -> List[Dict[str, Any]]:
    """Get possible routes between two locations"""
    # Simplified routing logic
    routes = []
    
    # Check if both are in Noida
    origin_lower = origin.lower()
    dest_lower = destination.lower()
    
    if "sector" in origin_lower and "sector" in dest_lower:
        # Internal Noida route
        routes.append({
            "name": "Via Internal Roads",
            "type": "arterial",
            "estimated_distance_km": 5.0,
            "toll": False
        })
    
    if "delhi" in dest_lower or "delhi" in origin_lower:
        # Routes to/from Delhi
        routes.extend([
            {
                "name": "Via DND Flyway",
                "type": "highway",
                "estimated_distance_km": 15.0,
                "toll": True,
                "toll_amount": 30
            },
            {
                "name": "Via Noida-Delhi Link Road",
                "type": "arterial",
                "estimated_distance_km": 18.0,
                "toll": False
            },
            {
                "name": "Via Kalindi Kunj",
                "type": "mixed",
                "estimated_distance_km": 20.0,
                "toll": False
            }
        ])
    
    if "indirapuram" in origin_lower or "indirapuram" in dest_lower:
        routes.append({
            "name": "Via NH24",
            "type": "highway",
            "estimated_distance_km": 12.0,
            "toll": False,
            "congestion_warning": True
        })
    
    if not routes:
        routes.append({
            "name": "Via Noida Expressway",
            "type": "highway",
            "estimated_distance_km": 10.0,
            "toll": True
        })
    
    return routes


# Export data for use by other modules
TRAFFIC_KNOWLEDGE = {
    "road_network": ROAD_NETWORK,
    "traffic_patterns": TRAFFIC_PATTERNS,
    "noida_grid": NOIDA_GRID,
    "indirapuram": INDIRAPURAM_DATA,
    "aqi_patterns": AQI_PATTERNS,
    "metro_network": METRO_NETWORK,
    "event_impacts": EVENT_IMPACTS
}


if __name__ == "__main__":
    # Test data access
    print("Current Traffic Pattern:")
    print(json.dumps(get_current_traffic_pattern(), indent=2))
    
    print("\nAQI Estimate (December, 9 AM, 5000 vehicles):")
    print(f"AQI: {get_aqi_estimate(12, 9, 5000):.0f}")
    
    print("\nRoutes from Sector 62 to Delhi:")
    routes = get_route_options("Sector 62", "Delhi")
    for route in routes:
        print(f"  - {route['name']}: {route['estimated_distance_km']} km")
