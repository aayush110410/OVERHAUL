"""
New Traffic God - Utility Functions
====================================
Common utilities used across the Traffic Foundation Model
"""

import os
import json
import logging
import hashlib
import pickle
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from functools import wraps
import math


# ============================================================================
# LOGGING UTILITIES
# ============================================================================

def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """Setup a logger with console and optional file output"""
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# Default logger
logger = setup_logger("traffic_god")


# ============================================================================
# TIME UTILITIES
# ============================================================================

def get_time_of_day(hour: int) -> str:
    """Get time of day category from hour"""
    if 5 <= hour < 8:
        return "early_morning"
    elif 8 <= hour < 11:
        return "morning_peak"
    elif 11 <= hour < 14:
        return "midday"
    elif 14 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 21:
        return "evening_peak"
    elif 21 <= hour < 24:
        return "night"
    else:
        return "late_night"


def get_day_type(date: datetime) -> str:
    """Get day type (weekday/weekend/holiday)"""
    # Indian holidays (approximate - should be updated annually)
    HOLIDAYS_2024 = [
        (1, 26),   # Republic Day
        (3, 25),   # Holi
        (4, 14),   # Ambedkar Jayanti
        (8, 15),   # Independence Day
        (10, 2),   # Gandhi Jayanti
        (10, 12),  # Dussehra
        (11, 1),   # Diwali
        (12, 25),  # Christmas
    ]
    
    if (date.month, date.day) in HOLIDAYS_2024:
        return "holiday"
    elif date.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return "weekend"
    else:
        return "weekday"


def is_peak_hour(hour: int) -> bool:
    """Check if hour is during peak traffic"""
    return (8 <= hour < 11) or (17 <= hour < 21)


def time_to_minutes(time_str: str) -> int:
    """Convert time string (HH:MM) to minutes from midnight"""
    parts = time_str.split(":")
    return int(parts[0]) * 60 + int(parts[1])


def minutes_to_time(minutes: int) -> str:
    """Convert minutes from midnight to time string (HH:MM)"""
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours:02d}:{mins:02d}"


def get_current_timestamp() -> str:
    """Get current timestamp string"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ============================================================================
# GEOGRAPHIC UTILITIES
# ============================================================================

def haversine_distance(
    lat1: float, lon1: float,
    lat2: float, lon2: float
) -> float:
    """Calculate distance between two points in km using Haversine formula"""
    R = 6371  # Earth radius in km
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = (math.sin(delta_lat / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) *
         math.sin(delta_lon / 2) ** 2)
    
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c


def calculate_travel_time(
    distance_km: float,
    avg_speed_kmh: float,
    traffic_factor: float = 1.0
) -> float:
    """Calculate travel time in minutes"""
    if avg_speed_kmh <= 0:
        return float('inf')
    
    effective_speed = avg_speed_kmh / traffic_factor
    return (distance_km / effective_speed) * 60


def grid_to_coords(
    grid_x: int, grid_y: int,
    origin_lat: float, origin_lon: float,
    cell_size_km: float = 0.1
) -> Tuple[float, float]:
    """Convert grid coordinates to lat/lon"""
    # Approximate: 1 degree lat ≈ 111 km
    lat = origin_lat + (grid_y * cell_size_km / 111)
    # Approximate: 1 degree lon ≈ 111 * cos(lat) km
    lon = origin_lon + (grid_x * cell_size_km / (111 * math.cos(math.radians(origin_lat))))
    return lat, lon


def coords_to_grid(
    lat: float, lon: float,
    origin_lat: float, origin_lon: float,
    cell_size_km: float = 0.1
) -> Tuple[int, int]:
    """Convert lat/lon to grid coordinates"""
    grid_y = int((lat - origin_lat) * 111 / cell_size_km)
    grid_x = int((lon - origin_lon) * 111 * math.cos(math.radians(origin_lat)) / cell_size_km)
    return grid_x, grid_y


# Noida reference point
NOIDA_ORIGIN = (28.5355, 77.3910)  # Sector 18


# ============================================================================
# CACHING UTILITIES
# ============================================================================

def compute_hash(data: Any) -> str:
    """Compute MD5 hash of data"""
    if isinstance(data, dict):
        data = json.dumps(data, sort_keys=True)
    return hashlib.md5(str(data).encode()).hexdigest()


class DiskCache:
    """Simple disk-based cache"""
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.pkl")
    
    def get(self, key: str) -> Optional[Any]:
        path = self._get_path(key)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
                if data.get('expiry') and datetime.now() > data['expiry']:
                    os.remove(path)
                    return None
                return data['value']
        return None
    
    def set(self, key: str, value: Any, ttl_seconds: int = 3600):
        path = self._get_path(key)
        data = {
            'value': value,
            'expiry': datetime.now() + timedelta(seconds=ttl_seconds)
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    def delete(self, key: str):
        path = self._get_path(key)
        if os.path.exists(path):
            os.remove(path)
    
    def clear(self):
        for f in os.listdir(self.cache_dir):
            os.remove(os.path.join(self.cache_dir, f))


def cached(ttl_seconds: int = 3600):
    """Decorator for caching function results"""
    cache = {}
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = compute_hash((args, kwargs))
            
            if key in cache:
                value, expiry = cache[key]
                if time.time() < expiry:
                    return value
            
            result = func(*args, **kwargs)
            cache[key] = (result, time.time() + ttl_seconds)
            return result
        
        return wrapper
    return decorator


# ============================================================================
# DATA UTILITIES
# ============================================================================

def normalize_value(
    value: float,
    min_val: float,
    max_val: float,
    clip: bool = True
) -> float:
    """Normalize value to [0, 1] range"""
    if max_val == min_val:
        return 0.5
    
    normalized = (value - min_val) / (max_val - min_val)
    
    if clip:
        normalized = max(0, min(1, normalized))
    
    return normalized


def denormalize_value(
    normalized: float,
    min_val: float,
    max_val: float
) -> float:
    """Denormalize value from [0, 1] range"""
    return normalized * (max_val - min_val) + min_val


def moving_average(values: List[float], window: int) -> List[float]:
    """Calculate moving average"""
    if len(values) < window:
        return values
    
    result = []
    for i in range(len(values) - window + 1):
        avg = sum(values[i:i + window]) / window
        result.append(avg)
    
    return result


def exponential_moving_average(
    values: List[float],
    alpha: float = 0.3
) -> List[float]:
    """Calculate exponential moving average"""
    if not values:
        return []
    
    result = [values[0]]
    for i in range(1, len(values)):
        ema = alpha * values[i] + (1 - alpha) * result[-1]
        result.append(ema)
    
    return result


# ============================================================================
# TRAFFIC-SPECIFIC UTILITIES
# ============================================================================

def calculate_congestion_level(
    current_flow: float,
    capacity: float
) -> str:
    """Calculate congestion level from flow/capacity ratio"""
    if capacity <= 0:
        return "unknown"
    
    ratio = current_flow / capacity
    
    if ratio < 0.5:
        return "free_flow"
    elif ratio < 0.7:
        return "light"
    elif ratio < 0.85:
        return "moderate"
    elif ratio < 0.95:
        return "heavy"
    else:
        return "congested"


def estimate_delay(
    free_flow_time: float,
    congestion_level: str
) -> float:
    """Estimate delay based on congestion level"""
    DELAY_FACTORS = {
        "free_flow": 1.0,
        "light": 1.2,
        "moderate": 1.5,
        "heavy": 2.0,
        "congested": 3.0,
        "unknown": 1.5
    }
    
    factor = DELAY_FACTORS.get(congestion_level, 1.5)
    return free_flow_time * factor


def calculate_aqi_health_impact(aqi: float) -> Dict[str, Any]:
    """Calculate health impact from AQI value"""
    if aqi <= 50:
        category = "Good"
        impact = "Air quality is satisfactory"
        advice = "Ideal for outdoor activities"
    elif aqi <= 100:
        category = "Moderate"
        impact = "Acceptable air quality"
        advice = "Sensitive groups may experience symptoms"
    elif aqi <= 150:
        category = "Unhealthy for Sensitive Groups"
        impact = "May affect sensitive individuals"
        advice = "Reduce prolonged outdoor exertion"
    elif aqi <= 200:
        category = "Unhealthy"
        impact = "Everyone may experience health effects"
        advice = "Avoid prolonged outdoor activities"
    elif aqi <= 300:
        category = "Very Unhealthy"
        impact = "Health alert"
        advice = "Stay indoors, use air purifiers"
    else:
        category = "Hazardous"
        impact = "Health emergency"
        advice = "Avoid all outdoor activities"
    
    return {
        "aqi": aqi,
        "category": category,
        "impact": impact,
        "advice": advice
    }


def get_vehicle_emission_factor(vehicle_type: str) -> float:
    """Get emission factor for vehicle type (g CO2/km)"""
    EMISSION_FACTORS = {
        "petrol_car": 120,
        "diesel_car": 140,
        "ev_car": 0,
        "hybrid_car": 80,
        "cng_car": 100,
        "petrol_2w": 60,
        "ev_2w": 0,
        "auto_rickshaw": 90,
        "ev_auto": 0,
        "bus": 800,
        "ev_bus": 0,
        "truck": 1000,
        "metro": 0  # Per passenger, already electric
    }
    
    return EMISSION_FACTORS.get(vehicle_type, 100)


# ============================================================================
# FILE UTILITIES
# ============================================================================

def ensure_dir(path: str):
    """Ensure directory exists"""
    os.makedirs(path, exist_ok=True)


def save_json(data: Any, path: str, indent: int = 2):
    """Save data as JSON"""
    ensure_dir(os.path.dirname(path))
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(path: str) -> Any:
    """Load data from JSON"""
    with open(path, 'r') as f:
        return json.load(f)


def save_pickle(data: Any, path: str):
    """Save data as pickle"""
    ensure_dir(os.path.dirname(path))
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(path: str) -> Any:
    """Load data from pickle"""
    with open(path, 'rb') as f:
        return pickle.load(f)


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def validate_coordinates(lat: float, lon: float) -> bool:
    """Validate lat/lon coordinates"""
    return -90 <= lat <= 90 and -180 <= lon <= 180


def validate_time_string(time_str: str) -> bool:
    """Validate time string format (HH:MM)"""
    try:
        parts = time_str.split(":")
        if len(parts) != 2:
            return False
        hour, minute = int(parts[0]), int(parts[1])
        return 0 <= hour < 24 and 0 <= minute < 60
    except:
        return False


def validate_noida_region(lat: float, lon: float) -> bool:
    """Check if coordinates are within Noida/NCR region"""
    # Approximate bounding box for Noida-NCR
    MIN_LAT, MAX_LAT = 28.3, 28.9
    MIN_LON, MAX_LON = 77.1, 77.7
    
    return MIN_LAT <= lat <= MAX_LAT and MIN_LON <= lon <= MAX_LON


# ============================================================================
# RETRY UTILITIES
# ============================================================================

def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple = (Exception,)
):
    """Decorator for retrying functions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts - 1:
                        raise
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {current_delay}s..."
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff
        
        return wrapper
    return decorator


# ============================================================================
# TIMING UTILITIES
# ============================================================================

class Timer:
    """Context manager for timing code blocks"""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.start = None
        self.end = None
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        self.end = time.time()
        elapsed = self.end - self.start
        if self.name:
            logger.info(f"{self.name}: {elapsed:.3f}s")
    
    @property
    def elapsed(self) -> float:
        if self.end:
            return self.end - self.start
        return time.time() - self.start


def timeit(func):
    """Decorator for timing functions"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.debug(f"{func.__name__}: {elapsed:.3f}s")
        return result
    return wrapper
