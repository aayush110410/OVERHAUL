"""
NCR Data Loader - Load and query local CSV data for Delhi NCR.

This module loads:
- AQI data: NCR_AQI_2024_2025_REAL_ANCHORED.csv
- Traffic data: delhi_ncr_traffic_2026-01-01_to_2026-01-07.csv
- City-specific traffic data: noida_traffic_fresh.csv, ghaziabad_traffic_fresh.csv
"""

import os
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

# Data directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TRAFFIC_DIR = DATA_DIR / "traffic"
AQI_DIR = DATA_DIR / "aqi"

# Cache for loaded data
_aqi_data: Optional[pd.DataFrame] = None
_traffic_data: Optional[pd.DataFrame] = None
_noida_traffic: Optional[pd.DataFrame] = None
_ghaziabad_traffic: Optional[pd.DataFrame] = None


def load_aqi_data() -> pd.DataFrame:
    """Load NCR AQI data from CSV."""
    global _aqi_data
    if _aqi_data is not None:
        return _aqi_data
    
    aqi_file = AQI_DIR / "NCR_AQI_2024_2025_REAL_ANCHORED.csv"
    if not aqi_file.exists():
        print(f"⚠ AQI data file not found: {aqi_file}")
        return pd.DataFrame()
    
    try:
        _aqi_data = pd.read_csv(aqi_file)
        _aqi_data['date'] = pd.to_datetime(_aqi_data['date'])
        print(f"✓ Loaded AQI data: {len(_aqi_data)} records")
        return _aqi_data
    except Exception as e:
        print(f"⚠ Error loading AQI data: {e}")
        return pd.DataFrame()


def load_traffic_data() -> pd.DataFrame:
    """Load Delhi NCR traffic data from CSV."""
    global _traffic_data
    if _traffic_data is not None:
        return _traffic_data
    
    traffic_file = TRAFFIC_DIR / "delhi_ncr_traffic_2026-01-01_to_2026-01-07.csv"
    if not traffic_file.exists():
        print(f"⚠ Traffic data file not found: {traffic_file}")
        return pd.DataFrame()
    
    try:
        _traffic_data = pd.read_csv(traffic_file)
        _traffic_data['Date'] = pd.to_datetime(_traffic_data['Date'])
        print(f"✓ Loaded traffic data: {len(_traffic_data)} records")
        return _traffic_data
    except Exception as e:
        print(f"⚠ Error loading traffic data: {e}")
        return pd.DataFrame()


def get_latest_aqi(city: Optional[str] = None, region: Optional[str] = None) -> Dict[str, Any]:
    """Get latest AQI data for a city/region."""
    df = load_aqi_data()
    if df.empty:
        return {"error": "No AQI data available"}
    
    # Get most recent data
    latest_date = df['date'].max()
    recent = df[df['date'] == latest_date]
    
    if city:
        recent = recent[recent['city'].str.lower() == city.lower()]
    if region:
        recent = recent[recent['region'].str.lower().str.contains(region.lower())]
    
    if recent.empty:
        return {"error": f"No data for {city} {region}"}
    
    # Aggregate by city
    result = {}
    for city_name in recent['city'].unique():
        city_data = recent[recent['city'] == city_name]
        result[city_name] = {
            "avg_pm25": round(city_data['pm25'].mean(), 1),
            "avg_pm10": round(city_data['pm10'].mean(), 1),
            "avg_aqi": round(city_data['aqi'].mean(), 0),
            "max_pm25": round(city_data['pm25'].max(), 1),
            "min_pm25": round(city_data['pm25'].min(), 1),
            "regions": city_data['region'].unique().tolist(),
            "dominant_cause": city_data['dominant_cause'].iloc[0] if 'dominant_cause' in city_data.columns else "Unknown",
            "date": latest_date.strftime('%Y-%m-%d'),
        }
    
    return result


def get_aqi_trend(city: str, days: int = 7) -> Dict[str, Any]:
    """Get AQI trend for a city over the past N days."""
    df = load_aqi_data()
    if df.empty:
        return {"error": "No AQI data available"}
    
    # Filter by city
    city_df = df[df['city'].str.lower() == city.lower()]
    if city_df.empty:
        return {"error": f"No data for city: {city}"}
    
    # Get last N days
    latest_date = city_df['date'].max()
    start_date = latest_date - timedelta(days=days)
    trend_df = city_df[(city_df['date'] >= start_date) & (city_df['date'] <= latest_date)]
    
    # Daily averages
    daily = trend_df.groupby('date').agg({
        'pm25': 'mean',
        'pm10': 'mean',
        'aqi': 'mean'
    }).round(1)
    
    return {
        "city": city,
        "period": f"{start_date.strftime('%Y-%m-%d')} to {latest_date.strftime('%Y-%m-%d')}",
        "trend": daily.to_dict('index'),
        "avg_pm25": round(trend_df['pm25'].mean(), 1),
        "avg_aqi": round(trend_df['aqi'].mean(), 0),
        "trend_direction": "improving" if trend_df.iloc[-1]['pm25'] < trend_df.iloc[0]['pm25'] else "worsening",
    }


def get_traffic_summary(city: Optional[str] = None, hour: int = None) -> Dict[str, Any]:
    """Get traffic summary for a city at a specific hour."""
    df = load_traffic_data()
    if df.empty:
        return {"error": "No traffic data available"}
    
    # Filter
    filtered = df.copy()
    if city:
        filtered = filtered[filtered['City'].str.lower() == city.lower()]
    if hour is not None:
        filtered = filtered[filtered['Hour'] == hour]
    
    if filtered.empty:
        return {"error": f"No data for {city} at hour {hour}"}
    
    # Aggregate
    result = {
        "avg_speed_kmph": round(filtered['Average_Speed_kmph'].mean(), 1),
        "avg_congestion_pct": round(filtered['Congestion_pct'].mean(), 1),
        "total_vehicles": int(filtered['Vehicles_Total'].mean()),
        "ev_count": int(filtered['EVs'].mean()),
        "ev_percentage": round(filtered['EVs'].sum() / filtered['Vehicles_Total'].sum() * 100, 2),
        "top_routes": filtered.groupby('Route')['Congestion_pct'].mean().nlargest(5).to_dict(),
    }
    
    return result


def get_ncr_summary() -> Dict[str, Any]:
    """Get a comprehensive summary of NCR data for all cities."""
    aqi_data = load_aqi_data()
    traffic_data = load_traffic_data()
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "aqi": {},
        "traffic": {},
    }
    
    # AQI summary by city
    if not aqi_data.empty:
        latest_date = aqi_data['date'].max()
        for city in ['Delhi', 'Noida', 'Ghaziabad']:
            city_data = aqi_data[(aqi_data['city'] == city) & (aqi_data['date'] == latest_date)]
            if not city_data.empty:
                summary["aqi"][city] = {
                    "pm25": round(city_data['pm25'].mean(), 1),
                    "pm10": round(city_data['pm10'].mean(), 1),
                    "aqi": round(city_data['aqi'].mean(), 0),
                    "category": get_aqi_category(city_data['aqi'].mean()),
                }
    
    # Traffic summary by city
    if not traffic_data.empty:
        current_hour = datetime.now().hour
        for city in ['Delhi', 'Noida', 'Ghaziabad']:
            city_data = traffic_data[(traffic_data['City'] == city) & (traffic_data['Hour'] == current_hour)]
            if not city_data.empty:
                summary["traffic"][city] = {
                    "avg_speed_kmph": round(city_data['Average_Speed_kmph'].mean(), 1),
                    "congestion_pct": round(city_data['Congestion_pct'].mean(), 1),
                    "total_vehicles": int(city_data['Vehicles_Total'].mean()),
                    "ev_percentage": round(city_data['EVs'].sum() / city_data['Vehicles_Total'].sum() * 100, 2),
                }
    
    return summary


def get_aqi_category(aqi: float) -> str:
    """Get AQI category from value."""
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"


def get_model_predictions() -> Dict[str, Any]:
    """Load trained ML models and produce predictions for each city at current conditions."""
    try:
        import joblib
        models_dir = PROJECT_ROOT / "models"
        
        aqi_model_path = models_dir / "ncr_aqi_model.pkl"
        speed_model_path = models_dir / "ncr_traffic_speed_model.pkl"
        congestion_model_path = models_dir / "ncr_congestion_model.pkl"
        metadata_path = models_dir / "feature_metadata.pkl"
        baselines_path = models_dir / "ncr_baselines.json"
        
        if not aqi_model_path.exists():
            return {}
        
        aqi_model = joblib.load(aqi_model_path)
        speed_model = joblib.load(speed_model_path)
        congestion_model = joblib.load(congestion_model_path)
        metadata = joblib.load(metadata_path)
        
        import json
        with open(baselines_path) as f:
            baselines = json.load(f)
        
        now = datetime.now()
        hour = now.hour
        is_peak = 1 if (8 <= hour <= 10 or 17 <= hour <= 20) else 0
        is_night = 1 if (hour >= 22 or hour <= 5) else 0
        
        city_encoder_aqi = metadata.get("aqi_city_encoder")
        city_encoder_traffic = metadata.get("traffic_city_encoder")
        
        predictions = {}
        
        for city in ["Delhi", "Noida", "Ghaziabad", "Gurugram"]:
            city_baselines = baselines.get(city, {})
            aqi_bl = city_baselines.get("aqi", {})
            traffic_bl = city_baselines.get("traffic", {})
            
            pred = {}
            
            # AQI prediction
            if city_encoder_aqi and city in city_encoder_aqi.classes_:
                city_code = city_encoder_aqi.transform([city])[0]
                pm25_val = aqi_bl.get("pm25_mean", 150)
                pm10_val = aqi_bl.get("pm10_mean", 300)
                aqi_features = pd.DataFrame([{
                    "pm25_real": pm25_val,
                    "pm10_real": pm10_val,
                    "month": now.month,
                    "day_of_week": now.weekday(),
                    "day_of_year": now.timetuple().tm_yday,
                    "is_weekend": 1 if now.weekday() >= 5 else 0,
                    "season_code": _get_season_code(now.month),
                    "city_code": city_code,
                    "festival_flag": 0,
                }])
                pred["predicted_aqi"] = round(float(aqi_model.predict(aqi_features)[0]), 1)
            
            # Traffic speed prediction
            if city_encoder_traffic and city in city_encoder_traffic.classes_:
                city_code_t = city_encoder_traffic.transform([city])[0]
                vehicles = int(traffic_bl.get("total_vehicles_mean", 12000))
                evs = int(vehicles * 0.03)
                speed_features = pd.DataFrame([{
                    "Hour": hour,
                    "Vehicles_Total": vehicles,
                    "is_peak": is_peak,
                    "is_night": is_night,
                    "city_code": city_code_t,
                    "EVs": evs,
                    "Weather_Flag": 0,
                    "Festival_Flag": 0,
                }])
                pred["predicted_speed_kmph"] = round(float(speed_model.predict(speed_features)[0]), 1)
                
                # Congestion prediction
                cong_features = speed_features[["Hour", "Vehicles_Total", "is_peak",
                                                 "is_night", "city_code", "EVs", "Weather_Flag"]]
                pred["predicted_congestion"] = {0: "Low", 1: "Moderate", 2: "High", 3: "Severe"}.get(
                    int(congestion_model.predict(cong_features)[0]), "Unknown"
                )
            
            if pred:
                predictions[city] = pred
        
        return predictions
    except Exception as e:
        print(f"⚠ Model prediction warning: {e}")
        return {}


def _get_season_code(month: int) -> int:
    """Map month to season code matching training data."""
    if month in (12, 1, 2):
        return 0  # Winter
    elif month in (3, 4, 5):
        return 1  # Summer
    elif month in (6, 7, 8, 9):
        return 2  # Monsoon
    else:
        return 3  # Post-Monsoon


def format_ncr_data_for_prompt() -> str:
    """Format NCR data as a string for inclusion in AI prompts."""
    summary = get_ncr_summary()
    
    lines = ["=== LIVE NCR DATA (from monitoring CSVs) ===", ""]
    
    # AQI section
    lines.append("AIR QUALITY (AQI) — measured values:")
    for city, data in summary.get("aqi", {}).items():
        lines.append(f"  {city}: PM2.5={data['pm25']} µg/m³, AQI={data['aqi']} ({data['category']})")
    
    lines.append("")
    
    # Traffic section
    lines.append("TRAFFIC — measured values:")
    for city, data in summary.get("traffic", {}).items():
        lines.append(f"  {city}: Speed={data['avg_speed_kmph']} km/h, Congestion={data['congestion_pct']}%, EVs={data['ev_percentage']}%")
    
    # ML model predictions
    predictions = get_model_predictions()
    if predictions:
        lines.append("")
        lines.append("ML MODEL PREDICTIONS (trained on 10,965 AQI + 2,688 traffic records):")
        for city, pred in predictions.items():
            parts = []
            if "predicted_aqi" in pred:
                parts.append(f"AQI={pred['predicted_aqi']}")
            if "predicted_speed_kmph" in pred:
                parts.append(f"Speed={pred['predicted_speed_kmph']} km/h")
            if "predicted_congestion" in pred:
                parts.append(f"Congestion={pred['predicted_congestion']}")
            if parts:
                lines.append(f"  {city}: {', '.join(parts)}")
    
    lines.append("")
    lines.append(f"Data timestamp: {summary['timestamp']}")
    
    return "\n".join(lines)


# Pre-load data on import
def init_data():
    """Initialize data loading."""
    load_aqi_data()
    load_traffic_data()


# Auto-init when module is imported
try:
    init_data()
except Exception as e:
    print(f"⚠ Data init warning: {e}")
