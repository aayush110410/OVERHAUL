"""
OVERHAUL Model Training Pipeline
=================================

Trains prediction models from real NCR CSV data:
1. AQI Prediction Model — Random Forest trained on 10,966 real AQI records
2. Traffic Speed Model — Gradient Boosting trained on traffic CSV data
3. Congestion Model — Classifier for congestion severity

Data sources:
- data/aqi/NCR_AQI_2024_2025_REAL_ANCHORED.csv (10,966 rows)
- data/traffic/delhi_ncr_traffic_2026-01-01_to_2026-01-07.csv (1,513 rows)
- data/traffic/noida_traffic_fresh.csv (505 rows)
- data/traffic/ghaziabad_traffic_fresh.csv
- data/traffic/gurugram_traffic_fresh.csv

Output:
- models/ncr_aqi_model.pkl — AQI predictor
- models/ncr_traffic_speed_model.pkl — Speed predictor
- models/ncr_congestion_model.pkl — Congestion classifier
- models/feature_metadata.pkl — Feature column info
- models/training_report.json — Training metrics
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_aqi_data() -> pd.DataFrame:
    """Load and preprocess the real AQI CSV data."""
    path = os.path.join(DATA_DIR, "aqi", "NCR_AQI_2024_2025_REAL_ANCHORED.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    
    # Feature engineering
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    
    # Season encoding (numerical for model)
    season_map = {"Winter": 0, "Summer": 1, "Monsoon": 2, "Post-Monsoon": 3, "Autumn": 3}
    df["season_code"] = df["season"].map(season_map).fillna(0).astype(int)
    
    # City encoding
    city_encoder = LabelEncoder()
    df["city_code"] = city_encoder.fit_transform(df["city"].fillna("Delhi"))
    
    # Festival flag
    df["festival_flag"] = df["festival_flag"].fillna(0).astype(int)
    
    # Clean numeric columns
    for col in ["pm25", "pm10", "aqi"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Use real-anchored columns if available
    if "pm25_adjusted_real" in df.columns:
        df["pm25_real"] = pd.to_numeric(df["pm25_adjusted_real"], errors="coerce").fillna(df["pm25"])
    else:
        df["pm25_real"] = df["pm25"]
    
    if "pm10_adjusted_real" in df.columns:
        df["pm10_real"] = pd.to_numeric(df["pm10_adjusted_real"], errors="coerce").fillna(df["pm10"])
    else:
        df["pm10_real"] = df["pm10"]
    
    df = df.dropna(subset=["aqi", "pm25_real"])
    
    return df, city_encoder


def load_traffic_data() -> pd.DataFrame:
    """Load and combine all traffic CSV files."""
    traffic_dir = os.path.join(DATA_DIR, "traffic")
    dfs = []
    
    for fname in os.listdir(traffic_dir):
        if fname.endswith(".csv"):
            fpath = os.path.join(traffic_dir, fname)
            try:
                df = pd.read_csv(fpath)
                dfs.append(df)
            except Exception as e:
                print(f"  Warning: Could not load {fname}: {e}")
    
    if not dfs:
        raise FileNotFoundError("No traffic CSV files found")
    
    df = pd.concat(dfs, ignore_index=True)
    
    # Feature engineering
    df["Hour"] = pd.to_numeric(df["Hour"], errors="coerce").fillna(12).astype(int)
    df["is_peak"] = df["Hour"].apply(lambda h: 1 if (8 <= h <= 10 or 17 <= h <= 20) else 0)
    df["is_night"] = df["Hour"].apply(lambda h: 1 if (h >= 22 or h <= 5) else 0)
    
    # City encoding
    city_encoder = LabelEncoder()
    df["city_code"] = city_encoder.fit_transform(df["City"].fillna("Delhi"))
    
    # Clean numeric columns
    for col in ["Vehicles_Total", "Average_Speed_kmph", "Congestion_pct", "EVs"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    df = df.dropna(subset=["Average_Speed_kmph", "Congestion_pct"])
    
    return df, city_encoder


def train_aqi_model(df: pd.DataFrame) -> dict:
    """Train AQI prediction model from real CSV data."""
    print("\n" + "=" * 60)
    print("Training AQI Prediction Model (Random Forest)")
    print("=" * 60)
    
    feature_cols = ["pm25_real", "pm10_real", "month", "day_of_week", "day_of_year",
                    "is_weekend", "season_code", "city_code", "festival_flag"]
    
    X = df[feature_cols].copy()
    y = df["aqi"].copy()
    
    # Remove rows with NaN in features or target
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]
    
    print(f"  Training samples: {len(X)}")
    print(f"  Features: {feature_cols}")
    print(f"  Target: AQI (range {y.min():.0f} - {y.max():.0f})")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1)
    cv_mae = -cv_scores.mean()
    
    print(f"\n  Results:")
    print(f"    MAE:  {mae:.2f} AQI points")
    print(f"    RMSE: {rmse:.2f}")
    print(f"    R²:   {r2:.4f}")
    print(f"    CV MAE (5-fold): {cv_mae:.2f}")
    
    # Feature importance
    importances = dict(zip(feature_cols, model.feature_importances_))
    sorted_imp = sorted(importances.items(), key=lambda x: -x[1])
    print(f"\n  Feature Importances:")
    for feat, imp in sorted_imp:
        print(f"    {feat}: {imp:.4f}")
    
    # Save model
    model_path = os.path.join(MODELS_DIR, "ncr_aqi_model.pkl")
    joblib.dump(model, model_path)
    print(f"\n  Saved: {model_path}")
    
    return {
        "model": "ncr_aqi_model",
        "type": "RandomForestRegressor",
        "samples": len(X),
        "features": feature_cols,
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4),
        "cv_mae": round(cv_mae, 2),
        "feature_importances": {k: round(v, 4) for k, v in sorted_imp},
    }


def train_traffic_speed_model(df: pd.DataFrame) -> dict:
    """Train traffic speed prediction model."""
    print("\n" + "=" * 60)
    print("Training Traffic Speed Model (Gradient Boosting)")
    print("=" * 60)
    
    feature_cols = ["Hour", "Vehicles_Total", "is_peak", "is_night", "city_code"]
    if "EVs" in df.columns and df["EVs"].notna().sum() > 100:
        feature_cols.append("EVs")
    if "Weather_Flag" in df.columns:
        feature_cols.append("Weather_Flag")
    if "Festival_Flag" in df.columns:
        feature_cols.append("Festival_Flag")
    
    X = df[feature_cols].copy().fillna(0)
    y = df["Average_Speed_kmph"].copy()
    
    mask = y.notna()
    X = X[mask]
    y = y[mask]
    
    print(f"  Training samples: {len(X)}")
    print(f"  Features: {feature_cols}")
    print(f"  Target: Speed (range {y.min():.1f} - {y.max():.1f} km/h)")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=42,
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1)
    cv_mae = -cv_scores.mean()
    
    print(f"\n  Results:")
    print(f"    MAE:  {mae:.2f} km/h")
    print(f"    RMSE: {rmse:.2f}")
    print(f"    R²:   {r2:.4f}")
    print(f"    CV MAE (5-fold): {cv_mae:.2f}")
    
    importances = dict(zip(feature_cols, model.feature_importances_))
    sorted_imp = sorted(importances.items(), key=lambda x: -x[1])
    print(f"\n  Feature Importances:")
    for feat, imp in sorted_imp:
        print(f"    {feat}: {imp:.4f}")
    
    model_path = os.path.join(MODELS_DIR, "ncr_traffic_speed_model.pkl")
    joblib.dump(model, model_path)
    print(f"\n  Saved: {model_path}")
    
    return {
        "model": "ncr_traffic_speed_model",
        "type": "GradientBoostingRegressor",
        "samples": len(X),
        "features": feature_cols,
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4),
        "cv_mae": round(cv_mae, 2),
        "feature_importances": {k: round(v, 4) for k, v in sorted_imp},
    }


def train_congestion_model(df: pd.DataFrame) -> dict:
    """Train congestion severity classifier."""
    print("\n" + "=" * 60)
    print("Training Congestion Severity Model (Gradient Boosting Classifier)")
    print("=" * 60)
    
    # Create severity labels from congestion percentage
    def congestion_label(pct):
        if pct < 20:
            return 0  # Low
        elif pct < 40:
            return 1  # Moderate
        elif pct < 60:
            return 2  # High
        else:
            return 3  # Severe
    
    df["congestion_severity"] = df["Congestion_pct"].apply(congestion_label)
    
    feature_cols = ["Hour", "Vehicles_Total", "is_peak", "is_night", "city_code"]
    if "EVs" in df.columns and df["EVs"].notna().sum() > 100:
        feature_cols.append("EVs")
    if "Weather_Flag" in df.columns:
        feature_cols.append("Weather_Flag")
    
    X = df[feature_cols].copy().fillna(0)
    y = df["congestion_severity"].copy()
    
    print(f"  Training samples: {len(X)}")
    print(f"  Classes: Low(0), Moderate(1), High(2), Severe(3)")
    print(f"  Distribution: {dict(y.value_counts().sort_index())}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        min_samples_split=5,
        random_state=42,
    )
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy", n_jobs=-1)
    cv_acc = cv_scores.mean()
    
    print(f"\n  Results:")
    print(f"    Accuracy: {acc:.4f}")
    print(f"    CV Accuracy (5-fold): {cv_acc:.4f}")
    
    model_path = os.path.join(MODELS_DIR, "ncr_congestion_model.pkl")
    joblib.dump(model, model_path)
    print(f"\n  Saved: {model_path}")
    
    return {
        "model": "ncr_congestion_model",
        "type": "GradientBoostingClassifier",
        "samples": len(X),
        "features": feature_cols,
        "accuracy": round(acc, 4),
        "cv_accuracy": round(cv_acc, 4),
        "classes": ["Low", "Moderate", "High", "Severe"],
    }


def compute_baseline_stats(aqi_df: pd.DataFrame, traffic_df: pd.DataFrame) -> dict:
    """Compute verified baseline statistics from real CSV data for engine calibration."""
    print("\n" + "=" * 60)
    print("Computing Baseline Statistics from Real Data")
    print("=" * 60)
    
    baselines = {}
    
    # AQI baselines per city (latest data, winter season)
    for city in aqi_df["city"].unique():
        city_data = aqi_df[aqi_df["city"] == city]
        latest = city_data.sort_values("date").tail(30)  # Last 30 days
        baselines[city] = {
            "aqi": {
                "mean": round(latest["aqi"].mean(), 1),
                "median": round(latest["aqi"].median(), 1),
                "min": round(latest["aqi"].min(), 1),
                "max": round(latest["aqi"].max(), 1),
                "pm25_mean": round(latest["pm25_real"].mean(), 1),
                "pm10_mean": round(latest["pm10_real"].mean(), 1) if "pm10_real" in latest.columns else None,
                "records": len(latest),
            }
        }
        print(f"  {city}: AQI={baselines[city]['aqi']['mean']}, PM2.5={baselines[city]['aqi']['pm25_mean']}")
    
    # Traffic baselines per city
    for city in traffic_df["City"].unique():
        city_data = traffic_df[traffic_df["City"] == city]
        if city not in baselines:
            baselines[city] = {}
        baselines[city]["traffic"] = {
            "avg_speed_kmph": round(city_data["Average_Speed_kmph"].mean(), 1),
            "congestion_pct_mean": round(city_data["Congestion_pct"].mean(), 1),
            "peak_congestion_pct": round(city_data[city_data["is_peak"] == 1]["Congestion_pct"].mean(), 1) if "is_peak" in city_data.columns else None,
            "offpeak_speed_kmph": round(city_data[city_data["is_night"] == 1]["Average_Speed_kmph"].mean(), 1) if "is_night" in city_data.columns else None,
            "total_vehicles_mean": round(city_data["Vehicles_Total"].mean(), 0),
            "records": len(city_data),
        }
        print(f"  {city}: Speed={baselines[city]['traffic']['avg_speed_kmph']} km/h, "
              f"Congestion={baselines[city]['traffic']['congestion_pct_mean']}%")
    
    # Save baselines for engine calibration
    baselines_path = os.path.join(MODELS_DIR, "ncr_baselines.json")
    
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Convert all numpy types  
    clean_baselines = json.loads(json.dumps(baselines, default=_convert))
    with open(baselines_path, "w") as f:
        json.dump(clean_baselines, f, indent=2)
    print(f"\n  Saved: {baselines_path}")
    
    return baselines


def main():
    """Run the full training pipeline."""
    print("=" * 60)
    print("OVERHAUL Model Training Pipeline")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "models": [],
        "baselines": {},
    }
    
    # 1. Load data
    print("\nLoading AQI data...")
    aqi_df, aqi_city_encoder = load_aqi_data()
    print(f"  Loaded {len(aqi_df)} AQI records across {aqi_df['city'].nunique()} cities")
    
    print("\nLoading traffic data...")
    traffic_df, traffic_city_encoder = load_traffic_data()
    print(f"  Loaded {len(traffic_df)} traffic records across {traffic_df['City'].nunique()} cities")
    
    # Save encoders
    joblib.dump({
        "aqi_city_encoder": aqi_city_encoder,
        "traffic_city_encoder": traffic_city_encoder,
    }, os.path.join(MODELS_DIR, "feature_metadata.pkl"))
    
    # 2. Train models
    aqi_report = train_aqi_model(aqi_df)
    report["models"].append(aqi_report)
    
    speed_report = train_traffic_speed_model(traffic_df)
    report["models"].append(speed_report)
    
    congestion_report = train_congestion_model(traffic_df)
    report["models"].append(congestion_report)
    
    # 3. Compute baselines
    baselines = compute_baseline_stats(aqi_df, traffic_df)
    report["baselines"] = baselines
    
    # 4. Save training report
    report_path = os.path.join(MODELS_DIR, "training_report.json")
    
    def _convert_numpy(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    clean_report = json.loads(json.dumps(report, default=_convert_numpy))
    with open(report_path, "w") as f:
        json.dump(clean_report, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nModels saved to: {MODELS_DIR}/")
    for m in report["models"]:
        model_type = m.get("type", "Unknown")
        key_metric = ""
        if "r2" in m:
            key_metric = f"R²={m['r2']}"
        elif "accuracy" in m:
            key_metric = f"Acc={m['accuracy']}"
        print(f"  {m['model']}: {model_type} ({key_metric}, {m['samples']} samples)")
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
