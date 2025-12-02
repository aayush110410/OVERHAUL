"""TrafficGod predictive brain trained on TomTom features."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV

# Updated Feature Set including Lags and Rolling Stats
FEATURE_COLUMNS = [
    "congestion_ratio",
    "delay_ratio",
    "speed_delta",
    "hour",
    "dow",
    "sin_hour",
    "cos_hour",
    "incident_count",
    "congestion_lag_1",
    "congestion_lag_2",
    "congestion_roll_mean_4",
    "congestion_roll_std_4",
    "congestion_x_incident"
]
TARGET_COLUMN = "target_congestion"


@dataclass
class TrainingStats:
    train_mae: float
    val_mae: float
    val_rmse: float
    val_r2: float


class TrafficGodModel:
    def __init__(self) -> None:
        # Base model, will be tuned
        self.model = HistGradientBoostingRegressor(random_state=42)

    def fit(self, df: pd.DataFrame, tune: bool = True) -> TrainingStats:
        # Ensure all features exist (fill missing with 0 if any new ones are NaN)
        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = 0.0
                
        mask = df[TARGET_COLUMN].notna()
        subset = df.loc[mask, FEATURE_COLUMNS + [TARGET_COLUMN]].dropna()
        X = subset[FEATURE_COLUMNS]
        y = subset[TARGET_COLUMN]
        
        # Time-based split (no shuffle) to respect causality
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        if tune:
            print("Tuning hyperparameters via RandomizedSearchCV...")
            param_dist = {
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_iter": [100, 300, 500, 1000],
                "max_depth": [3, 5, 7, 10, None],
                "l2_regularization": [0.0, 0.1, 1.0],
                "min_samples_leaf": [20, 50, 100]
            }
            search = RandomizedSearchCV(
                self.model, 
                param_dist, 
                n_iter=10, 
                cv=3, 
                scoring='neg_mean_absolute_error', 
                n_jobs=-1, 
                random_state=42
            )
            search.fit(X_train, y_train)
            self.model = search.best_estimator_
            print(f"Best params: {search.best_params_}")
        else:
            self.model.fit(X_train, y_train)
            
        train_preds = self.model.predict(X_train)
        val_preds = self.model.predict(X_val)
        
        stats = TrainingStats(
            train_mae=float(mean_absolute_error(y_train, train_preds)),
            val_mae=float(mean_absolute_error(y_val, val_preds)),
            val_rmse=float(np.sqrt(mean_squared_error(y_val, val_preds))),
            val_r2=float(r2_score(y_val, val_preds)),
        )
        return stats

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: Path) -> None:
        self.model = joblib.load(path)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        # Ensure columns exist
        X = df.copy()
        for col in FEATURE_COLUMNS:
            if col not in X.columns:
                X[col] = 0.0
        return self.model.predict(X[FEATURE_COLUMNS])


def rank_hotspots(df: pd.DataFrame, predictions: np.ndarray, top_k: int = 10) -> List[Dict[str, object]]:
    df = df.copy()
    df["predicted_congestion"] = predictions
    latest_ts = df["timestamp"].max()
    recent = df[df["timestamp"] >= latest_ts - pd.Timedelta(minutes=30)]
    ranked = recent.sort_values("predicted_congestion", ascending=False).head(top_k)
    hotspots: List[Dict[str, object]] = []
    for _, row in ranked.iterrows():
        if row["predicted_congestion"] <= 0:
            continue
        cause = "Incident pressure" if row.get("incident_count", 0) else "Recurring demand peak"
        suggestion = generate_action(row)
        hotspots.append(
            {
                "timestamp": row["timestamp"].isoformat(),
                "lat": row["lat"],
                "lon": row["lon"],
                "predicted_congestion": float(row["predicted_congestion"]),
                "current_congestion": float(row["congestion_ratio"]),
                "cause": cause,
                "suggestion": suggestion,
            }
        )
    return hotspots


def generate_action(row: pd.Series) -> str:
    if row.get("incident_count", 0):
        return "Dispatch clearance crew, broadcast reroute via VMS"
    if row["predicted_congestion"] > 0.6:
        return "Activate reversible lane + extend green splits"
    if row["predicted_congestion"] > 0.4:
        return "Deploy adaptive signal + ramp metering"
    return "Monitor"


def suggest_detours(hotspots: List[Dict[str, object]]) -> List[Dict[str, object]]:
    detours: List[Dict[str, object]] = []
    for hot in hotspots:
        lat = hot["lat"]
        lon = hot["lon"]
        detours.append(
            {
                "origin": [lat, lon],
                "avoid_cell": [lat, lon],
                "suggested_path": [
                    [lat + 0.01, lon - 0.01],
                    [lat + 0.02, lon + 0.02],
                ],
                "explanation": "Shift flows north-east to unload hotspot",
            }
        )
    return detours


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the TrafficGod model")
    parser.add_argument("--dataset", required=True, help="Path to tomtom_features parquet")
    parser.add_argument("--model-out", default="models/traffic_god.joblib", help="Model artifact path")
    parser.add_argument("--report-out", default="reports/traffic_god_report.json", help="JSON report path")
    parser.add_argument("--top-k", type=int, default=8, help="Number of hotspots to surface")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.dataset)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    model = TrafficGodModel()
    stats = model.fit(df, tune=args.tune)
    model.save(Path(args.model_out))
    
    # Predict on the latest slice for the report
    latest_slice = df.tail(min(len(df), 500)).copy()
    preds = model.predict(latest_slice)
    hotspots = rank_hotspots(latest_slice, preds, args.top_k)
    detours = suggest_detours(hotspots)
    
    report = {
        "dataset": args.dataset,
        "model_path": args.model_out,
        "train_mae": stats.train_mae,
        "val_mae": stats.val_mae,
        "val_rmse": stats.val_rmse,
        "val_r2": stats.val_r2,
        "hotspots": hotspots,
        "detours": detours,
    }
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
