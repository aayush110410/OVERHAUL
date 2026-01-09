from __future__ import annotations

"""F1-style baseline predictor training for city traffic.

Inspired by the external F1 repo pattern:
- load tabular data
- engineer simple features
- train Gradient Boosting
- evaluate with a holdout time split

This trains two targets when available:
- Average_Speed_kmph
- Congestion_pct

Outputs are written to models/traffic/ as joblib pickles.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


@dataclass(frozen=True)
class TrainResult:
    target: str
    mae: float
    model_path: Path


def _time_sort(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    if "timestamp" in cols:
        c = cols["timestamp"]
        df[c] = pd.to_datetime(df[c], errors="coerce")
        df = df.dropna(subset=[c]).sort_values(c)
        return df
    if "date" in cols and "hour" in cols:
        d = cols["date"]
        h = cols["hour"]
        df[d] = pd.to_datetime(df[d], errors="coerce")
        df = df.dropna(subset=[d]).sort_values([d, h])
        return df
    return df


def _build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    cols = {c.lower(): c for c in df.columns}

    feat = pd.DataFrame(index=df.index)

    # Common temporal features
    if "timestamp" in cols:
        ts = pd.to_datetime(df[cols["timestamp"]], errors="coerce")
        feat["dow"] = ts.dt.dayofweek
        feat["month"] = ts.dt.month
        feat["day"] = ts.dt.day
    elif "date" in cols:
        dt = pd.to_datetime(df[cols["date"]], errors="coerce")
        feat["dow"] = dt.dt.dayofweek
        feat["month"] = dt.dt.month
        feat["day"] = dt.dt.day
    if "hour" in cols:
        feat["hour"] = pd.to_numeric(df[cols["hour"]], errors="coerce")

    # Traffic composition features (if present)
    for key in [
        "vehicles_total",
        "cars",
        "two_wheelers",
        "buses",
        "trucks",
        "auto_rickshaws",
        "evs",
        "weather_flag",
        "festival_flag",
        "jam_length_km",
        "jam_count",
    ]:
        if key in cols:
            feat[key] = pd.to_numeric(df[cols[key]], errors="coerce")

    # Categorical route/city (simple hashing)
    if "route" in cols:
        feat["route_hash"] = df[cols["route"]].astype(str).apply(lambda s: hash(s) % 10_000)
    if "city" in cols:
        feat["city_hash"] = df[cols["city"]].astype(str).apply(lambda s: hash(s) % 10_000)

    feat = feat.fillna(0.0)
    return feat, cols


def train_one(df: pd.DataFrame, *, target_col: str, out_dir: Path, label: str) -> TrainResult:
    X, cols = _build_features(df)
    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0.0)

    n = len(df)
    split = int(max(1, n * 0.8))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    pred = model.predict(X_test) if len(X_test) else model.predict(X_train)
    truth = y_test if len(X_test) else y_train
    mae = float(mean_absolute_error(truth, pred))

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{label}__{target_col}.joblib"
    joblib.dump({"model": model, "feature_columns": list(X.columns)}, model_path)

    return TrainResult(target=target_col, mae=mae, model_path=model_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train simple city traffic predictors (GBR)")
    p.add_argument("--csv", required=True, help="Path to city CSV")
    p.add_argument("--out", default="models/traffic", help="Output directory")
    p.add_argument("--label", default=None, help="Label for model file names")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    out_dir = Path(args.out)

    df = pd.read_csv(csv_path)
    df = _time_sort(df)

    cols = {c.lower(): c for c in df.columns}
    label = args.label or csv_path.stem

    results: List[TrainResult] = []

    if "average_speed_kmph" in cols:
        results.append(train_one(df, target_col=cols["average_speed_kmph"], out_dir=out_dir, label=label))
    if "congestion_pct" in cols:
        results.append(train_one(df, target_col=cols["congestion_pct"], out_dir=out_dir, label=label))
    if "avg_speed" in cols:
        results.append(train_one(df, target_col=cols["avg_speed"], out_dir=out_dir, label=label))

    for r in results:
        print({"target": r.target, "mae": round(r.mae, 4), "model_path": str(r.model_path)})


if __name__ == "__main__":
    main()
