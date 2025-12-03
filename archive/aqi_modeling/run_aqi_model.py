#!/usr/bin/env python3
"""Train a data-aware AQI forecaster for Noida and benchmark it against Delhi CPCB data.

The pipeline performs the following steps:
1. Cleans the daily AQI spreadsheet (Noida, 2024) and reshapes it into a tidy time-series.
2. Engineers autoregressive and seasonal features suitable for tree-based regressors.
3. Trains a gradient boosted regressor with time-series cross-validation and reports metrics.
4. Forecasts forward for the requested horizon (default: 7 days) and stores the results.
5. Parses the supplied CPCB PDF bulletin to compare forecasts with the official 01-Jan-2025 snapshot.
6. Optionally hits the OpenAQ REST API to pull the most recent PM2.5/PM10 bulletin for extra context.

Example:
    python aqi_modeling/run_aqi_model.py \
        --xlsx AQI_daily_city_level_noida_2024_noida_2024.xlsx \
        --pdf AQ-NCR-01012025.pdf \
        --output-dir artifacts/aqi \
        --forecast-horizon 10 \
        --fetch-openaq
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from pypdf import PdfReader
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

MONTH_MAP = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}

AQI_CATEGORIES = [
    "Good",
    "Satisfactory",
    "Moderate",
    "Poor",
    "Very Poor",
    "Severe",
]

FEATURE_COLUMNS = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_7",
    "roll_7",
    "roll_14",
    "trend_7",
    "month",
    "dayofweek",
    "sin_doy",
    "cos_doy",
]


@dataclass
class CVFoldResult:
    fold: int
    mae: float
    rmse: float


def load_noida_sheet(xlsx_path: Path) -> pd.DataFrame:
    """Return tidy daily AQI series for Noida in 2024."""

    raw = pd.read_excel(xlsx_path)
    raw["day_num"] = pd.to_numeric(raw["Day"], errors="coerce")
    raw = raw.dropna(subset=["day_num"]).copy()
    raw["day_num"] = raw["day_num"].astype(int)

    value_cols = [c for c in raw.columns if c in MONTH_MAP]
    if not value_cols:
        raise ValueError("Spreadsheet does not contain month columns")

    tidy = (
        raw.melt(
            id_vars="day_num",
            value_vars=value_cols,
            var_name="month_name",
            value_name="aqi",
        )
        .dropna(subset=["aqi"])
        .copy()
    )
    tidy["month"] = tidy["month_name"].map(MONTH_MAP)
    tidy["date"] = pd.to_datetime(
        {
            "year": 2024,
            "month": tidy["month"],
            "day": tidy["day_num"],
        },
        errors="coerce",
    )
    tidy = tidy.dropna(subset=["date"]).sort_values("date")
    tidy = tidy[["date", "aqi"]].reset_index(drop=True)

    # Filter low sentinel values (e.g., classification rows)
    tidy = tidy[tidy["aqi"] > 10].reset_index(drop=True)
    return tidy


def add_temporal_features(series: pd.DataFrame) -> pd.DataFrame:
    """Add autoregressive, rolling, and seasonal features."""

    df = series.sort_values("date").copy()
    for lag in (1, 2, 3, 7):
        df[f"lag_{lag}"] = df["aqi"].shift(lag)

    df["roll_7"] = df["aqi"].rolling(7).mean().shift(1)
    df["roll_14"] = df["aqi"].rolling(14).mean().shift(1)
    df["trend_7"] = df["lag_1"] - df["lag_7"]

    df["month"] = df["date"].dt.month
    df["dayofweek"] = df["date"].dt.dayofweek
    df["dayofyear"] = df["date"].dt.dayofyear
    df["sin_doy"] = np.sin(2 * np.pi * df["dayofyear"] / 365.0)
    df["cos_doy"] = np.cos(2 * np.pi * df["dayofyear"] / 365.0)

    df = df.dropna().reset_index(drop=True)
    return df


def train_with_timeseries_cv(features: pd.DataFrame) -> Tuple[HistGradientBoostingRegressor, List[CVFoldResult], pd.DataFrame]:
    """Train the model and capture cross-validation diagnostics."""

    if features.empty:
        raise ValueError("No feature rows available for training.")

    X = features[FEATURE_COLUMNS]
    y = features["aqi"]

    n_splits = min(6, max(3, len(features) // 40))
    splitter = TimeSeriesSplit(n_splits=n_splits)

    fold_metrics: List[CVFoldResult] = []
    cv_rows: List[pd.Series] = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        model = HistGradientBoostingRegressor(
            max_depth=5,
            learning_rate=0.05,
            max_iter=400,
            l2_regularization=0.1,
            random_state=42,
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])
        mae = mean_absolute_error(y.iloc[test_idx], preds)
        rmse = math.sqrt(mean_squared_error(y.iloc[test_idx], preds))
        fold_metrics.append(CVFoldResult(fold=fold, mae=mae, rmse=rmse))

        chunk = features.iloc[test_idx][["date", "aqi"]].copy()
        chunk["prediction"] = preds
        chunk["fold"] = fold
        cv_rows.append(chunk)

    final_model = HistGradientBoostingRegressor(
        max_depth=5,
        learning_rate=0.05,
        max_iter=600,
        l2_regularization=0.05,
        random_state=7,
    )
    final_model.fit(X, y)

    cv_frame = pd.concat(cv_rows).sort_values("date").reset_index(drop=True)
    return final_model, fold_metrics, cv_frame


def compute_feature_vector(history: pd.DataFrame, target_date: pd.Timestamp) -> Optional[pd.DataFrame]:
    """Construct a feature row for the target date using historical values."""

    relevant = history[history["date"] < target_date].sort_values("date")
    if len(relevant) < 14:
        return None

    values = relevant["aqi"].to_numpy()

    def safe_lag(offset: int) -> float:
        if len(values) >= offset:
            return float(values[-offset])
        return float(values[-1])

    row = {
        "lag_1": safe_lag(1),
        "lag_2": safe_lag(2),
        "lag_3": safe_lag(3),
        "lag_7": safe_lag(7),
        "roll_7": float(pd.Series(values[-7:]).mean()),
        "roll_14": float(pd.Series(values[-14:]).mean()),
        "trend_7": float(safe_lag(1) - safe_lag(7)),
        "month": target_date.month,
        "dayofweek": target_date.dayofweek,
        "sin_doy": math.sin(2 * math.pi * target_date.timetuple().tm_yday / 365.0),
        "cos_doy": math.cos(2 * math.pi * target_date.timetuple().tm_yday / 365.0),
    }
    return pd.DataFrame([row])


def forecast_horizon(
    base_series: pd.DataFrame,
    model: HistGradientBoostingRegressor,
    days: int,
) -> pd.DataFrame:
    """Iteratively forecast the next `days` values."""

    history = base_series.sort_values("date").copy()
    forecasts: List[Dict[str, float]] = []

    next_date = history["date"].max()
    for step in range(1, days + 1):
        next_date = next_date + pd.Timedelta(days=1)
        feature_row = compute_feature_vector(history, next_date)
        if feature_row is None:
            break
        pred = float(model.predict(feature_row[FEATURE_COLUMNS])[0])
        forecasts.append({"date": next_date, "predicted_aqi": pred})
        history = pd.concat(
            [history, pd.DataFrame({"date": [next_date], "aqi": [pred]})],
            ignore_index=True,
        )

    return pd.DataFrame(forecasts)


def parse_pdf_snapshot(pdf_path: Path, cities: Sequence[str]) -> Dict[str, Dict[str, Optional[float]]]:
    """Extract AQI values for the requested cities from the CPCB bulletin."""

    reader = PdfReader(str(pdf_path))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    text = text.replace("\uf0de", " ").replace("\uf0dd", " ")

    result: Dict[str, Dict[str, Optional[float]]] = {}
    for line in text.splitlines():
        stripped = line.strip()
        for city in cities:
            if city in result:
                continue
            if stripped.lower().startswith(city.lower()):
                numbers = [float(n) for n in re.findall(r"\b(\d{2,3})\b", stripped)]
                cats = re.findall(r"(Good|Satisfactory|Moderate|Poor|Very Poor|Severe)", stripped, flags=re.I)
                entry = {
                    "category": cats[0] if cats else None,
                    "aqi_2025": numbers[0] if numbers else None,
                    "aqi_2024": numbers[1] if len(numbers) > 1 else None,
                    "aqi_2023": numbers[2] if len(numbers) > 2 else None,
                }
                result[city] = entry
    return result


def fetch_openaq_latest(city: str, limit: int = 50, token: Optional[str] = None) -> Dict[str, float]:
    """Call the OpenAQ API for the latest PM10/PM2.5 readings."""

    url = "https://api.openaq.org/v3/latest"
    params = {
        "city": city,
        "parameter": ["pm25", "pm10"],
        "limit": limit,
    }
    headers = {"Accept": "application/json"}
    if token:
        headers["X-API-Key"] = token

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        return {"error": f"OpenAQ request failed: {exc}"}

    payload = response.json()
    measurements: Dict[str, List[float]] = {"pm25": [], "pm10": []}
    for item in payload.get("results", []):
        for measurement in item.get("measurements", []):
            parameter = measurement.get("parameter")
            value = measurement.get("value")
            if parameter in measurements and isinstance(value, (int, float)):
                measurements[parameter].append(float(value))

    summary = {}
    for parameter, values in measurements.items():
        if values:
            summary[f"{parameter}_avg"] = float(np.mean(values))
            summary[f"{parameter}_max"] = float(np.max(values))
            summary[f"{parameter}_min"] = float(np.min(values))
            summary[f"{parameter}_count"] = len(values)
    summary["source_url"] = response.url
    return summary


def fetch_waqi_snapshot(city: str, token: Optional[str] = None) -> Dict[str, float]:
    """Fallback to World Air Quality Index (WAQI) if OpenAQ is unavailable."""

    url = f"https://api.waqi.info/feed/{city.lower().replace(' ', '-')}/"
    params = {"token": token or "demo"}
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        return {"error": f"WAQI request failed: {exc}"}

    payload = response.json()
    if payload.get("status") != "ok":
        return {"error": f"WAQI error: {payload.get('data')}"}

    data = payload.get("data", {})
    iaqi = data.get("iaqi", {})
    return {
        "aqi": data.get("aqi"),
        "dominentpol": data.get("dominentpol"),
        "pm25": (iaqi.get("pm25", {}) or {}).get("v"),
        "pm10": (iaqi.get("pm10", {}) or {}).get("v"),
        "last_update": data.get("time", {}).get("s"),
        "source_url": response.url,
    }


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_pipeline(args: argparse.Namespace) -> Dict[str, object]:
    xlsx_path = Path(args.xlsx)
    pdf_path = Path(args.pdf)
    output_dir = ensure_output_dir(Path(args.output_dir))

    noida_series = load_noida_sheet(xlsx_path)
    feature_frame = add_temporal_features(noida_series)
    model, folds, cv_frame = train_with_timeseries_cv(feature_frame)

    forecasts = forecast_horizon(
        base_series=noida_series,
        model=model,
        days=args.forecast_horizon,
    )

    pdf_snapshot = parse_pdf_snapshot(pdf_path, cities=("Delhi", "Noida"))

    external_context: Optional[Dict[str, float]] = None
    if args.fetch_openaq:
        external_context = fetch_openaq_latest(args.city, token=args.openaq_token)
        if not external_context or "error" in external_context:
            waqi = fetch_waqi_snapshot(args.city, token=args.waqi_token)
            external_context = {
                "openaq": external_context,
                "waqi": waqi,
            }

    cv_path = output_dir / "cv_predictions.csv"
    forecast_path = output_dir / "forecast.csv"
    metrics_path = output_dir / "metrics.json"

    cv_frame.to_csv(cv_path, index=False)
    if not forecasts.empty:
        forecasts.to_csv(forecast_path, index=False)

    metrics_payload = {
        "folds": [fold.__dict__ for fold in folds],
        "cv_mae_mean": float(np.mean([fold.mae for fold in folds])),
        "cv_rmse_mean": float(np.mean([fold.rmse for fold in folds])),
        "pdf_snapshot": pdf_snapshot,
        "openaq": external_context,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2))

    comparison = None
    if forecasts is not None and not forecasts.empty and "Noida" in pdf_snapshot:
        jan1 = pd.Timestamp("2025-01-01")
        jan1_row = forecasts[forecasts["date"] == jan1]
        if not jan1_row.empty:
            comparison = {
                "forecasted_aqi": float(jan1_row.iloc[0]["predicted_aqi"]),
                "observed_aqi": pdf_snapshot["Noida"].get("aqi_2025"),
                "delta": float(jan1_row.iloc[0]["predicted_aqi"]) - float(
                    pdf_snapshot["Noida"].get("aqi_2025") or 0
                ),
            }

    return {
        "cv_predictions": str(cv_path),
        "forecasts": str(forecast_path) if forecasts is not None and not forecasts.empty else None,
        "metrics": str(metrics_path),
        "comparison": comparison,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and forecast AQI for Noida.")
    parser.add_argument("--xlsx", required=True, help="Path to the Noida AQI XLSX file")
    parser.add_argument("--pdf", required=True, help="Path to the CPCB Delhi AQI PDF bulletin")
    parser.add_argument(
        "--output-dir",
        default="artifacts/aqi",
        help="Directory where metrics/predictions will be stored",
    )
    parser.add_argument(
        "--forecast-horizon",
        type=int,
        default=7,
        help="Number of days to forecast beyond the last observation",
    )
    parser.add_argument(
        "--city",
        default="Noida",
        help="City name to use when querying OpenAQ",
    )
    parser.add_argument(
        "--fetch-openaq",
        action="store_true",
        help="Call the OpenAQ API for the freshest PM2.5/PM10 context",
    )
    parser.add_argument(
        "--openaq-token",
        default=None,
        help="Optional OpenAQ API token (set if your account requires one)",
    )
    parser.add_argument(
        "--waqi-token",
        default=None,
        help="Optional WAQI token (falls back to the public 'demo' token if omitted)",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    outputs = run_pipeline(args)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
