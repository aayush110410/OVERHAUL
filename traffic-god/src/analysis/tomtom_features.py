"""Turn TomTom flow snapshots into ML-ready features."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TomTom feature matrix")
    parser.add_argument(
        "--input",
        required=True,
        help="Folder containing flows_*.parquet and incidents_*.jsonl files",
    )
    parser.add_argument(
        "--output",
        default="data/processed/tomtom_features.parquet",
        help="Destination parquet file",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=30,
        help="Minimum samples required to keep a grid cell",
    )
    return parser.parse_args()


def load_parquet_stack(folder: Path) -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in folder.glob("flows_*.parquet")]
    if not frames:
        raise FileNotFoundError("No flow parquet files found")
    return pd.concat(frames, ignore_index=True)


def load_incident_records(folder: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for path in folder.glob("incidents_*.jsonl"):
        for line in path.read_text().splitlines():
            rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "geometry" in df.columns:
        coords = df["geometry"].apply(lambda g: g.get("coordinates", [None, None]) if isinstance(g, dict) else [None, None])
        df["longitude"] = coords.apply(lambda pair: pair[0])
        df["latitude"] = coords.apply(lambda pair: pair[1])
    if "startTime" in df.columns:
        df["startTime"] = pd.to_datetime(df["startTime"], errors="coerce", utc=True)
    if "endTime" in df.columns:
        df["endTime"] = pd.to_datetime(df["endTime"], errors="coerce", utc=True)
    return df


def engineer_features(flow_df: pd.DataFrame, incidents_df: pd.DataFrame, min_samples: int) -> pd.DataFrame:
    df = flow_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.dropna(subset=["current_speed", "free_flow_speed"]).copy()
    df = df[df["free_flow_speed"] > 0]

    df["congestion_ratio"] = 1.0 - (df["current_speed"] / df["free_flow_speed"])
    df["delay_ratio"] = (
        (df["current_travel_time"] - df["free_flow_travel_time"]) / df["free_flow_travel_time"].replace(0, np.nan)
    ).fillna(0.0)
    df["speed_delta"] = df["free_flow_speed"] - df["current_speed"]
    df["hour"] = df["timestamp"].dt.hour
    df["dow"] = df["timestamp"].dt.dayofweek
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)

    if not incidents_df.empty and {"latitude", "longitude"}.issubset(incidents_df.columns):
        if "startTime" in incidents_df.columns:
            incidents_df["startTime"] = pd.to_datetime(incidents_df["startTime"], utc=True)
        if "endTime" in incidents_df.columns:
            incidents_df["endTime"] = pd.to_datetime(incidents_df["endTime"], utc=True)
        active_counts = []
        for _, row in df.iterrows():
            ts = row["timestamp"]
            mask = (incidents_df["startTime"] <= ts) & (
                (incidents_df["endTime"].isna()) | (incidents_df["endTime"] >= ts)
            )
            active_counts.append(int(mask.sum()))
        df["incident_count"] = active_counts
    else:
        df["incident_count"] = 0

    df.sort_values(["lat", "lon", "timestamp"], inplace=True)
    
    # --- ENHANCED FEATURE ENGINEERING ---
    # 1. Lag Features: What happened 1 step ago? 2 steps ago?
    #    (Assuming uniform sampling, e.g., 15 min intervals)
    df["congestion_lag_1"] = df.groupby(["lat", "lon"])["congestion_ratio"].shift(1)
    df["congestion_lag_2"] = df.groupby(["lat", "lon"])["congestion_ratio"].shift(2)
    
    # 2. Rolling Statistics: Trends over the last window (e.g., 4 steps = 1 hour)
    #    We use min_periods=1 to get values even at the start
    df["congestion_roll_mean_4"] = df.groupby(["lat", "lon"])["congestion_ratio"].transform(lambda x: x.rolling(4, min_periods=1).mean())
    df["congestion_roll_std_4"] = df.groupby(["lat", "lon"])["congestion_ratio"].transform(lambda x: x.rolling(4, min_periods=1).std()).fillna(0)
    
    # 3. Interaction Features
    df["congestion_x_incident"] = df["congestion_ratio"] * df["incident_count"]
    
    # Target: Next step congestion
    df["target_congestion"] = df.groupby(["lat", "lon"])["congestion_ratio"].shift(-1)
    
    # Drop rows where we don't have enough history for lags or future for target
    df = df.dropna(subset=["target_congestion", "congestion_lag_1", "congestion_lag_2"])

    counts = df.groupby(["lat", "lon"]).size().reset_index(name="samples")
    keep = counts[counts["samples"] >= min_samples][["lat", "lon"]]
    df = df.merge(keep, on=["lat", "lon"], how="inner")
    return df


def main() -> None:
    args = parse_args()
    folder = Path(args.input)
    flows = load_parquet_stack(folder)
    incidents = load_incident_records(folder)
    features = engineer_features(flows, incidents, args.min_samples)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(args.output, index=False)
    if features.empty:
        print(
            "No feature rows met the min_samples threshold; collect more snapshots or lower --min-samples",
        )
    else:
        print(f"Wrote {len(features)} feature rows to {args.output}")


if __name__ == "__main__":  # pragma: no cover
    main()
