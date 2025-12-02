"""CLI for harvesting TomTom flow + incident data.

Usage example:
    python -m src.data.tomtom.ingest \
        --bbox 28.55 28.75 77.25 77.45 \
        --grid 4 4 \
        --snapshots 6 \
        --sleep-seconds 30 \
        --output data/processed/tomtom
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .client import BoundingBox, TomTomClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download TomTom traffic snapshots")
    parser.add_argument(
        "--api-key",
        default=None,
        help="TomTom API key (falls back to TOMTOM_API_KEY environment variable)",
    )
    parser.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("S", "N", "W", "E"))
    parser.add_argument(
        "--grid",
        nargs=2,
        type=int,
        default=(3, 3),
        metavar=("LAT", "LON"),
        help="Number of sampling points along latitude and longitude",
    )
    parser.add_argument("--snapshots", type=int, default=1, help="How many snapshots to capture")
    parser.add_argument(
        "--sleep-seconds",
        type=int,
        default=60,
        help="Delay between snapshots to avoid rate-limits",
    )
    parser.add_argument(
        "--zoom",
        type=int,
        default=12,
        help="TomTom zoom level for flowSegmentData (5-22)",
    )
    parser.add_argument(
        "--output",
        default="data/processed/tomtom",
        help="Directory where parquet/jsonl outputs are stored",
    )
    return parser.parse_args()


def ensure_output(path: str | Path) -> Path:
    folder = Path(path)
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def sample_points(bbox: BoundingBox, lat_samples: int, lon_samples: int) -> List[Tuple[float, float]]:
    lats = np.linspace(bbox.south, bbox.north, lat_samples)
    lons = np.linspace(bbox.west, bbox.east, lon_samples)
    return [(round(lat, 6), round(lon, 6)) for lat in lats for lon in lons]


def harvest_snapshot(
    client: TomTomClient,
    bbox: BoundingBox,
    points: Iterable[Tuple[float, float]],
    zoom: int,
) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
    timestamp = dt.datetime.now(dt.timezone.utc)
    flow_rows: List[Dict[str, object]] = []
    for lat, lon in points:
        point = f"{lat},{lon}"
        try:
            payload = client.flow_segment_data(point=point, zoom=zoom)
        except Exception as exc:  # pragma: no cover - graceful degrade
            flow_rows.append(
                {
                    "timestamp": timestamp,
                    "lat": lat,
                    "lon": lon,
                    "error": str(exc),
                }
            )
            continue
        segment = payload.get("flowSegmentData", {})
        flow_rows.append(
            {
                "timestamp": timestamp,
                "lat": lat,
                "lon": lon,
                "current_speed": segment.get("currentSpeed"),
                "free_flow_speed": segment.get("freeFlowSpeed"),
                "current_travel_time": segment.get("currentTravelTime"),
                "free_flow_travel_time": segment.get("freeFlowTravelTime"),
                "confidence": segment.get("confidence"),
                "road_closure": bool(segment.get("roadClosure", False)),
                "frc": segment.get("frc"),
            }
        )
    incident_payload = client.incidents_box(bbox)
    incidents = incident_payload.get("incidents", [])
    return pd.DataFrame(flow_rows), incidents


def main() -> None:
    args = parse_args()
    api_key = args.api_key or os.environ.get("TOMTOM_API_KEY")
    if not api_key:
        raise SystemExit("Missing API key. Pass --api-key or set TOMTOM_API_KEY env var.")

    bbox = BoundingBox(north=args.bbox[1], south=args.bbox[0], west=args.bbox[2], east=args.bbox[3])
    points = sample_points(bbox, lat_samples=args.grid[0], lon_samples=args.grid[1])
    client = TomTomClient(api_key)
    output_dir = ensure_output(args.output)

    for snap in range(1, args.snapshots + 1):
        flow_df, incidents = harvest_snapshot(client, bbox, points, args.zoom)
        if flow_df.empty:
            print(f"[snapshot {snap}] No flow data returned")
        else:
            file_stub = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            flow_path = output_dir / f"flows_{file_stub}.parquet"
            incident_path = output_dir / f"incidents_{file_stub}.jsonl"
            flow_df.to_parquet(flow_path, index=False)
            if incidents:
                incident_path.write_text("\n".join(json.dumps(obj) for obj in incidents))
            print(
                f"[snapshot {snap}] Stored {len(flow_df)} flow rows -> {flow_path.name} | {len(incidents)} incidents",
            )
        if snap < args.snapshots:
            time.sleep(max(1, args.sleep_seconds))


if __name__ == "__main__":  # pragma: no cover
    main()
