"""Unit tests for TomTom ingestion + feature engineering."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no cover - import hook
    sys.path.append(str(ROOT))

from src.analysis.tomtom_features import engineer_features
from src.data.tomtom.client import BoundingBox
from src.data.tomtom.ingest import sample_points


def test_sample_points_grid() -> None:
    bbox = BoundingBox(north=10.0, south=9.0, east=77.0, west=76.0)
    pts = sample_points(bbox, lat_samples=2, lon_samples=3)
    assert len(pts) == 6
    assert pts[0] == (9.0, 76.0)
    assert pts[-1][0] == 10.0


def test_engineer_features_generates_target() -> None:
    flow = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=6, freq="5min"),
            "lat": [28.6] * 3 + [28.7] * 3,
            "lon": [77.3] * 3 + [77.4] * 3,
            "current_speed": [20, 18, 15, 30, 28, 26],
            "free_flow_speed": [40, 40, 40, 45, 45, 45],
            "current_travel_time": [110, 120, 140, 80, 85, 90],
            "free_flow_travel_time": [90, 90, 90, 70, 70, 70],
        }
    )
    incidents = pd.DataFrame(
        {
            "startTime": [pd.Timestamp("2025-01-01 00:00:00")],
            "endTime": [pd.Timestamp("2025-01-01 01:00:00")],
            "latitude": [28.6],
            "longitude": [77.3],
        }
    )
    feats = engineer_features(flow, incidents, min_samples=1)
    assert "target_congestion" in feats.columns
    assert feats["target_congestion"].notna().any()
    assert feats["incident_count"].iloc[0] >= 1
