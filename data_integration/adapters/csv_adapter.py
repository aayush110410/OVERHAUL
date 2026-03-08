"""
CSV & JSON Adapters
====================

Adapters for local static data files:
  - NCRAqiCSVAdapter       — AQI data from NCR_AQI_2024_2025_REAL_ANCHORED.csv
  - NCRTrafficCSVAdapter   — Traffic data from delhi_ncr_traffic + city-specific CSVs
  - HistoricalJSONAdapter  — Historical metrics from data/historical_metrics.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from data_integration.adapters.base import (
    DataAdapter,
    DataDomain,
    DataResult,
    SourceType,
    RefreshPolicy,
)

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"


# ══════════════════════════════════════════════════════════════════
#  AQI CSV Adapter
# ══════════════════════════════════════════════════════════════════

class NCRAqiCSVAdapter(DataAdapter):
    """
    Loads NCR AQI data from CSV.

    Kwargs accepted by fetch():
      city:   filter by city name (case-insensitive)
      days:   number of most-recent days to return (default: all)
    """

    def __init__(self):
        super().__init__(ttl_seconds=0)  # static, never expires
        self._df: Optional[pd.DataFrame] = None

    @property
    def name(self) -> str:
        return "ncr_aqi_csv"

    @property
    def source_type(self) -> SourceType:
        return SourceType.CSV

    @property
    def domains(self) -> List[DataDomain]:
        return [DataDomain.AQI]

    @property
    def refresh_policy(self) -> RefreshPolicy:
        return RefreshPolicy.STATIC

    def _load(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df
        path = _DATA_DIR / "aqi" / "NCR_AQI_2024_2025_REAL_ANCHORED.csv"
        if not path.exists():
            raise FileNotFoundError(f"AQI file not found: {path}")
        self._df = pd.read_csv(path)
        self._df["date"] = pd.to_datetime(self._df["date"])
        return self._df

    async def fetch(self, **kwargs) -> DataResult:
        df = self._load()
        city: Optional[str] = kwargs.get("city")
        days: Optional[int] = kwargs.get("days")

        if city:
            df = df[df["city"].str.lower() == city.lower()]

        if days and days > 0:
            latest = df["date"].max()
            cutoff = latest - pd.Timedelta(days=days)
            df = df[df["date"] >= cutoff]

        latest_date = df["date"].max()
        recent = df[df["date"] == latest_date]

        summary: Dict[str, Any] = {}
        for city_name in recent["city"].unique():
            cd = recent[recent["city"] == city_name]
            summary[city_name] = {
                "pm25": round(cd["pm25"].mean(), 1),
                "pm10": round(cd["pm10"].mean(), 1),
                "aqi": round(cd["aqi"].mean(), 0),
                "category": _aqi_category(cd["aqi"].mean()),
                "dominant_cause": cd["dominant_cause"].iloc[0] if "dominant_cause" in cd.columns else "Unknown",
                "date": latest_date.strftime("%Y-%m-%d"),
                "regions": cd["region"].unique().tolist(),
            }

        return DataResult(
            data=summary,
            source="NCR AQI CSV (2024-2025)",
            source_type=SourceType.CSV,
            domain=DataDomain.AQI,
            meta={"rows": len(self._load()), "cities": list(summary.keys())},
        )

    async def health(self) -> Dict[str, Any]:
        base = await super().health()
        try:
            df = self._load()
            base["rows"] = len(df)
            base["cities"] = df["city"].unique().tolist()
        except Exception as e:
            base["status"] = "error"
            base["error"] = str(e)[:100]
        return base


# ══════════════════════════════════════════════════════════════════
#  Traffic CSV Adapter
# ══════════════════════════════════════════════════════════════════

class NCRTrafficCSVAdapter(DataAdapter):
    """
    Loads NCR traffic data from CSV files.

    Automatically loads all available traffic CSVs:
      - delhi_ncr_traffic_*.csv  (main, all cities)
      - noida_traffic_fresh.csv
      - ghaziabad_traffic_fresh.csv
      - gurugram_traffic_fresh.csv

    Kwargs accepted by fetch():
      city:  filter by city name
      hour:  filter by hour (0-23)
    """

    def __init__(self):
        super().__init__(ttl_seconds=0)
        self._df: Optional[pd.DataFrame] = None

    @property
    def name(self) -> str:
        return "ncr_traffic_csv"

    @property
    def source_type(self) -> SourceType:
        return SourceType.CSV

    @property
    def domains(self) -> List[DataDomain]:
        return [DataDomain.TRAFFIC]

    @property
    def refresh_policy(self) -> RefreshPolicy:
        return RefreshPolicy.STATIC

    def _load(self) -> pd.DataFrame:
        if self._df is not None:
            return self._df

        traffic_dir = _DATA_DIR / "traffic"
        frames: List[pd.DataFrame] = []

        # Load all traffic CSV files
        csv_patterns = [
            "delhi_ncr_traffic_*.csv",
            "noida_traffic_fresh.csv",
            "ghaziabad_traffic_fresh.csv",
            "gurugram_traffic_fresh.csv",
        ]
        for pattern in csv_patterns:
            for path in sorted(traffic_dir.glob(pattern)):
                try:
                    df = pd.read_csv(path)
                    df["Date"] = pd.to_datetime(df["Date"])
                    df["_source_file"] = path.name
                    frames.append(df)
                except Exception:
                    continue

        if not frames:
            raise FileNotFoundError("No traffic CSV files found")

        self._df = pd.concat(frames, ignore_index=True)
        # De-duplicate: keep the row from the most specific file per (City, Route, Date, Hour)
        self._df = self._df.drop_duplicates(
            subset=["City", "Route", "Date", "Hour"], keep="last"
        )
        return self._df

    async def fetch(self, **kwargs) -> DataResult:
        df = self._load()
        city: Optional[str] = kwargs.get("city")
        hour: Optional[int] = kwargs.get("hour")

        filtered = df.copy()
        if city:
            filtered = filtered[filtered["City"].str.lower() == city.lower()]
        if hour is not None:
            filtered = filtered[filtered["Hour"] == hour]
        if filtered.empty:
            hour = None  # fallback to all hours if specific hour yields no data
            filtered = df.copy()
            if city:
                filtered = filtered[filtered["City"].str.lower() == city.lower()]

        summary: Dict[str, Any] = {}
        for city_name in filtered["City"].unique():
            cd = filtered[filtered["City"] == city_name]
            total_vehicles = cd["Vehicles_Total"].sum()
            summary[city_name] = {
                "avg_speed_kmph": round(cd["Average_Speed_kmph"].mean(), 1),
                "congestion_pct": round(cd["Congestion_pct"].mean(), 1),
                "total_vehicles": int(cd["Vehicles_Total"].mean()),
                "ev_percentage": round(cd["EVs"].sum() / total_vehicles * 100, 2) if total_vehicles > 0 else 0,
                "top_routes": (
                    cd.groupby("Route")["Congestion_pct"]
                    .mean()
                    .nlargest(5)
                    .round(1)
                    .to_dict()
                ),
            }

        source_files = self._load()["_source_file"].unique().tolist()
        return DataResult(
            data=summary,
            source="NCR Traffic CSV",
            source_type=SourceType.CSV,
            domain=DataDomain.TRAFFIC,
            meta={"rows": len(self._load()), "source_files": source_files, "cities": list(summary.keys())},
        )

    async def health(self) -> Dict[str, Any]:
        base = await super().health()
        try:
            df = self._load()
            base["rows"] = len(df)
            base["cities"] = df["City"].unique().tolist()
            base["source_files"] = df["_source_file"].unique().tolist()
        except Exception as e:
            base["status"] = "error"
            base["error"] = str(e)[:100]
        return base


# ══════════════════════════════════════════════════════════════════
#  Historical JSON Adapter
# ══════════════════════════════════════════════════════════════════

class HistoricalJSONAdapter(DataAdapter):
    """
    Loads yearly historical metrics from data/historical_metrics.json.

    Kwargs accepted by fetch():
      (none — returns the full payload)
    """

    def __init__(self):
        super().__init__(ttl_seconds=0)
        self._data: Optional[Dict[str, Any]] = None

    @property
    def name(self) -> str:
        return "historical_json"

    @property
    def source_type(self) -> SourceType:
        return SourceType.JSON

    @property
    def domains(self) -> List[DataDomain]:
        return [DataDomain.HISTORICAL]

    @property
    def refresh_policy(self) -> RefreshPolicy:
        return RefreshPolicy.STATIC

    def _load(self) -> Dict[str, Any]:
        if self._data is not None:
            return self._data
        path = _DATA_DIR / "historical_metrics.json"
        if not path.exists():
            raise FileNotFoundError(f"Historical metrics file not found: {path}")
        with open(path) as f:
            self._data = json.load(f)
        return self._data

    async def fetch(self, **kwargs) -> DataResult:
        data = self._load()
        return DataResult(
            data=data,
            source="historical_metrics.json",
            source_type=SourceType.JSON,
            domain=DataDomain.HISTORICAL,
            meta={"records": len(data.get("records", []))},
        )


# ── Helpers ───────────────────────────────────────────────────────

def _aqi_category(aqi: float) -> str:
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Satisfactory"
    if aqi <= 200:
        return "Moderate"
    if aqi <= 300:
        return "Poor"
    if aqi <= 400:
        return "Very Poor"
    return "Severe"
