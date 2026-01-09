from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class CsvSummary:
    path: Path
    rows: int
    cols: int
    columns: List[str]
    preview: str
    stats_text: str


def _format_preview(df: pd.DataFrame, n: int = 8) -> str:
    head = df.head(n)
    return head.to_csv(index=False)


def _safe_numeric_stats(df: pd.DataFrame) -> str:
    numeric = df.select_dtypes(include=["number"])
    if numeric.empty:
        return "No numeric columns detected."

    desc = numeric.describe(percentiles=[0.1, 0.5, 0.9]).transpose()
    # Keep it compact
    keep_cols = [c for c in ["count", "mean", "std", "min", "10%", "50%", "90%", "max"] if c in desc.columns]
    desc = desc[keep_cols]
    return desc.to_string(max_rows=40)


def _try_time_trend(df: pd.DataFrame) -> Optional[str]:
    # Heuristic support for common schemas in your repo:
    # 1) TomTom daily: timestamp, avg_speed, jam_length_km, jam_count
    # 2) Delhi NCR hourly: Date, Hour, Average_Speed_kmph, Congestion_pct, Vehicles_Total

    cols = {c.lower(): c for c in df.columns}

    if "timestamp" in cols:
        ts = cols["timestamp"]
        try:
            tmp = df.copy()
            tmp[ts] = pd.to_datetime(tmp[ts], errors="coerce")
            tmp = tmp.dropna(subset=[ts])
            tmp = tmp.sort_values(ts)
            if tmp.empty:
                return None

            # daily aggregates
            day = tmp[ts].dt.date
            out = ["Time trend (by day):"]
            if "avg_speed" in cols:
                out.append("- avg_speed: start→end = " + f"{tmp[cols['avg_speed']].iloc[0]:.2f} → {tmp[cols['avg_speed']].iloc[-1]:.2f}")
            if "jam_length_km" in cols:
                out.append("- jam_length_km: start→end = " + f"{tmp[cols['jam_length_km']].iloc[0]:.2f} → {tmp[cols['jam_length_km']].iloc[-1]:.2f}")
            if "jam_count" in cols:
                out.append("- jam_count: start→end = " + f"{tmp[cols['jam_count']].iloc[0]:.0f} → {tmp[cols['jam_count']].iloc[-1]:.0f}")

            recent = tmp.tail(min(14, len(tmp)))
            out.append("Recent window (last rows):")
            show_cols = [c for c in [ts, cols.get("avg_speed"), cols.get("jam_length_km"), cols.get("jam_count")] if c]
            out.append(recent[show_cols].to_string(index=False))
            return "\n".join(out)
        except Exception:
            return None

    if "date" in cols and "hour" in cols:
        try:
            tmp = df.copy()
            tmp[cols["date"]] = pd.to_datetime(tmp[cols["date"]], errors="coerce")
            tmp = tmp.dropna(subset=[cols["date"]])
            tmp = tmp.sort_values([cols["date"], cols["hour"]])
            if tmp.empty:
                return None

            out = ["Time trend (Date+Hour):"]
            # speed / congestion columns if present
            speed_col = None
            for key in ["average_speed_kmph", "avg_speed", "speed_kmph"]:
                if key in cols:
                    speed_col = cols[key]
                    break
            cong_col = cols.get("congestion_pct")
            veh_col = cols.get("vehicles_total")

            if speed_col:
                out.append(f"- {speed_col}: min={tmp[speed_col].min():.2f}, max={tmp[speed_col].max():.2f}, mean={tmp[speed_col].mean():.2f}")
            if cong_col:
                out.append(f"- {cong_col}: min={tmp[cong_col].min():.2f}, max={tmp[cong_col].max():.2f}, mean={tmp[cong_col].mean():.2f}")
            if veh_col:
                out.append(f"- {veh_col}: min={tmp[veh_col].min():.0f}, max={tmp[veh_col].max():.0f}, mean={tmp[veh_col].mean():.0f}")

            # last day summary
            last_day = tmp[cols["date"]].dt.date.max()
            day_slice = tmp[tmp[cols["date"]].dt.date == last_day]
            out.append(f"Latest day ({last_day}) snapshot (top 8 rows):")
            out.append(day_slice.head(8).to_string(index=False))
            return "\n".join(out)
        except Exception:
            return None

    return None


def summarize_csv(path: Path, *, max_rows: int = 100_000) -> CsvSummary:
    df = pd.read_csv(path)
    if len(df) > max_rows:
        df = df.head(max_rows)

    preview = _format_preview(df)
    stats = _safe_numeric_stats(df)
    trend = _try_time_trend(df)

    stats_text = stats
    if trend:
        stats_text = stats_text + "\n\n" + trend

    return CsvSummary(
        path=path,
        rows=int(df.shape[0]),
        cols=int(df.shape[1]),
        columns=[str(c) for c in df.columns.tolist()],
        preview=preview,
        stats_text=stats_text,
    )


def csv_to_documents(summary: CsvSummary) -> List[Dict[str, str]]:
    base = {
        "source_path": str(summary.path.as_posix()),
        "type": "csv",
    }

    docs: List[Dict[str, str]] = []

    docs.append(
        {
            **base,
            "title": "CSV schema",
            "text": (
                f"CSV file: {summary.path.name}\n"
                f"Rows: {summary.rows}, Columns: {summary.cols}\n"
                f"Columns: {', '.join(summary.columns)}"
            ),
        }
    )

    docs.append(
        {
            **base,
            "title": "CSV preview",
            "text": (
                f"CSV file: {summary.path.name}\n"
                "Preview (first rows):\n" + summary.preview
            ),
        }
    )

    docs.append(
        {
            **base,
            "title": "CSV statistics",
            "text": (
                f"CSV file: {summary.path.name}\n"
                "Numeric statistics + trend notes:\n" + summary.stats_text
            ),
        }
    )

    return docs
