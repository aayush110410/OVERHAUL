"""Core KPI utilities for traffic-god analysis stack."""
from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np


def travel_time_stats(durations: Sequence[float]) -> dict:
    """Return avg/min/max travel times in seconds."""

    arr = np.array(list(durations), dtype=np.float64)
    if arr.size == 0:
        return {"avg": 0.0, "min": 0.0, "max": 0.0}
    return {"avg": float(arr.mean()), "min": float(arr.min()), "max": float(arr.max())}


def throughput(counts: Sequence[int], interval_seconds: float) -> float:
    """Vehicles per hour computed from counts per interval."""

    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    total = sum(counts)
    hours = (len(counts) * interval_seconds) / 3600.0
    return total / hours if hours else 0.0


def average_delay(travel_times: Sequence[float], free_flow_time: float) -> float:
    """Average delay per vehicle relative to free-flow travel time."""

    if free_flow_time <= 0:
        raise ValueError("free_flow_time must be positive")
    if not travel_times:
        return 0.0
    delays = [max(0.0, tt - free_flow_time) for tt in travel_times]
    return float(sum(delays) / len(delays))


def mota(tp: int, fp: int, fn: int, id_switches: int) -> float:
    """Compute Multiple Object Tracking Accuracy (MOTA)."""

    denominator = tp + fn
    if denominator == 0:
        return 0.0
    return 1.0 - (fp + fn + id_switches) / denominator
