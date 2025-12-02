"""TomTom Traffic API client.

This module wraps the Flow and Incident endpoints to obtain speed, congestion,
and incident metadata for configurable bounding boxes and intervals.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import requests

FLOW_BASE_URL = "https://api.tomtom.com/traffic/services/4"
INCIDENT_BASE_URL = "https://api.tomtom.com/traffic/services/5"


@dataclass
class BoundingBox:
    north: float
    south: float
    east: float
    west: float

    def as_params(self) -> str:
        return f"{self.north},{self.west},{self.south},{self.east}"


class TomTomClient:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def _request(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        merged = {"key": self.api_key, "unit": "KMPH"}
        if params:
            merged.update(params)
        response = requests.get(url, params=merged, timeout=15)
        response.raise_for_status()
        return response.json()

    def flow_segment_data(self, point: str, zoom: int = 12) -> Dict[str, Any]:
        params = {"point": point, "zoom": zoom}
        url = f"{FLOW_BASE_URL}/flowSegmentData/absolute/10/json"
        return self._request(url, params)

    def incidents_box(self, bbox: BoundingBox, fields: Optional[Iterable[str]] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "bbox": bbox.as_params(),
            "language": "en-GB",
            "relatedIncidents": "false",
        }
        if fields:
            params["fields"] = ",".join(fields)
        url = f"{INCIDENT_BASE_URL}/incidentDetails"
        return self._request(url, params)

    def flow_box(self, bbox: BoundingBox) -> Dict[str, Any]:
        params = {"bbox": bbox.as_params(), "flowSegmentData": "true"}
        url = f"{FLOW_BASE_URL}/flowSegmentData/absolute/10/json"
        return self._request(url, params)
