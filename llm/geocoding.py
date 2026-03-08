"""Free geocoding via Nominatim (OpenStreetMap).

Replaces Azure Maps geocoding – no API key required.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

_NOMINATIM_BASE = "https://nominatim.openstreetmap.org"
_HEADERS = {"User-Agent": "OVERHAUL-TrafficIntelligence/1.0"}


async def geocode(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Forward geocode – address/place name → coordinates."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            f"{_NOMINATIM_BASE}/search",
            params={"q": query, "format": "json", "limit": limit, "addressdetails": 1},
            headers=_HEADERS,
        )
        resp.raise_for_status()
        results = resp.json()

    return [
        {
            "display_name": r.get("display_name", ""),
            "lat": float(r["lat"]),
            "lon": float(r["lon"]),
            "type": r.get("type", ""),
            "address": r.get("address", {}),
        }
        for r in results
    ]


async def reverse_geocode(lat: float, lon: float) -> Dict[str, Any]:
    """Reverse geocode – coordinates → address."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            f"{_NOMINATIM_BASE}/reverse",
            params={"lat": lat, "lon": lon, "format": "json", "addressdetails": 1},
            headers=_HEADERS,
        )
        resp.raise_for_status()
        data = resp.json()

    return {
        "display_name": data.get("display_name", ""),
        "lat": float(data.get("lat", lat)),
        "lon": float(data.get("lon", lon)),
        "address": data.get("address", {}),
    }
