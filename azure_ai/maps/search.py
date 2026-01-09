"""Azure Maps search helpers (geocoding / reverse geocoding)."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

import httpx

from azure_ai.config import AzureConfig, azure_maps_enabled, load_azure_config

_SAFE_QUERY_RE = re.compile(r"[^\w\s,.-]", re.UNICODE)


async def azure_maps_geocode(
    *,
    query: str,
    cfg: Optional[AzureConfig] = None,
    limit: int = 1,
) -> Dict[str, Any]:
    cfg = cfg or load_azure_config()
    if not azure_maps_enabled(cfg):
        raise RuntimeError("Azure Maps is not configured. Set AZURE_MAPS_KEY.")

    q = (query or "").strip()
    q = _SAFE_QUERY_RE.sub(" ", q)
    q = re.sub(r"\s+", " ", q).strip()
    if not q:
        raise ValueError("Query is required")

    url = "https://atlas.microsoft.com/search/address/json"
    params = {
        "api-version": "1.0",
        "subscription-key": cfg.azure_maps_key,
        "query": q,
        "limit": str(max(1, min(limit, 10))),
    }

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


async def azure_maps_reverse_geocode(
    *,
    lat: float,
    lon: float,
    cfg: Optional[AzureConfig] = None,
) -> Dict[str, Any]:
    cfg = cfg or load_azure_config()
    if not azure_maps_enabled(cfg):
        raise RuntimeError("Azure Maps is not configured. Set AZURE_MAPS_KEY.")

    url = "https://atlas.microsoft.com/search/address/reverse/json"
    params = {
        "api-version": "1.0",
        "subscription-key": cfg.azure_maps_key,
        "query": f"{lat},{lon}",
    }

    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()
