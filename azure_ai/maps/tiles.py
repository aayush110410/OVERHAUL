"""Azure Maps tile proxy helper.

This is used by FastAPI endpoints to fetch Azure Maps tiles server-side.
"""

from __future__ import annotations

from typing import Optional

import httpx

from azure_ai.config import AzureConfig, azure_maps_enabled, load_azure_config


async def azure_maps_fetch_tile(
    *,
    tileset_id: str,
    z: int,
    x: int,
    y: int,
    cfg: Optional[AzureConfig] = None,
) -> bytes:
    cfg = cfg or load_azure_config()
    if not azure_maps_enabled(cfg):
        raise RuntimeError("Azure Maps is not configured. Set AZURE_MAPS_KEY.")

    url = "https://atlas.microsoft.com/map/tile"
    params = {
        "api-version": "2.1",
        "tilesetId": tileset_id,
        "zoom": str(z),
        "x": str(x),
        "y": str(y),
    }
    headers = {"subscription-key": cfg.azure_maps_key}

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url, params=params, headers=headers)
        resp.raise_for_status()
        return resp.content
