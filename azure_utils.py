"""Azure integration helpers.

This module centralizes calls to Microsoft services used by OVERHAUL:
- Azure OpenAI Service: LDRAGO natural-language reasoning.
- Azure Maps: spatial context (geocoding / reverse geocoding).

Design goals:
- No secrets in code; everything via env vars.
- Safe defaults, strict input validation, explicit timeouts.
- Small, reusable functions to keep FastAPI endpoints clean.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx


def _env(name: str) -> Optional[str]:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


@dataclass(frozen=True)
class AzureConfig:
    azure_openai_endpoint: Optional[str]
    azure_openai_key: Optional[str]
    azure_openai_deployment: Optional[str]
    azure_openai_api_version: str

    azure_resource_group: Optional[str]
    azure_maps_key: Optional[str]

    """Compatibility shim.

    All Azure integration code now lives under the `azure_ai/` package to keep the
    repo clean and make rollback easy.

    This file remains only to avoid breaking older imports during transition.
    Prefer importing from `azure_ai.*` directly.
    """

    from azure_ai.config import AzureConfig, azure_maps_enabled, azure_openai_enabled, load_azure_config
    from azure_ai.maps.search import azure_maps_geocode, azure_maps_reverse_geocode
    from azure_ai.openai.chat import azure_openai_chat_json, azure_openai_chat_text

    __all__ = [
        "AzureConfig",
        "load_azure_config",
        "azure_openai_enabled",
        "azure_openai_chat_json",
        "azure_openai_chat_text",
        "azure_maps_enabled",
        "azure_maps_geocode",
        "azure_maps_reverse_geocode",
    ]
async def azure_openai_chat_json(
