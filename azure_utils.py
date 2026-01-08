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

    # Optional: Azure ML online endpoint for surrogate inference
    azure_ml_endpoint: Optional[str]
    azure_ml_key: Optional[str]


def load_azure_config() -> AzureConfig:
    return AzureConfig(
        azure_openai_endpoint=_env("AZURE_OPENAI_ENDPOINT"),
        azure_openai_key=_env("AZURE_OPENAI_KEY"),
        azure_openai_deployment=_env("AZURE_OPENAI_DEPLOYMENT"),
        azure_openai_api_version=_env("AZURE_OPENAI_API_VERSION") or "2024-10-21",
        azure_resource_group=_env("AZURE_RESOURCE_GROUP"),
        azure_maps_key=_env("AZURE_MAPS_KEY"),
        azure_ml_endpoint=_env("AZURE_ML_ENDPOINT"),
        azure_ml_key=_env("AZURE_ML_KEY"),
    )


def azure_openai_enabled(cfg: Optional[AzureConfig] = None) -> bool:
    cfg = cfg or load_azure_config()
    return bool(cfg.azure_openai_endpoint and cfg.azure_openai_key and cfg.azure_openai_deployment)


async def azure_openai_chat_json(
    *,
    prompt: str,
    system: str,
    cfg: Optional[AzureConfig] = None,
    temperature: float = 0.2,
    max_output_tokens: int = 900,
) -> Dict[str, Any]:
    """Call Azure OpenAI chat and request a JSON object response.

    Returns a dict parsed from the model output.
    """

    cfg = cfg or load_azure_config()
    if not azure_openai_enabled(cfg):
        raise RuntimeError("Azure OpenAI is not configured. Set AZURE_OPENAI_ENDPOINT/AZURE_OPENAI_KEY/AZURE_OPENAI_DEPLOYMENT.")

    # Import lazily so local dev without openai installed can still import module.
    from openai import AsyncAzureOpenAI  # type: ignore

    client = AsyncAzureOpenAI(
        azure_endpoint=cfg.azure_openai_endpoint,
        api_key=cfg.azure_openai_key,
        api_version=cfg.azure_openai_api_version,
    )

    resp = await client.chat.completions.create(
        model=cfg.azure_openai_deployment,
        temperature=temperature,
        max_tokens=max_output_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        # Best-effort JSON mode; if the deployment/model doesn't support it,
        # we still attempt to parse JSON from the text content.
        response_format={"type": "json_object"},
    )

    content = (resp.choices[0].message.content or "").strip()
    if not content:
        raise RuntimeError("Azure OpenAI returned empty content")

    # Some models wrap JSON in code fences. Extract the first JSON object.
    content = content.replace("```json", "```")
    if "```" in content:
        parts = content.split("```")
        # take the largest fenced block as best signal
        content = max((p.strip() for p in parts if p.strip()), key=len)

    # Conservative extraction: find first '{' ... last '}'.
    start = content.find("{")
    end = content.rfind("}")
    if start >= 0 and end > start:
        content = content[start : end + 1]

    import json

    return json.loads(content)


async def azure_openai_chat_text(
    *,
    prompt: str,
    system: str,
    cfg: Optional[AzureConfig] = None,
    temperature: float = 0.2,
    max_output_tokens: int = 900,
) -> str:
    """Call Azure OpenAI chat and return plain text."""

    cfg = cfg or load_azure_config()
    if not azure_openai_enabled(cfg):
        raise RuntimeError(
            "Azure OpenAI is not configured. Set AZURE_OPENAI_ENDPOINT/AZURE_OPENAI_KEY/AZURE_OPENAI_DEPLOYMENT."
        )

    from openai import AsyncAzureOpenAI  # type: ignore

    client = AsyncAzureOpenAI(
        azure_endpoint=cfg.azure_openai_endpoint,
        api_key=cfg.azure_openai_key,
        api_version=cfg.azure_openai_api_version,
    )

    resp = await client.chat.completions.create(
        model=cfg.azure_openai_deployment,
        temperature=temperature,
        max_tokens=max_output_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )

    content = (resp.choices[0].message.content or "").strip()
    if not content:
        raise RuntimeError("Azure OpenAI returned empty content")
    return content


def azure_maps_enabled(cfg: Optional[AzureConfig] = None) -> bool:
    cfg = cfg or load_azure_config()
    return bool(cfg.azure_maps_key)


_SAFE_QUERY_RE = re.compile(r"[^\w\s,.-]", re.UNICODE)


async def azure_maps_geocode(
    *,
    query: str,
    cfg: Optional[AzureConfig] = None,
    limit: int = 1,
) -> Dict[str, Any]:
    """Geocode a query using Azure Maps.

    Returns raw Azure Maps JSON response.
    """

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


async def azure_ml_predict(
    *,
    payload: Dict[str, Any],
    cfg: Optional[AzureConfig] = None,
) -> Dict[str, Any]:
    """Optional Azure ML online endpoint invocation.

    This is an integration hook to run surrogate models on Azure ML.
    It is intentionally optional and only used when endpoint+key are configured.
    """

    cfg = cfg or load_azure_config()
    if not (cfg.azure_ml_endpoint and cfg.azure_ml_key):
        raise RuntimeError("Azure ML endpoint not configured. Set AZURE_ML_ENDPOINT and AZURE_ML_KEY.")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {cfg.azure_ml_key}",
    }
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(cfg.azure_ml_endpoint, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()
