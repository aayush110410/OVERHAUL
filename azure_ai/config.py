"""Azure configuration.

All secrets must be provided via environment variables.

Required for Azure OpenAI:
- AZURE_OPENAI_ENDPOINT: e.g. https://<resource>.openai.azure.com/
- AZURE_OPENAI_KEY
- AZURE_OPENAI_DEPLOYMENT: the *deployment name* you created in Azure AI Foundry
Optional:
- AZURE_OPENAI_API_VERSION (default: 2024-10-21)

Required for Azure Maps:
- AZURE_MAPS_KEY

Optional:
- AZURE_ML_ENDPOINT / AZURE_ML_KEY
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


_DEFAULT_DOTENV_PATH = Path(__file__).resolve().parents[1] / ".env"


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
    azure_openai_deployment_ldrago: Optional[str]
    azure_openai_deployment_weather: Optional[str]
    azure_openai_deployment_behavior: Optional[str]
    azure_openai_deployment_policy_econ: Optional[str]
    azure_openai_api_version: str

    azure_maps_key: Optional[str]

    azure_ml_endpoint: Optional[str]
    azure_ml_key: Optional[str]


def load_azure_config() -> AzureConfig:
    # Local dev convenience: load .env from repo root if present.
    # In production, real environment variables should be used.
    try:
        if _DEFAULT_DOTENV_PATH.exists():
            from dotenv import load_dotenv  # type: ignore

            dotenv_override = os.getenv("OVERHAUL_DOTENV_OVERRIDE", "1").strip() != "0"
            load_dotenv(dotenv_path=_DEFAULT_DOTENV_PATH, override=dotenv_override)
    except Exception:
        pass

    return AzureConfig(
        azure_openai_endpoint=_env("AZURE_OPENAI_ENDPOINT"),
        azure_openai_key=_env("AZURE_OPENAI_KEY"),
        azure_openai_deployment=_env("AZURE_OPENAI_DEPLOYMENT"),
        azure_openai_deployment_ldrago=_env("AZURE_OPENAI_DEPLOYMENT_LDRAGO"),
        azure_openai_deployment_weather=_env("AZURE_OPENAI_DEPLOYMENT_WEATHER"),
        azure_openai_deployment_behavior=_env("AZURE_OPENAI_DEPLOYMENT_BEHAVIOR"),
        azure_openai_deployment_policy_econ=_env("AZURE_OPENAI_DEPLOYMENT_POLICY_ECON"),
        azure_openai_api_version=_env("AZURE_OPENAI_API_VERSION") or "2024-10-21",
        azure_maps_key=_env("AZURE_MAPS_KEY"),
        azure_ml_endpoint=_env("AZURE_ML_ENDPOINT"),
        azure_ml_key=_env("AZURE_ML_KEY"),
    )


def azure_openai_deployment_for(role: str, cfg: Optional[AzureConfig] = None) -> Optional[str]:
    """Return the Azure OpenAI deployment name for a given role.

    Role-specific env vars allow you to route different domain agents to different
    deployments while still supporting a single shared deployment via AZURE_OPENAI_DEPLOYMENT.
    """
    cfg = cfg or load_azure_config()
    role_norm = (role or "").strip().lower()
    if role_norm in {"ldrago", "master", "orchestrator"}:
        return cfg.azure_openai_deployment_ldrago or cfg.azure_openai_deployment
    if role_norm in {"weather"}:
        return cfg.azure_openai_deployment_weather or cfg.azure_openai_deployment
    if role_norm in {"behavior", "behaviour", "behavioral"}:
        return cfg.azure_openai_deployment_behavior or cfg.azure_openai_deployment
    if role_norm in {"policy", "policy_econ", "economics", "policy_economics"}:
        return cfg.azure_openai_deployment_policy_econ or cfg.azure_openai_deployment
    return cfg.azure_openai_deployment


def azure_openai_enabled(cfg: Optional[AzureConfig] = None) -> bool:
    cfg = cfg or load_azure_config()
    return bool(cfg.azure_openai_endpoint and cfg.azure_openai_key and cfg.azure_openai_deployment)


def azure_maps_enabled(cfg: Optional[AzureConfig] = None) -> bool:
    cfg = cfg or load_azure_config()
    return bool(cfg.azure_maps_key)
