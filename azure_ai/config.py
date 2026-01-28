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
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import threading


_DEFAULT_DOTENV_PATH = Path(__file__).resolve().parents[1] / ".env"

# ============================================================
# SINGLETON CACHED CONFIG - Loaded ONCE at startup
# This fixes the intermittent API key issues on Render
# ============================================================
_CACHED_CONFIG: Optional["AzureConfig"] = None
_CONFIG_LOCK = threading.Lock()
_DOTENV_LOADED = False


def _load_dotenv_once():
    """Load .env file exactly once at startup (for local dev only)."""
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    
    try:
        # Only load .env in local development, NOT in production
        # Render sets RENDER=true in the environment
        is_render = os.getenv("RENDER", "").lower() == "true"
        is_production = os.getenv("PRODUCTION", "").lower() == "true"
        
        if not is_render and not is_production and _DEFAULT_DOTENV_PATH.exists():
            from dotenv import load_dotenv
            dotenv_override = os.getenv("OVERHAUL_DOTENV_OVERRIDE", "1").strip() != "0"
            load_dotenv(dotenv_path=_DEFAULT_DOTENV_PATH, override=dotenv_override)
            print(f"[Azure Config] Loaded .env from {_DEFAULT_DOTENV_PATH}")
        elif is_render:
            print("[Azure Config] Running on Render - using environment variables directly")
        
        _DOTENV_LOADED = True
    except Exception as e:
        print(f"[Azure Config] Warning: Could not load .env: {e}")
        _DOTENV_LOADED = True  # Mark as done to avoid repeated attempts


def _env(name: str) -> Optional[str]:
    """Get environment variable, stripping whitespace."""
    value = os.environ.get(name)  # Use os.environ directly for reliability
    if value is None:
        return None
    value = value.strip()
    return value or None


def _mask_key(key: Optional[str]) -> str:
    """Mask API key for safe logging."""
    if not key:
        return "(not set)"
    if len(key) <= 8:
        return "****"
    return f"{key[:4]}...{key[-4:]}"


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
    
    def debug_info(self) -> dict:
        """Return safe debug info (keys masked)."""
        return {
            "azure_openai_endpoint": self.azure_openai_endpoint or "(not set)",
            "azure_openai_key": _mask_key(self.azure_openai_key),
            "azure_openai_deployment": self.azure_openai_deployment or "(not set)",
            "azure_openai_api_version": self.azure_openai_api_version,
            "azure_maps_key": _mask_key(self.azure_maps_key),
            "openai_enabled": bool(self.azure_openai_endpoint and self.azure_openai_key and self.azure_openai_deployment),
            "maps_enabled": bool(self.azure_maps_key),
        }


def _build_config() -> AzureConfig:
    """Build config from environment variables."""
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


def load_azure_config(force_reload: bool = False) -> AzureConfig:
    """
    Load Azure configuration from environment variables.
    
    IMPORTANT: Config is cached at startup for reliability.
    Use force_reload=True only for debugging.
    """
    global _CACHED_CONFIG
    
    # Fast path: return cached config
    if _CACHED_CONFIG is not None and not force_reload:
        return _CACHED_CONFIG
    
    with _CONFIG_LOCK:
        # Double-check after acquiring lock
        if _CACHED_CONFIG is not None and not force_reload:
            return _CACHED_CONFIG
        
        # Load .env once (for local dev)
        _load_dotenv_once()
        
        # Build and cache config
        _CACHED_CONFIG = _build_config()
        
        # Log configuration status at startup
        debug = _CACHED_CONFIG.debug_info()
        print(f"[Azure Config] ═══════════════════════════════════════════")
        print(f"[Azure Config] OpenAI Endpoint: {debug['azure_openai_endpoint']}")
        print(f"[Azure Config] OpenAI Key: {debug['azure_openai_key']}")
        print(f"[Azure Config] OpenAI Deployment: {debug['azure_openai_deployment']}")
        print(f"[Azure Config] OpenAI API Version: {debug['azure_openai_api_version']}")
        print(f"[Azure Config] Maps Key: {debug['azure_maps_key']}")
        print(f"[Azure Config] ───────────────────────────────────────────")
        print(f"[Azure Config] OpenAI Enabled: {debug['openai_enabled']}")
        print(f"[Azure Config] Maps Enabled: {debug['maps_enabled']}")
        print(f"[Azure Config] ═══════════════════════════════════════════")
        
        return _CACHED_CONFIG


def get_config_debug_info() -> dict:
    """Get debug info about current configuration (for API endpoint)."""
    cfg = load_azure_config()
    return {
        **cfg.debug_info(),
        "environment": {
            "RENDER": os.environ.get("RENDER", "(not set)"),
            "PRODUCTION": os.environ.get("PRODUCTION", "(not set)"),
            "OVERHAUL_AZURE_ONLY": os.environ.get("OVERHAUL_AZURE_ONLY", "(not set)"),
        },
        "cached": _CACHED_CONFIG is not None,
    }


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
    """Check if Azure OpenAI is properly configured."""
    cfg = cfg or load_azure_config()
    enabled = bool(cfg.azure_openai_endpoint and cfg.azure_openai_key and cfg.azure_openai_deployment)
    return enabled


def azure_maps_enabled(cfg: Optional[AzureConfig] = None) -> bool:
    """Check if Azure Maps is properly configured."""
    cfg = cfg or load_azure_config()
    enabled = bool(cfg.azure_maps_key)
    return enabled


# ============================================================
# PRELOAD CONFIG AT MODULE IMPORT
# This ensures config is loaded immediately when the module is imported
# ============================================================
def _preload_config():
    """Preload configuration at module import time."""
    try:
        load_azure_config()
    except Exception as e:
        print(f"[Azure Config] Warning: Failed to preload config: {e}")

# Preload on import
_preload_config()
