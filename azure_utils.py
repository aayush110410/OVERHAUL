"""Compatibility shim for Azure integration.

All Azure integration code now lives under the `azure_ai/` package to keep the
repo clean and make rollback easy.

This file remains only to avoid breaking older imports during transition.
Prefer importing from `azure_ai.*` directly.
"""

from azure_ai.config import AzureConfig, azure_maps_enabled, azure_openai_enabled, load_azure_config, get_config_debug_info
from azure_ai.maps.search import azure_maps_geocode, azure_maps_reverse_geocode
from azure_ai.openai.chat import azure_openai_chat_json, azure_openai_chat_text

__all__ = [
    "AzureConfig",
    "load_azure_config",
    "get_config_debug_info",
    "azure_openai_enabled",
    "azure_openai_chat_json",
    "azure_openai_chat_text",
    "azure_maps_enabled",
    "azure_maps_geocode",
    "azure_maps_reverse_geocode",
]
