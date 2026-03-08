"""LLM configuration.

Central config for all AI model providers used by OVERHAUL.

Providers (via OpenRouter):
  1. Qwen 3 4B               – Fast reasoning (chat, JSON)
  2. Llama 3.3 70B Instruct  – Heavy-duty analysis & predictions (free)
  3. GPT-OSS-120B            – Cross-validation & second opinion (free)

Providers (via Google AI):
  4. Gemini 3.1 Pro Preview  – Deep reasoning, orchestration, agents
  5. Imagen 3                – AI-generated map overlays

All secrets come from environment variables.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


_DEFAULT_DOTENV_PATH = Path(__file__).resolve().parents[1] / ".env"
_CONFIG_LOCK = threading.Lock()
_CACHED_CONFIG: Optional["LLMConfig"] = None
_DOTENV_LOADED = False


def _load_dotenv_once():
    """Load .env file exactly once (local dev only)."""
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    try:
        is_render = os.getenv("RENDER", "").lower() == "true"
        is_production = os.getenv("PRODUCTION", "").lower() == "true"
        if not is_render and not is_production and _DEFAULT_DOTENV_PATH.exists():
            from dotenv import load_dotenv
            load_dotenv(dotenv_path=_DEFAULT_DOTENV_PATH, override=True)
            print(f"[LLM Config] Loaded .env from {_DEFAULT_DOTENV_PATH}")
        elif is_render:
            print("[LLM Config] Running on Render – using environment variables directly")
        _DOTENV_LOADED = True
    except Exception as e:
        print(f"[LLM Config] Warning: Could not load .env: {e}")
        _DOTENV_LOADED = True


def _env(name: str) -> Optional[str]:
    val = os.environ.get(name)
    if val is None:
        return None
    val = val.strip()
    return val or None


def _mask(key: Optional[str]) -> str:
    if not key:
        return "(not set)"
    if len(key) <= 8:
        return "****"
    return f"{key[:4]}...{key[-4:]}"


@dataclass(frozen=True)
class LLMConfig:
    # ── OpenRouter models ──
    openrouter_api_key: Optional[str]
    qwen_model: str       # Fast chat: "qwen/qwen3-4b:free"
    llama_model: str      # Heavy analysis: "meta-llama/llama-3.3-70b-instruct:free"
    gpt_oss_model: str    # Cross-validation: "openai/gpt-oss-120b:free"

    # ── Google AI models ──
    gemini_api_key: Optional[str]
    gemini_model: str     # Deep reasoning: "gemini-3.1-pro-preview"

    def debug_info(self) -> dict:
        return {
            "openrouter_key": _mask(self.openrouter_api_key),
            "qwen_model": self.qwen_model,
            "llama_model": self.llama_model,
            "gpt_oss_model": self.gpt_oss_model,
            "gemini_key": _mask(self.gemini_api_key),
            "gemini_model": self.gemini_model,
            "qwen_enabled": bool(self.openrouter_api_key),
            "gemini_enabled": bool(self.gemini_api_key),
            "models_available": sum([
                bool(self.openrouter_api_key),  # gives access to 3 models
                bool(self.gemini_api_key),
            ]),
        }


def _build_config() -> LLMConfig:
    return LLMConfig(
        openrouter_api_key=_env("OPENROUTER_API_KEY"),
        qwen_model=_env("QWEN_MODEL") or "qwen/qwen3-4b:free",
        llama_model=_env("LLAMA_MODEL") or "meta-llama/llama-3.3-70b-instruct:free",
        gpt_oss_model=_env("GPT_OSS_MODEL") or "openai/gpt-oss-120b:free",
        gemini_api_key=_env("GEMINI_API_KEY"),
        gemini_model=_env("GEMINI_MODEL") or "gemini-3.1-pro-preview",
    )


def load_llm_config(force_reload: bool = False) -> LLMConfig:
    """Load LLM configuration (cached singleton)."""
    global _CACHED_CONFIG
    if _CACHED_CONFIG is not None and not force_reload:
        return _CACHED_CONFIG
    with _CONFIG_LOCK:
        if _CACHED_CONFIG is not None and not force_reload:
            return _CACHED_CONFIG
        _load_dotenv_once()
        _CACHED_CONFIG = _build_config()
        d = _CACHED_CONFIG.debug_info()
        print(f"[LLM Config] ═══════════════════════════════════════════")
        print(f"[LLM Config] OpenRouter Key: {d['openrouter_key']}")
        print(f"[LLM Config] Qwen Model:     {d['qwen_model']}")
        print(f"[LLM Config] Llama Model:    {d['llama_model']}")
        print(f"[LLM Config] GPT-OSS Model:  {d['gpt_oss_model']}")
        print(f"[LLM Config] Gemini Key:     {d['gemini_key']}")
        print(f"[LLM Config] Gemini Model:   {d['gemini_model']}")
        print(f"[LLM Config] ───────────────────────────────────────────")
        print(f"[LLM Config] OpenRouter OK:  {d['qwen_enabled']}")
        print(f"[LLM Config] Gemini OK:      {d['gemini_enabled']}")
        print(f"[LLM Config] ═══════════════════════════════════════════")
        return _CACHED_CONFIG


def qwen_enabled(cfg: Optional[LLMConfig] = None) -> bool:
    cfg = cfg or load_llm_config()
    return bool(cfg.openrouter_api_key)


def gemini_enabled(cfg: Optional[LLMConfig] = None) -> bool:
    cfg = cfg or load_llm_config()
    return bool(cfg.gemini_api_key)


def get_config_debug_info() -> dict:
    cfg = load_llm_config()
    return {
        **cfg.debug_info(),
        "environment": {
            "RENDER": os.environ.get("RENDER", "(not set)"),
            "PRODUCTION": os.environ.get("PRODUCTION", "(not set)"),
        },
        "cached": _CACHED_CONFIG is not None,
    }


# Preload on import
def _preload():
    try:
        load_llm_config()
    except Exception as e:
        print(f"[LLM Config] Warning: Failed to preload: {e}")

_preload()
