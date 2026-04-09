"""Chat helpers for OVERHAUL's multi-model LLM stack.

Models available:
  Via OpenRouter:
    - qwen_chat_text()    – Qwen 3 4B        (fast reasoning, chat)
    - llama_chat_text()   – Llama 3.3 70B    (heavy analysis, predictions)
    - gpt_oss_chat_text() – GPT-OSS-120B     (cross-validation, second opinion)
  Via Google AI:
    - gemini_chat_text()  – Gemini 3.1 Pro   (deep reasoning, agents, search)

Unified helpers:
    - llm_chat_text()     – auto-picks based on task type
    - llm_chat_json()     – auto-picks, returns parsed JSON
    - llm_ensemble()      – runs multiple models & merges results
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional

import httpx

from llm.config import LLMConfig, gemini_enabled, load_llm_config, qwen_enabled
from llm.usage_tracker import record_qwen_usage, record_gemini_usage, display_usage

# ── Retry config ──
_MAX_RETRIES = 2
_BACKOFF_BASE = 2.0


# ═══════════════════════════════════════════════════════════════
# QWEN 3 4B  (via OpenRouter – https://openrouter.ai)
# ═══════════════════════════════════════════════════════════════

async def qwen_chat_text(
    *,
    prompt: str,
    system: str = "",
    cfg: Optional[LLMConfig] = None,
    model: Optional[str] = None,
    max_output_tokens: int = 12000,
) -> str:
    """Call Qwen 3 4B via OpenRouter and return plain text."""
    cfg = cfg or load_llm_config()
    if not cfg.openrouter_api_key:
        raise RuntimeError("OpenRouter API key not set (OPENROUTER_API_KEY)")

    model_name = model or cfg.qwen_model
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {cfg.openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://overhaul-ncr.onrender.com",
        "X-Title": "OVERHAUL Traffic Intelligence",
    }

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_output_tokens,
        "temperature": 0.7,
    }

    last_error: Optional[Exception] = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                resp = await client.post(url, headers=headers, json=payload)

            if resp.status_code in (429, 500, 502, 503, 504):
                last_error = RuntimeError(f"OpenRouter API error {resp.status_code}: {resp.text[:300]}")
                if attempt < _MAX_RETRIES:
                    wait = _BACKOFF_BASE * (2 ** attempt)
                    if resp.status_code == 429:
                        retry_after = resp.headers.get("Retry-After")
                        if retry_after:
                            try:
                                wait = max(wait, float(retry_after))
                            except ValueError:
                                pass
                    print(f"[Qwen] Retrying in {wait:.1f}s (attempt {attempt+1}/{_MAX_RETRIES}, status={resp.status_code})")
                    await asyncio.sleep(wait)
                    continue
                raise last_error

            if resp.status_code != 200:
                raise RuntimeError(f"OpenRouter API error {resp.status_code}: {resp.text[:500]}")

            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                raise RuntimeError("OpenRouter returned no choices")

            # Track token usage
            usage = data.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            record_qwen_usage(input_tokens, output_tokens)

            # Display usage periodically (every 10 calls)
            from llm.usage_tracker import get_usage_tracker
            stats = get_usage_tracker().get_stats()
            if stats.total_calls % 10 == 1:  # Show on 1st, 11th, 21st call, etc.
                display_usage(1000000)  # Assuming 1M token limit

            content = choices[0].get("message", {}).get("content", "")
            # GPT-OSS-120B is a reasoning model: content may be null,
            # with the actual output in the "reasoning" field.
            if not content:
                content = choices[0].get("message", {}).get("reasoning", "")
            content = (content or "").strip()
            if not content:
                return "Analysis completed. Please check the data visualizations for insights."
            return content

        except httpx.TimeoutException:
            last_error = RuntimeError("Qwen request timed out (180s)")
            if attempt < _MAX_RETRIES:
                await asyncio.sleep(_BACKOFF_BASE * (2 ** attempt))
                continue
            raise last_error
        except httpx.ConnectError as exc:
            last_error = RuntimeError(f"OpenRouter connection error: {exc}")
            if attempt < _MAX_RETRIES:
                await asyncio.sleep(_BACKOFF_BASE * (2 ** attempt))
                continue
            raise last_error

    raise last_error or RuntimeError("Qwen call failed after retries")


# ════════════════════════    exit
# ═══════════════════════════════════════
# GEMINI 3 PRO PREVIEW  (via Google AI API)
# ═══════════════════════════════════════════════════════════════

async def gemini_chat_text(
    *,
    prompt: str,
    system: str = "",
    cfg: Optional[LLMConfig] = None,
    model: Optional[str] = None,
    max_output_tokens: int = 12000,
    enable_search: bool = False,
) -> str:
    """Call Gemini 3.1 Pro Preview via Google AI API and return plain text."""
    cfg = cfg or load_llm_config()
    if not cfg.gemini_api_key:
        raise RuntimeError("Gemini API key not set (GEMINI_API_KEY)")

    model_name = model or cfg.gemini_model
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"

    contents = []
    if system:
        contents.append({"role": "user", "parts": [{"text": f"System instructions: {system}"}]})
        contents.append({"role": "model", "parts": [{"text": "Understood. I will follow these instructions."}]})
    contents.append({"role": "user", "parts": [{"text": prompt}]})

    payload: Dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": max_output_tokens,
        },
    }

    if enable_search:
        payload["tools"] = [{"googleSearch": {}}]

    last_error: Optional[Exception] = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                resp = await client.post(url, params={"key": cfg.gemini_api_key}, json=payload)

            if resp.status_code in (429, 500, 502, 503, 504):
                last_error = RuntimeError(f"Gemini API error {resp.status_code}: {resp.text[:300]}")
                if attempt < _MAX_RETRIES:
                    wait = _BACKOFF_BASE * (2 ** attempt)
                    print(f"[Gemini] Retrying in {wait:.1f}s (attempt {attempt+1}/{_MAX_RETRIES})")
                    await asyncio.sleep(wait)
                    continue
                raise last_error

            if resp.status_code != 200:
                raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text[:500]}")

            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                raise RuntimeError("Gemini returned no candidates")

            # Track token usage
            usage = data.get("usageMetadata", {})
            input_tokens = usage.get("promptTokenCount", 0)
            output_tokens = usage.get("candidatesTokenCount", 0)
            record_gemini_usage(input_tokens, output_tokens)

            # Display usage periodically (every 10 calls)
            from llm.usage_tracker import get_usage_tracker
            stats = get_usage_tracker().get_stats()
            if stats.total_calls % 10 == 1:  # Show on 1st, 11th, 21st call, etc.
                display_usage(1000000)  # Assuming 1M token limit

            parts = candidates[0].get("content", {}).get("parts", [])
            text_parts = [p.get("text", "") for p in parts if "text" in p]
            result = "\n".join(text_parts).strip()
            return result or "Analysis completed."

        except httpx.TimeoutException:
            last_error = RuntimeError("Gemini request timed out (180s)")
            if attempt < _MAX_RETRIES:
                await asyncio.sleep(_BACKOFF_BASE * (2 ** attempt))
                continue
            raise last_error
        except httpx.ConnectError as exc:
            last_error = RuntimeError(f"Gemini connection error: {exc}")
            if attempt < _MAX_RETRIES:
                await asyncio.sleep(_BACKOFF_BASE * (2 ** attempt))
                continue
            raise last_error

    raise last_error or RuntimeError("Gemini call failed after retries")


# ═══════════════════════════════════════════════════════════════
# LLAMA 3.3 70B INSTRUCT  (via OpenRouter – free)
# ═══════════════════════════════════════════════════════════════

async def llama_chat_text(
    *,
    prompt: str,
    system: str = "",
    cfg: Optional[LLMConfig] = None,
    max_output_tokens: int = 12000,
) -> str:
    """Call Llama 3.3 70B Instruct via OpenRouter. Best for deep analysis."""
    cfg = cfg or load_llm_config()
    return await qwen_chat_text(
        prompt=prompt,
        system=system,
        cfg=cfg,
        model=cfg.llama_model,
        max_output_tokens=max_output_tokens,
    )


# ═══════════════════════════════════════════════════════════════
# GPT-OSS-120B  (via OpenRouter – free)
# ═══════════════════════════════════════════════════════════════

async def gpt_oss_chat_text(
    *,
    prompt: str,
    system: str = "",
    cfg: Optional[LLMConfig] = None,
    max_output_tokens: int = 12000,
) -> str:
    """Call GPT-OSS-120B via OpenRouter. Best for cross-validation."""
    cfg = cfg or load_llm_config()
    return await qwen_chat_text(
        prompt=prompt,
        system=system,
        cfg=cfg,
        model=cfg.gpt_oss_model,
        max_output_tokens=max_output_tokens,
    )


# ═══════════════════════════════════════════════════════════════
# UNIFIED HELPERS – smart model routing
# ═══════════════════════════════════════════════════════════════

# Task → model mapping for maximum power:
#   "fast"       → Qwen 3 4B       (sub-second, chat responses)
#   "analysis"   → Llama 3.3 70B   (deep data analysis, predictions)
#   "validate"   → GPT-OSS-120B    (cross-check numbers, second opinion)
#   "reason"     → Gemini 3.1 Pro  (orchestration, search grounding)
#   "qwen"/"gemini" → legacy compat

_TASK_MODEL_MAP = {
    "fast": "qwen",
    "analysis": "llama",
    "validate": "gpt_oss",
    "reason": "gemini",
    "qwen": "qwen",
    "gemini": "gemini",
    "llama": "llama",
    "gpt_oss": "gpt_oss",
}


async def llm_chat_text(
    *,
    prompt: str,
    system: str = "",
    cfg: Optional[LLMConfig] = None,
    max_output_tokens: int = 12000,
    prefer: str = "qwen",
) -> str:
    """Call the best available LLM and return plain text.

    prefer values: "qwen", "gemini", "llama", "gpt_oss",
                   "fast", "analysis", "validate", "reason"
    Falls back through the chain: preferred → alternatives → error.
    """
    cfg = cfg or load_llm_config()
    resolved = _TASK_MODEL_MAP.get(prefer, prefer)

    # Build ordered fallback chain
    _CHAIN = {
        "qwen":    [qwen_chat_text, llama_chat_text, gpt_oss_chat_text, gemini_chat_text],
        "llama":   [llama_chat_text, gpt_oss_chat_text, qwen_chat_text, gemini_chat_text],
        "gpt_oss": [gpt_oss_chat_text, llama_chat_text, qwen_chat_text, gemini_chat_text],
        "gemini":  [gemini_chat_text, llama_chat_text, qwen_chat_text, gpt_oss_chat_text],
    }
    chain = _CHAIN.get(resolved, _CHAIN["qwen"])

    last_error = None
    for fn in chain:
        # Check if the provider is available
        if fn in (qwen_chat_text, llama_chat_text, gpt_oss_chat_text) and not qwen_enabled(cfg):
            continue
        if fn is gemini_chat_text and not gemini_enabled(cfg):
            continue
        try:
            return await fn(prompt=prompt, system=system, cfg=cfg, max_output_tokens=max_output_tokens)
        except Exception as e:
            last_error = e
            print(f"[LLM] {fn.__name__} failed: {e}, trying next...")

    raise last_error or RuntimeError(
        "No LLM configured. Set OPENROUTER_API_KEY and/or GEMINI_API_KEY."
    )


async def llm_chat_json(
    *,
    prompt: str,
    system: str = "",
    cfg: Optional[LLMConfig] = None,
    max_output_tokens: int = 16000,
    prefer: str = "qwen",
) -> Dict[str, Any]:
    """Call best available LLM and return parsed JSON dict."""
    json_system = f"{system}\n\nIMPORTANT: Respond with valid JSON only, no markdown or explanation."

    content = await llm_chat_text(
        prompt=prompt,
        system=json_system,
        cfg=cfg,
        max_output_tokens=max_output_tokens,
        prefer=prefer,
    )

    # Handle fallback/placeholder responses
    if content.startswith("[Analysis") or content.startswith("Analysis completed"):
        return {"summary": content, "error": "model_fallback"}

    # Strip code fences
    content = content.replace("```json", "```")
    if "```" in content:
        parts = content.split("```")
        content = max((p.strip() for p in parts if p.strip()), key=len)

    start = content.find("{")
    end = content.rfind("}")
    if start >= 0 and end > start:
        content = content[start : end + 1]

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"summary": content, "error": "json_parse_failed"}


async def llm_ensemble(
    *,
    prompt: str,
    system: str = "",
    cfg: Optional[LLMConfig] = None,
    models: Optional[list] = None,
    max_output_tokens: int = 12000,
) -> Dict[str, Any]:
    """Run the same prompt through multiple models in parallel and return all results.

    This enables cross-validation: run Llama for analysis, GPT-OSS for verification,
    and Qwen for a quick take — then compare/merge in the caller.

    Returns: {"model_name": "response_text", ...}
    Default models: ["llama", "gpt_oss", "qwen"]
    """
    cfg = cfg or load_llm_config()
    models = models or ["llama", "gpt_oss", "qwen"]

    _FN_MAP = {
        "qwen": qwen_chat_text,
        "llama": llama_chat_text,
        "gpt_oss": gpt_oss_chat_text,
        "gemini": gemini_chat_text,
    }

    async def _call(name: str) -> tuple:
        fn = _FN_MAP.get(name)
        if not fn:
            return name, f"Unknown model: {name}"
        try:
            result = await fn(prompt=prompt, system=system, cfg=cfg, max_output_tokens=max_output_tokens)
            return name, result
        except Exception as e:
            return name, f"[ERROR] {e}"

    # Filter to available models
    available = []
    for m in models:
        if m in ("qwen", "llama", "gpt_oss") and qwen_enabled(cfg):
            available.append(m)
        elif m == "gemini" and gemini_enabled(cfg):
            available.append(m)

    results = await asyncio.gather(*[_call(m) for m in available])
    return dict(results)
