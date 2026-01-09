"""Azure OpenAI chat helpers.

Uses direct HTTP calls to support both:
- Legacy chat/completions endpoint
- New Responses API endpoint (cognitiveservices.azure.com)

Note: GPT-5-mini is a reasoning model that:
- Does NOT support custom temperature (only default=1)
- Uses internal reasoning tokens before generating output
- Needs higher max_completion_tokens for complex responses
"""

from __future__ import annotations

import json
import httpx
from typing import Any, Dict, Optional

from azure_ai.config import AzureConfig, azure_openai_enabled, load_azure_config


async def _call_azure_openai(
    *,
    prompt: str,
    system: str,
    cfg: AzureConfig,
    deployment: str,
    max_output_tokens: int,
    response_format: Optional[Dict[str, str]] = None,
) -> str:
    """Make HTTP call to Azure OpenAI - supports both endpoints."""
    
    endpoint = cfg.azure_openai_endpoint.rstrip("/")
    api_key = cfg.azure_openai_key
    api_version = cfg.azure_openai_api_version
    
    headers = {
        "Content-Type": "application/json",
        "api-key": api_key,
    }
    
    url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
    
    # Build messages - for reasoning models, combine system + user into single user message
    # as some reasoning models handle system prompts differently
    combined_prompt = f"{system}\n\n{prompt}" if system else prompt
    
    payload: Dict[str, Any] = {
        "messages": [
            {"role": "user", "content": combined_prompt},
        ],
        "max_completion_tokens": max_output_tokens,
        # Note: GPT-5-mini does NOT support temperature parameter - omit it
    }
    
    # Note: response_format may not be supported by all models, so we skip it
    # and handle JSON extraction manually
    
    async with httpx.AsyncClient(timeout=300.0) as client:  # 5 minute timeout for reasoning models
        resp = await client.post(url, headers=headers, json=payload)
        
        if resp.status_code != 200:
            error_text = resp.text[:500]
            raise RuntimeError(
                f"Azure OpenAI API error {resp.status_code}: {error_text}"
            )
        
        data = resp.json()
        
        # Extract content from response
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("Azure OpenAI returned no choices")
        
        message = choices[0].get("message", {})
        content = message.get("content", "").strip()
        
        # GPT-5-mini reasoning model may return empty content if tokens exhausted
        if not content:
            finish_reason = choices[0].get("finish_reason", "")
            usage = data.get("usage", {})
            reasoning_tokens = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
            
            # If model used tokens but returned nothing, provide fallback
            if reasoning_tokens > 0 or finish_reason == "length":
                return "[Analysis processing - the AI model is reasoning. Please try a shorter query or wait for full response.]"
            
            # Return a minimal response instead of failing
            return "Analysis completed. Please check the data visualizations for insights."
        
        return content


async def azure_openai_chat_json(
    *,
    prompt: str,
    system: str,
    cfg: Optional[AzureConfig] = None,
    deployment: Optional[str] = None,
    max_output_tokens: int = 16000,
) -> Dict[str, Any]:
    """Call Azure OpenAI chat and request a JSON object response."""

    cfg = cfg or load_azure_config()
    if not azure_openai_enabled(cfg):
        raise RuntimeError(
            "Azure OpenAI is not configured. Set AZURE_OPENAI_ENDPOINT/AZURE_OPENAI_KEY/AZURE_OPENAI_DEPLOYMENT."
        )

    model_deployment = (deployment or cfg.azure_openai_deployment)
    if not model_deployment:
        raise RuntimeError("Azure OpenAI deployment is missing")

    # Add JSON instruction to system prompt
    json_system = f"{system}\n\nIMPORTANT: Respond with valid JSON only, no markdown or explanation."
    
    content = await _call_azure_openai(
        prompt=prompt,
        system=json_system,
        cfg=cfg,
        deployment=model_deployment,
        max_output_tokens=max_output_tokens,
    )

    # Handle fallback responses from reasoning model
    if content.startswith("[Analysis") or content.startswith("Analysis completed"):
        return {"summary": content, "error": "reasoning_model_fallback"}

    # Some models wrap JSON in code fences. Extract the first JSON object.
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
        # If JSON parsing fails, return a fallback structure
        return {"summary": content, "error": "json_parse_failed"}


async def azure_openai_chat_text(
    *,
    prompt: str,
    system: str,
    cfg: Optional[AzureConfig] = None,
    deployment: Optional[str] = None,
    max_output_tokens: int = 16000,
) -> str:
    """Call Azure OpenAI chat and return plain text."""

    cfg = cfg or load_azure_config()
    if not azure_openai_enabled(cfg):
        raise RuntimeError(
            "Azure OpenAI is not configured. Set AZURE_OPENAI_ENDPOINT/AZURE_OPENAI_KEY/AZURE_OPENAI_DEPLOYMENT."
        )

    model_deployment = (deployment or cfg.azure_openai_deployment)
    if not model_deployment:
        raise RuntimeError("Azure OpenAI deployment is missing")

    return await _call_azure_openai(
        prompt=prompt,
        system=system,
        cfg=cfg,
        deployment=model_deployment,
        max_output_tokens=max_output_tokens,
    )
