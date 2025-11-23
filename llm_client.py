"""LLM Client abstraction.

This module centralizes all interactions with LLM providers so we can swap between
OpenAI, Gemini, local models, etc. The coordinator and downstream agents only call
`LLMClient.generate()` with structured prompts.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional


class LLMClient:
    """Simple pluggable LLM wrapper.

    TODO: Replace the placeholder logic with real SDK calls (OpenAI, Gemini, Llama.cpp, etc.)
    once API keys and deployment endpoints are available.
    """

    def __init__(self) -> None:
        self.provider = os.getenv("LLM_PROVIDER", "mock")
        self.model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Return a synthetic completion for now.

        Agentic design pattern linkage:
        - Prompt chaining: upstream stages build structured prompts, this client executes them.
        - Routing: the coordinator decides which agent should call `generate` based on task type.
        - Critic/safety: critic agent can call back into the LLM client for red-team variants.
        """

        if self.provider == "mock":
            return (
                "[LLM MOCK RESPONSE]\n"
                f"model={self.model}, temp={self.temperature}\n"
                f"prompt_snippet={prompt[:256]}..."
            )
        # TODO: implement concrete providers (OpenAI, Gemini, etc.) using official SDKs
        raise NotImplementedError(
            "Configure a supported LLM provider or extend `LLMClient` with custom logic."
        )


llm_client = LLMClient()
