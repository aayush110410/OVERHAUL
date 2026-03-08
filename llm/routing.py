"""Model routing strategy for OVERHAUL's multi-model LLM stack.

This module defines the intelligent routing logic that assigns the
optimal model to each cognitive task based on:
  - Task complexity (simple→Qwen, complex→Llama, synthesis→Gemini)
  - Latency budget (streaming chat→Qwen, deep analysis→Llama)
  - Cost optimization (free models preferred, Gemini for high-value)
  - Reliability (fallback chains for rate-limited free models)

Model Capabilities Matrix:
┌──────────────────────┬──────────┬───────────┬───────────┬──────────────┐
│ Capability           │ Qwen 4B  │ Llama 70B │ GPT-OSS   │ Gemini 3.1   │
├──────────────────────┼──────────┼───────────┼───────────┼──────────────┤
│ JSON parsing         │ ★★★★☆   │ ★★★☆☆    │ ★★☆☆☆    │ ★★★★★       │
│ Fast chat            │ ★★★★★   │ ★★★☆☆    │ ★★☆☆☆    │ ★★★★☆       │
│ Deep reasoning       │ ★★☆☆☆   │ ★★★★★    │ ★★★★☆    │ ★★★★★       │
│ Data analysis        │ ★★★☆☆   │ ★★★★★    │ ★★★★☆    │ ★★★★☆       │
│ Cross-validation     │ ★★☆☆☆   │ ★★★★☆    │ ★★★★★    │ ★★★☆☆       │
│ Multi-source synth.  │ ★★☆☆☆   │ ★★★☆☆    │ ★★★☆☆    │ ★★★★★       │
│ Web search grounding │ ☆☆☆☆☆   │ ☆☆☆☆☆    │ ☆☆☆☆☆    │ ★★★★★       │
│ Cost                 │ FREE     │ FREE      │ FREE      │ Pay-per-use  │
│ Latency (median)     │ <1s      │ ~3s       │ ~5s       │ ~3s          │
│ Context window       │ 32K      │ 128K      │ 128K      │ 1M           │
└──────────────────────┴──────────┴───────────┴───────────┴──────────────┘

LDRAGO v2 Pipeline Assignment:
  Parser      → Qwen 3 4B        (needs speed + JSON output)
  Planner     → Heuristic rules   (no LLM needed for most queries)
  Researcher  → Data APIs         (no LLM needed)
  Reasoner    → Llama 3.3 70B    (needs deep analysis capability)
  Critic      → GPT-OSS-120B     (independent model for cross-validation)
  Synthesizer → Gemini 3.1 Pro   (needs multi-source synthesis + search)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


class TaskType(str, Enum):
    """Cognitive task types that can be routed to models."""
    PARSE = "parse"
    PLAN = "plan"
    RESEARCH = "research"
    ANALYZE = "analyze"
    VALIDATE = "validate"
    SYNTHESIZE = "synthesize"
    CHAT = "chat"
    FORECAST = "forecast"


class RoutingPriority(str, Enum):
    """What to optimize for."""
    SPEED = "speed"       # Minimize latency
    QUALITY = "quality"   # Maximize reasoning depth
    COST = "cost"         # Minimize API spend
    BALANCED = "balanced" # Default trade-off


@dataclass(frozen=True)
class ModelProfile:
    """Static profile of a model's capabilities."""
    name: str
    provider: str
    prefer_key: str            # Key used in llm_chat_text(prefer=...)
    params_b: float            # Parameter count in billions
    context_window: int        # Max tokens
    cost_per_1k_tokens: float  # USD (0 = free)
    median_latency_ms: int     # Typical response time
    strengths: List[str]
    weaknesses: List[str]


# ── Model Profiles ──
QWEN = ModelProfile(
    name="qwen/qwen3-4b:free", provider="openrouter", prefer_key="fast",
    params_b=4, context_window=32_768, cost_per_1k_tokens=0.0,
    median_latency_ms=800,
    strengths=["fast", "json_output", "chat"],
    weaknesses=["shallow_reasoning", "limited_context"],
)

LLAMA = ModelProfile(
    name="meta-llama/llama-3.3-70b-instruct:free", provider="openrouter",
    prefer_key="analysis", params_b=70, context_window=128_000,
    cost_per_1k_tokens=0.0, median_latency_ms=3000,
    strengths=["deep_reasoning", "data_analysis", "long_context"],
    weaknesses=["slower", "rate_limited"],
)

GPT_OSS = ModelProfile(
    name="openai/gpt-oss-120b:free", provider="openrouter",
    prefer_key="validate", params_b=120, context_window=128_000,
    cost_per_1k_tokens=0.0, median_latency_ms=5000,
    strengths=["cross_validation", "reasoning_model", "independent_opinion"],
    weaknesses=["reasoning_only_output", "slow", "rate_limited"],
)

GEMINI = ModelProfile(
    name="gemini-3.1-pro-preview", provider="google",
    prefer_key="reason", params_b=0, context_window=1_000_000,
    cost_per_1k_tokens=0.00125, median_latency_ms=3000,
    strengths=["synthesis", "multi_source", "web_search", "huge_context"],
    weaknesses=["costs_money", "quota_limited"],
)

ALL_MODELS = [QWEN, LLAMA, GPT_OSS, GEMINI]


# ── Task → Model Routing Table ──
_ROUTING_TABLE: Dict[TaskType, Dict[RoutingPriority, str]] = {
    TaskType.PARSE: {
        RoutingPriority.SPEED: "fast",
        RoutingPriority.QUALITY: "fast",
        RoutingPriority.COST: "fast",
        RoutingPriority.BALANCED: "fast",
    },
    TaskType.ANALYZE: {
        RoutingPriority.SPEED: "fast",
        RoutingPriority.QUALITY: "analysis",
        RoutingPriority.COST: "analysis",
        RoutingPriority.BALANCED: "analysis",
    },
    TaskType.VALIDATE: {
        RoutingPriority.SPEED: "fast",
        RoutingPriority.QUALITY: "validate",
        RoutingPriority.COST: "validate",
        RoutingPriority.BALANCED: "validate",
    },
    TaskType.SYNTHESIZE: {
        RoutingPriority.SPEED: "analysis",
        RoutingPriority.QUALITY: "reason",
        RoutingPriority.COST: "analysis",
        RoutingPriority.BALANCED: "reason",
    },
    TaskType.CHAT: {
        RoutingPriority.SPEED: "fast",
        RoutingPriority.QUALITY: "analysis",
        RoutingPriority.COST: "fast",
        RoutingPriority.BALANCED: "fast",
    },
    TaskType.FORECAST: {
        RoutingPriority.SPEED: "analysis",
        RoutingPriority.QUALITY: "reason",
        RoutingPriority.COST: "analysis",
        RoutingPriority.BALANCED: "analysis",
    },
}


def route_model(
    task: TaskType,
    priority: RoutingPriority = RoutingPriority.BALANCED,
) -> str:
    """Return the prefer= key for llm_chat_text() based on task and priority."""
    return _ROUTING_TABLE.get(task, {}).get(priority, "fast")


def get_pipeline_cost_estimate() -> Dict[str, Any]:
    """Estimate cost per LDRAGO v2 pipeline execution."""
    # Approximate token usage per agent
    _TOKENS_PER_AGENT = {
        "parser": 500,
        "planner": 0,        # Heuristic
        "researcher": 0,     # Data APIs
        "reasoner": 6000,
        "critic": 3000,
        "synthesizer": 6000,
    }

    total_tokens = sum(_TOKENS_PER_AGENT.values())
    free_tokens = _TOKENS_PER_AGENT["parser"] + _TOKENS_PER_AGENT["reasoner"] + _TOKENS_PER_AGENT["critic"]
    paid_tokens = _TOKENS_PER_AGENT["synthesizer"]

    return {
        "total_tokens_per_query": total_tokens,
        "free_tokens": free_tokens,
        "paid_tokens_gemini": paid_tokens,
        "cost_per_query_usd": round(paid_tokens / 1000 * GEMINI.cost_per_1k_tokens, 5),
        "queries_per_dollar": round(1.0 / max(paid_tokens / 1000 * GEMINI.cost_per_1k_tokens, 0.0001), 0),
    }
