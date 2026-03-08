"""LDRAGO v2 — Cognitive Agent System.

Multi-agent reasoning architecture with specialized cognitive roles:
  - Parser     : Interprets user intent, extracts entities
  - Planner    : Decomposes questions into simulation tasks
  - Researcher : Gathers data and context
  - Reasoner   : Deep analysis and causal inference
  - Critic     : Validates, challenges, finds inconsistencies
  - Synthesizer: Merges all outputs into coherent insight

Each agent is a stateless async function that takes context and
returns structured output. The Orchestrator coordinates them.
"""

from agents.cognitive.roles import (
    AgentRole,
    CognitiveAgent,
    AgentContext,
    AgentOutput,
)
from agents.cognitive.orchestrator import LDRAGOv2

__all__ = [
    "AgentRole",
    "CognitiveAgent",
    "AgentContext",
    "AgentOutput",
    "LDRAGOv2",
]
