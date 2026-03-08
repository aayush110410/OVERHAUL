"""Cognitive agent role definitions and base types.

Every agent in the LDRAGO v2 system conforms to these types.
Agents are stateless functions — all state lives in AgentContext.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional


class AgentRole(str, Enum):
    """Cognitive roles in the LDRAGO reasoning pipeline."""
    PARSER = "parser"
    PLANNER = "planner"
    RESEARCHER = "researcher"
    REASONER = "reasoner"
    CRITIC = "critic"
    SYNTHESIZER = "synthesizer"


@dataclass
class AgentContext:
    """Shared context passed through the agent pipeline.

    Each agent reads what it needs and appends its output.
    The context accumulates knowledge as it flows through the pipeline.
    """

    # ── User input ──
    query: str
    city: str = "delhi"
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── Accumulated by agents ──
    parsed_intent: Dict[str, Any] = field(default_factory=dict)
    execution_plan: Dict[str, Any] = field(default_factory=dict)
    research_data: Dict[str, Any] = field(default_factory=dict)
    reasoning_output: Dict[str, Any] = field(default_factory=dict)
    critique: Dict[str, Any] = field(default_factory=dict)
    synthesis: str = ""

    # ── Engine results (filled by orchestrator) ──
    engine_results: Dict[str, Any] = field(default_factory=dict)

    # ── Telemetry ──
    agent_logs: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def log(self, role: str, message: str, duration_ms: float = 0):
        self.agent_logs.append({
            "role": role,
            "message": message,
            "duration_ms": round(duration_ms, 1),
            "timestamp": time.time(),
        })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "city": self.city,
            "parsed_intent": self.parsed_intent,
            "execution_plan": self.execution_plan,
            "engine_results": {k: v for k, v in self.engine_results.items()
                               if k != "raw"},
            "reasoning_output": self.reasoning_output,
            "critique": self.critique,
            "synthesis": self.synthesis,
            "agent_logs": self.agent_logs,
            "errors": self.errors,
        }


@dataclass
class AgentOutput:
    """Structured output from a cognitive agent."""
    role: AgentRole
    result: Dict[str, Any]
    confidence: float = 1.0
    duration_ms: float = 0.0
    model_used: str = ""
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


# Type alias for an agent function
CognitiveAgent = Callable[
    [AgentContext],
    Coroutine[Any, Any, AgentOutput],
]
