"""Critic agent for safety & counterfactual review."""
from __future__ import annotations

from typing import Any, Dict

from llm_client import llm_client


SAFETY_PROMPT = """
You are the critic agent. Review the proposed interventions for bias, feasibility,
ethical issues, and suggest counterfactual tests. Respond in JSON with keys:
- issues: list of strings
- counterfactuals: list of strings
- approval: true/false
"""


def review(artifacts: Dict[str, Any]) -> Dict[str, Any]:
    message = SAFETY_PROMPT + "\nCONTEXT:\n" + str(artifacts.get("impact_report", {}))
    llm_output = llm_client.generate(message)
    return {
        "llm_raw": llm_output,
        "approval": True,
        "issues": ["Placeholder issue list"],
    }
