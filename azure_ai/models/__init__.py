"""Domain models that report to LDRAGo.

These are "modelized" domain agents (weather, behavior, policy/econ, etc.).
They may use deterministic calculations + optional Azure OpenAI calls for
structured reasoning and explanation.

All code lives under azure_ai/ for clean rollback.
"""

from .orchestrator import run_domain_models
