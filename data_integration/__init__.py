"""Data Integration Layer — bridges data sources ↔ simulation engines."""

from data_integration.bridge import (
    ncr_data_to_engine_input,
    prompt_to_interventions,
    build_scenario_from_prompt,
    format_engine_results_for_chat,
)

__all__ = [
    "ncr_data_to_engine_input",
    "prompt_to_interventions",
    "build_scenario_from_prompt",
    "format_engine_results_for_chat",
]
