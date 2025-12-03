"""Coordinator agent that routes tasks to specialist agents.

Patterns:
- Routing: classifier selects the appropriate agent for each task.
- Prompt chaining: decomposes workflow (collect -> clean -> embed -> plan -> simulate -> review).
- Planning: builds execution plan graph before invoking agents.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

from llm_client import llm_client
from privacy_guard import privacy_guard
from agents import data_collector, cleaner, embeddings_rag, behavior_model, simulator_adapter, impact_estimator, critic

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def load_config() -> Dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text())


def simple_router(task: str) -> str:
    """Return agent key based on heuristic routing.

    TODO: replace with embedding similarity or mini-classifier per Agentic spec.
    """
    routing_table = {
        "collect": "data_collector",
        "clean": "cleaner",
        "embed": "embeddings_rag",
        "segment": "behavior_model",
        "simulate": "simulator_adapter",
        "estimate": "impact_estimator",
        "critic": "critic",
    }
    return routing_table.get(task, "critic")


def orchestrate(prompt: str, user_id: str = "anonymous") -> Dict[str, Any]:
    config = load_config()
    privacy_guard.assert_safe(user_id, ["socio_economic_bracket"])  # Example check

    plan = [
        {"task": "collect", "description": "Fetch EV/AQI datasets"},
        {"task": "clean", "description": "Normalize and deduplicate"},
        {"task": "embed", "description": "Index for RAG"},
        {"task": "segment", "description": "Behavior modeling"},
        {"task": "simulate", "description": "Run SUMO baseline + scenarios"},
        {"task": "estimate", "description": "Compute impacts"},
        {"task": "critic", "description": "Safety and quality review"},
    ]

    artifacts: Dict[str, Any] = {"prompt": prompt, "plan": plan}
    for step in plan:
        agent_key = simple_router(step["task"])
        if agent_key == "data_collector":
            artifacts["raw_data"] = data_collector.collect_sources(config, prompt)
        elif agent_key == "cleaner":
            artifacts["clean_data"] = cleaner.clean(artifacts.get("raw_data", []))
        elif agent_key == "embeddings_rag":
            embeddings_rag.ensure_index(config)
            artifacts["rag_metadata"] = embeddings_rag.upsert_documents(artifacts.get("clean_data", []))
        elif agent_key == "behavior_model":
            artifacts["behavior_profiles"] = behavior_model.segment(config)
        elif agent_key == "simulator_adapter":
            artifacts["simulation_runs"] = simulator_adapter.run_all(config)
        elif agent_key == "impact_estimator":
            artifacts["impact_report"] = impact_estimator.estimate(config, artifacts)
        elif agent_key == "critic":
            artifacts["critic"] = critic.review(artifacts)

    summary_prompt = json.dumps({
        "task": "summarize",
        "region": config["region"]["name"],
        "impacts": artifacts.get("impact_report", {}),
        "critic": artifacts.get("critic", {}),
    })
    artifacts["llm_summary"] = llm_client.generate(summary_prompt)
    return artifacts
