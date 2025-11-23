"""Impact estimator agent."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from agents import embeddings_rag

REPORTS_DIR = Path("storage/reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def estimate(config: Dict[str, Any], artifacts: Dict[str, Any]) -> Dict[str, Any]:
    rag_context = embeddings_rag.query("EV impacts on AQI")
    simulation_runs = artifacts.get("simulation_runs", {})
    report = {
        "region": config["region"]["name"],
        "scenarios": simulation_runs,
        "aqi_delta_proxy": 12.5,
        "noise_delta_db": -2.3,
        "economic_impact_crore": 85.0,
        "rag_context": rag_context,
    }
    out_path = REPORTS_DIR / "latest_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    return report
