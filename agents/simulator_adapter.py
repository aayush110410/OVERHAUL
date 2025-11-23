"""SUMO simulator adapter."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List

SUMO_DIR = Path("sumo_net")
SUMO_DIR.mkdir(exist_ok=True)

BASELINE_NET = SUMO_DIR / "corridor.net.xml"
BASELINE_ROUTES = SUMO_DIR / "corridor.rou.xml"
BASELINE_CFG = SUMO_DIR / "corridor.sumo.cfg"


def ensure_baseline(config: Dict[str, Any]) -> None:
    if BASELINE_NET.exists() and BASELINE_ROUTES.exists():
        return
    builder = SUMO_DIR / "build_from_osm.py"
    if not builder.exists():
        raise FileNotFoundError("SUMO network builder script missing")
    subprocess.run(["python", str(builder)], check=True)


def _run_sumo(additional_opts: List[str]) -> Dict[str, Any]:
    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        return {"warning": "SUMO_HOME not set. Skipping actual simulation."}
    sumo_bin = Path(sumo_home) / "bin" / "sumo"
    cmd = [str(sumo_bin), "-c", str(BASELINE_CFG), *additional_opts]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "cmd": " ".join(cmd),
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }


def run_baseline(config: Dict[str, Any]) -> Dict[str, Any]:
    ensure_baseline(config)
    return _run_sumo(["--duration-log.disable" "true"])


def apply_intervention(config: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, Any]:
    ensure_baseline(config)
    # TODO: Modify route files or use TraCI hooks to apply interventions.
    return _run_sumo(["--duration-log.disable" "true"])


def collect_metrics(sim_output: Dict[str, Any]) -> Dict[str, Any]:
    # Placeholder heuristics until TraCI metrics wired up
    return {
        "mean_travel_time": 18.5,
        "queue_length": 42,
        "throughput": 1200,
        "emissions_proxy": 0.85,
        "noise_db": 68.0,
        "raw": sim_output,
    }


def run_all(config: Dict[str, Any]) -> Dict[str, Any]:
    runs = {}
    baseline = run_baseline(config)
    runs["baseline"] = collect_metrics(baseline)
    runs["scenario_a"] = collect_metrics(apply_intervention(config, config["simulation"]["interventions"]["scenario_a"]))
    runs["scenario_b"] = collect_metrics(apply_intervention(config, config["simulation"]["interventions"]["scenario_b"]))
    runs["scenario_c"] = collect_metrics(apply_intervention(config, config["simulation"]["interventions"]["scenario_c"]))
    return runs
