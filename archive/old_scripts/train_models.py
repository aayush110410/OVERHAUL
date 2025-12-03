"""Train surrogate models for OVERHAUL.

Generates synthetic scenarios using the core simulator in app.py and trains
regressors for travel time, vehicle-km travelled (VKT), corridor PM2.5,

and candidate ranking (meta-model). The resulting `.pkl` artifacts are dropped
alongside this script so `app.py` can load them on startup.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

import app  # Reuse simulator + candidate definitions

SCENARIO_FEATURES = [
    "demand",
    "reroute_penalty",
    "signal_opt_pct",
    "bus_lane",
    "bus_lane_pct",
    "demand_shift_pct",
    "weather_factor",
]

OUTPUT_DIR = Path(__file__).resolve().parent


def scenario_to_features(scenario: Dict[str, float]) -> List[float]:
    return [float(scenario.get(key, 0.0)) for key in SCENARIO_FEATURES]


def seeded_random_scenario(rng: random.Random) -> Dict[str, float]:
    scenario = app.interpret_prompt_to_scenario("optimize corridor")
    scenario["demand"] = rng.randint(800, 2200)
    scenario["signal_opt_pct"] = round(rng.uniform(0.0, 30.0), 2)
    scenario["signal_edges"] = ["e2", "e3"]
    scenario["bus_lane"] = 1 if rng.random() < 0.5 else 0
    scenario["bus_lane_pct"] = round(rng.uniform(5.0, 35.0), 2) if scenario["bus_lane"] else 0.0
    scenario["buslane_edges"] = ["e2"]
    scenario["reroute_penalty"] = round(rng.uniform(0.0, 1.0), 3)
    scenario["penalize_edges"] = ["e1", "e2", "e3"]
    scenario["demand_shift_pct"] = round(rng.uniform(0.0, 25.0), 2)
    scenario["weather_factor"] = round(rng.uniform(0.1, 0.7), 2)
    scenario["od_from"] = "A"
    scenario["od_to"] = "F"
    scenario["EF_g_per_km"] = 240 + rng.uniform(-20.0, 20.0)
    return scenario


@dataclass
class TrainingMatrices:
    x_tt: List[List[float]]
    y_tt: List[float]
    x_vkt: List[List[float]]
    y_vkt: List[float]
    x_pm: List[List[float]]
    y_pm: List[float]
    x_meta: List[List[float]]
    y_meta: List[float]


def generate_training_data(samples: int, seed: int) -> TrainingMatrices:
    rng = random.Random(seed)
    candidates = app.propose_candidates()

    x_tt: List[List[float]] = []
    y_tt: List[float] = []
    x_vkt: List[List[float]] = []
    y_vkt: List[float] = []
    x_pm: List[List[float]] = []  # [vkt, demand, weather, demand_shift, signal]
    y_pm: List[float] = []
    x_meta: List[List[float]] = []
    y_meta: List[float] = []

    for _ in range(samples):
        scenario = seeded_random_scenario(rng)
        baseline = app.simulate_once(scenario)
        if not baseline:
            continue
        features = scenario_to_features(scenario)

        x_tt.append(features)
        y_tt.append(baseline["avg_travel_time_min"])

        x_vkt.append(features)
        y_vkt.append(baseline["total_vkt"])

        x_pm.append(
            [
                baseline["total_vkt"],
                scenario["demand"],
                scenario["weather_factor"],
                scenario["demand_shift_pct"],
                scenario["signal_opt_pct"],
            ]
        )
        y_pm.append(baseline["pm25"])

        for idx, candidate in enumerate(candidates):
            candidate_scenario = dict(scenario)
            candidate_scenario.update(candidate["params"])
            candidate_result = app.simulate_once(candidate_scenario)
            if not candidate_result:
                continue
            delta = baseline["avg_travel_time_min"] - candidate_result["avg_travel_time_min"]
            x_meta.append(features + [idx])
            y_meta.append(delta)

    return TrainingMatrices(x_tt, y_tt, x_vkt, y_vkt, x_pm, y_pm, x_meta, y_meta)


def train_model(name: str, x: List[List[float]], y: List[float], model, output_name: str) -> Dict[str, float]:
    if not x:
        raise RuntimeError(f"No training samples available for {name}")
    X = np.array(x, dtype=np.float32)
    y_arr = np.array(y, dtype=np.float32)
    X_train, X_test, y_train, y_test = train_test_split(X, y_arr, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {
        "r2": float(r2_score(y_test, preds)),
        "mae": float(mean_absolute_error(y_test, preds)),
        "n_samples": int(len(x)),
        "output_file": output_name,
    }
    joblib.dump(model, OUTPUT_DIR / output_name)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train OVERHAUL surrogate models")
    parser.add_argument("--samples", type=int, default=2500, help="Synthetic scenarios to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print(f"Generating {args.samples} synthetic scenarios…")
    matrices = generate_training_data(args.samples, args.seed)

    report = {
        "samples_requested": args.samples,
        "seed": args.seed,
        "models": {},
    }

    report["models"]["traffic_tt_model.pkl"] = train_model(
        "travel_time",
        matrices.x_tt,
        matrices.y_tt,
        GradientBoostingRegressor(random_state=args.seed, max_depth=3, n_estimators=500),
        "traffic_tt_model.pkl",
    )

    report["models"]["traffic_vkt_model.pkl"] = train_model(
        "vehicle_km",
        matrices.x_vkt,
        matrices.y_vkt,
        RandomForestRegressor(
            n_estimators=600,
            min_samples_leaf=2,
            random_state=args.seed,
            n_jobs=-1,
        ),
        "traffic_vkt_model.pkl",
    )

    report["models"]["pollution_model.pkl"] = train_model(
        "pollution",
        matrices.x_pm,
        matrices.y_pm,
        GradientBoostingRegressor(random_state=args.seed, max_depth=3, n_estimators=400),
        "pollution_model.pkl",
    )

    report["models"]["ldra_go_meta_model.pkl"] = train_model(
        "candidate_meta",
        matrices.x_meta,
        matrices.y_meta,
        RandomForestRegressor(
            n_estimators=800,
            max_depth=12,
            random_state=args.seed,
            n_jobs=-1,
        ),
        "ldra_go_meta_model.pkl",
    )

    report_path = OUTPUT_DIR / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print("Training complete →", report_path)


if __name__ == "__main__":
    main()
