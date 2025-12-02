# traffic-god

End-to-end research sandbox for building intelligent traffic operations driven by computer vision, calibrated microsimulation, and reinforcement learning. The initial milestone focuses on a single corridor (one camera, one intersection) and grows iteratively as new sensors and control policies come online.

## Project Modules

- **Perception (`src/perception/`)** – Video ingestion, YOLOv8-based detection, multi-object tracking, and trajectory export utilities.
- **Analysis (`src/analysis/`)** – Feature aggregation, KPI computation (travel time, throughput, delay, MOTA), and calibration helpers for SUMO.
- **Simulation (`src/sim/`)** – SUMO wrappers, calibration scripts, and `gym.Env`-compatible control surfaces powered by TraCI.
- **Control (`src/control/`)** – Baseline logic and RL agents (PPO via Stable-Baselines3) that act on SUMO observations.
- **Server (`src/server/`)** – FastAPI + Streamlit surfaces for dashboards, scenario review, and remote job triggers.

## Quickstart

```bash
# 1) Clone and enter repo
mkdir traffic-god && cd traffic-god
python -m venv .venv && source .venv/bin/activate

# 2) Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3) Smoke-test perception pipeline on a sample clip
python src/perception/infer.py --video data/raw/sample.mp4 --out trajectories.csv

# 4) Step a mock SUMO environment (requires SUMO + TraCI and DISPLAY when using GUI)
python src/sim/sumo_env.py --dry-run

# 5) Launch a placeholder RL training loop (dry mode)
python src/control/train_rl.py --episodes 1 --dry-run
```

## Dev Containers & Docker

- `.devcontainer/devcontainer.json` targets the official VS Code Python image and installs requirements automatically.
- `docker/Dockerfile` bakes a slim runtime image for CI smoke tests and reproducible batch jobs.

## Next Steps

1. Drop sample videos under `data/raw/` (NGSIM, UA-DETRAC, or in-house captures).
2. Build SUMO networks for the chosen intersection and store configs under `configs/sumo/`.
3. Implement calibration, dashboard, and RL training loops following the step-by-step playbook in the project brief.

## TomTom-powered Traffic Brain

We now have an ingestion → feature-engineering → modeling toolchain that turns live TomTom data into actionable controls. Set your API key once:

```bash
export TOMTOM_API_KEY="p5ZzkdRmzx3RAJMK9I9Kq8RdSzYIuWHy"
```

Then run the pipeline from inside `traffic-god/`:

```bash
# 1) Harvest flow + incident snapshots for the Noida–Delhi corridor
python -m src.data.tomtom.ingest \
	--bbox 28.50 28.75 77.25 77.50 \
	--grid 4 4 \
	--snapshots 6 \
	--sleep-seconds 30 \
	--output data/processed/tomtom

# 2) Build ML-ready features (lags, congestion ratios, incident counts)
python -m src.analysis.tomtom_features \
	--input data/processed/tomtom \
	--output data/processed/tomtom_features.parquet

# 3) Train the TrafficGod model + emit hotspot report
python -m src.control.traffic_god \
	--dataset data/processed/tomtom_features.parquet \
	--model-out data/models/traffic_god.joblib \
	--report-out data/reports/traffic_god_report.json
```

The final JSON report includes:

- Validation metrics (MAE / R²)
- Ranked congestion hotspots with probable cause (incident vs. demand)
- Tactical actions (signal tweaks, reversible lanes, clearance crews)
- Heuristic detour paths you can feed into SUMO or the FastAPI server.

Use the stored parquet/model artifacts to simulate counterfactuals inside `src/sim/sumo_env.py` or to seed new RL runs under `src/control/train_rl.py`.
