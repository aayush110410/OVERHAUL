#!/usr/bin/env bash
set -euo pipefail

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/perception/infer.py --video data/raw/sample.mp4 --out trajectories.csv --dry-run
python src/sim/sumo_env.py --dry-run --max-steps 5
python src/control/train_rl.py --dry-run --episodes 1
