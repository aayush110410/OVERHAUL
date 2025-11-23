"""Bridge utilities that expose traffic-god modules to OVERHAUL FastAPI services.

All heavy dependencies live under `traffic-god/`. This bridge lazily injects that
folder into `sys.path`, imports the perception / RL helpers, and provides safe wrappers
with better error messages for the main app.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

TRAFFIC_GOD_ROOT = Path(__file__).parent / "traffic-god"
TRAFFIC_GOD_SRC = TRAFFIC_GOD_ROOT / "src"

if TRAFFIC_GOD_SRC.exists() and str(TRAFFIC_GOD_SRC) not in sys.path:
    sys.path.append(str(TRAFFIC_GOD_SRC))

try:  # pragma: no cover - imports validated via unit tests instead
    from perception.infer import InferenceConfig, run_inference  # type: ignore
except Exception as exc:  # noqa: BLE001
    InferenceConfig = None  # type: ignore
    _PERCEPTION_IMPORT_ERROR = exc
else:
    _PERCEPTION_IMPORT_ERROR = None

try:  # pragma: no cover
    from control.train_rl import build_training_config, train  # type: ignore
except Exception as exc:  # noqa: BLE001
    build_training_config = None  # type: ignore
    train = None  # type: ignore
    _RL_IMPORT_ERROR = exc
else:
    _RL_IMPORT_ERROR = None


class TrafficGodService:
    """Facade for triggering traffic-god pipelines from OVERHAUL."""

    def __init__(self, root_dir: Optional[Path] = None) -> None:
        self.root_dir = root_dir or TRAFFIC_GOD_ROOT
        self.data_dir = self.root_dir / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ Perception
    def run_perception(self, video_path: str, output_csv: Optional[str] = None, dry_run: bool = False) -> Dict[str, Any]:
        if InferenceConfig is None or run_inference is None:  # type: ignore[arg-type]
            raise RuntimeError(f"Perception stack unavailable: {_PERCEPTION_IMPORT_ERROR}")

        video = Path(video_path)
        if not video.exists() and not dry_run:
            raise FileNotFoundError(f"Video not found: {video}")

        output = Path(output_csv) if output_csv else (self.root_dir / "data" / "processed" / "trajectories.csv")
        output.parent.mkdir(parents=True, exist_ok=True)

        cfg = InferenceConfig(video_path=video, output_csv=output, dry_run=dry_run)
        csv_path = run_inference(cfg)
        return {
            "video": str(video.resolve()),
            "output_csv": str(csv_path.resolve()),
            "dry_run": dry_run,
        }

    # ------------------------------------------------------------------ RL / control
    def train_rl(self, overrides: Optional[Dict[str, Any]] = None, dry_run: bool = False) -> Dict[str, Any]:
        if build_training_config is None or train is None:
            raise RuntimeError(f"RL stack unavailable: {_RL_IMPORT_ERROR}")

        config = build_training_config(overrides or {})
        result = train(config, dry_run=dry_run)
        return {"config": config, "result": result}
