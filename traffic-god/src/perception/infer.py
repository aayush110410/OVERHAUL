"""Video detection + tracking pipeline for traffic-god.

The module exposes a thin CLI wrapper that couples a YOLOv8 detector with Norfair's
multi-object tracker, then writes per-track trajectories to CSV for downstream use
(calibration, SUMO demand synthesis, RL reward shaping, etc.).

The `run_inference` function can operate in a dry-run mode (skipping heavy ML imports)
so unit tests can validate bookkeeping without needing GPU drivers or model weights.
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

LOGGER = logging.getLogger("traffic_god.perception")


@dataclass
class InferenceConfig:
    """Runtime configuration for the perception CLI."""

    video_path: Path
    output_csv: Path
    model_path: str = "yolov8n.pt"
    conf_threshold: float = 0.4
    tracker_distance_threshold: float = 50.0
    seed: int = 42
    dry_run: bool = False


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and random for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)


def flatten_trajectories(history: Dict[int, List[Tuple[int, float, float]]]) -> List[Dict[str, float]]:
    """Flatten tracker history into row dictionaries.

    Parameters
    ----------
    history:
        Mapping of track id -> list of (frame, cx, cy) samples.
    """

    rows: List[Dict[str, float]] = []
    for track_id, samples in history.items():
        for frame_idx, cx, cy in samples:
            rows.append({"track_id": track_id, "frame": frame_idx, "x": cx, "y": cy})
    return rows


def _load_backends(dry_run: bool):
    if dry_run:
        return None, None, None

    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("ultralytics is required for inference") from exc

    try:
        from norfair import Detection, Tracker  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("norfair is required for tracking") from exc

    return YOLO, Detection, Tracker


def run_inference(cfg: InferenceConfig) -> Path:
    """Execute detection + tracking loop and write trajectories to CSV."""

    set_global_seed(cfg.seed)
    cfg.output_csv.parent.mkdir(parents=True, exist_ok=True)

    YOLO, Detection, Tracker = _load_backends(cfg.dry_run)

    if cfg.dry_run:
        LOGGER.info("Dry run: skipping model execution and emitting empty CSV")
        cfg.output_csv.write_text("track_id,frame,x,y\n", encoding="utf-8")
        return cfg.output_csv

    try:  # Local import to avoid forcing OpenCV on environments that only run dry tests
        import importlib

        cv2 = importlib.import_module("cv2")
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("OpenCV (cv2) is required for inference") from exc

    model = YOLO(cfg.model_path)
    tracker = Tracker(distance_function=None, distance_threshold=cfg.tracker_distance_threshold)

    cap = cv2.VideoCapture(str(cfg.video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {cfg.video_path}")

    frame_idx = 0
    trajectories: Dict[int, List[Tuple[int, float, float]]] = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        results = model.predict(source=frame, stream=False, imgsz=640, conf=cfg.conf_threshold)
        result = results[0]
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            detections.append(Detection(points=np.array([[cx, cy]]), scores=np.array([box.conf[0].item()]), data={"bbox": (x1, y1, x2, y2)}))
        active_tracks = tracker.update(detections)
        for track in active_tracks:
            cx, cy = track.estimate[0]
            trajectories.setdefault(track.id, []).append((frame_idx, float(cx), float(cy)))

    cap.release()

    rows = flatten_trajectories(trajectories)
    with cfg.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["track_id", "frame", "x", "y"])
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("Wrote %d trajectory rows to %s", len(rows), cfg.output_csv)
    return cfg.output_csv


def parse_args(argv: Sequence[str] | None = None) -> InferenceConfig:
    parser = argparse.ArgumentParser(description="Run YOLOv8 + Norfair tracking on a video")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--out", default="trajectories.csv", help="Output CSV path")
    parser.add_argument("--weights", default="yolov8n.pt", help="Path to YOLO weights")
    parser.add_argument("--conf", type=float, default=0.4, help="Detection confidence threshold")
    parser.add_argument("--distance-th", type=float, default=50.0, help="Tracker distance threshold")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Skip model execution (useful for CI tests)")
    args = parser.parse_args(argv)
    return InferenceConfig(
        video_path=Path(args.video),
        output_csv=Path(args.out),
        model_path=args.weights,
        conf_threshold=args.conf,
        tracker_distance_threshold=args.distance_th,
        seed=args.seed,
        dry_run=args.dry_run,
    )


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    cfg = parse_args(argv)
    run_inference(cfg)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
