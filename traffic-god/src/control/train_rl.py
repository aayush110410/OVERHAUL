"""Placeholder RL training entry-point.

Implements a tiny CLI that would normally launch PPO against the SUMO gym env.
For now, the script supports a `--dry-run` flag so unit tests can execute without
SUMO or GPU dependencies.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

from src.sim.sumo_env import SumoIntersectionEnv, build_env_config

LOGGER = logging.getLogger("traffic_god.control")


def build_training_config(overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    base = {
        "total_timesteps": 10_000,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "seed": 7,
        "policy": "MlpPolicy",
    }
    if overrides:
        base.update(overrides)
    return base


def train(cfg: Dict[str, Any], dry_run: bool = False) -> Dict[str, Any]:
    """Run (or simulate) PPO training and return metadata."""

    env = SumoIntersectionEnv(dry_run=True)
    metadata = {"episodes": 0, "timesteps": 0, "dry_run": dry_run}
    if dry_run:
        LOGGER.info("Dry run: skipping PPO initialization")
        return metadata

    try:
        from stable_baselines3 import PPO  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("stable-baselines3 is required for RL training") from exc

    model = PPO(cfg["policy"], env, learning_rate=cfg["learning_rate"], gamma=cfg["gamma"], verbose=1, seed=cfg["seed"])
    model.learn(total_timesteps=cfg["total_timesteps"])
    metadata.update({"timesteps": cfg["total_timesteps"], "episodes": cfg["total_timesteps"] // env.cfg["max_steps"]})
    env.close()
    return metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO agent on SUMO env")
    parser.add_argument("--config", type=Path, help="Optional JSON config override", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Skip PPO init for CI")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes for metadata output")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    args = parse_args()
    cfg = build_training_config()
    if args.config and args.config.exists():
        overrides = json.loads(args.config.read_text())
        cfg.update(overrides)
    if args.episodes:
        cfg["total_timesteps"] = args.episodes * build_env_config()["max_steps"]
    train(cfg, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
