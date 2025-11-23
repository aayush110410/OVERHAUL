"""Gym-compatible SUMO environment scaffold.

This module exposes:
- `SumoIntersectionEnv`: minimal `gym.Env` skeleton that can run in a dry mode for unit tests.
- `build_env_config`: helper that normalizes config dictionaries (unit-tested).
- CLI entry-point allowing quick smoke tests or benchmarking without SUMO installed.
"""
from __future__ import annotations

import argparse
import importlib
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
def _load_gym_backend() -> Tuple[Any, Any]:
    for module_name in ("gym", "gymnasium"):
        try:
            module = importlib.import_module(module_name)
            spaces_module = importlib.import_module(f"{module_name}.spaces")
            return module, spaces_module
        except ImportError:
            continue
    raise RuntimeError("gym (or gymnasium) is required for sumo_env module")


gym, spaces = _load_gym_backend()


LOGGER = logging.getLogger("traffic_god.sumo_env")


def build_env_config(overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Return a normalized environment config (pure function for tests)."""

    base = {
        "sumo_cfg": "configs/sumo/sample.sumocfg",
        "use_gui": False,
        "max_steps": 900,
        "delta_time": 1.0,
        "reward_weights": {"queue": -0.1, "speed": 0.01},
    }
    if overrides:
        base.update(overrides)
    return base


@dataclass
class EnvState:
    step_count: int = 0
    last_reward: float = 0.0


class SumoIntersectionEnv(gym.Env):
    """Minimal SUMO -> Gym bridge (mocked for now)."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, config: Dict[str, Any] | None = None, dry_run: bool = False) -> None:
        super().__init__()
        self.cfg = build_env_config(config)
        self.dry_run = dry_run
        self.state = EnvState()

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):  # type: ignore[override]
        super().reset(seed=seed)
        self.state = EnvState()
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, {}

    def step(self, action: int):  # type: ignore[override]
        self.state.step_count += 1
        congestion = np.random.rand()
        reward = -congestion + 0.01 * action
        self.state.last_reward = reward
        done = self.state.step_count >= self.cfg["max_steps"]
        info = {"congestion_proxy": congestion}
        obs = np.random.rand(*self.observation_space.shape).astype(np.float32)
        return obs, reward, done, False, info

    def render(self):  # pragma: no cover - placeholder
        LOGGER.info("Rendering step %s (dry=%s)", self.state.step_count, self.dry_run)

    def close(self):  # pragma: no cover
        LOGGER.info("Closing SUMO env")


def parse_args():
    parser = argparse.ArgumentParser(description="Dry-run SUMO gym env")
    parser.add_argument("--max-steps", type=int, default=5, help="Number of steps to simulate")
    parser.add_argument("--dry-run", action="store_true", help="Skip SUMO attachments")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    env = SumoIntersectionEnv({"max_steps": args.max_steps}, dry_run=args.dry_run or True)
    obs, _ = env.reset()
    LOGGER.info("Initial obs shape: %s", obs.shape)
    done = False
    total_reward = 0.0
    while not done:
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        LOGGER.info("step=%d reward=%.3f info=%s", env.state.step_count, reward, info)
    LOGGER.info("Finished roll-out. total_reward=%.3f", total_reward)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
