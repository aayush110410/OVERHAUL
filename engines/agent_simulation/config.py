"""
Agent Simulation Configuration
================================

Internal configuration for the agent-based simulation layer.
Manages OASIS runtime settings, Zep memory config, and agent parameters.
All framework references are internal — never exposed in API output.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ZepConfig:
    """Zep Cloud memory configuration for agent persistent memory."""
    api_key: str = field(default_factory=lambda: os.getenv("ZEP_API_KEY", ""))
    enabled: bool = field(default_factory=lambda: bool(os.getenv("ZEP_API_KEY")))
    collection_name: str = "overhaul_agent_memory"
    # Fallback to in-memory store when Zep unavailable
    fallback_to_local: bool = True


@dataclass
class OASISConfig:
    """OASIS simulation runtime configuration."""
    # LLM config for PolicyAgent (only agent type using LLM)
    llm_api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    llm_model: str = "qwen/qwen3-4b"
    llm_base_url: str = "https://openrouter.ai/api/v1"
    # Simulation platform — we repurpose OASIS social sim for urban sim
    platform: str = "urban_simulation"
    # IPC directory for agent interviews
    ipc_base_dir: str = field(
        default_factory=lambda: os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "agent_sim"
        )
    )


@dataclass
class SwarmConfig:
    """Configuration for the urban agent swarm."""
    # Agent population sizing
    commuter_count: int = 2000           # Sampled from 7M NCR commuters
    freight_count: int = 200             # Sampled from 420K daily freight trips
    policy_agent_count: int = 1          # Per active intervention
    enable_environment_agent: bool = True

    # Timestep configuration
    timesteps: int = 10                  # Number of simulation steps
    step_duration_minutes: int = 15      # Simulated time per step
    # 10 steps × 15 min = 2.5 hours (morning rush simulation)

    # Agent behavior parameters
    congestion_memory_weight: float = 0.3   # How much past congestion affects route choice
    peer_influence_weight: float = 0.15     # Swarm intelligence — segment info sharing
    adaptation_threshold: float = 0.2       # Flow/capacity ratio change triggering reroute
    mode_shift_threshold: float = 0.4       # Congestion level triggering mode switch

    # Calibration
    max_calibration_iterations: int = 3     # Re-run if deviation > tolerance
    calibration_tolerance: float = 0.20     # 20% deviation threshold vs ML model predictions
    baseline_tolerance: float = 0.15        # 15% deviation threshold vs ncr_baselines.json

    # Performance
    parallel_agent_batches: int = 4         # Process agents in N parallel batches
    enable_persistent_memory: bool = True   # Use Zep for cross-run memory

    # Knowledge graph
    enable_knowledge_graph: bool = True     # Store NCR network in Zep graph
    enable_agent_interviews: bool = True    # IPC system for mid-sim agent queries
    enable_report_generation: bool = True   # MiroFish ReportAgent for post-sim analysis
    enable_swarm_intelligence: bool = True  # Agents share info within segments


@dataclass
class AgentSimConfig:
    """Top-level agent simulation configuration."""
    zep: ZepConfig = field(default_factory=ZepConfig)
    oasis: OASISConfig = field(default_factory=OASISConfig)
    swarm: SwarmConfig = field(default_factory=SwarmConfig)

    # Feature flag — allows disabling agent sim globally
    enabled: bool = True

    # Output sanitization — strip all framework references
    sanitize_output: bool = True
    blocked_terms: tuple = (
        "mirofish", "oasis", "camel", "zep", "camel-ai",
        "camel_ai", "camel_oasis", "zep_cloud", "zep-cloud",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serializable config (with sensitive fields masked)."""
        return {
            "enabled": self.enabled,
            "swarm": {
                "commuter_count": self.swarm.commuter_count,
                "freight_count": self.swarm.freight_count,
                "timesteps": self.swarm.timesteps,
                "step_duration_minutes": self.swarm.step_duration_minutes,
                "enable_persistent_memory": self.swarm.enable_persistent_memory,
                "enable_knowledge_graph": self.swarm.enable_knowledge_graph,
                "enable_swarm_intelligence": self.swarm.enable_swarm_intelligence,
            },
            "zep_enabled": self.zep.enabled,
            "llm_configured": bool(self.oasis.llm_api_key),
        }


# Singleton
_CONFIG: Optional[AgentSimConfig] = None


def get_agent_sim_config() -> AgentSimConfig:
    """Get global agent simulation configuration."""
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = AgentSimConfig()
    return _CONFIG
