"""
OVERHAUL Simulation Engines
===========================

Modular, composable domain simulation engines for urban system modeling.

Architecture:
  engines/
    base.py           - Abstract base classes & shared types
    registry.py       - Engine discovery & registration
    transport/        - Transportation network simulation
    environment/      - Environmental impact (AQI, emissions, noise)
    infrastructure/   - Urban infrastructure (flyovers, metro, roads)
    logistics/        - Supply chain & freight network
    energy/           - Energy systems & EV transition
    economic/         - Economic impact modeling
    population/       - Population segment behavior modeling
"""

from engines.base import (
    SimulationEngine,
    SimulationResult,
    EngineCapability,
    Scenario,
    Intervention,
)
from engines.registry import EngineRegistry, get_registry

__all__ = [
    "SimulationEngine",
    "SimulationResult",
    "EngineCapability",
    "Scenario",
    "Intervention",
    "EngineRegistry",
    "get_registry",
]
