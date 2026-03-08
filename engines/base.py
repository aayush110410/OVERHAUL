"""
Engine Base Classes & Shared Types
===================================

All simulation engines inherit from SimulationEngine and return SimulationResult.
This ensures composability — any engine can feed into any other engine.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class EngineCapability(str, Enum):
    """What a simulation engine can do."""
    TRANSPORT = "transport"
    ENVIRONMENT = "environment"
    INFRASTRUCTURE = "infrastructure"
    LOGISTICS = "logistics"
    ENERGY = "energy"
    ECONOMIC = "economic"
    POPULATION = "population"


@dataclass
class Intervention:
    """A single policy or infrastructure change to simulate."""
    name: str
    domain: EngineCapability
    parameters: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "domain": self.domain.value,
            "parameters": self.parameters,
            "description": self.description,
        }


@dataclass
class Scenario:
    """A complete simulation scenario with one or more interventions."""
    name: str
    description: str = ""
    city: str = "Delhi NCR"
    interventions: List[Intervention] = field(default_factory=list)
    time_horizon_days: int = 365
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "city": self.city,
            "interventions": [i.to_dict() for i in self.interventions],
            "time_horizon_days": self.time_horizon_days,
            "metadata": self.metadata,
        }


@dataclass
class SimulationResult:
    """Standardized output from any simulation engine."""
    engine: str
    domain: EngineCapability
    scenario_name: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    impacts: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    confidence: float = 0.0  # 0.0 to 1.0
    warnings: List[str] = field(default_factory=list)
    runtime_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine": self.engine,
            "domain": self.domain.value,
            "scenario": self.scenario_name,
            "metrics": self.metrics,
            "impacts": self.impacts,
            "recommendations": self.recommendations,
            "confidence": round(self.confidence, 3),
            "warnings": self.warnings,
            "runtime_seconds": round(self.runtime_seconds, 3),
            "metadata": self.metadata,
        }


class SimulationEngine(ABC):
    """Abstract base for all OVERHAUL simulation engines.

    Every engine must:
    1. Declare its capabilities
    2. Accept a Scenario and return a SimulationResult
    3. Report what data it needs and what it can estimate
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable engine name."""

    @property
    @abstractmethod
    def capabilities(self) -> List[EngineCapability]:
        """Which domains this engine covers."""

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def required_data(self) -> List[str]:
        """Data fields this engine needs. Override per engine."""
        return []

    @property
    def optional_data(self) -> List[str]:
        """Data fields this engine can use but doesn't require."""
        return []

    async def simulate(self, scenario: Scenario, data: Dict[str, Any] | None = None) -> SimulationResult:
        """Run simulation with timing wrapper."""
        t0 = time.perf_counter()
        result = await self._run(scenario, data or {})
        result.runtime_seconds = time.perf_counter() - t0
        return result

    @abstractmethod
    async def _run(self, scenario: Scenario, data: Dict[str, Any]) -> SimulationResult:
        """Engine-specific simulation logic. Override this."""

    def validate_data(self, data: Dict[str, Any]) -> List[str]:
        """Check if required data is present. Returns list of missing fields."""
        missing = [f for f in self.required_data if f not in data]
        return missing

    def estimate_missing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate missing data fields using defaults/heuristics. Override per engine."""
        return data

    def info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "capabilities": [c.value for c in self.capabilities],
            "required_data": self.required_data,
            "optional_data": self.optional_data,
        }
