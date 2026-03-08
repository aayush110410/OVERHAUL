"""
OVERHAUL Microservices — Shared Contracts & Types
=================================================

Defines the inter-service communication contracts (request/response schemas)
used by all microservices. Each service imports from here to ensure
type consistency across boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


# ── Service identifiers ──────────────────────────────────────────

class ServiceName(str, Enum):
    GATEWAY = "gateway"
    SIMULATION = "simulation"
    LLM = "llm"
    DATA = "data"
    VALIDATION = "validation"
    TRAFFIC_GOD = "traffic_god"


# ── Simulation contracts ─────────────────────────────────────────

@dataclass
class SimulationRequest:
    prompt: str = ""
    city: str = "delhi"
    interventions: Optional[List[Dict[str, Any]]] = None
    engines: Optional[List[str]] = None
    time_horizon_days: int = 365
    template_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class SimulationResponse:
    scenario: str
    city: str
    interventions: List[Dict[str, Any]]
    results: Dict[str, Any]
    geojson: Dict[str, Any] = field(default_factory=dict)
    cached: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── LLM contracts ────────────────────────────────────────────────

@dataclass
class LLMChatRequest:
    prompt: str
    mode: str = "fast"  # fast | deep | agents
    context: Dict[str, Any] = field(default_factory=dict)
    ncr_data: Dict[str, Any] = field(default_factory=dict)
    engine_results: Dict[str, Any] = field(default_factory=dict)
    max_tokens: int = 12000

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LLMChatResponse:
    response: str
    mode: str
    models_used: Dict[str, str] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Data contracts ────────────────────────────────────────────────

@dataclass
class NCRDataRequest:
    city: str = "delhi"
    include_aqi: bool = True
    include_traffic: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NCRDataResponse:
    city: str
    aqi: Dict[str, Any] = field(default_factory=dict)
    traffic: Dict[str, Any] = field(default_factory=dict)
    engine_input: Dict[str, Any] = field(default_factory=dict)
    formatted_prompt: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LiveDataRequest:
    lat: float = 28.62
    lon: float = 77.35
    data_type: str = "aqi"  # aqi | route | both

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GeocodingRequest:
    query: str = ""
    lat: Optional[float] = None
    lon: Optional[float] = None
    reverse: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


# ── Validation contracts ──────────────────────────────────────────

@dataclass
class ValidationEntry:
    user_name: str
    city: str
    data_type: str
    value: Any
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Chat (Gateway aggregated) contracts ───────────────────────────

@dataclass
class ChatRequest:
    prompt: str
    mode: str = "fast"
    scenario: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ChatResponse:
    summary: str
    impact_cards: List[Dict[str, Any]] = field(default_factory=list)
    domains: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    geojson: Dict[str, Any] = field(default_factory=dict)
    ncr_data: Dict[str, Any] = field(default_factory=dict)
    live_context: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    manifest: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Health check ──────────────────────────────────────────────────

@dataclass
class HealthStatus:
    service: str
    status: str = "ok"  # ok | degraded | down
    version: str = "1.0.0"
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
