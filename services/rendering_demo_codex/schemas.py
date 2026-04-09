from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field


LayerId = Literal[
    "satellites",
    "flights",
    "traffic",
    "weather",
    "simulation",
    "buildings",
    "intelligence",
]

SimulationInterventionType = Literal[
    "new_road",
    "lane_expansion",
    "flyover",
    "road_closure",
    "signal_optimization",
]


class GeoPoint(BaseModel):
    longitude: float
    latitude: float
    altitude: float = 0.0


class GeoLine(BaseModel):
    coordinates: List[GeoPoint] = Field(default_factory=list)


class LayerStats(BaseModel):
    object_count: int = 0
    update_frequency_hz: float = 0.0
    active: bool = True
    source: str = "demo"
    last_updated: datetime = Field(default_factory=datetime.utcnow)


class DatasetRecord(BaseModel):
    id: str
    position: GeoPoint | None = None
    path: GeoLine | None = None
    properties: Dict[str, Any] = Field(default_factory=dict)


class LayerSnapshot(BaseModel):
    layer_id: LayerId
    label: str
    kind: Literal["points", "lines", "grid", "hybrid"] = "points"
    stats: LayerStats = Field(default_factory=LayerStats)
    items: List[DatasetRecord] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)


class PerformanceBudget(BaseModel):
    target_fps: int = 60
    preferred_resolution_scale: float = 1.0
    max_particles: int = 24000
    max_instanced_objects: int = 24000


class WorldSnapshot(BaseModel):
    sequence: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    focus: Dict[str, Any]
    budget: PerformanceBudget = Field(default_factory=PerformanceBudget)
    layers: Dict[LayerId, LayerSnapshot]


class TimelineState(BaseModel):
    label: str
    year_offset: int
    metrics: Dict[str, float] = Field(default_factory=dict)
    heatmap: List[DatasetRecord] = Field(default_factory=list)
    flows: List[DatasetRecord] = Field(default_factory=list)
    infrastructure: List[DatasetRecord] = Field(default_factory=list)


class SimulationIntervention(BaseModel):
    kind: SimulationInterventionType
    name: str
    target_segment_id: str | None = None
    coordinates: List[GeoPoint] = Field(default_factory=list)
    lane_delta: int = 0
    capacity_delta: float = 0.0
    speed_delta: float = 0.0
    notes: str = ""


class SimulationRequest(BaseModel):
    prompt: str = ""
    focus: GeoPoint | None = None
    focus_name: str | None = None
    years: List[int] = Field(default_factory=lambda: [0, 1, 5])
    interventions: List[SimulationIntervention] = Field(default_factory=list)


class SimulationReport(BaseModel):
    summary: str
    traffic_improvement_pct: float
    travel_time_delta_pct: float
    congestion_delta_pct: float
    pollution_delta_pct: float
    active_models: List[str] = Field(default_factory=list)
    timeline: List[TimelineState] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class AgentTrace(BaseModel):
    agent: str
    model: str
    summary: str
    confidence: float


class OrchestrationResponse(BaseModel):
    prompt: str
    parsed_location: Dict[str, Any] = Field(default_factory=dict)
    selected_models: List[str] = Field(default_factory=list)
    agent_trace: List[AgentTrace] = Field(default_factory=list)
    simulation: SimulationReport
    visualization_commands: Dict[str, Any] = Field(default_factory=dict)


class BootstrapResponse(BaseModel):
    snapshot: WorldSnapshot
    available_layers: List[Dict[str, Any]] = Field(default_factory=list)
    presets: List[Dict[str, Any]] = Field(default_factory=list)
    service_status: Dict[str, Any] = Field(default_factory=dict)
