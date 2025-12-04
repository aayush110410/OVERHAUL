"""
API Module
"""

from .routes import (
    app,
    create_app,
    # Request models
    TrafficQuery,
    RouteRequest,
    ScenarioRequest,
    InfrastructureRequest,
    PredictionRequest,
    Location,
    QueryType,
    # Response models
    TrafficCondition,
    RouteOption,
    PredictionResponse,
    AnalysisResponse
)

__all__ = [
    "app",
    "create_app",
    # Request models
    "TrafficQuery",
    "RouteRequest",
    "ScenarioRequest",
    "InfrastructureRequest",
    "PredictionRequest",
    "Location",
    "QueryType",
    # Response models
    "TrafficCondition",
    "RouteOption",
    "PredictionResponse",
    "AnalysisResponse"
]
