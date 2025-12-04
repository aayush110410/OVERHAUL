"""
NEW TRAFFIC GOD - Advanced Traffic Intelligence Foundation Model
================================================================

A large-scale foundation model for traffic simulation, prediction, and optimization
specifically designed for Indian urban scenarios (Noida, Indirapuram, NCR region).

Architecture:
- Transformer-based foundation model with traffic-specific encodings
- Mixture-of-Experts (MoE) for scalable capacity
- Retrieval-Augmented Generation (RAG) for real-time data
- World Model / Simulator for counterfactual analysis
- Reward-based continuous learning

Capabilities:
1. Natural Language Understanding - Parse any traffic-related query
2. Traffic Flow Prediction - Real-time and forecasted conditions
3. Route Optimization - Best routes with alternatives
4. Infrastructure Analysis - Evaluate and suggest improvements
5. Scenario Simulation - What-if analysis with full physics
6. Policy Impact Assessment - Government scheme evaluation

Author: OVERHAUL Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "OVERHAUL"

# Core components
from .core.foundation_model import TrafficFoundationModel, TrafficModelConfig
from .core.tokenizer import TrafficTokenizer

# Simulation
from .simulation.world_model import WorldModel, ScenarioPlanner, TrafficState

# Retrieval
from .retrieval.rag_system import TrafficRAG, KnowledgeBase

# Training
from .training.trainer import Trainer, RLHFTrainer, ContinuousLearner

# Inference
from .inference.predictor import TrafficPredictor, TrafficPrediction

# Evaluation
from .evaluation.evaluator import Evaluator, TrafficBenchmark

# Configuration
from .configs.settings import (
    ModelConfig,
    TrainingConfig,
    InferenceConfig,
    ConfigLoader,
    get_config,
    NOIDA_SETTINGS
)

# Data
from .data.noida_traffic_data import (
    TRAFFIC_KNOWLEDGE,
    get_current_traffic_pattern,
    get_aqi_estimate,
    get_route_options
)

# Main orchestrator
from .traffic_god import TrafficGod, TrafficGodResponse, create_traffic_god, quick_query

__all__ = [
    # Version info
    "__version__",
    "__author__",
    
    # Core
    "TrafficFoundationModel",
    "TrafficModelConfig",
    "TrafficTokenizer",
    
    # Simulation
    "WorldModel",
    "ScenarioPlanner",
    "TrafficState",
    
    # Retrieval
    "TrafficRAG",
    "KnowledgeBase",
    
    # Training
    "Trainer",
    "RLHFTrainer",
    "ContinuousLearner",
    
    # Inference
    "TrafficPredictor",
    "TrafficPrediction",
    
    # Evaluation
    "Evaluator",
    "TrafficBenchmark",
    
    # Configuration
    "ModelConfig",
    "TrainingConfig",
    "InferenceConfig",
    "ConfigLoader",
    "get_config",
    "NOIDA_SETTINGS",
    
    # Data
    "TRAFFIC_KNOWLEDGE",
    "get_current_traffic_pattern",
    "get_aqi_estimate",
    "get_route_options",
    
    # Main
    "TrafficGod",
    "TrafficGodResponse",
    "create_traffic_god",
    "quick_query"
]
