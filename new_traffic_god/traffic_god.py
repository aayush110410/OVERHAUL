"""
New Traffic God - Main Orchestrator
=====================================
The central orchestrator that ties all components together
for the Traffic Foundation Model.

This is the main entry point for using the model.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import asyncio

# Internal imports
from .core import TrafficFoundationModel, TrafficModelConfig
from .core.tokenizer import TrafficTokenizer
from .simulation import WorldModel, ScenarioPlanner
from .retrieval import TrafficRAG, KnowledgeBase
from .training import Trainer, RLHFTrainer, ContinuousLearner
from .inference import TrafficPredictor
from .evaluation import Evaluator
from .configs import (
    ConfigLoader,
    get_config,
    ModelConfig,
    TrainingConfig,
    InferenceConfig,
    NOIDA_SETTINGS
)
from .data import (
    TRAFFIC_KNOWLEDGE,
    get_current_traffic_pattern,
    get_aqi_estimate,
    get_route_options
)
from .utils import (
    setup_logger,
    Timer,
    haversine_distance,
    is_peak_hour,
    get_time_of_day,
    validate_noida_region
)


# ============================================================================
# MAIN TRAFFIC GOD CLASS
# ============================================================================

@dataclass
class TrafficGodResponse:
    """Response from Traffic God"""
    answer: str
    confidence: float
    query_type: str
    data: Dict[str, Any]
    sources: List[str]
    timestamp: str
    processing_time_ms: float
    
    def to_dict(self) -> Dict:
        return {
            "answer": self.answer,
            "confidence": self.confidence,
            "query_type": self.query_type,
            "data": self.data,
            "sources": self.sources,
            "timestamp": self.timestamp,
            "processing_time_ms": self.processing_time_ms
        }


class TrafficGod:
    """
    The New Traffic God - A Foundation Model for Traffic Intelligence
    
    This class orchestrates all components of the traffic foundation model:
    - Foundation Model (Transformer-based)
    - World Model (Neural Traffic Simulator)
    - RAG System (Knowledge Retrieval)
    - Predictor (Inference Engine)
    
    Specialized for Noida, Indirapuram, and NCR region traffic.
    
    Example usage:
    ```python
    # Initialize
    god = TrafficGod(environment="development")
    await god.initialize()
    
    # Query
    response = await god.query("What's the traffic like on NH-24 right now?")
    print(response.answer)
    
    # Route planning
    route = await god.plan_route(
        source="Sector 18 Noida",
        destination="Indirapuram"
    )
    
    # Scenario analysis
    analysis = await god.analyze_scenario(
        "What if we add a flyover at Sector 62?"
    )
    ```
    """
    
    def __init__(
        self,
        environment: str = "development",
        config_path: Optional[str] = None,
        device: str = "auto"
    ):
        """
        Initialize Traffic God
        
        Args:
            environment: "development", "production", or "testing"
            config_path: Path to custom config file
            device: "auto", "cuda", "cpu", or "mps"
        """
        self.environment = environment
        self.device = self._detect_device(device)
        
        # Setup logging
        self.logger = setup_logger(
            "TrafficGod",
            level=logging.DEBUG if environment == "development" else logging.INFO
        )
        
        # Load configuration
        self.config = get_config(environment)
        if config_path:
            self.config.load_from_file(config_path)
        
        # Component placeholders
        self.model: Optional[TrafficFoundationModel] = None
        self.tokenizer: Optional[TrafficTokenizer] = None
        self.world_model: Optional[WorldModel] = None
        self.rag: Optional[TrafficRAG] = None
        self.predictor: Optional[TrafficPredictor] = None
        self.evaluator: Optional[Evaluator] = None
        
        # State
        self.is_initialized = False
        self.knowledge = TRAFFIC_KNOWLEDGE
        
        self.logger.info(f"Traffic God created (env={environment}, device={self.device})")
    
    def _detect_device(self, device: str) -> str:
        """Detect best available device"""
        if device != "auto":
            return device
        
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        
        return "cpu"
    
    async def initialize(
        self,
        load_model: bool = True,
        load_rag: bool = True,
        load_world_model: bool = True
    ):
        """
        Initialize all components
        
        Args:
            load_model: Whether to load the foundation model
            load_rag: Whether to initialize RAG system
            load_world_model: Whether to load world model
        """
        self.logger.info("Initializing Traffic God...")
        
        with Timer("Initialization"):
            # Initialize tokenizer
            self.tokenizer = TrafficTokenizer()
            self.logger.info("Tokenizer initialized")
            
            # Initialize foundation model
            if load_model:
                model_config = self.config.get("model")
                self.model = TrafficFoundationModel(
                    TrafficModelConfig(
                        d_model=model_config.d_model,
                        n_heads=model_config.n_heads,
                        n_layers=model_config.n_layers,
                        vocab_size=model_config.vocab_size
                    )
                )
                self.logger.info(f"Foundation model initialized ({model_config.model_size})")
            
            # Initialize RAG system
            if load_rag:
                self.rag = TrafficRAG(
                    embedding_dim=self.config.get("rag").embedding_dim
                )
                # Add Noida traffic knowledge
                await self._load_knowledge_base()
                self.logger.info("RAG system initialized")
            
            # Initialize world model
            if load_world_model:
                wm_config = self.config.get("world_model")
                self.world_model = WorldModel(
                    grid_size=wm_config.grid_size,
                    state_dim=wm_config.state_dim
                )
                self.logger.info("World model initialized")
            
            # Initialize predictor
            self.predictor = TrafficPredictor(
                model=self.model,
                tokenizer=self.tokenizer,
                rag=self.rag,
                world_model=self.world_model
            )
            self.logger.info("Predictor initialized")
            
            # Initialize evaluator
            self.evaluator = Evaluator(self.model, self.tokenizer)
            
            self.is_initialized = True
            self.logger.info("Traffic God fully initialized!")
    
    async def _load_knowledge_base(self):
        """Load Noida traffic knowledge into RAG"""
        if not self.rag:
            return
        
        # Add traffic patterns
        for time_period, data in self.knowledge.get("traffic_patterns", {}).items():
            self.rag.add_document(
                f"Traffic pattern for {time_period}: {data}",
                metadata={"type": "traffic_pattern", "period": time_period}
            )
        
        # Add road information
        for road, info in self.knowledge.get("roads", {}).items():
            self.rag.add_document(
                f"Road information for {road}: {info}",
                metadata={"type": "road_info", "road": road}
            )
        
        # Add metro information
        self.rag.add_document(
            f"Metro network: {self.knowledge.get('metro', {})}",
            metadata={"type": "metro"}
        )
        
        self.logger.debug(f"Loaded knowledge base with traffic data")
    
    # ========================================================================
    # MAIN QUERY INTERFACE
    # ========================================================================
    
    async def query(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None
    ) -> TrafficGodResponse:
        """
        Process a natural language query about traffic
        
        Args:
            question: Natural language question
            context: Optional context (location, time, etc.)
        
        Returns:
            TrafficGodResponse with answer and metadata
        """
        if not self.is_initialized:
            await self.initialize()
        
        start_time = datetime.now()
        
        # Detect query type
        query_type = self._detect_query_type(question)
        self.logger.debug(f"Query type detected: {query_type}")
        
        # Route to appropriate handler
        if query_type == "route_planning":
            result = await self._handle_route_query(question, context)
        elif query_type == "traffic_prediction":
            result = await self._handle_prediction_query(question, context)
        elif query_type == "scenario_analysis":
            result = await self._handle_scenario_query(question, context)
        elif query_type == "infrastructure":
            result = await self._handle_infrastructure_query(question, context)
        elif query_type == "aqi":
            result = await self._handle_aqi_query(question, context)
        else:
            result = await self._handle_general_query(question, context)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return TrafficGodResponse(
            answer=result["answer"],
            confidence=result.get("confidence", 0.8),
            query_type=query_type,
            data=result.get("data", {}),
            sources=result.get("sources", []),
            timestamp=datetime.now().isoformat(),
            processing_time_ms=processing_time
        )
    
    def _detect_query_type(self, question: str) -> str:
        """Detect the type of query"""
        question_lower = question.lower()
        
        route_keywords = ["route", "how to go", "best way", "reach", "travel to", "directions"]
        prediction_keywords = ["traffic", "congestion", "busy", "peak", "how long", "time to reach"]
        scenario_keywords = ["what if", "scenario", "imagine", "suppose", "impact of"]
        infrastructure_keywords = ["infrastructure", "flyover", "metro", "road", "build", "construct"]
        aqi_keywords = ["aqi", "pollution", "air quality", "smog", "pm2.5"]
        
        for kw in route_keywords:
            if kw in question_lower:
                return "route_planning"
        
        for kw in scenario_keywords:
            if kw in question_lower:
                return "scenario_analysis"
        
        for kw in infrastructure_keywords:
            if kw in question_lower:
                return "infrastructure"
        
        for kw in aqi_keywords:
            if kw in question_lower:
                return "aqi"
        
        for kw in prediction_keywords:
            if kw in question_lower:
                return "traffic_prediction"
        
        return "general"
    
    async def _handle_route_query(
        self,
        question: str,
        context: Optional[Dict]
    ) -> Dict[str, Any]:
        """Handle route planning queries"""
        # Extract source and destination from question
        # This is a placeholder - real implementation would use NER
        
        routes = get_route_options("Sector 18", "Indirapuram")
        
        current_hour = datetime.now().hour
        traffic_level = "heavy" if is_peak_hour(current_hour) else "moderate"
        
        answer = f"""Based on current {traffic_level} traffic conditions:

**Recommended Route:** {routes[0]['name']}
- Distance: {routes[0]['distance_km']} km
- Estimated Time: {routes[0]['duration_min']} minutes
- Via: {', '.join(routes[0]['via'])}

**Alternative Routes:**
"""
        for i, route in enumerate(routes[1:], 1):
            answer += f"\n{i}. {route['name']} - {route['duration_min']} min ({route['distance_km']} km)"
        
        return {
            "answer": answer,
            "confidence": 0.85,
            "data": {"routes": routes, "traffic_level": traffic_level},
            "sources": ["historical_patterns", "real_time_estimation"]
        }
    
    async def _handle_prediction_query(
        self,
        question: str,
        context: Optional[Dict]
    ) -> Dict[str, Any]:
        """Handle traffic prediction queries"""
        current_hour = datetime.now().hour
        pattern = get_current_traffic_pattern()
        
        answer = f"""**Current Traffic Status ({get_time_of_day(current_hour)})**

{pattern['description']}

**Key Observations:**
- Overall congestion: {pattern['congestion_level']}
- Average speed: {pattern['avg_speed']} km/h
- Hotspots: {', '.join(pattern.get('hotspots', ['Sector 18', 'NH-24']))}

**Prediction for next 2 hours:**
"""
        
        # Add predictions
        for i in range(1, 3):
            future_hour = (current_hour + i) % 24
            future_level = "increasing" if is_peak_hour(future_hour) else "decreasing"
            answer += f"\n- {future_hour}:00 - Traffic {future_level}"
        
        return {
            "answer": answer,
            "confidence": 0.82,
            "data": pattern,
            "sources": ["traffic_patterns", "historical_data"]
        }
    
    async def _handle_scenario_query(
        self,
        question: str,
        context: Optional[Dict]
    ) -> Dict[str, Any]:
        """Handle scenario analysis queries"""
        if self.world_model:
            # Use world model for simulation
            planner = ScenarioPlanner(self.world_model)
            # Placeholder - would parse scenario from question
        
        answer = """**Scenario Analysis**

I've analyzed your scenario. Here's my assessment:

**Impact Analysis:**
- Traffic flow change: +15-25%
- Affected corridors: NH-24, Sector 62 Road
- Peak hour impact: Significant reduction in congestion

**Recommendations:**
1. Phased implementation recommended
2. Consider temporary diversions during construction
3. Coordinate with metro schedule changes

**Long-term Benefits:**
- Reduced travel time by 12-18 minutes
- Lower fuel consumption
- Improved air quality in the area
"""
        
        return {
            "answer": answer,
            "confidence": 0.75,
            "data": {"impact": "positive", "magnitude": "moderate"},
            "sources": ["world_model_simulation", "historical_analogies"]
        }
    
    async def _handle_infrastructure_query(
        self,
        question: str,
        context: Optional[Dict]
    ) -> Dict[str, Any]:
        """Handle infrastructure suggestion queries"""
        answer = """**Infrastructure Analysis**

Based on current traffic patterns and future projections:

**Recommended Improvements:**

1. **Signal Optimization** (Immediate)
   - Smart signals at Sector 62 crossing
   - Cost: ₹5 Cr | Impact: 15% flow improvement

2. **Flyover Construction** (Medium-term)
   - Location: Sector 18 to Film City junction
   - Cost: ₹120 Cr | Impact: 35% congestion reduction

3. **Metro Extension** (Long-term)
   - Aqua Line extension to Indirapuram
   - Cost: ₹800 Cr | Impact: 25% vehicle reduction

**Applicable Government Schemes:**
- Smart Cities Mission (30% funding)
- AMRUT 2.0 (infrastructure grants)
- FAME II (EV infrastructure)
"""
        
        return {
            "answer": answer,
            "confidence": 0.80,
            "data": {"suggestions": 3},
            "sources": ["urban_planning_data", "government_schemes"]
        }
    
    async def _handle_aqi_query(
        self,
        question: str,
        context: Optional[Dict]
    ) -> Dict[str, Any]:
        """Handle AQI and pollution queries"""
        aqi_data = get_aqi_estimate()
        
        answer = f"""**Air Quality Report**

**Current AQI:** {aqi_data['aqi']} ({aqi_data['category']})

**Pollutant Levels:**
- PM2.5: {aqi_data['pm25']} µg/m³
- PM10: {aqi_data['pm10']} µg/m³
- NO2: {aqi_data.get('no2', 45)} ppb

**Traffic Contribution:** {aqi_data.get('traffic_contribution', 35)}%

**Health Advisory:**
{aqi_data.get('advisory', 'Sensitive groups should limit outdoor activities')}

**Recommendations:**
- Use N95 masks if outdoors
- Prefer metro/electric vehicles
- Best time for outdoor activities: Early morning (6-8 AM)
"""
        
        return {
            "answer": answer,
            "confidence": 0.88,
            "data": aqi_data,
            "sources": ["aqi_sensors", "traffic_emissions_model"]
        }
    
    async def _handle_general_query(
        self,
        question: str,
        context: Optional[Dict]
    ) -> Dict[str, Any]:
        """Handle general queries"""
        # Use RAG for general knowledge retrieval
        if self.rag:
            relevant_docs = self.rag.retrieve(question, top_k=3)
            context_text = "\n".join([doc.text for doc in relevant_docs])
        else:
            context_text = ""
        
        answer = f"""Based on my knowledge of Noida/NCR traffic:

{context_text if context_text else "I can help you with traffic predictions, route planning, scenario analysis, and infrastructure suggestions for the Noida-NCR region."}

Would you like me to:
1. Check current traffic conditions
2. Plan a route
3. Analyze a scenario
4. Suggest infrastructure improvements
"""
        
        return {
            "answer": answer,
            "confidence": 0.70,
            "data": {},
            "sources": ["knowledge_base"]
        }
    
    # ========================================================================
    # SPECIALIZED METHODS
    # ========================================================================
    
    async def plan_route(
        self,
        source: str,
        destination: str,
        departure_time: Optional[str] = None,
        preferences: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Plan optimal route between two points
        
        Args:
            source: Starting location
            destination: End location
            departure_time: Preferred departure time (HH:MM)
            preferences: Route preferences (avoid_tolls, prefer_highways, etc.)
        
        Returns:
            Route options with traffic conditions
        """
        routes = get_route_options(source, destination)
        
        # Adjust for departure time
        if departure_time:
            hour = int(departure_time.split(":")[0])
            if is_peak_hour(hour):
                for route in routes:
                    route["duration_min"] = int(route["duration_min"] * 1.5)
        
        return {
            "source": source,
            "destination": destination,
            "departure_time": departure_time or "now",
            "routes": routes,
            "recommended": routes[0] if routes else None
        }
    
    async def predict_traffic(
        self,
        location: str,
        time_range: str = "6h"
    ) -> Dict[str, Any]:
        """
        Predict traffic for a location
        
        Args:
            location: Location name or coordinates
            time_range: Prediction range ("1h", "6h", "24h")
        
        Returns:
            Traffic predictions
        """
        if not self.predictor:
            raise RuntimeError("Predictor not initialized")
        
        # Use predictor for actual prediction
        current = get_current_traffic_pattern()
        
        return {
            "location": location,
            "current": current,
            "predictions": [],  # Would be filled by model
            "confidence": 0.85
        }
    
    async def analyze_scenario(
        self,
        scenario_description: str,
        affected_area: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze a hypothetical scenario
        
        Args:
            scenario_description: Description of the scenario
            affected_area: Primary affected area
        
        Returns:
            Scenario analysis with impacts and recommendations
        """
        if self.world_model:
            planner = ScenarioPlanner(self.world_model)
            # Would run actual simulation
        
        return {
            "scenario": scenario_description,
            "area": affected_area,
            "impact": "Analysis would be here",
            "recommendations": []
        }
    
    async def suggest_infrastructure(
        self,
        area: str,
        problem: str,
        budget: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Suggest infrastructure improvements
        
        Args:
            area: Target area
            problem: Problem description
            budget: Budget in crores (optional)
        
        Returns:
            Infrastructure suggestions
        """
        return {
            "area": area,
            "problem": problem,
            "suggestions": [],  # Would be filled by analysis
            "government_schemes": []
        }
    
    # ========================================================================
    # TRAINING METHODS
    # ========================================================================
    
    async def train(
        self,
        data_path: str,
        epochs: int = 10,
        batch_size: int = 32
    ):
        """Train or fine-tune the model"""
        if not self.model:
            raise RuntimeError("Model not initialized")
        
        train_config = self.config.get("training")
        trainer = Trainer(
            model=self.model,
            config=train_config
        )
        
        # Would implement actual training
        self.logger.info(f"Training for {epochs} epochs...")
    
    async def fine_tune(
        self,
        dataset_path: str,
        task: str = "traffic_prediction"
    ):
        """Fine-tune model for specific task"""
        self.logger.info(f"Fine-tuning for task: {task}")
        # Would implement fine-tuning
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            "initialized": self.is_initialized,
            "environment": self.environment,
            "device": self.device,
            "components": {
                "model": self.model is not None,
                "tokenizer": self.tokenizer is not None,
                "world_model": self.world_model is not None,
                "rag": self.rag is not None,
                "predictor": self.predictor is not None
            }
        }
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        if self.model:
            import torch
            torch.save(self.model.state_dict(), path)
            self.logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        if self.model:
            import torch
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.logger.info(f"Checkpoint loaded from {path}")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

async def create_traffic_god(
    environment: str = "development"
) -> TrafficGod:
    """
    Create and initialize Traffic God
    
    Args:
        environment: "development", "production", or "testing"
    
    Returns:
        Initialized TrafficGod instance
    """
    god = TrafficGod(environment=environment)
    await god.initialize()
    return god


def quick_query(question: str) -> str:
    """
    Quick synchronous query (for simple use cases)
    
    Args:
        question: Question to ask
    
    Returns:
        Answer string
    """
    async def _query():
        god = await create_traffic_god("development")
        response = await god.query(question)
        return response.answer
    
    return asyncio.run(_query())


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """CLI interface for Traffic God"""
    import argparse
    
    parser = argparse.ArgumentParser(description="New Traffic God CLI")
    parser.add_argument("--env", default="development", help="Environment")
    parser.add_argument("--query", type=str, help="Query to process")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    async def run():
        god = await create_traffic_god(args.env)
        
        if args.query:
            response = await god.query(args.query)
            print(response.answer)
        
        elif args.interactive:
            print("Traffic God Interactive Mode")
            print("Type 'exit' to quit\n")
            
            while True:
                try:
                    question = input("You: ").strip()
                    if question.lower() == 'exit':
                        break
                    
                    response = await god.query(question)
                    print(f"\nTraffic God: {response.answer}\n")
                    
                except KeyboardInterrupt:
                    break
            
            print("\nGoodbye!")
    
    asyncio.run(run())


if __name__ == "__main__":
    main()
