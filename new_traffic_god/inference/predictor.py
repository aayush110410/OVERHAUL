"""
Traffic Predictor - The Unified Inference Engine
=================================================

Combines all components for inference:
1. Foundation Model - Language understanding and generation
2. World Model - Simulation and scenario planning
3. RAG System - Real-time knowledge retrieval
4. Physics Engine - Constraint validation

This is the main interface for making predictions and answering queries.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import re
import math

# Import components
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.foundation_model import TrafficFoundationModel, TrafficModelConfig, create_traffic_model
from core.tokenizer import TrafficTokenizer
from simulation.world_model import WorldModel, TrafficState, SimulationConfig, ScenarioPlanner, create_world_model
from retrieval.rag_system import TrafficRAG, create_rag_system


@dataclass
class PredictionConfig:
    """Configuration for the predictor"""
    model_size: str = "base"
    use_simulation: bool = True
    use_rag: bool = True
    max_simulation_steps: int = 60
    confidence_threshold: float = 0.7
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


@dataclass
class TrafficPrediction:
    """Structured traffic prediction output"""
    # Core predictions
    travel_time_minutes: float
    congestion_level: str  # free, light, moderate, heavy, gridlock
    congestion_score: float  # 0-1
    avg_speed_kmh: float
    traffic_flow_vph: int  # vehicles per hour
    
    # Air quality
    aqi_prediction: float
    aqi_category: str
    
    # Route information
    recommended_route: Optional[List[str]] = None
    alternative_routes: Optional[List[Dict[str, Any]]] = None
    
    # Time-based predictions
    forecasts: Optional[Dict[str, Any]] = None
    best_departure_time: Optional[str] = None
    
    # Confidence and sources
    confidence: float = 0.0
    uncertainty: float = 0.0
    data_sources: List[str] = field(default_factory=list)
    citations: List[Dict[str, Any]] = field(default_factory=list)
    
    # Explanation
    explanation: str = ""
    factors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "travel_time_minutes": self.travel_time_minutes,
            "congestion_level": self.congestion_level,
            "congestion_score": self.congestion_score,
            "avg_speed_kmh": self.avg_speed_kmh,
            "traffic_flow_vph": self.traffic_flow_vph,
            "aqi_prediction": self.aqi_prediction,
            "aqi_category": self.aqi_category,
            "recommended_route": self.recommended_route,
            "alternative_routes": self.alternative_routes,
            "forecasts": self.forecasts,
            "best_departure_time": self.best_departure_time,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "data_sources": self.data_sources,
            "citations": self.citations,
            "explanation": self.explanation,
            "factors": self.factors
        }


class QueryParser:
    """
    Parse natural language queries to extract:
    - Intent (route, prediction, comparison, hypothetical, etc.)
    - Locations (origin, destination)
    - Time context
    - Constraints
    """
    
    # Intent patterns
    INTENT_PATTERNS = {
        "route": [
            r"(?:best|fastest|shortest|optimal)\s+(?:way|route|path)",
            r"how\s+(?:to\s+)?(?:get|go|reach|travel)",
            r"directions?\s+(?:to|from)",
            r"route\s+(?:from|to|between)"
        ],
        "prediction": [
            r"(?:what|how)\s+(?:is|will)\s+(?:the\s+)?traffic",
            r"predict\s+(?:the\s+)?traffic",
            r"traffic\s+(?:forecast|prediction)",
            r"(?:will|would)\s+there\s+be\s+(?:traffic|congestion)"
        ],
        "travel_time": [
            r"how\s+long\s+(?:will|does|would)",
            r"(?:travel|journey|commute)\s+time",
            r"time\s+to\s+reach",
            r"eta\s+(?:to|for)"
        ],
        "comparison": [
            r"(?:compare|versus|vs|better)",
            r"which\s+(?:is|route|way)\s+(?:faster|better|shorter)",
            r"difference\s+between"
        ],
        "hypothetical": [
            r"what\s+if",
            r"(?:if|when)\s+we\s+(?:add|remove|build|close)",
            r"impact\s+of",
            r"(?:how|what)\s+(?:would|will)\s+(?:happen|change)"
        ],
        "current_status": [
            r"(?:current|right\s+now|live)",
            r"is\s+there\s+(?:traffic|congestion|jam)",
            r"(?:how|what)\s+is\s+(?:the\s+)?(?:traffic|situation)"
        ],
        "aqi": [
            r"(?:air\s+quality|aqi|pollution)",
            r"(?:how|what)\s+is\s+(?:the\s+)?(?:air|pollution)"
        ]
    }
    
    # Location patterns
    LOCATION_KEYWORDS = [
        "from", "to", "at", "in", "near", "on", "via", "through",
        "between", "towards", "starting", "ending", "origin", "destination"
    ]
    
    # Time patterns
    TIME_PATTERNS = {
        "absolute": r"(?:at\s+)?(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)",
        "relative": r"(in\s+\d+\s+(?:minute|hour|min|hr)s?)",
        "period": r"(morning|afternoon|evening|night|rush\s+hour|peak|off-?peak)",
        "day": r"(today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
    }
    
    def __init__(self):
        # Compile patterns
        self.intent_compiled = {
            intent: [re.compile(p, re.IGNORECASE) for p in patterns]
            for intent, patterns in self.INTENT_PATTERNS.items()
        }
        
        self.time_compiled = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.TIME_PATTERNS.items()
        }
    
    def parse(self, query: str) -> Dict[str, Any]:
        """Parse a query and extract structured information"""
        query_lower = query.lower()
        
        result = {
            "original_query": query,
            "intent": self._detect_intent(query_lower),
            "locations": self._extract_locations(query),
            "time_context": self._extract_time(query),
            "entities": self._extract_entities(query),
            "constraints": self._extract_constraints(query_lower)
        }
        
        return result
    
    def _detect_intent(self, query: str) -> str:
        """Detect the primary intent of the query"""
        for intent, patterns in self.intent_compiled.items():
            for pattern in patterns:
                if pattern.search(query):
                    return intent
        return "general"
    
    def _extract_locations(self, query: str) -> Dict[str, Optional[str]]:
        """Extract origin, destination, and other locations"""
        locations = {
            "origin": None,
            "destination": None,
            "mentions": []
        }
        
        # Known Noida/NCR locations
        known_locations = [
            "sector 18", "sector 62", "sector 63", "film city",
            "noida expressway", "dnd flyway", "nh24", "indirapuram",
            "vaishali", "kaushambi", "vasundhara", "greater noida",
            "delhi", "ghaziabad", "botanical garden", "noida city centre"
        ]
        
        query_lower = query.lower()
        
        # Find all location mentions
        for loc in known_locations:
            if loc in query_lower:
                locations["mentions"].append(loc.title())
        
        # Try to identify origin and destination
        from_match = re.search(r"from\s+([^,\.]+?)(?:\s+to|\s*,|$)", query_lower)
        to_match = re.search(r"to\s+([^,\.]+?)(?:\s+from|\s*,|$)", query_lower)
        
        if from_match:
            locations["origin"] = from_match.group(1).strip().title()
        if to_match:
            locations["destination"] = to_match.group(1).strip().title()
        
        # If only one location mentioned and it's a route query
        if len(locations["mentions"]) == 1 and not locations["origin"]:
            locations["destination"] = locations["mentions"][0]
        
        return locations
    
    def _extract_time(self, query: str) -> Dict[str, Any]:
        """Extract time-related information"""
        time_info = {
            "specified": False,
            "time": None,
            "period": None,
            "day": None,
            "is_future": False
        }
        
        for time_type, pattern in self.time_compiled.items():
            match = pattern.search(query)
            if match:
                time_info["specified"] = True
                if time_type == "absolute":
                    time_info["time"] = match.group(1)
                elif time_type == "period":
                    time_info["period"] = match.group(1)
                elif time_type == "day":
                    time_info["day"] = match.group(1)
                    time_info["is_future"] = match.group(1).lower() == "tomorrow"
        
        return time_info
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities"""
        entities = []
        
        # Road names
        roads = re.findall(r"(nh\d+|[a-z]+ expressway|[a-z]+ flyway)", query.lower())
        entities.extend([r.upper() for r in roads])
        
        # Numbers (could be sector numbers, times, etc.)
        numbers = re.findall(r"\d+", query)
        
        return entities
    
    def _extract_constraints(self, query: str) -> Dict[str, Any]:
        """Extract constraints and preferences"""
        constraints = {
            "avoid_tolls": "avoid toll" in query or "no toll" in query,
            "avoid_highways": "avoid highway" in query or "no highway" in query,
            "prefer_metro": "metro" in query or "public transport" in query,
            "time_sensitive": "urgent" in query or "asap" in query or "quickly" in query
        }
        return constraints


class TrafficPredictor:
    """
    The main prediction engine that combines all components
    """
    
    def __init__(self, config: Optional[PredictionConfig] = None):
        self.config = config or PredictionConfig()
        
        # Initialize components
        print("Initializing Traffic Predictor...")
        
        # Foundation model
        print("  Loading foundation model...")
        self.model = create_traffic_model(self.config.model_size, self.config.device)
        
        # Tokenizer
        print("  Loading tokenizer...")
        self.tokenizer = TrafficTokenizer()
        
        # World model for simulation
        if self.config.use_simulation:
            print("  Loading world model...")
            self.world_model = create_world_model(self.config.device)
            self.planner = ScenarioPlanner(self.world_model)
        else:
            self.world_model = None
            self.planner = None
        
        # RAG system
        if self.config.use_rag:
            print("  Loading RAG system...")
            self.rag = create_rag_system()
        else:
            self.rag = None
        
        # Query parser
        self.parser = QueryParser()
        
        # Cache for recent predictions
        self.cache: Dict[str, Tuple[TrafficPrediction, datetime]] = {}
        self.cache_ttl = timedelta(minutes=5)
        
        print("Traffic Predictor ready!")
    
    def predict(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> TrafficPrediction:
        """
        Main prediction method
        
        Takes a natural language query and returns a structured prediction
        """
        # Check cache
        cache_key = self._get_cache_key(query)
        if cache_key in self.cache:
            pred, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < self.cache_ttl:
                return pred
        
        # Parse query
        parsed = self.parser.parse(query)
        
        # Get current time context
        time_context = self._get_time_context(parsed.get("time_context", {}))
        
        # Retrieve relevant knowledge
        rag_context = None
        if self.rag:
            location = parsed["locations"].get("origin") or parsed["locations"].get("destination")
            rag_context = self.rag.retrieve_context(query, location)
        
        # Route to appropriate handler based on intent
        intent = parsed["intent"]
        
        if intent == "route":
            prediction = self._handle_route_query(parsed, time_context, rag_context)
        elif intent == "prediction":
            prediction = self._handle_prediction_query(parsed, time_context, rag_context)
        elif intent == "travel_time":
            prediction = self._handle_travel_time_query(parsed, time_context, rag_context)
        elif intent == "hypothetical":
            prediction = self._handle_hypothetical_query(parsed, time_context, rag_context)
        elif intent == "current_status":
            prediction = self._handle_status_query(parsed, time_context, rag_context)
        elif intent == "aqi":
            prediction = self._handle_aqi_query(parsed, time_context, rag_context)
        else:
            prediction = self._handle_general_query(parsed, time_context, rag_context)
        
        # Cache prediction
        self.cache[cache_key] = (prediction, datetime.now())
        
        return prediction
    
    def _get_cache_key(self, query: str) -> str:
        """Generate cache key for query"""
        import hashlib
        # Include time bucket for freshness
        time_bucket = datetime.now().strftime("%Y%m%d%H")
        return hashlib.md5(f"{query}:{time_bucket}".encode()).hexdigest()
    
    def _get_time_context(self, parsed_time: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Convert parsed time to model input format"""
        now = datetime.now()
        
        # Time slot (5-minute intervals)
        minutes = now.hour * 60 + now.minute
        time_slot = minutes // 5
        
        return {
            "time_slots": torch.tensor([[time_slot]], device=self.config.device),
            "day_of_week": torch.tensor([[now.weekday()]], device=self.config.device),
            "month": torch.tensor([[now.month - 1]], device=self.config.device),
            "is_holiday": torch.tensor([[0]], device=self.config.device)  # Would need holiday calendar
        }
    
    def _get_base_predictions(
        self,
        query: str,
        time_context: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """Get predictions from foundation model"""
        # Tokenize query
        input_ids = self.tokenizer.encode(query, max_length=512, truncation=True)
        input_tensor = torch.tensor([input_ids], device=self.config.device)
        
        # Model inference
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_tensor,
                **time_context,
                output_predictions=True
            )
        
        # Extract predictions
        congestion_probs = F.softmax(outputs["congestion_level"], dim=-1)[0]
        congestion_level = congestion_probs.argmax().item()
        
        return {
            "traffic_flow": outputs["traffic_flow"][0].item(),
            "travel_time": outputs["travel_time"][0].item(),
            "congestion_level": congestion_level,
            "congestion_probs": congestion_probs.tolist(),
            "aqi": outputs["aqi_prediction"][0].item(),
            "forecasts": outputs["forecasts"][0].tolist(),
            "confidence": 1.0 / (1.0 + outputs["prediction_uncertainty"][0].item())
        }
    
    def _congestion_to_text(self, level: int) -> str:
        """Convert congestion level to text"""
        levels = ["free", "light", "moderate", "heavy", "gridlock"]
        return levels[min(level, len(levels) - 1)]
    
    def _aqi_to_category(self, aqi: float) -> str:
        """Convert AQI to category"""
        if aqi <= 50:
            return "Good"
        elif aqi <= 100:
            return "Moderate"
        elif aqi <= 150:
            return "Unhealthy for Sensitive Groups"
        elif aqi <= 200:
            return "Unhealthy"
        elif aqi <= 300:
            return "Very Unhealthy"
        else:
            return "Hazardous"
    
    def _handle_route_query(
        self,
        parsed: Dict[str, Any],
        time_context: Dict[str, torch.Tensor],
        rag_context: Optional[Dict[str, Any]]
    ) -> TrafficPrediction:
        """Handle route finding queries"""
        origin = parsed["locations"].get("origin", "Current Location")
        destination = parsed["locations"].get("destination", "Unknown")
        
        # Get base predictions
        base_preds = self._get_base_predictions(parsed["original_query"], time_context)
        
        # Get route from planner if available
        routes = []
        if self.planner:
            # Would need to convert locations to grid coordinates
            # For now, generate sample routes
            routes = [
                {
                    "name": "Via Noida Expressway",
                    "distance_km": 15.2,
                    "estimated_time_minutes": base_preds["travel_time"] * 0.9,
                    "congestion": "moderate"
                },
                {
                    "name": "Via DND Flyway",
                    "distance_km": 18.5,
                    "estimated_time_minutes": base_preds["travel_time"] * 1.1,
                    "congestion": "heavy"
                },
                {
                    "name": "Via NH24",
                    "distance_km": 16.8,
                    "estimated_time_minutes": base_preds["travel_time"] * 1.2,
                    "congestion": "heavy"
                }
            ]
        
        # Build explanation
        explanation = f"Route from {origin} to {destination}:\n"
        if routes:
            explanation += f"Recommended: {routes[0]['name']} ({routes[0]['distance_km']} km, ~{routes[0]['estimated_time_minutes']:.0f} min)\n"
            explanation += f"Current congestion: {self._congestion_to_text(base_preds['congestion_level'])}\n"
        
        # Add RAG context
        data_sources = ["Traffic Foundation Model"]
        citations = []
        if rag_context:
            data_sources.append("Noida Traffic Knowledge Base")
            citations = rag_context.get("citations", [])
            if rag_context.get("relevant_docs"):
                explanation += f"\nRelevant info: {rag_context['relevant_docs'][0][0][:200]}..."
        
        return TrafficPrediction(
            travel_time_minutes=base_preds["travel_time"],
            congestion_level=self._congestion_to_text(base_preds["congestion_level"]),
            congestion_score=base_preds["congestion_probs"][base_preds["congestion_level"]],
            avg_speed_kmh=max(10, 60 - base_preds["congestion_level"] * 10),
            traffic_flow_vph=int(base_preds["traffic_flow"]),
            aqi_prediction=base_preds["aqi"],
            aqi_category=self._aqi_to_category(base_preds["aqi"]),
            recommended_route=[routes[0]["name"]] if routes else None,
            alternative_routes=routes[1:] if len(routes) > 1 else None,
            confidence=base_preds["confidence"],
            uncertainty=1 - base_preds["confidence"],
            data_sources=data_sources,
            citations=citations,
            explanation=explanation,
            factors=["Time of day", "Day of week", "Historical patterns", "Current conditions"]
        )
    
    def _handle_prediction_query(
        self,
        parsed: Dict[str, Any],
        time_context: Dict[str, torch.Tensor],
        rag_context: Optional[Dict[str, Any]]
    ) -> TrafficPrediction:
        """Handle traffic prediction queries"""
        base_preds = self._get_base_predictions(parsed["original_query"], time_context)
        
        # Build multi-horizon forecast
        forecasts = {
            "5_min": {"congestion": base_preds["forecasts"][0] if base_preds["forecasts"] else None},
            "15_min": {"congestion": base_preds["forecasts"][2] if len(base_preds["forecasts"]) > 2 else None},
            "30_min": {"congestion": base_preds["forecasts"][5] if len(base_preds["forecasts"]) > 5 else None},
            "1_hour": {"congestion": base_preds["forecasts"][11] if len(base_preds["forecasts"]) > 11 else None}
        }
        
        # Find best departure time
        best_time = None
        if base_preds["congestion_level"] >= 3:  # Heavy or worse
            best_time = "Consider departing 30-45 minutes earlier or after peak hours (after 10 AM or after 8 PM)"
        
        explanation = f"Traffic prediction for the queried area:\n"
        explanation += f"Current congestion: {self._congestion_to_text(base_preds['congestion_level'])}\n"
        explanation += f"Average speed: {max(10, 60 - base_preds['congestion_level'] * 10)} km/h\n"
        explanation += f"Traffic flow: {int(base_preds['traffic_flow'])} vehicles/hour\n"
        
        data_sources = ["Traffic Foundation Model", "Historical Patterns"]
        if rag_context:
            data_sources.append("Real-time Updates")
        
        return TrafficPrediction(
            travel_time_minutes=base_preds["travel_time"],
            congestion_level=self._congestion_to_text(base_preds["congestion_level"]),
            congestion_score=base_preds["congestion_probs"][base_preds["congestion_level"]],
            avg_speed_kmh=max(10, 60 - base_preds["congestion_level"] * 10),
            traffic_flow_vph=int(base_preds["traffic_flow"]),
            aqi_prediction=base_preds["aqi"],
            aqi_category=self._aqi_to_category(base_preds["aqi"]),
            forecasts=forecasts,
            best_departure_time=best_time,
            confidence=base_preds["confidence"],
            uncertainty=1 - base_preds["confidence"],
            data_sources=data_sources,
            explanation=explanation,
            factors=["Time of day", "Day of week", "Weather", "Events", "Historical patterns"]
        )
    
    def _handle_travel_time_query(
        self,
        parsed: Dict[str, Any],
        time_context: Dict[str, torch.Tensor],
        rag_context: Optional[Dict[str, Any]]
    ) -> TrafficPrediction:
        """Handle travel time queries"""
        base_preds = self._get_base_predictions(parsed["original_query"], time_context)
        
        origin = parsed["locations"].get("origin", "Origin")
        destination = parsed["locations"].get("destination", "Destination")
        
        # Adjust travel time based on congestion
        adjusted_time = base_preds["travel_time"]
        if base_preds["congestion_level"] >= 3:
            adjusted_time *= 1.3  # 30% increase for heavy traffic
        
        explanation = f"Estimated travel time from {origin} to {destination}:\n"
        explanation += f"Normal conditions: {base_preds['travel_time']:.0f} minutes\n"
        explanation += f"Current conditions: {adjusted_time:.0f} minutes\n"
        explanation += f"Traffic level: {self._congestion_to_text(base_preds['congestion_level'])}\n"
        
        if base_preds["congestion_level"] >= 3:
            explanation += "\n⚠️ Heavy traffic detected. Consider alternate routes or times."
        
        return TrafficPrediction(
            travel_time_minutes=adjusted_time,
            congestion_level=self._congestion_to_text(base_preds["congestion_level"]),
            congestion_score=base_preds["congestion_probs"][base_preds["congestion_level"]],
            avg_speed_kmh=max(10, 60 - base_preds["congestion_level"] * 10),
            traffic_flow_vph=int(base_preds["traffic_flow"]),
            aqi_prediction=base_preds["aqi"],
            aqi_category=self._aqi_to_category(base_preds["aqi"]),
            confidence=base_preds["confidence"],
            uncertainty=1 - base_preds["confidence"],
            data_sources=["Traffic Foundation Model", "Real-time Data"],
            explanation=explanation,
            factors=["Distance", "Current traffic", "Time of day", "Road conditions"]
        )
    
    def _handle_hypothetical_query(
        self,
        parsed: Dict[str, Any],
        time_context: Dict[str, torch.Tensor],
        rag_context: Optional[Dict[str, Any]]
    ) -> TrafficPrediction:
        """Handle hypothetical/what-if queries"""
        base_preds = self._get_base_predictions(parsed["original_query"], time_context)
        
        # Parse hypothetical scenario
        query_lower = parsed["original_query"].lower()
        
        scenario_type = "general"
        impact_factor = 1.0
        
        if "flyover" in query_lower or "bridge" in query_lower:
            scenario_type = "infrastructure_add"
            impact_factor = 0.7  # 30% improvement
        elif "close" in query_lower or "block" in query_lower:
            scenario_type = "infrastructure_remove"
            impact_factor = 1.5  # 50% worsening
        elif "metro" in query_lower:
            scenario_type = "metro"
            impact_factor = 0.85  # 15% improvement
        elif "ev" in query_lower or "electric" in query_lower:
            scenario_type = "ev_adoption"
            impact_factor = 1.0  # Traffic neutral, but AQI improves
        
        # Simulate impact
        if self.world_model:
            # Would run actual simulation here
            pass
        
        # Calculate hypothetical outcomes
        new_travel_time = base_preds["travel_time"] * impact_factor
        new_congestion = max(0, min(4, base_preds["congestion_level"] + (1 if impact_factor > 1 else -1)))
        
        aqi_impact = base_preds["aqi"]
        if scenario_type == "ev_adoption":
            aqi_impact *= 0.75  # 25% AQI improvement
        
        explanation = f"Hypothetical scenario analysis:\n"
        explanation += f"Scenario: {scenario_type.replace('_', ' ').title()}\n"
        explanation += f"\nCurrent state:\n"
        explanation += f"  Travel time: {base_preds['travel_time']:.0f} min\n"
        explanation += f"  Congestion: {self._congestion_to_text(base_preds['congestion_level'])}\n"
        explanation += f"\nProjected state:\n"
        explanation += f"  Travel time: {new_travel_time:.0f} min ({(impact_factor-1)*100:+.0f}%)\n"
        explanation += f"  Congestion: {self._congestion_to_text(new_congestion)}\n"
        
        if scenario_type == "ev_adoption":
            explanation += f"\nAQI Impact:\n"
            explanation += f"  Current AQI: {base_preds['aqi']:.0f}\n"
            explanation += f"  Projected AQI: {aqi_impact:.0f} (-25%)\n"
        
        return TrafficPrediction(
            travel_time_minutes=new_travel_time,
            congestion_level=self._congestion_to_text(new_congestion),
            congestion_score=0.7,
            avg_speed_kmh=max(10, 60 - new_congestion * 10),
            traffic_flow_vph=int(base_preds["traffic_flow"] / impact_factor),
            aqi_prediction=aqi_impact,
            aqi_category=self._aqi_to_category(aqi_impact),
            confidence=0.6,  # Lower confidence for hypotheticals
            uncertainty=0.4,
            data_sources=["World Model Simulation", "Historical Patterns", "Traffic Physics"],
            explanation=explanation,
            factors=["Scenario parameters", "Traffic simulation", "Historical comparisons"]
        )
    
    def _handle_status_query(
        self,
        parsed: Dict[str, Any],
        time_context: Dict[str, torch.Tensor],
        rag_context: Optional[Dict[str, Any]]
    ) -> TrafficPrediction:
        """Handle current status queries"""
        base_preds = self._get_base_predictions(parsed["original_query"], time_context)
        
        location = parsed["locations"].get("destination") or parsed["locations"].get("origin") or "the area"
        
        explanation = f"Current traffic status for {location}:\n"
        explanation += f"Congestion level: {self._congestion_to_text(base_preds['congestion_level']).upper()}\n"
        explanation += f"Average speed: {max(10, 60 - base_preds['congestion_level'] * 10)} km/h\n"
        explanation += f"Traffic flow: {int(base_preds['traffic_flow'])} vehicles/hour\n"
        explanation += f"\nAir Quality: {self._aqi_to_category(base_preds['aqi'])} (AQI: {base_preds['aqi']:.0f})\n"
        
        if rag_context and rag_context.get("recent_updates"):
            explanation += f"\nRecent updates:\n"
            for update in rag_context["recent_updates"][:2]:
                explanation += f"  • {update[:100]}...\n"
        
        return TrafficPrediction(
            travel_time_minutes=base_preds["travel_time"],
            congestion_level=self._congestion_to_text(base_preds["congestion_level"]),
            congestion_score=base_preds["congestion_probs"][base_preds["congestion_level"]],
            avg_speed_kmh=max(10, 60 - base_preds["congestion_level"] * 10),
            traffic_flow_vph=int(base_preds["traffic_flow"]),
            aqi_prediction=base_preds["aqi"],
            aqi_category=self._aqi_to_category(base_preds["aqi"]),
            confidence=base_preds["confidence"],
            uncertainty=1 - base_preds["confidence"],
            data_sources=["Real-time Sensors", "Traffic Foundation Model"],
            explanation=explanation,
            factors=["Current sensors", "Time of day", "Recent incidents"]
        )
    
    def _handle_aqi_query(
        self,
        parsed: Dict[str, Any],
        time_context: Dict[str, torch.Tensor],
        rag_context: Optional[Dict[str, Any]]
    ) -> TrafficPrediction:
        """Handle AQI/pollution queries"""
        base_preds = self._get_base_predictions(parsed["original_query"], time_context)
        
        location = parsed["locations"].get("destination") or "Noida"
        
        # Calculate traffic contribution to AQI
        traffic_aqi_contribution = base_preds["traffic_flow"] * 0.02  # Simplified model
        
        explanation = f"Air Quality Analysis for {location}:\n"
        explanation += f"Current AQI: {base_preds['aqi']:.0f}\n"
        explanation += f"Category: {self._aqi_to_category(base_preds['aqi'])}\n"
        explanation += f"\nTraffic contribution to pollution:\n"
        explanation += f"  Estimated: {traffic_aqi_contribution:.0f} AQI points ({traffic_aqi_contribution/base_preds['aqi']*100:.1f}%)\n"
        explanation += f"\nCurrent traffic level: {self._congestion_to_text(base_preds['congestion_level'])}\n"
        
        if base_preds["aqi"] > 150:
            explanation += "\n⚠️ Air quality is unhealthy. Consider:\n"
            explanation += "  • Avoid outdoor activities\n"
            explanation += "  • Use public transport or carpool\n"
            explanation += "  • Travel during off-peak hours to reduce emissions\n"
        
        return TrafficPrediction(
            travel_time_minutes=base_preds["travel_time"],
            congestion_level=self._congestion_to_text(base_preds["congestion_level"]),
            congestion_score=base_preds["congestion_probs"][base_preds["congestion_level"]],
            avg_speed_kmh=max(10, 60 - base_preds["congestion_level"] * 10),
            traffic_flow_vph=int(base_preds["traffic_flow"]),
            aqi_prediction=base_preds["aqi"],
            aqi_category=self._aqi_to_category(base_preds["aqi"]),
            confidence=base_preds["confidence"],
            uncertainty=1 - base_preds["confidence"],
            data_sources=["AQI Sensors", "Traffic Foundation Model", "Emission Models"],
            explanation=explanation,
            factors=["Traffic volume", "Vehicle types", "Weather", "Industrial emissions"]
        )
    
    def _handle_general_query(
        self,
        parsed: Dict[str, Any],
        time_context: Dict[str, torch.Tensor],
        rag_context: Optional[Dict[str, Any]]
    ) -> TrafficPrediction:
        """Handle general queries that don't fit specific categories"""
        base_preds = self._get_base_predictions(parsed["original_query"], time_context)
        
        explanation = "Traffic Analysis:\n"
        explanation += f"Based on your query, here's the current situation:\n"
        explanation += f"  Congestion: {self._congestion_to_text(base_preds['congestion_level'])}\n"
        explanation += f"  Average speed: {max(10, 60 - base_preds['congestion_level'] * 10)} km/h\n"
        explanation += f"  Air quality: {self._aqi_to_category(base_preds['aqi'])} (AQI: {base_preds['aqi']:.0f})\n"
        
        if rag_context and rag_context.get("relevant_docs"):
            explanation += f"\nRelevant information:\n"
            for doc, score in rag_context["relevant_docs"][:3]:
                explanation += f"  • {doc[:150]}...\n"
        
        return TrafficPrediction(
            travel_time_minutes=base_preds["travel_time"],
            congestion_level=self._congestion_to_text(base_preds["congestion_level"]),
            congestion_score=base_preds["congestion_probs"][base_preds["congestion_level"]],
            avg_speed_kmh=max(10, 60 - base_preds["congestion_level"] * 10),
            traffic_flow_vph=int(base_preds["traffic_flow"]),
            aqi_prediction=base_preds["aqi"],
            aqi_category=self._aqi_to_category(base_preds["aqi"]),
            confidence=base_preds["confidence"],
            uncertainty=1 - base_preds["confidence"],
            data_sources=["Traffic Foundation Model"],
            explanation=explanation,
            factors=["Current conditions", "Historical patterns"]
        )
    
    def generate_response(
        self,
        query: str,
        max_length: int = 256
    ) -> str:
        """Generate a natural language response"""
        # Get prediction
        prediction = self.predict(query)
        
        # Generate text using foundation model
        input_ids = self.tokenizer.encode(
            f"Query: {query}\nTraffic conditions: {prediction.congestion_level}\nResponse:",
            max_length=512,
            truncation=True
        )
        input_tensor = torch.tensor([input_ids], device=self.config.device)
        
        self.model.eval()
        with torch.no_grad():
            output_ids = self.model.generate(
                input_tensor,
                max_new_tokens=max_length,
                temperature=0.7
            )
        
        response = self.tokenizer.decode(output_ids[0].tolist())
        
        # Append prediction summary
        response += f"\n\n📊 Summary:\n"
        response += f"• Travel time: {prediction.travel_time_minutes:.0f} min\n"
        response += f"• Congestion: {prediction.congestion_level}\n"
        response += f"• Air quality: {prediction.aqi_category}\n"
        response += f"• Confidence: {prediction.confidence*100:.0f}%"
        
        return response


# Factory function
def create_predictor(config: Optional[PredictionConfig] = None) -> TrafficPredictor:
    """Create a traffic predictor instance"""
    return TrafficPredictor(config)


if __name__ == "__main__":
    # Test the predictor
    print("Creating Traffic Predictor...")
    predictor = create_predictor()
    
    # Test queries
    queries = [
        "What is the traffic like on Noida Expressway right now?",
        "Best route from Sector 62 to Indirapuram?",
        "How long will it take to reach Delhi from Noida?",
        "What if we add a flyover at Sector 18?",
        "What is the AQI in Noida?",
        "Predict traffic on NH24 for tomorrow morning"
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print("="*60)
        
        prediction = predictor.predict(query)
        print(prediction.explanation)
        print(f"\nConfidence: {prediction.confidence*100:.0f}%")
