"""
Traffic God LLM - Integrated Interface
======================================

This module integrates the trained Traffic God LLM
into the OVERHAUL application.

NO EXTERNAL APIs - This is YOUR custom trained model.
"""

import torch
import os
from typing import Optional, Dict, Any
from datetime import datetime


class TrafficGodLLM:
    """
    Your Custom Traffic LLM - ChatGPT-like for Traffic
    
    Usage:
        llm = TrafficGodLLM()
        response = llm.chat("What's the traffic at Sector 18?")
        print(response)
    """
    
    def __init__(
        self,
        model_path: str = "./new_traffic_god/checkpoints/trained_model",
        device: str = "auto"
    ):
        """
        Initialize the Traffic God LLM
        
        Args:
            model_path: Path to trained model checkpoint
            device: "auto", "cuda", "mps", or "cpu"
        """
        # Import here to avoid circular imports
        from new_traffic_god.core.foundation_model import TrafficFoundationModel, TrafficModelConfig
        from new_traffic_god.core.tokenizer import TrafficTokenizer
        
        # Detect device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        print(f"🚦 Loading Traffic God LLM on {self.device}...")
        
        # Load tokenizer
        self.tokenizer = TrafficTokenizer()
        
        # Load model
        config_path = os.path.join(model_path, "config.pt")
        model_file = os.path.join(model_path, "model.pt")
        
        # Default config (always use this for consistency)
        self.config = TrafficModelConfig(
            vocab_size=5000,
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=512,
            num_experts=2,
            max_position_embeddings=256
        )
        
        self.model = TrafficFoundationModel(self.config)
        
        if os.path.exists(model_file):
            state_dict = torch.load(model_file, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state_dict)
            print("✅ Loaded trained model!")
        else:
            print("⚠️ No trained model found, using random weights")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Conversation history
        self.history = []
        
        print("✅ Traffic God LLM ready!")
    
    def chat(
        self,
        message: str,
        temperature: float = 0.7,
        max_tokens: int = 150,
        include_history: bool = True
    ) -> str:
        """
        Chat with Traffic God
        
        Args:
            message: User's message
            temperature: Generation temperature (0.1-1.0)
            max_tokens: Maximum tokens to generate
            include_history: Include conversation history
        
        Returns:
            Generated response
        """
        # Build prompt
        if include_history and self.history:
            context = "\n".join([
                f"User: {h['user']}\nTraffic God: {h['assistant']}"
                for h in self.history[-3:]  # Last 3 turns
            ])
            prompt = f"{context}\nUser: {message}\nTraffic God:"
        else:
            prompt = f"User: {message}\nTraffic God:"
        
        # Tokenize
        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([input_ids[:150]]).to(self.device)
        
        # Generate
        with torch.no_grad():
            for _ in range(max_tokens):
                outputs = self.model(input_ids, output_predictions=False)
                logits = outputs["logits"][:, -1, :] / temperature
                
                # Top-k sampling
                top_k = 40
                top_logits, top_indices = torch.topk(logits, top_k)
                probs = torch.softmax(top_logits, dim=-1)
                next_token_idx = torch.multinomial(probs, num_samples=1)
                next_token = top_indices.gather(-1, next_token_idx)
                
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                # Stop conditions
                if input_ids.shape[1] > 250:
                    break
                # Check for end patterns
                decoded = self.tokenizer.decode(input_ids[0, -10:].tolist())
                if "User:" in decoded or "\n\n\n" in decoded:
                    break
        
        # Decode response
        full_response = self.tokenizer.decode(input_ids[0].tolist())
        
        # Extract just the response
        response = full_response[len(prompt):].strip()
        
        # Clean up
        if "User:" in response:
            response = response.split("User:")[0].strip()
        
        # Add to history
        self.history.append({
            "user": message,
            "assistant": response
        })
        
        return response
    
    def get_traffic(self, location: str) -> Dict[str, Any]:
        """Get traffic information for a location"""
        response = self.chat(f"What is the traffic like at {location}?")
        
        return {
            "location": location,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
    
    def plan_route(self, source: str, destination: str) -> Dict[str, Any]:
        """Plan a route between two locations"""
        response = self.chat(f"Best route from {source} to {destination}")
        
        return {
            "source": source,
            "destination": destination,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_aqi(self, location: str) -> Dict[str, Any]:
        """Get AQI information for a location"""
        response = self.chat(f"What's the AQI at {location}?")
        
        return {
            "location": location,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
    
    def analyze_scenario(self, scenario: str) -> Dict[str, Any]:
        """Analyze a traffic scenario"""
        response = self.chat(f"Analyze this scenario: {scenario}")
        
        return {
            "scenario": scenario,
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
    
    def clear_history(self):
        """Clear conversation history"""
        self.history = []
    
    def get_predictions(self, input_text: str) -> Dict[str, Any]:
        """Get traffic predictions from the model"""
        input_ids = self.tokenizer.encode(input_text)
        input_ids = torch.tensor([input_ids[:100]]).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids, output_predictions=True)
        
        return {
            "traffic_flow": outputs["traffic_flow"].item(),
            "travel_time": outputs["travel_time"].item(),
            "congestion_probs": torch.softmax(outputs["congestion_level"], dim=-1).tolist()[0],
            "aqi_prediction": outputs["aqi_prediction"].item(),
            "confidence": 1.0 / (1.0 + outputs["prediction_uncertainty"].item())
        }


# Global instance for easy access
_llm_instance: Optional[TrafficGodLLM] = None


def get_traffic_god() -> TrafficGodLLM:
    """Get or create Traffic God LLM instance"""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = TrafficGodLLM()
    return _llm_instance


def chat(message: str) -> str:
    """Quick chat function"""
    return get_traffic_god().chat(message)


# ============================================================================
# FASTAPI INTEGRATION
# ============================================================================

def create_llm_routes():
    """Create FastAPI routes for the LLM"""
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel
    
    router = APIRouter(prefix="/llm", tags=["Traffic God LLM"])
    
    class ChatRequest(BaseModel):
        message: str
        temperature: float = 0.7
        max_tokens: int = 150
    
    class ChatResponse(BaseModel):
        response: str
        timestamp: str
    
    @router.post("/chat", response_model=ChatResponse)
    async def llm_chat(request: ChatRequest):
        """Chat with Traffic God LLM"""
        try:
            llm = get_traffic_god()
            response = llm.chat(
                request.message,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            return ChatResponse(
                response=response,
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/traffic/{location}")
    async def get_traffic(location: str):
        """Get traffic for location"""
        llm = get_traffic_god()
        return llm.get_traffic(location)
    
    @router.get("/route")
    async def plan_route(source: str, destination: str):
        """Plan a route"""
        llm = get_traffic_god()
        return llm.plan_route(source, destination)
    
    @router.get("/aqi/{location}")
    async def get_aqi(location: str):
        """Get AQI for location"""
        llm = get_traffic_god()
        return llm.get_aqi(location)
    
    @router.post("/clear")
    async def clear_history():
        """Clear conversation history"""
        llm = get_traffic_god()
        llm.clear_history()
        return {"status": "cleared"}
    
    return router
