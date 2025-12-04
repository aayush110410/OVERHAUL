"""
Text Generation Engine - ChatGPT-like Response Generation
==========================================================

This module provides the actual text generation capabilities
to make Traffic God respond like a conversational AI.

NO external LLM APIs - this is YOUR own model generating text.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple, Generator
from dataclasses import dataclass
import re
from collections import deque
import json


@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    max_new_tokens: int = 512
    min_new_tokens: int = 10
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    no_repeat_ngram_size: int = 3
    do_sample: bool = True
    num_beams: int = 1  # 1 = greedy/sampling, >1 = beam search
    early_stopping: bool = True
    
    # Special tokens
    eos_token_id: int = 50256
    pad_token_id: int = 50256
    bos_token_id: int = 50256


class ConversationHistory:
    """Manages conversation context"""
    
    def __init__(self, max_turns: int = 10, max_tokens: int = 1500):
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.history: deque = deque(maxlen=max_turns * 2)  # User + Assistant per turn
        self.system_prompt: Optional[str] = None
    
    def set_system_prompt(self, prompt: str):
        """Set the system prompt for the conversation"""
        self.system_prompt = prompt
    
    def add_user_message(self, message: str):
        """Add a user message"""
        self.history.append({"role": "user", "content": message})
    
    def add_assistant_message(self, message: str):
        """Add an assistant message"""
        self.history.append({"role": "assistant", "content": message})
    
    def get_context(self) -> str:
        """Get formatted conversation context"""
        parts = []
        
        if self.system_prompt:
            parts.append(f"System: {self.system_prompt}\n")
        
        for msg in self.history:
            role = "User" if msg["role"] == "user" else "Traffic God"
            parts.append(f"{role}: {msg['content']}")
        
        return "\n".join(parts)
    
    def clear(self):
        """Clear conversation history"""
        self.history.clear()


class TextGenerator:
    """
    Core text generation engine
    
    This is what makes the model actually generate responses like ChatGPT.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        config: Optional[GenerationConfig] = None,
        device: str = "cuda"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or GenerationConfig()
        self.device = device
        
        # Move model to device
        self.model = self.model.to(device)
        self.model.eval()
        
        # Response templates for traffic domain
        self.response_templates = self._load_response_templates()
    
    def _load_response_templates(self) -> Dict[str, List[str]]:
        """Load domain-specific response templates"""
        return {
            "traffic_status": [
                "Based on current conditions, {location} is experiencing {level} traffic.",
                "The traffic at {location} is currently {level}.",
                "Right now, {location} has {level} congestion levels."
            ],
            "route_suggestion": [
                "I recommend taking {route} which should take about {time} minutes.",
                "The best route would be via {route}, estimated {time} minutes.",
                "Consider going through {route} - it's currently the fastest at {time} minutes."
            ],
            "prediction": [
                "Traffic is expected to {trend} over the next {period}.",
                "I predict {trend} traffic conditions in the coming {period}.",
                "Based on patterns, expect {trend} traffic for the next {period}."
            ],
            "aqi_report": [
                "The current AQI in {location} is {value}, which is {category}.",
                "Air quality at {location}: AQI {value} ({category}).",
                "{location} has an AQI of {value}, classified as {category}."
            ]
        }
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> str:
        """
        Generate text response from prompt
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling
            stream: Whether to yield tokens as they're generated
            
        Returns:
            Generated text response
        """
        # Use config defaults if not specified
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.temperature
        top_p = top_p or self.config.top_p
        top_k = top_k or self.config.top_k
        
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([input_ids], device=self.device)
        
        # Track generated tokens for repetition penalty
        generated_tokens = []
        past_key_values = None
        
        if stream:
            return self._generate_stream(
                input_ids, max_new_tokens, temperature, top_p, top_k, **kwargs
            )
        
        # Standard generation
        for _ in range(max_new_tokens):
            # Forward pass
            with torch.cuda.amp.autocast(enabled=True):
                outputs = self.model(
                    input_ids if past_key_values is None else input_ids[:, -1:],
                    use_cache=True,
                    past_key_values=past_key_values,
                    output_predictions=False,
                    **kwargs
                )
            
            next_token_logits = outputs["logits"][:, -1, :]
            past_key_values = outputs.get("past_key_values")
            
            # Apply repetition penalty
            if generated_tokens and self.config.repetition_penalty != 1.0:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits,
                    generated_tokens,
                    self.config.repetition_penalty
                )
            
            # Apply no-repeat ngram
            if self.config.no_repeat_ngram_size > 0:
                next_token_logits = self._apply_no_repeat_ngram(
                    next_token_logits,
                    generated_tokens,
                    self.config.no_repeat_ngram_size
                )
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply top-k
            if top_k > 0:
                next_token_logits = self._top_k_filtering(next_token_logits, top_k)
            
            # Apply top-p (nucleus sampling)
            if top_p < 1.0:
                next_token_logits = self._top_p_filtering(next_token_logits, top_p)
            
            # Sample or argmax
            if self.config.do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Track and append
            token_id = next_token.item()
            generated_tokens.append(token_id)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Check for EOS
            if token_id == self.config.eos_token_id:
                break
        
        # Decode generated tokens
        generated_text = self.tokenizer.decode(generated_tokens)
        
        return self._post_process(generated_text)
    
    def _generate_stream(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        **kwargs
    ) -> Generator[str, None, None]:
        """Stream tokens as they're generated"""
        generated_tokens = []
        past_key_values = None
        
        for _ in range(max_new_tokens):
            outputs = self.model(
                input_ids if past_key_values is None else input_ids[:, -1:],
                use_cache=True,
                past_key_values=past_key_values,
                output_predictions=False,
                **kwargs
            )
            
            next_token_logits = outputs["logits"][:, -1, :] / temperature
            past_key_values = outputs.get("past_key_values")
            
            if top_k > 0:
                next_token_logits = self._top_k_filtering(next_token_logits, top_k)
            if top_p < 1.0:
                next_token_logits = self._top_p_filtering(next_token_logits, top_p)
            
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            token_id = next_token.item()
            generated_tokens.append(token_id)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Yield the decoded token
            token_text = self.tokenizer.decode([token_id])
            yield token_text
            
            if token_id == self.config.eos_token_id:
                break
    
    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        generated_tokens: List[int],
        penalty: float
    ) -> torch.Tensor:
        """Penalize already generated tokens"""
        for token_id in set(generated_tokens):
            if logits[0, token_id] < 0:
                logits[0, token_id] *= penalty
            else:
                logits[0, token_id] /= penalty
        return logits
    
    def _apply_no_repeat_ngram(
        self,
        logits: torch.Tensor,
        generated_tokens: List[int],
        ngram_size: int
    ) -> torch.Tensor:
        """Prevent repeating n-grams"""
        if len(generated_tokens) < ngram_size:
            return logits
        
        # Get the last (ngram_size - 1) tokens
        prev_ngram = tuple(generated_tokens[-(ngram_size - 1):])
        
        # Find all n-grams in generated tokens
        for i in range(len(generated_tokens) - ngram_size + 1):
            ngram = tuple(generated_tokens[i:i + ngram_size - 1])
            if ngram == prev_ngram:
                # Ban the token that would complete the repeated n-gram
                banned_token = generated_tokens[i + ngram_size - 1]
                logits[0, banned_token] = float('-inf')
        
        return logits
    
    def _top_k_filtering(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """Keep only top-k tokens"""
        values, _ = torch.topk(logits, k)
        min_value = values[:, -1].unsqueeze(-1)
        return torch.where(logits < min_value, torch.full_like(logits, float('-inf')), logits)
    
    def _top_p_filtering(self, logits: torch.Tensor, p: float) -> torch.Tensor:
        """Nucleus (top-p) filtering"""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative prob above threshold
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False
        
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
        
        return logits
    
    def _post_process(self, text: str) -> str:
        """Clean up generated text"""
        # Remove incomplete sentences at the end
        if text and text[-1] not in '.!?':
            last_punct = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
            if last_punct > len(text) // 2:
                text = text[:last_punct + 1]
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text


class ChatTrafficGod:
    """
    ChatGPT-like interface for Traffic God
    
    This is the main class that provides conversational AI capabilities.
    """
    
    SYSTEM_PROMPT = """You are Traffic God, an advanced AI assistant specialized in traffic and urban mobility for Noida, Indirapuram, and the Delhi NCR region.

Your capabilities:
- Real-time traffic analysis and predictions
- Route planning and optimization
- Air quality impact assessment
- Infrastructure recommendations
- Traffic pattern analysis

You have deep knowledge of:
- All Noida sectors (1-168) and their traffic patterns
- Major roads: DND Flyway, Noida Expressway, NH-24, Film City Road
- Metro network: Blue Line, Aqua Line stations
- Peak hours: 8-11 AM and 5-9 PM
- Local events, schools, offices that affect traffic

Respond naturally and helpfully. Give specific, actionable advice based on time of day and current conditions."""
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: str = "cuda"
    ):
        self.generator = TextGenerator(model, tokenizer, device=device)
        self.conversation = ConversationHistory()
        self.conversation.set_system_prompt(self.SYSTEM_PROMPT)
        
        # Knowledge base for responses
        self.knowledge_base = self._load_knowledge_base()
    
    def _load_knowledge_base(self) -> Dict[str, Any]:
        """Load traffic knowledge for grounding responses"""
        from ..data.noida_traffic_data import TRAFFIC_KNOWLEDGE
        return TRAFFIC_KNOWLEDGE
    
    def chat(
        self,
        message: str,
        include_context: bool = True,
        **generation_kwargs
    ) -> str:
        """
        Process a chat message and generate response
        
        Args:
            message: User's message
            include_context: Whether to include conversation history
            
        Returns:
            Assistant's response
        """
        # Add to history
        self.conversation.add_user_message(message)
        
        # Build prompt
        if include_context:
            prompt = self.conversation.get_context() + "\nTraffic God:"
        else:
            prompt = f"User: {message}\nTraffic God:"
        
        # Generate response
        response = self.generator.generate(prompt, **generation_kwargs)
        
        # Clean and add to history
        response = self._clean_response(response)
        self.conversation.add_assistant_message(response)
        
        return response
    
    def _clean_response(self, response: str) -> str:
        """Clean up the generated response"""
        # Remove any continuation of user text
        if "User:" in response:
            response = response.split("User:")[0]
        
        # Remove role prefixes if present
        for prefix in ["Traffic God:", "Assistant:", "AI:"]:
            if response.startswith(prefix):
                response = response[len(prefix):]
        
        return response.strip()
    
    def get_traffic_info(self, location: str, time: Optional[str] = None) -> str:
        """Get traffic information for a location"""
        prompt = f"""Provide current traffic information for {location}.
Time: {time or 'now'}
Include: traffic level, estimated delays, best routes, AQI impact.

Traffic God:"""
        
        return self.generator.generate(prompt)
    
    def plan_route(
        self,
        source: str,
        destination: str,
        departure_time: Optional[str] = None
    ) -> str:
        """Plan a route between two locations"""
        prompt = f"""Plan the best route:
From: {source}
To: {destination}
Departure: {departure_time or 'now'}

Consider: current traffic, alternative routes, metro options, estimated time.

Traffic God:"""
        
        return self.generator.generate(prompt)
    
    def analyze_scenario(self, scenario: str) -> str:
        """Analyze a traffic scenario"""
        prompt = f"""Analyze this traffic scenario:
{scenario}

Provide: impact assessment, affected areas, recommendations, timeline.

Traffic God:"""
        
        return self.generator.generate(prompt)
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation.clear()
    
    def set_system_prompt(self, prompt: str):
        """Update the system prompt"""
        self.conversation.set_system_prompt(prompt)


class BeamSearchGenerator:
    """
    Beam search for higher quality generation
    
    Used when you need more coherent, less random outputs.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        num_beams: int = 4,
        device: str = "cuda"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.num_beams = num_beams
        self.device = device
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        length_penalty: float = 1.0,
        early_stopping: bool = True
    ) -> str:
        """Generate using beam search"""
        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([input_ids], device=self.device)
        
        # Initialize beams
        beams = [(input_ids, 0.0)]  # (sequence, score)
        
        for _ in range(max_new_tokens):
            all_candidates = []
            
            for seq, score in beams:
                outputs = self.model(seq, output_predictions=False)
                logits = outputs["logits"][:, -1, :]
                log_probs = F.log_softmax(logits, dim=-1)
                
                # Get top-k tokens for this beam
                top_log_probs, top_tokens = torch.topk(log_probs, self.num_beams)
                
                for i in range(self.num_beams):
                    token = top_tokens[0, i].unsqueeze(0).unsqueeze(0)
                    new_seq = torch.cat([seq, token], dim=1)
                    new_score = score + top_log_probs[0, i].item()
                    
                    # Apply length penalty
                    normalized_score = new_score / (new_seq.shape[1] ** length_penalty)
                    
                    all_candidates.append((new_seq, new_score, normalized_score))
            
            # Keep top beams
            all_candidates.sort(key=lambda x: x[2], reverse=True)
            beams = [(c[0], c[1]) for c in all_candidates[:self.num_beams]]
            
            # Check for EOS in best beam
            if early_stopping and beams[0][0][0, -1].item() == 50256:
                break
        
        # Return best sequence
        best_seq = beams[0][0][0, len(input_ids[0]):]
        return self.tokenizer.decode(best_seq.tolist())


# ============================================================================
# PRE-TRAINED RESPONSE GENERATION (For faster inference without full model)
# ============================================================================

class TemplateBasedGenerator:
    """
    Fast template-based response generation
    
    Uses the model to fill in templates rather than generating from scratch.
    Much faster but less flexible.
    """
    
    TEMPLATES = {
        "traffic_current": """The current traffic at {location} is {level}. 
Average speed is {speed} km/h with an estimated delay of {delay} minutes.
{recommendations}""",

        "route_simple": """Route from {source} to {destination}:
• Distance: {distance} km
• Estimated time: {time} minutes
• Via: {via}
• Current status: {status}

Recommendation: {recommendation}""",

        "aqi_report": """Air Quality at {location}:
• AQI: {aqi} ({category})
• PM2.5: {pm25} μg/m³
• Traffic contribution: {contribution}%

Health Advisory: {advisory}""",

        "prediction": """Traffic Prediction for {location}:

Current: {current_level}

Next 1 hour: {hour_1}
Next 3 hours: {hour_3}
Next 6 hours: {hour_6}

Best time to travel: {best_time}
Confidence: {confidence}%"""
    }
    
    def __init__(self, knowledge_base: Dict[str, Any]):
        self.knowledge = knowledge_base
    
    def generate_traffic_report(self, location: str, hour: int) -> str:
        """Generate a traffic report using templates"""
        # Get data from knowledge base
        level = self._get_traffic_level(location, hour)
        speed = self._get_average_speed(level)
        delay = self._get_delay(level)
        
        return self.TEMPLATES["traffic_current"].format(
            location=location,
            level=level,
            speed=speed,
            delay=delay,
            recommendations=self._get_recommendations(level)
        )
    
    def _get_traffic_level(self, location: str, hour: int) -> str:
        if 8 <= hour < 11 or 17 <= hour < 21:
            return "heavy"
        elif 11 <= hour < 17:
            return "moderate"
        else:
            return "light"
    
    def _get_average_speed(self, level: str) -> int:
        speeds = {"heavy": 15, "moderate": 30, "light": 50}
        return speeds.get(level, 30)
    
    def _get_delay(self, level: str) -> int:
        delays = {"heavy": 25, "moderate": 10, "light": 0}
        return delays.get(level, 10)
    
    def _get_recommendations(self, level: str) -> str:
        if level == "heavy":
            return "Consider using Metro or delaying your trip by 1-2 hours."
        elif level == "moderate":
            return "Normal traffic conditions. Use navigation for best route."
        else:
            return "Good time to travel. Roads are clear."
