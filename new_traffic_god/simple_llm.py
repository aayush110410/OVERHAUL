"""
Traffic God LLM - Simple Interface
==================================

Your custom trained LLM for traffic queries.
NO external APIs - 100% your own model.
"""

import torch
import json
import os
import re
from typing import Optional, Dict, Any
from datetime import datetime


class SimpleTokenizer:
    """Word-level tokenizer for Traffic God"""
    
    def __init__(self, vocab_path: Optional[str] = None):
        self.word_to_id = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        self.id_to_word = {0: "<PAD>", 1: "<UNK>", 2: "<BOS>", 3: "<EOS>"}
        self.vocab_size = 4
        
        if vocab_path and os.path.exists(vocab_path):
            self.load(vocab_path)
    
    def load(self, path: str):
        with open(path, 'r') as f:
            data = json.load(f)
        self.word_to_id = data['word_to_id']
        self.id_to_word = {int(k): v for k, v in data['id_to_word'].items()}
        self.vocab_size = data['vocab_size']
    
    def _tokenize(self, text: str):
        text = text.lower()
        return re.findall(r'\w+|[^\w\s]', text)
    
    def encode(self, text: str):
        return [self.word_to_id.get(w, 1) for w in self._tokenize(text)]
    
    def decode(self, ids):
        words = [self.id_to_word.get(i, "<UNK>") for i in ids if i != 0]
        result = []
        for i, w in enumerate(words):
            if w in ".,!?:;)" or (i > 0 and result and result[-1] and result[-1][-1] in "($"):
                result.append(w)
            else:
                result.append(" " + w if result else w)
        return "".join(result)


class TrafficGodLLM:
    """
    Your Custom Traffic LLM
    
    Usage:
        llm = TrafficGodLLM()
        response = llm.chat("What's the traffic at Sector 18?")
    """
    
    def __init__(self, model_path: str = "./new_traffic_god/checkpoints/trained_model"):
        from new_traffic_god.core.foundation_model import TrafficFoundationModel, TrafficModelConfig
        
        # Device
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        print(f"🚦 Loading Traffic God on {self.device}...")
        
        # Load tokenizer
        tokenizer_path = os.path.join(model_path, "tokenizer.json")
        self.tokenizer = SimpleTokenizer(tokenizer_path)
        
        # Load config
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            config = TrafficModelConfig(
                vocab_size=cfg['vocab_size'],
                hidden_size=cfg['hidden_size'],
                num_hidden_layers=cfg['num_hidden_layers'],
                num_attention_heads=cfg['num_attention_heads'],
                intermediate_size=cfg['intermediate_size'],
                num_experts=cfg.get('num_experts', 4),
                max_position_embeddings=cfg['max_position_embeddings']
            )
        else:
            config = TrafficModelConfig(vocab_size=self.tokenizer.vocab_size)
        
        # Load model
        self.model = TrafficFoundationModel(config)
        model_file = os.path.join(model_path, "model.pt")
        if os.path.exists(model_file):
            state_dict = torch.load(model_file, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state_dict)
            print("✅ Model loaded!")
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.history = []
        print("✅ Traffic God ready!")
    
    def chat(self, message: str, temperature: float = 0.7, max_tokens: int = 80) -> str:
        """Chat with Traffic God - improved generation with repetition penalty"""
        prompt = f"User: {message}\nTraffic God:"
        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([input_ids]).to(self.device)
        
        prompt_len = input_ids.shape[1]
        generated_tokens = []
        recent_tokens = []  # For repetition penalty
        
        with torch.no_grad():
            for step in range(max_tokens):
                if input_ids.shape[1] >= 200:
                    break
                
                outputs = self.model(input_ids, output_predictions=False)
                logits = outputs["logits"][:, -1, :] / temperature
                
                # Apply repetition penalty
                for token_id in set(recent_tokens[-20:]):
                    logits[0, token_id] /= 2.0
                
                # Top-k sampling
                k = min(50, logits.shape[-1])
                top_logits, top_indices = torch.topk(logits, k)
                probs = torch.softmax(top_logits, dim=-1)
                next_idx = torch.multinomial(probs, num_samples=1)
                next_token = top_indices.gather(-1, next_idx)
                
                token_id = next_token.item()
                generated_tokens.append(token_id)
                recent_tokens.append(token_id)
                
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                # Stop conditions
                if step > 25:
                    decoded = self.tokenizer.decode(input_ids[0, -5:].tolist()).lower()
                    if "user:" in decoded or "." in decoded[-2:]:
                        break
        
        response = self.tokenizer.decode(input_ids[0].tolist())
        response = response[len(prompt):].strip()
        
        # Clean up
        if "user:" in response.lower():
            response = response[:response.lower().find("user:")].strip()
        
        # Format nicely
        response = self._clean_response(response)
        
        self.history.append({"user": message, "assistant": response})
        return response
    
    def _clean_response(self, text: str) -> str:
        """Clean and format the generated response"""
        # Remove extra asterisks
        text = re.sub(r'\s*\*\s*', ' ', text)
        # Remove multiple spaces  
        text = re.sub(r'\s+', ' ', text)
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]
        # Ensure proper ending
        if text and text[-1] not in '.!?':
            text = text.rstrip() + '.'
        return text.strip()
    
    def get_traffic(self, location: str) -> str:
        return self.chat(f"What is the traffic like at {location}?")
    
    def plan_route(self, source: str, destination: str) -> str:
        return self.chat(f"How do I go from {source} to {destination}?")
    
    def get_aqi(self, location: str) -> str:
        return self.chat(f"What's the AQI in {location}?")
    
    def clear_history(self):
        self.history = []


# Singleton
_instance = None

def get_llm() -> TrafficGodLLM:
    global _instance
    if _instance is None:
        _instance = TrafficGodLLM()
    return _instance


def chat(message: str) -> str:
    """Quick chat function"""
    return get_llm().chat(message)


# ============================================================================
# INTERACTIVE CLI
# ============================================================================

def interactive_chat():
    """Run interactive chat session"""
    print("\n" + "="*60)
    print("🚦 TRAFFIC GOD - Interactive Chat")
    print("="*60)
    print("Your custom LLM for Noida/NCR traffic")
    print("Commands: /quit, /clear, /help")
    print("="*60 + "\n")
    
    llm = TrafficGodLLM()
    
    while True:
        try:
            user_input = input("🚗 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['/quit', '/exit', 'quit', 'exit']:
                print("\n👋 Goodbye! Drive safe!")
                break
            
            if user_input.lower() == '/clear':
                llm.clear_history()
                print("✅ History cleared")
                continue
            
            if user_input.lower() == '/help':
                print("""
Commands:
  /quit   - Exit chat
  /clear  - Clear history
  /help   - Show this help

Ask about:
  - Traffic at any Noida location
  - Routes between places
  - AQI and air quality
  - Peak hours and best times to travel
""")
                continue
            
            print("🚦 Traffic God: ", end="", flush=True)
            response = llm.chat(user_input)
            print(response)
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break


if __name__ == "__main__":
    interactive_chat()
