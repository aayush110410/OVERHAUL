#!/usr/bin/env python3
"""
Traffic God - Interactive Chat Interface
=========================================

Run this to chat with YOUR custom traffic LLM.
No external APIs - this is your own trained model.

Usage:
    python -m new_traffic_god.chat
    python -m new_traffic_god.chat --model_path ./checkpoints/final
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from new_traffic_god.core.foundation_model import TrafficFoundationModel, TrafficModelConfig, create_traffic_model
from new_traffic_god.core.tokenizer import TrafficTokenizer
from new_traffic_god.inference.generation import TextGenerator, ChatTrafficGod, GenerationConfig


def print_banner():
    """Print welcome banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   🚦 TRAFFIC GOD - Your Custom Traffic LLM 🚦                   ║
║                                                                  ║
║   Specialized for Noida, Indirapuram & NCR Region               ║
║   100% Custom Model - No External APIs                          ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

Commands:
  /clear  - Clear conversation history
  /help   - Show this help
  /route  - Plan a route (e.g., /route Sector 18 to Indirapuram)
  /traffic - Get traffic status (e.g., /traffic Sector 62)
  /quit   - Exit chat

Type your question or command:
"""
    print(banner)


def load_model(model_path: str = None, model_size: str = "tiny"):
    """Load model from checkpoint or create new"""
    print("Loading Traffic God model...")
    
    # Initialize tokenizer
    tokenizer = TrafficTokenizer()
    
    if model_path and os.path.exists(model_path):
        # Load from checkpoint
        print(f"Loading from checkpoint: {model_path}")
        
        config = TrafficModelConfig()  # Load config from checkpoint ideally
        model = TrafficFoundationModel(config)
        
        model_file = os.path.join(model_path, "model.pt")
        if os.path.exists(model_file):
            state_dict = torch.load(model_file, map_location="cpu")
            model.load_state_dict(state_dict)
            print("Model loaded successfully!")
    else:
        # Create new model (for testing/demo)
        print(f"Creating new {model_size} model...")
        model = create_traffic_model(size=model_size)
        print("Note: Using untrained model. Train first for better responses!")
    
    return model, tokenizer


def handle_command(command: str, chat: ChatTrafficGod) -> str:
    """Handle special commands"""
    parts = command.split(maxsplit=1)
    cmd = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""
    
    if cmd == "/clear":
        chat.clear_history()
        return "Conversation history cleared."
    
    elif cmd == "/help":
        return """
Available commands:
  /clear           - Clear conversation history
  /route <from> to <to> - Plan a route
  /traffic <location>   - Get traffic status
  /aqi <location>       - Get air quality
  /quit            - Exit chat
        """
    
    elif cmd == "/route":
        if " to " in args:
            source, dest = args.split(" to ", 1)
            return chat.plan_route(source.strip(), dest.strip())
        else:
            return "Usage: /route <source> to <destination>"
    
    elif cmd == "/traffic":
        if args:
            return chat.get_traffic_info(args.strip())
        else:
            return "Usage: /traffic <location>"
    
    elif cmd == "/aqi":
        if args:
            return chat.generator.generate(
                f"What is the current AQI at {args}? Provide health advisory."
            )
        else:
            return "Usage: /aqi <location>"
    
    return None  # Not a command


def main():
    parser = argparse.ArgumentParser(description="Chat with Traffic God")
    parser.add_argument("--model_path", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--model_size", type=str, default="tiny", choices=["tiny", "base", "large"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu", "mps"])
    args = parser.parse_args()
    
    # Detect device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load model
    model, tokenizer = load_model(args.model_path, args.model_size)
    model = model.to(device)
    model.eval()
    
    # Create chat interface
    chat = ChatTrafficGod(model, tokenizer, device=device)
    
    # Print banner
    print_banner()
    
    # Chat loop
    while True:
        try:
            user_input = input("\n🚗 You: ").strip()
            
            if not user_input:
                continue
            
            # Check for quit
            if user_input.lower() in ["/quit", "/exit", "quit", "exit"]:
                print("\nGoodbye! Drive safe! 🚦")
                break
            
            # Check for commands
            if user_input.startswith("/"):
                response = handle_command(user_input, chat)
                if response:
                    print(f"\n🚦 Traffic God: {response}")
                    continue
            
            # Regular chat
            print("\n🚦 Traffic God: ", end="", flush=True)
            
            # Generate response
            response = chat.chat(user_input)
            print(response)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! Drive safe! 🚦")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue


# ============================================================================
# QUICK TEST MODE
# ============================================================================

def test_generation():
    """Quick test of text generation"""
    print("Testing text generation...")
    
    # Create small model
    config = TrafficModelConfig(
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=512,
        num_experts=2,
        vocab_size=5000,
        max_position_embeddings=256
    )
    
    model = TrafficFoundationModel(config)
    tokenizer = TrafficTokenizer()
    
    # Test input
    test_prompts = [
        "What is the traffic like at Sector 18?",
        "How do I go from Noida to Delhi?",
        "What's the AQI today?",
    ]
    
    gen_config = GenerationConfig(
        max_new_tokens=50,
        temperature=0.8,
        top_k=20
    )
    
    generator = TextGenerator(model, tokenizer, gen_config, device="cpu")
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        
        # Encode and generate
        full_prompt = f"User: {prompt}\nTraffic God:"
        
        # Simple forward pass test
        input_ids = tokenizer.encode(full_prompt)
        input_ids = torch.tensor([input_ids[:100]])  # Limit length
        
        with torch.no_grad():
            outputs = model(input_ids, output_predictions=False)
            logits = outputs["logits"]
            
            # Sample next token
            next_token_logits = logits[0, -1, :]
            probs = torch.softmax(next_token_logits / 0.8, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            print(f"Next token ID: {next_token.item()}")
            print(f"Logits shape: {logits.shape}")
    
    print("\n✅ Generation test passed!")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_generation()
    else:
        main()
