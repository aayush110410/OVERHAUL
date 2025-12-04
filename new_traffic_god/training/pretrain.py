"""
Pre-Training Script for Traffic God LLM
========================================

This script trains YOUR OWN custom LLM from scratch.
No external APIs - pure PyTorch training.

Usage:
    python -m new_traffic_god.training.pretrain --config configs/config.yaml
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from new_traffic_god.core.foundation_model import TrafficFoundationModel, TrafficModelConfig
from new_traffic_god.core.tokenizer import TrafficTokenizer


# ============================================================================
# DATASET
# ============================================================================

class TrafficTextDataset(Dataset):
    """Dataset for traffic text data"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: TrafficTokenizer,
        max_length: int = 512
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        # Load data
        if os.path.isfile(data_path):
            with open(data_path, 'r') as f:
                data = json.load(f)
        else:
            # Generate synthetic data if file doesn't exist
            from new_traffic_god.data.training_data_generator import TrafficDataGenerator
            generator = TrafficDataGenerator()
            data = generator.generate_dataset(10000)
        
        # Process samples
        for item in data:
            # Format as instruction-response pair
            text = f"User: {item['instruction']}\nAssistant: {item['response']}"
            self.samples.append(text)
        
        print(f"Loaded {len(self.samples)} training samples")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.samples[idx]
        
        # Tokenize
        token_ids = self.tokenizer.encode(text)
        
        # Pad or truncate
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            token_ids = token_ids + [self.tokenizer.pad_token_id] * (self.max_length - len(token_ids))
        
        input_ids = torch.tensor(token_ids[:-1], dtype=torch.long)
        labels = torch.tensor(token_ids[1:], dtype=torch.long)
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).float()
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask
        }


# ============================================================================
# TRAINING LOOP
# ============================================================================

class PreTrainer:
    """
    Pre-trains the Traffic Foundation Model
    
    This is where the actual LLM training happens.
    """
    
    def __init__(
        self,
        model: TrafficFoundationModel,
        tokenizer: TrafficTokenizer,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        config: Optional[Dict] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        # Default config
        self.config = config or {
            "batch_size": 8,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "epochs": 10,
            "warmup_steps": 1000,
            "gradient_accumulation": 4,
            "max_grad_norm": 1.0,
            "fp16": True,
            "save_steps": 1000,
            "eval_steps": 500,
            "output_dir": "./checkpoints"
        }
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"]
        )
        
        # Setup scheduler
        total_steps = len(train_dataset) // self.config["batch_size"] * self.config["epochs"]
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps)
        
        # Mixed precision
        self.scaler = GradScaler() if self.config["fp16"] else None
        
        # Setup logging
        self.logger = logging.getLogger("PreTrainer")
        logging.basicConfig(level=logging.INFO)
        
        # Metrics
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting pre-training...")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        os.makedirs(self.config["output_dir"], exist_ok=True)
        
        for epoch in range(self.config["epochs"]):
            self.model.train()
            epoch_loss = 0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                loss = self._training_step(batch)
                epoch_loss += loss
                num_batches += 1
                
                # Logging
                if self.global_step % 100 == 0:
                    avg_loss = epoch_loss / num_batches
                    self.logger.info(
                        f"Epoch {epoch+1} | Step {self.global_step} | "
                        f"Loss: {loss:.4f} | Avg Loss: {avg_loss:.4f}"
                    )
                
                # Evaluation
                if self.val_dataset and self.global_step % self.config["eval_steps"] == 0:
                    val_loss = self._evaluate()
                    self.logger.info(f"Validation Loss: {val_loss:.4f}")
                    
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self._save_checkpoint("best")
                
                # Save checkpoint
                if self.global_step % self.config["save_steps"] == 0:
                    self._save_checkpoint(f"step_{self.global_step}")
                
                self.global_step += 1
            
            # End of epoch
            avg_epoch_loss = epoch_loss / num_batches
            self.logger.info(f"Epoch {epoch+1} completed | Avg Loss: {avg_epoch_loss:.4f}")
            self._save_checkpoint(f"epoch_{epoch+1}")
        
        self.logger.info("Pre-training completed!")
        self._save_checkpoint("final")
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step"""
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        
        if self.config["fp16"]:
            with autocast():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    output_predictions=False
                )
                loss = outputs["loss"]
                loss = loss / self.config["gradient_accumulation"]
            
            self.scaler.scale(loss).backward()
            
            if (self.global_step + 1) % self.config["gradient_accumulation"] == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config["max_grad_norm"]
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_predictions=False
            )
            loss = outputs["loss"]
            loss = loss / self.config["gradient_accumulation"]
            loss.backward()
            
            if (self.global_step + 1) % self.config["gradient_accumulation"] == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config["max_grad_norm"]
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
        
        return loss.item() * self.config["gradient_accumulation"]
    
    @torch.no_grad()
    def _evaluate(self) -> float:
        """Evaluate on validation set"""
        self.model.eval()
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False
        )
        
        total_loss = 0
        num_batches = 0
        
        for batch in val_loader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_predictions=False
            )
            
            total_loss += outputs["loss"].item()
            num_batches += 1
        
        self.model.train()
        return total_loss / num_batches
    
    def _save_checkpoint(self, name: str):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(self.config["output_dir"], f"checkpoint_{name}")
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save model
        torch.save(
            self.model.state_dict(),
            os.path.join(checkpoint_path, "model.pt")
        )
        
        # Save tokenizer
        self.tokenizer.save(os.path.join(checkpoint_path, "tokenizer.json"))
        
        # Save training state
        training_state = {
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "config": self.config
        }
        with open(os.path.join(checkpoint_path, "training_state.json"), 'w') as f:
            json.dump(training_state, f)
        
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")


# ============================================================================
# FINE-TUNING (Supervised Fine-Tuning)
# ============================================================================

class SupervisedFineTuner(PreTrainer):
    """
    Supervised Fine-Tuning (SFT) for instruction following
    
    Takes a pre-trained model and fine-tunes it on
    instruction-response pairs.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Lower learning rate for fine-tuning
        self.config["learning_rate"] = 1e-5
        
        # Rebuild optimizer with new LR
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"]
        )
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Training step with instruction masking"""
        # Only compute loss on the response part
        # (This is a simplified version - full implementation would
        # mask the instruction tokens in the loss computation)
        return super()._training_step(batch)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Pre-train Traffic God LLM")
    parser.add_argument("--data_dir", type=str, default="./new_traffic_god/training_data")
    parser.add_argument("--output_dir", type=str, default="./new_traffic_god/checkpoints")
    parser.add_argument("--model_size", type=str, default="base", choices=["tiny", "base", "large"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--generate_data", action="store_true", help="Generate training data first")
    args = parser.parse_args()
    
    # Generate training data if needed
    if args.generate_data or not os.path.exists(f"{args.data_dir}/train.json"):
        print("Generating training data...")
        from new_traffic_god.data.training_data_generator import TrafficDataGenerator
        generator = TrafficDataGenerator()
        generator.generate_and_save(args.data_dir)
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = TrafficTokenizer()
    
    # Initialize model
    print(f"Creating {args.model_size} model...")
    config_map = {
        "tiny": TrafficModelConfig(
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=1024,
            num_experts=4
        ),
        "base": TrafficModelConfig(
            hidden_size=512,
            num_hidden_layers=8,
            num_attention_heads=8,
            intermediate_size=2048,
            num_experts=8
        ),
        "large": TrafficModelConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            num_experts=8
        )
    }
    
    model_config = config_map[args.model_size]
    model = TrafficFoundationModel(model_config)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = TrafficTextDataset(
        f"{args.data_dir}/train.json",
        tokenizer,
        max_length=512
    )
    
    val_dataset = None
    if os.path.exists(f"{args.data_dir}/val.json"):
        val_dataset = TrafficTextDataset(
            f"{args.data_dir}/val.json",
            tokenizer,
            max_length=512
        )
    
    # Training config
    training_config = {
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": 0.01,
        "epochs": args.epochs,
        "warmup_steps": 500,
        "gradient_accumulation": 4,
        "max_grad_norm": 1.0,
        "fp16": torch.cuda.is_available(),
        "save_steps": 1000,
        "eval_steps": 500,
        "output_dir": args.output_dir
    }
    
    # Train!
    trainer = PreTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=training_config
    )
    
    trainer.train()
    
    print("\n" + "="*50)
    print("Training completed!")
    print(f"Model saved to: {args.output_dir}")
    print("="*50)


if __name__ == "__main__":
    main()
