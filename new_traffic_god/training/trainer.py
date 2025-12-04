"""
Training Module - Distributed Training Pipeline
================================================

Implements:
1. Pretraining - Language modeling on traffic corpus
2. Supervised fine-tuning - QA, forecasting, simulation
3. RLHF - Reinforcement Learning from Human Feedback
4. Continuous learning - Incremental updates
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
import math
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Model
    model_size: str = "base"
    
    # Data
    train_data_path: str = "data/train"
    val_data_path: str = "data/val"
    max_seq_length: int = 2048
    
    # Optimization
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Schedule
    warmup_steps: int = 1000
    max_steps: int = 100000
    eval_steps: int = 1000
    save_steps: int = 5000
    
    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "bfloat16"
    
    # Distributed
    num_workers: int = 4
    
    # Logging
    log_steps: int = 100
    output_dir: str = "outputs"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class TrafficDataset(Dataset):
    """
    Dataset for traffic model training
    
    Supports:
    - Language modeling (next token prediction)
    - Traffic prediction (regression)
    - Question answering
    - Scenario simulation
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: Any,
        max_length: int = 2048,
        task: str = "lm"  # lm, qa, prediction, simulation
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task = task
        
        # Load data
        self.samples = self._load_data()
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load training data"""
        samples = []
        
        # Check if path exists
        if not self.data_path.exists():
            # Generate synthetic data for demo
            samples = self._generate_synthetic_data()
        else:
            # Load from files
            for file_path in self.data_path.glob("*.json"):
                with open(file_path) as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        samples.extend(data)
                    else:
                        samples.append(data)
        
        return samples
    
    def _generate_synthetic_data(self) -> List[Dict[str, Any]]:
        """Generate synthetic training data for demo"""
        samples = []
        
        # Traffic queries
        query_templates = [
            "What is the traffic like on {road} at {time}?",
            "Best route from {origin} to {destination}?",
            "How long will it take to reach {destination} from {origin}?",
            "Is there congestion on {road} right now?",
            "What is the AQI in {location}?",
            "Predict traffic on {road} for {time}",
            "What if we add a flyover at {location}?",
            "How does {event} affect traffic?"
        ]
        
        roads = ["Noida Expressway", "DND Flyway", "NH24", "Greater Noida Expressway"]
        locations = ["Sector 18", "Sector 62", "Indirapuram", "Vaishali", "Film City"]
        times = ["8 AM", "9 AM", "6 PM", "7 PM", "2 PM"]
        events = ["rain", "fog", "festival", "match", "VIP movement"]
        
        for i in range(1000):
            template = np.random.choice(query_templates)
            
            sample = {
                "id": f"synthetic_{i}",
                "query": template.format(
                    road=np.random.choice(roads),
                    origin=np.random.choice(locations),
                    destination=np.random.choice(locations),
                    location=np.random.choice(locations),
                    time=np.random.choice(times),
                    event=np.random.choice(events)
                ),
                "task": self.task
            }
            
            # Add labels based on task
            if self.task == "prediction":
                sample["traffic_flow"] = np.random.randint(500, 2000)
                sample["travel_time"] = np.random.randint(10, 60)
                sample["congestion_level"] = np.random.randint(0, 5)
            elif self.task == "qa":
                sample["answer"] = f"Based on current conditions, the traffic is {'heavy' if np.random.random() > 0.5 else 'moderate'}."
            
            samples.append(sample)
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Tokenize
        text = sample.get("query", "") + " " + sample.get("answer", "")
        encoded = self.tokenizer.encode(
            text,
            max_length=self.max_length,
            padding=True,
            truncation=True
        )
        
        # Create tensors
        input_ids = torch.tensor(encoded, dtype=torch.long)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()  # For language modeling
        }
        
        # Add task-specific targets
        if self.task == "prediction":
            result["traffic_flow_target"] = torch.tensor(sample.get("traffic_flow", 0), dtype=torch.float)
            result["travel_time_target"] = torch.tensor(sample.get("travel_time", 0), dtype=torch.float)
            result["congestion_target"] = torch.tensor(sample.get("congestion_level", 0), dtype=torch.long)
        
        return result


class Trainer:
    """
    Training orchestrator
    
    Handles:
    - Single GPU and distributed training
    - Mixed precision
    - Gradient accumulation
    - Checkpointing
    - Logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: TrainingConfig
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # Move model to device
        self.model = self.model.to(config.device)
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.warmup_steps,
            T_mult=2
        )
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.use_amp and config.device == "cuda" else None
        
        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_optimizer(self) -> AdamW:
        """Create AdamW optimizer with weight decay"""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'bias' in name or 'norm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        optimizer_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ]
        
        return AdamW(optimizer_groups, lr=self.config.learning_rate)
    
    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        callbacks: Optional[List[Callable]] = None
    ):
        """
        Main training loop
        """
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers
            )
        
        # Training loop
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        logger.info(f"Starting training for {self.config.max_steps} steps")
        
        while self.global_step < self.config.max_steps:
            for batch in train_loader:
                loss = self._training_step(batch)
                total_loss += loss
                num_batches += 1
                
                # Gradient accumulation
                if num_batches % self.config.gradient_accumulation_steps == 0:
                    self._optimizer_step()
                    self.global_step += 1
                    
                    # Logging
                    if self.global_step % self.config.log_steps == 0:
                        avg_loss = total_loss / num_batches
                        logger.info(f"Step {self.global_step}: loss = {avg_loss:.4f}")
                        total_loss = 0
                        num_batches = 0
                    
                    # Evaluation
                    if val_loader and self.global_step % self.config.eval_steps == 0:
                        val_loss = self._evaluate(val_loader)
                        logger.info(f"Step {self.global_step}: val_loss = {val_loss:.4f}")
                        
                        if val_loss < self.best_loss:
                            self.best_loss = val_loss
                            self.save_checkpoint("best")
                        
                        self.model.train()
                    
                    # Save checkpoint
                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint(f"step_{self.global_step}")
                    
                    # Callbacks
                    if callbacks:
                        for callback in callbacks:
                            callback(self)
                    
                    if self.global_step >= self.config.max_steps:
                        break
        
        logger.info("Training complete!")
        self.save_checkpoint("final")
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step"""
        # Move to device
        batch = {k: v.to(self.config.device) for k, v in batch.items()}
        
        # Forward pass
        with torch.cuda.amp.autocast(enabled=self.config.use_amp):
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch.get("labels")
            )
            loss = outputs.get("loss", torch.tensor(0.0))
            
            # Add auxiliary losses
            if "traffic_flow_target" in batch:
                flow_loss = F.mse_loss(
                    outputs.get("traffic_flow", torch.zeros(1)),
                    batch["traffic_flow_target"]
                )
                loss = loss + 0.1 * flow_loss
            
            if "travel_time_target" in batch:
                time_loss = F.mse_loss(
                    outputs.get("travel_time", torch.zeros(1)),
                    batch["travel_time_target"]
                )
                loss = loss + 0.1 * time_loss
            
            if "congestion_target" in batch:
                cong_loss = F.cross_entropy(
                    outputs.get("congestion_level", torch.zeros(1, 5)),
                    batch["congestion_target"]
                )
                loss = loss + 0.1 * cong_loss
        
        # Backward pass
        if self.scaler:
            self.scaler.scale(loss / self.config.gradient_accumulation_steps).backward()
        else:
            (loss / self.config.gradient_accumulation_steps).backward()
        
        return loss.item()
    
    def _optimizer_step(self):
        """Optimizer step with gradient clipping"""
        if self.scaler:
            self.scaler.unscale_(self.optimizer)
        
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm
        )
        
        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        self.scheduler.step()
        self.optimizer.zero_grad()
    
    @torch.no_grad()
    def _evaluate(self, val_loader: DataLoader) -> float:
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for batch in val_loader:
            batch = {k: v.to(self.config.device) for k, v in batch.items()}
            
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch.get("labels")
            )
            
            total_loss += outputs.get("loss", torch.tensor(0.0)).item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint"""
        checkpoint_path = self.output_dir / f"checkpoint_{name}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), checkpoint_path / "model.pt")
        
        # Save optimizer
        torch.save(self.optimizer.state_dict(), checkpoint_path / "optimizer.pt")
        
        # Save config
        with open(checkpoint_path / "config.json", 'w') as f:
            json.dump({
                "global_step": self.global_step,
                "best_loss": self.best_loss,
                "training_config": self.config.__dict__
            }, f)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint_path = Path(path)
        
        # Load model
        self.model.load_state_dict(torch.load(checkpoint_path / "model.pt"))
        
        # Load optimizer
        self.optimizer.load_state_dict(torch.load(checkpoint_path / "optimizer.pt"))
        
        # Load config
        with open(checkpoint_path / "config.json") as f:
            config = json.load(f)
            self.global_step = config["global_step"]
            self.best_loss = config["best_loss"]
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")


class RewardModel(nn.Module):
    """
    Reward model for RLHF
    
    Predicts human preference scores for model outputs
    """
    
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        
        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Add reward head
        hidden_size = base_model.config.hidden_size
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute reward score"""
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_predictions=False
        )
        
        hidden_states = outputs["hidden_states"]
        
        # Use last token representation
        last_hidden = hidden_states[:, -1, :]
        
        reward = self.reward_head(last_hidden)
        return reward.squeeze(-1)


class RLHFTrainer:
    """
    RLHF training using PPO
    """
    
    def __init__(
        self,
        model: nn.Module,
        reward_model: RewardModel,
        tokenizer: Any,
        config: TrainingConfig
    ):
        self.model = model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.config = config
        
        # Reference model (frozen)
        self.ref_model = type(model)(model.config)
        self.ref_model.load_state_dict(model.state_dict())
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # Move to device
        self.model = self.model.to(config.device)
        self.ref_model = self.ref_model.to(config.device)
        self.reward_model = self.reward_model.to(config.device)
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate * 0.1  # Lower LR for RLHF
        )
        
        # PPO parameters
        self.kl_coef = 0.02
        self.clip_range = 0.2
    
    def compute_rewards(
        self,
        input_ids: torch.Tensor,
        response_ids: torch.Tensor
    ) -> torch.Tensor:
        """Compute rewards for generated responses"""
        # Combine input and response
        full_ids = torch.cat([input_ids, response_ids], dim=1)
        
        # Get reward
        with torch.no_grad():
            reward = self.reward_model(full_ids)
        
        return reward
    
    def ppo_step(
        self,
        prompts: List[str],
        num_generations: int = 4
    ) -> Dict[str, float]:
        """Single PPO training step"""
        # Tokenize prompts
        prompt_ids = []
        for prompt in prompts:
            ids = self.tokenizer.encode(prompt, max_length=256, truncation=True)
            prompt_ids.append(torch.tensor(ids))
        
        prompt_ids = torch.nn.utils.rnn.pad_sequence(
            prompt_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        ).to(self.config.device)
        
        # Generate responses
        self.model.eval()
        with torch.no_grad():
            response_ids = self.model.generate(
                prompt_ids,
                max_new_tokens=128,
                temperature=0.8
            )
        
        # Get old log probs
        with torch.no_grad():
            old_outputs = self.model(response_ids, output_predictions=False)
            old_logits = old_outputs["logits"]
            old_log_probs = F.log_softmax(old_logits, dim=-1)
            old_log_probs = torch.gather(
                old_log_probs[:, :-1],
                dim=-1,
                index=response_ids[:, 1:].unsqueeze(-1)
            ).squeeze(-1)
        
        # Get reference log probs
        with torch.no_grad():
            ref_outputs = self.ref_model(response_ids, output_predictions=False)
            ref_logits = ref_outputs["logits"]
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            ref_log_probs = torch.gather(
                ref_log_probs[:, :-1],
                dim=-1,
                index=response_ids[:, 1:].unsqueeze(-1)
            ).squeeze(-1)
        
        # Compute rewards
        rewards = self.compute_rewards(prompt_ids, response_ids[:, prompt_ids.size(1):])
        
        # PPO update
        self.model.train()
        
        for _ in range(4):  # PPO epochs
            # Get new log probs
            outputs = self.model(response_ids, output_predictions=False)
            logits = outputs["logits"]
            log_probs = F.log_softmax(logits, dim=-1)
            log_probs = torch.gather(
                log_probs[:, :-1],
                dim=-1,
                index=response_ids[:, 1:].unsqueeze(-1)
            ).squeeze(-1)
            
            # Compute advantages (simplified - just use rewards)
            advantages = rewards.unsqueeze(1).expand_as(log_probs)
            
            # PPO ratio
            ratio = torch.exp(log_probs - old_log_probs)
            
            # Clipped objective
            pg_loss1 = -advantages * ratio
            pg_loss2 = -advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            
            # KL penalty
            kl = (ref_log_probs - log_probs).mean()
            
            # Total loss
            loss = pg_loss + self.kl_coef * kl
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        
        return {
            "loss": loss.item(),
            "reward": rewards.mean().item(),
            "kl": kl.item()
        }


class ContinuousLearner:
    """
    Continuous learning system
    
    Handles:
    - Incremental data ingestion
    - Online fine-tuning
    - Catastrophic forgetting prevention
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: TrainingConfig
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
        # EWC (Elastic Weight Consolidation) for preventing forgetting
        self.fisher_info: Dict[str, torch.Tensor] = {}
        self.old_params: Dict[str, torch.Tensor] = {}
        self.ewc_lambda = 1000
        
        # Experience replay buffer
        self.replay_buffer: List[Dict[str, Any]] = []
        self.max_buffer_size = 10000
    
    def compute_fisher_info(self, dataloader: DataLoader):
        """Compute Fisher information for EWC"""
        self.model.train()
        
        for name, param in self.model.named_parameters():
            self.fisher_info[name] = torch.zeros_like(param)
            self.old_params[name] = param.data.clone()
        
        for batch in dataloader:
            batch = {k: v.to(self.config.device) for k, v in batch.items()}
            
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch.get("labels")
            )
            
            loss = outputs["loss"]
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.fisher_info[name] += param.grad.data.pow(2)
            
            self.model.zero_grad()
        
        # Average
        for name in self.fisher_info:
            self.fisher_info[name] /= len(dataloader)
    
    def ewc_loss(self) -> torch.Tensor:
        """Compute EWC penalty"""
        loss = 0
        
        for name, param in self.model.named_parameters():
            if name in self.fisher_info:
                loss += (
                    self.fisher_info[name] *
                    (param - self.old_params[name]).pow(2)
                ).sum()
        
        return self.ewc_lambda * loss
    
    def add_to_replay_buffer(self, sample: Dict[str, Any]):
        """Add sample to experience replay buffer"""
        if len(self.replay_buffer) >= self.max_buffer_size:
            # Remove oldest
            self.replay_buffer.pop(0)
        self.replay_buffer.append(sample)
    
    def get_replay_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """Sample from replay buffer"""
        if len(self.replay_buffer) < batch_size:
            return self.replay_buffer
        
        indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        return [self.replay_buffer[i] for i in indices]
    
    def online_update(
        self,
        new_data: List[Dict[str, Any]],
        num_steps: int = 100
    ):
        """Perform online update with new data"""
        self.model.train()
        
        optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate * 0.1)
        
        for step in range(num_steps):
            # Mix new data with replay
            batch_data = new_data[:self.config.batch_size // 2]
            replay_data = self.get_replay_batch(self.config.batch_size // 2)
            
            all_data = batch_data + replay_data
            
            # Forward pass
            for sample in all_data:
                self.add_to_replay_buffer(sample)
            
            # Training step (simplified)
            total_loss = 0
            for sample in all_data:
                ids = self.tokenizer.encode(sample.get("text", ""), max_length=512)
                input_ids = torch.tensor([ids], device=self.config.device)
                
                outputs = self.model(input_ids=input_ids, labels=input_ids)
                loss = outputs["loss"]
                
                # Add EWC loss
                loss += self.ewc_loss()
                
                total_loss += loss.item()
                loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            if (step + 1) % 10 == 0:
                logger.info(f"Online update step {step + 1}: loss = {total_loss / len(all_data):.4f}")


# Factory functions
def create_trainer(
    model: nn.Module,
    tokenizer: Any,
    config: Optional[TrainingConfig] = None
) -> Trainer:
    """Create a trainer instance"""
    if config is None:
        config = TrainingConfig()
    return Trainer(model, tokenizer, config)


def create_rlhf_trainer(
    model: nn.Module,
    tokenizer: Any,
    config: Optional[TrainingConfig] = None
) -> RLHFTrainer:
    """Create an RLHF trainer instance"""
    if config is None:
        config = TrainingConfig()
    
    reward_model = RewardModel(model)
    return RLHFTrainer(model, reward_model, tokenizer, config)
