"""
Core Transformer Architecture - Built from Scratch
===================================================

This implements the full transformer architecture with:
- Multi-Head Self-Attention
- Feed-Forward Networks
- Layer Normalization
- Positional Encodings (including temporal)
- Traffic-specific embeddings

Designed for traffic understanding, prediction, and simulation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass


@dataclass
class TrafficModelConfig:
    """Configuration for the Traffic Foundation Model"""
    vocab_size: int = 50257  # GPT-2 style vocab
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    max_position_embeddings: int = 2048
    layer_norm_eps: float = 1e-6
    
    # Traffic-specific
    num_time_slots: int = 288  # 5-minute intervals in a day
    num_locations: int = 1000  # Grid cells for Noida/NCR
    num_road_types: int = 10   # Highway, arterial, local, etc.
    num_weather_states: int = 8
    
    # MoE (Mixture of Experts)
    num_experts: int = 8
    num_experts_per_token: int = 2
    expert_capacity_factor: float = 1.25
    
    # Prediction heads
    num_forecast_horizons: int = 12  # 5min to 1hr predictions
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) - Better than absolute positional embeddings
    Allows the model to understand relative positions naturally
    """
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Build cache
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
    
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input"""
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to queries and keys"""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention with Rotary Embeddings
    
    Core attention mechanism:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    """
    def __init__(self, config: TrafficModelConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim ** -0.5
        
        # QKV projections
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        # Rotary embeddings
        self.rotary_emb = RotaryPositionalEmbedding(self.head_dim)
        
        # Dropout
        self.dropout = nn.Dropout(config.attention_dropout_prob)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape to (batch, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings
        cos, sin = self.rotary_emb(q, seq_len)
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Handle KV cache for efficient inference
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        
        if use_cache:
            present_key_value = (k, v)
        else:
            present_key_value = None
        
        # Attention scores: QK^T / sqrt(d_k)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask (for autoregressive generation)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.dropout(attn_weights)
        
        # Output: attention_weights @ V
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        
        # Output projection
        attn_output = self.o_proj(attn_output)
        
        return attn_output, present_key_value


class SwiGLU(nn.Module):
    """
    SwiGLU Activation - Better than GELU for transformers
    SwiGLU(x) = (x * W1) * σ(x * W_gate)
    """
    def __init__(self, config: TrafficModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ExpertLayer(nn.Module):
    """Single Expert in Mixture-of-Experts"""
    def __init__(self, config: TrafficModelConfig):
        super().__init__()
        self.ffn = SwiGLU(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts Layer
    
    Routes each token to top-k experts for processing.
    Allows massive parameter scaling with constant compute.
    """
    def __init__(self, config: TrafficModelConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_token = config.num_experts_per_token
        self.capacity_factor = config.expert_capacity_factor
        
        # Expert networks
        self.experts = nn.ModuleList([
            ExpertLayer(config) for _ in range(config.num_experts)
        ])
        
        # Router (gating network)
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Compute router logits
        router_logits = self.router(hidden_states)  # (batch, seq, num_experts)
        
        # Get top-k experts per token
        router_probs = F.softmax(router_logits, dim=-1)
        expert_weights, expert_indices = torch.topk(router_probs, self.num_experts_per_token, dim=-1)
        
        # Normalize weights
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
        
        # Compute load balancing loss (for training)
        load_balance_loss = self._compute_load_balance_loss(router_probs, expert_indices)
        
        # Route and process
        output = torch.zeros_like(hidden_states)
        
        for i, expert in enumerate(self.experts):
            # Find tokens routed to this expert
            expert_mask = (expert_indices == i).any(dim=-1)
            if expert_mask.any():
                expert_input = hidden_states[expert_mask]
                expert_output = expert(expert_input)
                
                # Weight by router probability
                weight_mask = (expert_indices == i).float()
                weights = (expert_weights * weight_mask).sum(dim=-1, keepdim=True)
                output[expert_mask] += weights[expert_mask] * expert_output
        
        return output, load_balance_loss
    
    def _compute_load_balance_loss(self, router_probs: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
        """Auxiliary loss to encourage balanced expert utilization"""
        num_tokens = router_probs.shape[0] * router_probs.shape[1]
        
        # Fraction of tokens dispatched to each expert
        expert_usage = torch.zeros(self.num_experts, device=router_probs.device)
        for i in range(self.num_experts):
            expert_usage[i] = (expert_indices == i).float().sum() / num_tokens
        
        # Mean routing probability per expert
        mean_routing_prob = router_probs.mean(dim=(0, 1))
        
        # Load balance loss
        loss = (expert_usage * mean_routing_prob).sum() * self.num_experts
        return loss


class TransformerBlock(nn.Module):
    """
    Single Transformer Block with:
    - Multi-Head Self-Attention
    - Mixture of Experts OR Standard FFN
    - Layer Normalization (RMSNorm for efficiency)
    - Residual Connections
    """
    def __init__(self, config: TrafficModelConfig, use_moe: bool = False):
        super().__init__()
        self.config = config
        self.use_moe = use_moe
        
        # Pre-normalization (RMSNorm)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Self-attention
        self.self_attn = MultiHeadSelfAttention(config)
        
        # FFN or MoE
        if use_moe:
            self.ffn = MixtureOfExperts(config)
        else:
            self.ffn = SwiGLU(config)
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]], Optional[torch.Tensor]]:
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, present_key_value = self.self_attn(
            hidden_states, 
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_value=past_key_value
        )
        hidden_states = residual + self.dropout(attn_output)
        
        # FFN/MoE with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        aux_loss = None
        if self.use_moe:
            hidden_states, aux_loss = self.ffn(hidden_states)
        else:
            hidden_states = self.ffn(hidden_states)
        
        hidden_states = residual + self.dropout(hidden_states)
        
        return hidden_states, present_key_value, aux_loss


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization - More efficient than LayerNorm"""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS = sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class TrafficEmbeddings(nn.Module):
    """
    Traffic-Specific Embeddings
    
    Combines:
    - Token embeddings (words/subwords)
    - Temporal embeddings (time of day, day of week)
    - Spatial embeddings (location grid)
    - Road type embeddings
    - Weather embeddings
    """
    def __init__(self, config: TrafficModelConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Temporal embeddings
        self.time_slot_embeddings = nn.Embedding(config.num_time_slots, config.hidden_size // 4)
        self.day_of_week_embeddings = nn.Embedding(7, config.hidden_size // 4)
        self.month_embeddings = nn.Embedding(12, config.hidden_size // 4)
        self.is_holiday_embeddings = nn.Embedding(2, config.hidden_size // 4)
        
        # Spatial embeddings (for location-aware processing)
        self.location_embeddings = nn.Embedding(config.num_locations, config.hidden_size // 2)
        
        # Contextual embeddings
        self.road_type_embeddings = nn.Embedding(config.num_road_types, config.hidden_size // 4)
        self.weather_embeddings = nn.Embedding(config.num_weather_states, config.hidden_size // 4)
        
        # Projection to combine all embeddings
        self.projection = nn.Linear(
            config.hidden_size + config.hidden_size + config.hidden_size // 2 + config.hidden_size // 2,
            config.hidden_size
        )
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Layer norm
        self.layer_norm = RMSNorm(config.hidden_size)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        time_slots: Optional[torch.Tensor] = None,
        day_of_week: Optional[torch.Tensor] = None,
        month: Optional[torch.Tensor] = None,
        is_holiday: Optional[torch.Tensor] = None,
        location_ids: Optional[torch.Tensor] = None,
        road_types: Optional[torch.Tensor] = None,
        weather_states: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Word embeddings
        word_embeds = self.word_embeddings(input_ids)
        
        # Default temporal context if not provided
        if time_slots is None:
            time_slots = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        if day_of_week is None:
            day_of_week = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        if month is None:
            month = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        if is_holiday is None:
            is_holiday = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        
        # Temporal embeddings (broadcast across sequence)
        time_embeds = torch.cat([
            self.time_slot_embeddings(time_slots),
            self.day_of_week_embeddings(day_of_week),
            self.month_embeddings(month),
            self.is_holiday_embeddings(is_holiday)
        ], dim=-1).expand(-1, seq_len, -1)
        
        # Spatial embeddings
        if location_ids is None:
            location_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        loc_embeds = self.location_embeddings(location_ids).expand(-1, seq_len, -1)
        
        # Context embeddings
        if road_types is None:
            road_types = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        if weather_states is None:
            weather_states = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        
        context_embeds = torch.cat([
            self.road_type_embeddings(road_types),
            self.weather_embeddings(weather_states)
        ], dim=-1).expand(-1, seq_len, -1)
        
        # Combine all embeddings
        combined = torch.cat([word_embeds, time_embeds, loc_embeds, context_embeds], dim=-1)
        
        # Project to hidden size
        embeddings = self.projection(combined)
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class TrafficFoundationModel(nn.Module):
    """
    Traffic Foundation Model - The Core Intelligence
    
    A transformer-based foundation model designed specifically for:
    1. Understanding natural language queries about traffic
    2. Predicting traffic conditions across time horizons
    3. Simulating traffic scenarios
    4. Optimizing routes and infrastructure
    
    Architecture:
    - Traffic-aware embeddings
    - Transformer blocks with MoE
    - Multiple prediction heads
    """
    def __init__(self, config: TrafficModelConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.embeddings = TrafficEmbeddings(config)
        
        # Transformer blocks (some with MoE for scale)
        self.layers = nn.ModuleList([
            TransformerBlock(
                config,
                use_moe=(i % 2 == 1)  # Every other layer uses MoE
            )
            for i in range(config.num_hidden_layers)
        ])
        
        # Final normalization
        self.final_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Output heads
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Traffic-specific prediction heads
        self.traffic_flow_head = nn.Linear(config.hidden_size, 1)  # Vehicles/hour
        self.travel_time_head = nn.Linear(config.hidden_size, 1)   # Minutes
        self.congestion_head = nn.Linear(config.hidden_size, 5)    # 5 congestion levels
        self.aqi_head = nn.Linear(config.hidden_size, 1)           # AQI prediction
        
        # Multi-horizon forecasting head
        self.forecast_head = nn.Linear(
            config.hidden_size,
            config.num_forecast_horizons * 4  # 4 metrics per horizon
        )
        
        # Confidence/uncertainty head
        self.uncertainty_head = nn.Linear(config.hidden_size, 2)  # mean, log_var
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Tie embeddings
        self.lm_head.weight = self.embeddings.word_embeddings.weight
    
    def _init_weights(self, module: nn.Module):
        """Initialize weights with scaled normal distribution"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        time_slots: Optional[torch.Tensor] = None,
        day_of_week: Optional[torch.Tensor] = None,
        month: Optional[torch.Tensor] = None,
        is_holiday: Optional[torch.Tensor] = None,
        location_ids: Optional[torch.Tensor] = None,
        road_types: Optional[torch.Tensor] = None,
        weather_states: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        output_predictions: bool = True
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create causal attention mask
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=device)
        
        # Expand mask for attention: (batch, 1, seq, seq)
        causal_mask = self._create_causal_mask(seq_len, device)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = attention_mask * causal_mask
        attention_mask = (1.0 - attention_mask) * torch.finfo(torch.float32).min
        
        # Embeddings
        hidden_states = self.embeddings(
            input_ids,
            time_slots=time_slots,
            day_of_week=day_of_week,
            month=month,
            is_holiday=is_holiday,
            location_ids=location_ids,
            road_types=road_types,
            weather_states=weather_states
        )
        
        # Transformer layers
        all_aux_losses = []
        present_key_values = [] if use_cache else None
        
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            hidden_states, present_kv, aux_loss = layer(
                hidden_states,
                attention_mask=attention_mask,
                use_cache=use_cache,
                past_key_value=past_kv
            )
            if use_cache:
                present_key_values.append(present_kv)
            if aux_loss is not None:
                all_aux_losses.append(aux_loss)
        
        # Final norm
        hidden_states = self.final_norm(hidden_states)
        
        # Outputs
        outputs = {
            "hidden_states": hidden_states,
            "past_key_values": present_key_values
        }
        
        # Language model logits
        lm_logits = self.lm_head(hidden_states)
        outputs["logits"] = lm_logits
        
        # Compute loss if labels provided
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
            
            # Add MoE auxiliary loss
            if all_aux_losses:
                aux_loss = torch.stack(all_aux_losses).mean()
                loss = loss + 0.01 * aux_loss
            
            outputs["loss"] = loss
        
        # Traffic predictions (from last hidden state)
        if output_predictions:
            last_hidden = hidden_states[:, -1, :]  # Use last token representation
            
            outputs["traffic_flow"] = self.traffic_flow_head(last_hidden).squeeze(-1)
            outputs["travel_time"] = F.softplus(self.travel_time_head(last_hidden)).squeeze(-1)
            outputs["congestion_level"] = self.congestion_head(last_hidden)
            outputs["aqi_prediction"] = F.softplus(self.aqi_head(last_hidden)).squeeze(-1)
            
            # Multi-horizon forecasts
            forecasts = self.forecast_head(last_hidden)
            outputs["forecasts"] = forecasts.view(batch_size, self.config.num_forecast_horizons, 4)
            
            # Uncertainty estimates
            uncertainty = self.uncertainty_head(last_hidden)
            outputs["prediction_mean"] = uncertainty[:, 0]
            outputs["prediction_uncertainty"] = F.softplus(uncertainty[:, 1])
        
        return outputs
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask (lower triangular)"""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        **kwargs
    ) -> torch.Tensor:
        """
        Autoregressive text generation with sampling
        """
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                outputs = self.forward(input_ids, output_predictions=False, **kwargs)
                next_token_logits = outputs["logits"][:, -1, :]
                
                # Apply temperature
                next_token_logits = next_token_logits / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Stop on EOS (token 50256 for GPT-2)
                if next_token.item() == 50256:
                    break
        
        return input_ids
    
    @torch.no_grad()
    def predict_traffic(
        self,
        query_embedding: torch.Tensor,
        time_context: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Make traffic predictions from query embedding
        """
        outputs = self.forward(
            query_embedding,
            **time_context,
            output_predictions=True
        )
        
        return {
            "traffic_flow": outputs["traffic_flow"].item(),
            "travel_time_minutes": outputs["travel_time"].item(),
            "congestion_level": F.softmax(outputs["congestion_level"], dim=-1).tolist(),
            "aqi": outputs["aqi_prediction"].item(),
            "forecasts": outputs["forecasts"].tolist(),
            "confidence": 1.0 / (1.0 + outputs["prediction_uncertainty"].item())
        }


# Factory function
def create_traffic_model(
    size: str = "base",
    device: Optional[str] = None
) -> TrafficFoundationModel:
    """
    Create a Traffic Foundation Model
    
    Sizes:
    - tiny: 25M params (for testing)
    - base: 125M params
    - large: 350M params
    - xl: 1.3B params
    """
    configs = {
        "tiny": TrafficModelConfig(
            hidden_size=256,
            num_hidden_layers=6,
            num_attention_heads=4,
            intermediate_size=1024,
            num_experts=4
        ),
        "base": TrafficModelConfig(
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            num_experts=8
        ),
        "large": TrafficModelConfig(
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=4096,
            num_experts=16
        ),
        "xl": TrafficModelConfig(
            hidden_size=2048,
            num_hidden_layers=36,
            num_attention_heads=32,
            intermediate_size=8192,
            num_experts=32
        )
    }
    
    config = configs.get(size, configs["base"])
    if device:
        config.device = device
    
    model = TrafficFoundationModel(config)
    model = model.to(config.device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Created Traffic Foundation Model ({size}): {num_params:,} parameters")
    
    return model
