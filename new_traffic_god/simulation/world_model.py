"""
World Model - Traffic Simulation Engine
========================================

A neural world model that simulates traffic dynamics.
Used for:
1. Counterfactual reasoning ("What if we add a flyover?")
2. Scenario planning
3. Infrastructure impact assessment
4. Multi-step prediction

This is not just a predictor - it's a full simulation engine
that understands traffic physics, flow dynamics, and urban patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import math
from collections import defaultdict


@dataclass
class SimulationConfig:
    """Configuration for traffic simulation"""
    # Grid configuration (Noida/NCR)
    grid_size: Tuple[int, int] = (100, 100)  # 100x100 cells
    cell_size_meters: float = 100.0  # Each cell = 100m x 100m
    
    # Time configuration
    time_step_seconds: int = 60  # 1-minute time steps
    simulation_horizon: int = 60  # Simulate 60 steps ahead (1 hour)
    
    # Traffic parameters
    max_vehicles_per_cell: int = 50
    avg_vehicle_length_meters: float = 4.5
    reaction_time_seconds: float = 1.5
    
    # Road network
    num_road_types: int = 5  # Highway, arterial, collector, local, lane
    road_capacities: List[int] = field(default_factory=lambda: [2000, 1200, 800, 400, 200])  # veh/hour
    road_speeds: List[float] = field(default_factory=lambda: [80, 50, 40, 30, 20])  # km/h
    
    # Model dimensions
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class TrafficState:
    """
    Represents the state of traffic in the simulation grid
    
    State variables per cell:
    - vehicle_count: Number of vehicles
    - avg_speed: Average speed (km/h)
    - density: Vehicles per km
    - flow: Vehicles per hour
    - queue_length: Vehicles waiting at signals
    - travel_time: Time to traverse cell
    - emission_rate: CO2 equivalent
    """
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.grid_h, self.grid_w = config.grid_size
        
        # State tensors
        self.vehicle_count = torch.zeros(self.grid_h, self.grid_w)
        self.avg_speed = torch.zeros(self.grid_h, self.grid_w)
        self.density = torch.zeros(self.grid_h, self.grid_w)
        self.flow = torch.zeros(self.grid_h, self.grid_w)
        self.queue_length = torch.zeros(self.grid_h, self.grid_w)
        self.signal_state = torch.zeros(self.grid_h, self.grid_w)  # 0=red, 1=green
        self.road_type = torch.zeros(self.grid_h, self.grid_w, dtype=torch.long)
        self.emission_rate = torch.zeros(self.grid_h, self.grid_w)
    
    def to_tensor(self) -> torch.Tensor:
        """Convert state to single tensor for neural network input"""
        return torch.stack([
            self.vehicle_count,
            self.avg_speed,
            self.density,
            self.flow,
            self.queue_length,
            self.signal_state,
            self.road_type.float(),
            self.emission_rate
        ], dim=0)  # (8, H, W)
    
    @classmethod
    def from_tensor(cls, tensor: torch.Tensor, config: SimulationConfig) -> "TrafficState":
        """Create state from tensor"""
        state = cls(config)
        state.vehicle_count = tensor[0]
        state.avg_speed = tensor[1]
        state.density = tensor[2]
        state.flow = tensor[3]
        state.queue_length = tensor[4]
        state.signal_state = tensor[5]
        state.road_type = tensor[6].long()
        state.emission_rate = tensor[7]
        return state
    
    def to(self, device: str) -> "TrafficState":
        """Move state to device"""
        self.vehicle_count = self.vehicle_count.to(device)
        self.avg_speed = self.avg_speed.to(device)
        self.density = self.density.to(device)
        self.flow = self.flow.to(device)
        self.queue_length = self.queue_length.to(device)
        self.signal_state = self.signal_state.to(device)
        self.road_type = self.road_type.to(device)
        self.emission_rate = self.emission_rate.to(device)
        return self


class TrafficPhysicsEngine:
    """
    Physics-based traffic flow calculations
    
    Implements fundamental traffic flow equations:
    - Greenshields model for speed-density relationship
    - Fundamental diagram (flow = density × speed)
    - Queue propagation
    - Signal timing effects
    """
    
    def __init__(self, config: SimulationConfig):
        self.config = config
    
    def greenshields_speed(
        self,
        density: torch.Tensor,
        free_flow_speed: torch.Tensor,
        jam_density: torch.Tensor
    ) -> torch.Tensor:
        """
        Greenshields model: v = v_f × (1 - k/k_j)
        
        Args:
            density: Current traffic density (veh/km)
            free_flow_speed: Speed with no traffic (km/h)
            jam_density: Density at gridlock (veh/km)
        
        Returns:
            Speed (km/h)
        """
        # Clamp density to valid range
        density = torch.clamp(density, min=0, max=jam_density)
        speed = free_flow_speed * (1 - density / jam_density)
        return torch.clamp(speed, min=0)
    
    def calculate_flow(
        self,
        density: torch.Tensor,
        speed: torch.Tensor
    ) -> torch.Tensor:
        """
        Fundamental relation: q = k × v
        
        Args:
            density: Traffic density (veh/km)
            speed: Traffic speed (km/h)
        
        Returns:
            Flow (veh/hour)
        """
        return density * speed
    
    def queue_dynamics(
        self,
        arrival_rate: torch.Tensor,
        service_rate: torch.Tensor,
        current_queue: torch.Tensor,
        time_step: float
    ) -> torch.Tensor:
        """
        Queue evolution: Q(t+1) = max(0, Q(t) + (λ - μ) × Δt)
        
        Args:
            arrival_rate: Vehicles arriving per hour
            service_rate: Vehicles served per hour (capacity)
            current_queue: Current queue length
            time_step: Time step in hours
        
        Returns:
            New queue length
        """
        queue_change = (arrival_rate - service_rate) * time_step
        new_queue = current_queue + queue_change
        return torch.clamp(new_queue, min=0)
    
    def emission_model(
        self,
        speed: torch.Tensor,
        acceleration: torch.Tensor,
        vehicle_count: torch.Tensor
    ) -> torch.Tensor:
        """
        Simple emission model based on speed and acceleration
        
        Emissions increase with:
        - Stop-and-go traffic (high acceleration variance)
        - Very low speeds (inefficient)
        - Very high speeds (high fuel consumption)
        
        Returns emission rate in arbitrary units
        """
        # Optimal speed around 50-60 km/h
        speed_factor = torch.abs(speed - 55) / 55 + 0.5
        
        # Acceleration increases emissions
        accel_factor = 1 + 0.5 * torch.abs(acceleration)
        
        # Per-vehicle emission
        emission_per_vehicle = speed_factor * accel_factor
        
        return emission_per_vehicle * vehicle_count


class SpatialConvBlock(nn.Module):
    """Convolutional block for spatial traffic patterns"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        
        return F.gelu(x + residual)


class TemporalAttention(nn.Module):
    """Attention over temporal dimension for sequence modeling"""
    
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, hidden)
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class WorldModel(nn.Module):
    """
    Neural World Model for Traffic Simulation
    
    Architecture:
    1. Spatial encoder (CNN) - Captures road network patterns
    2. Temporal encoder (Transformer) - Captures time dynamics
    3. Action encoder - Encodes interventions/changes
    4. Dynamics predictor - Predicts next state
    5. Reward predictor - Evaluates state quality
    
    The model learns to simulate traffic flow and predict
    the impact of interventions.
    """
    
    def __init__(self, config: SimulationConfig):
        super().__init__()
        self.config = config
        self.physics = TrafficPhysicsEngine(config)
        
        state_channels = 8  # From TrafficState.to_tensor()
        hidden_dim = config.hidden_dim
        
        # Spatial encoder (processes grid state)
        self.spatial_encoder = nn.Sequential(
            SpatialConvBlock(state_channels, 32),
            SpatialConvBlock(32, 64),
            SpatialConvBlock(64, 128),
            nn.AdaptiveAvgPool2d((10, 10)),  # Reduce to 10x10
            nn.Flatten(),
            nn.Linear(128 * 10 * 10, hidden_dim)
        )
        
        # Temporal encoder (processes sequence of states)
        self.temporal_encoder = nn.Sequential(
            *[TemporalAttention(hidden_dim, config.num_heads) 
              for _ in range(config.num_layers)]
        )
        
        # Action encoder (encodes interventions)
        self.action_encoder = nn.Sequential(
            nn.Linear(64, hidden_dim // 2),  # Action features
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Context encoder (time of day, weather, events)
        self.context_encoder = nn.Sequential(
            nn.Linear(32, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # Dynamics head (predicts next state)
        self.dynamics_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 128 * 10 * 10)
        )
        
        # State decoder (upsamples prediction back to grid)
        self.state_decoder = nn.Sequential(
            nn.Unflatten(1, (128, 10, 10)),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 20x20
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 40x40
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),   # 80x80
            nn.GELU(),
            nn.Conv2d(16, state_channels, 3, padding=1)
        )
        
        # Reward head (evaluates state quality)
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 4)  # [travel_time, congestion, emissions, safety]
        )
        
        # Uncertainty head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1)
        )
    
    def encode_state(self, state: torch.Tensor) -> torch.Tensor:
        """Encode a traffic state tensor"""
        return self.spatial_encoder(state)
    
    def encode_sequence(self, states: torch.Tensor) -> torch.Tensor:
        """
        Encode a sequence of states
        
        Args:
            states: (batch, time, channels, H, W)
        
        Returns:
            Sequence encoding: (batch, time, hidden)
        """
        batch, time, channels, h, w = states.shape
        
        # Encode each time step
        states_flat = states.view(batch * time, channels, h, w)
        encoded = self.spatial_encoder(states_flat)
        encoded = encoded.view(batch, time, -1)
        
        # Temporal attention
        encoded = self.temporal_encoder(encoded)
        
        return encoded
    
    def predict_next_state(
        self,
        current_state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        history: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict the next traffic state
        
        Args:
            current_state: Current state tensor (batch, 8, H, W)
            action: Action/intervention tensor (batch, 64)
            context: Context features (batch, 32)
            history: Historical states (batch, time, 8, H, W)
        
        Returns:
            next_state: Predicted next state
            reward: State quality metrics
            uncertainty: Prediction uncertainty
        """
        batch_size = current_state.shape[0]
        device = current_state.device
        
        # Encode current state
        state_encoding = self.encode_state(current_state)
        
        # Encode action (or use zeros)
        if action is None:
            action = torch.zeros(batch_size, 64, device=device)
        action_encoding = self.action_encoder(action)
        
        # Encode context (or use zeros)
        if context is None:
            context = torch.zeros(batch_size, 32, device=device)
        context_encoding = self.context_encoder(context)
        
        # Combine encodings
        combined = torch.cat([state_encoding, action_encoding, context_encoding], dim=-1)
        
        # Predict dynamics
        dynamics = self.dynamics_head(combined)
        
        # Decode to next state
        next_state = self.state_decoder(dynamics)
        
        # Resize to match input (in case of size mismatch)
        if next_state.shape[-2:] != current_state.shape[-2:]:
            next_state = F.interpolate(
                next_state, 
                size=current_state.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            )
        
        # Apply physics constraints
        next_state = self._apply_physics_constraints(next_state, current_state)
        
        # Predict reward
        reward = self.reward_head(state_encoding)
        
        # Predict uncertainty
        uncertainty = F.softplus(self.uncertainty_head(state_encoding))
        
        return next_state, reward, uncertainty
    
    def _apply_physics_constraints(
        self,
        predicted_state: torch.Tensor,
        current_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply physics-based constraints to ensure realistic predictions
        """
        # Extract components
        pred_count = predicted_state[:, 0:1]
        pred_speed = predicted_state[:, 1:2]
        pred_density = predicted_state[:, 2:3]
        pred_flow = predicted_state[:, 3:4]
        pred_queue = predicted_state[:, 4:5]
        pred_signal = predicted_state[:, 5:6]
        pred_road = predicted_state[:, 6:7]
        pred_emission = predicted_state[:, 7:8]
        
        current_count = current_state[:, 0:1]
        
        # Vehicle count must be non-negative
        pred_count = F.relu(pred_count)
        
        # Speed must be non-negative and bounded
        pred_speed = torch.clamp(pred_speed, min=0, max=120)
        
        # Density must be non-negative
        pred_density = F.relu(pred_density)
        
        # Flow = density × speed (enforce fundamental relation)
        pred_flow = pred_density * pred_speed
        
        # Queue must be non-negative
        pred_queue = F.relu(pred_queue)
        
        # Signal state: sigmoid to [0, 1]
        pred_signal = torch.sigmoid(pred_signal)
        
        # Road type: keep from current (doesn't change)
        pred_road = current_state[:, 6:7]
        
        # Emission rate: must be non-negative
        pred_emission = F.relu(pred_emission)
        
        # Conservation of vehicles (approximate)
        # Total vehicles shouldn't change dramatically in one step
        total_pred = pred_count.sum(dim=(-2, -1), keepdim=True)
        total_curr = current_count.sum(dim=(-2, -1), keepdim=True)
        
        if total_curr.sum() > 0:
            scale = (total_curr / (total_pred + 1e-6)).clamp(0.9, 1.1)
            pred_count = pred_count * scale
        
        # Reconstruct state
        constrained = torch.cat([
            pred_count, pred_speed, pred_density, pred_flow,
            pred_queue, pred_signal, pred_road, pred_emission
        ], dim=1)
        
        return constrained
    
    def simulate(
        self,
        initial_state: TrafficState,
        actions: Optional[List[torch.Tensor]] = None,
        contexts: Optional[List[torch.Tensor]] = None,
        num_steps: Optional[int] = None
    ) -> List[TrafficState]:
        """
        Run a full simulation from initial state
        
        Args:
            initial_state: Starting traffic state
            actions: List of actions per time step
            contexts: List of contexts per time step
            num_steps: Number of steps to simulate
        
        Returns:
            List of simulated states
        """
        if num_steps is None:
            num_steps = self.config.simulation_horizon
        
        device = self.config.device
        states = [initial_state]
        current_tensor = initial_state.to_tensor().unsqueeze(0).to(device)
        
        self.eval()
        with torch.no_grad():
            for t in range(num_steps):
                action = actions[t] if actions and t < len(actions) else None
                context = contexts[t] if contexts and t < len(contexts) else None
                
                next_tensor, _, _ = self.predict_next_state(
                    current_tensor,
                    action=action,
                    context=context
                )
                
                next_state = TrafficState.from_tensor(
                    next_tensor.squeeze(0).cpu(),
                    self.config
                )
                states.append(next_state)
                current_tensor = next_tensor
        
        return states
    
    def evaluate_intervention(
        self,
        initial_state: TrafficState,
        intervention: Dict[str, Any],
        horizon: int = 60
    ) -> Dict[str, Any]:
        """
        Evaluate the impact of an intervention
        
        Args:
            initial_state: Current traffic state
            intervention: Description of the intervention
            horizon: Evaluation horizon in steps
        
        Returns:
            Impact assessment with metrics
        """
        # Encode intervention as action
        action = self._encode_intervention(intervention)
        
        # Simulate with intervention
        states_with = self.simulate(
            initial_state,
            actions=[action] * horizon,
            num_steps=horizon
        )
        
        # Simulate without intervention (baseline)
        states_without = self.simulate(
            initial_state,
            num_steps=horizon
        )
        
        # Calculate metrics
        def avg_metrics(states):
            total_flow = sum(s.flow.mean().item() for s in states) / len(states)
            total_speed = sum(s.avg_speed.mean().item() for s in states) / len(states)
            total_queue = sum(s.queue_length.mean().item() for s in states) / len(states)
            total_emission = sum(s.emission_rate.mean().item() for s in states) / len(states)
            return {
                "avg_flow": total_flow,
                "avg_speed": total_speed,
                "avg_queue": total_queue,
                "avg_emission": total_emission
            }
        
        metrics_with = avg_metrics(states_with)
        metrics_without = avg_metrics(states_without)
        
        # Calculate deltas
        impact = {
            "flow_change": metrics_with["avg_flow"] - metrics_without["avg_flow"],
            "speed_change": metrics_with["avg_speed"] - metrics_without["avg_speed"],
            "queue_change": metrics_with["avg_queue"] - metrics_without["avg_queue"],
            "emission_change": metrics_with["avg_emission"] - metrics_without["avg_emission"],
            "with_intervention": metrics_with,
            "without_intervention": metrics_without,
            "intervention": intervention
        }
        
        return impact
    
    def _encode_intervention(self, intervention: Dict[str, Any]) -> torch.Tensor:
        """Encode an intervention as an action tensor"""
        action = torch.zeros(1, 64, device=self.config.device)
        
        # Map intervention types to action dimensions
        intervention_type = intervention.get("type", "none")
        
        type_mapping = {
            "add_lane": 0,
            "remove_lane": 1,
            "add_signal": 2,
            "optimize_signal": 3,
            "add_flyover": 4,
            "add_underpass": 5,
            "speed_limit": 6,
            "bus_lane": 7,
            "metro_station": 8,
            "ev_charging": 9,
            "congestion_pricing": 10
        }
        
        type_idx = type_mapping.get(intervention_type, 0)
        action[0, type_idx] = 1.0
        
        # Encode magnitude
        magnitude = intervention.get("magnitude", 1.0)
        action[0, 32] = magnitude
        
        # Encode location
        location = intervention.get("location", (50, 50))
        action[0, 33] = location[0] / 100.0
        action[0, 34] = location[1] / 100.0
        
        return action


class ScenarioPlanner:
    """
    Planner that uses the world model to evaluate scenarios
    
    Capabilities:
    - Route optimization
    - Infrastructure planning
    - Event impact assessment
    - Policy evaluation
    """
    
    def __init__(self, world_model: WorldModel):
        self.world_model = world_model
        self.config = world_model.config
    
    def find_best_route(
        self,
        state: TrafficState,
        origin: Tuple[int, int],
        destination: Tuple[int, int],
        num_alternatives: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Find optimal routes from origin to destination
        
        Returns top N routes with estimated travel times
        """
        routes = []
        
        # Simple A* pathfinding with traffic-aware costs
        def heuristic(pos):
            return math.sqrt((pos[0] - destination[0])**2 + (pos[1] - destination[1])**2)
        
        def get_neighbors(pos):
            neighbors = []
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = pos[0] + dx, pos[1] + dy
                if 0 <= nx < self.config.grid_size[0] and 0 <= ny < self.config.grid_size[1]:
                    neighbors.append((nx, ny))
            return neighbors
        
        def get_cost(pos):
            # Cost based on current traffic conditions
            speed = state.avg_speed[pos[0], pos[1]].item()
            density = state.density[pos[0], pos[1]].item()
            
            if speed < 1:
                return 100  # Gridlock
            
            # Travel time through cell
            cell_length = self.config.cell_size_meters / 1000  # km
            travel_time = cell_length / speed * 60  # minutes
            
            return travel_time
        
        # Run A* to find path
        import heapq
        
        open_set = [(0, origin, [origin])]
        visited = set()
        
        while open_set and len(routes) < num_alternatives:
            _, current, path = heapq.heappop(open_set)
            
            if current == destination:
                total_time = sum(get_cost(p) for p in path)
                routes.append({
                    "path": path,
                    "travel_time_minutes": total_time,
                    "distance_km": len(path) * self.config.cell_size_meters / 1000
                })
                continue
            
            if current in visited:
                continue
            visited.add(current)
            
            for neighbor in get_neighbors(current):
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    cost = sum(get_cost(p) for p in new_path) + heuristic(neighbor)
                    heapq.heappush(open_set, (cost, neighbor, new_path))
        
        return sorted(routes, key=lambda r: r["travel_time_minutes"])
    
    def evaluate_infrastructure(
        self,
        state: TrafficState,
        proposals: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple infrastructure proposals
        
        Returns ranked list with cost-benefit analysis
        """
        results = []
        
        for proposal in proposals:
            impact = self.world_model.evaluate_intervention(
                state,
                proposal,
                horizon=60
            )
            
            # Calculate benefit score
            benefit = (
                impact["speed_change"] * 2 +  # Speed improvement
                impact["flow_change"] * 1.5 - # Flow improvement
                impact["queue_change"] * 3 -  # Queue reduction
                impact["emission_change"] * 2  # Emission reduction
            )
            
            # Estimate cost (simplified)
            cost_factors = {
                "add_lane": 10,
                "add_flyover": 100,
                "add_underpass": 80,
                "add_signal": 5,
                "optimize_signal": 1,
                "metro_station": 200,
                "bus_lane": 15
            }
            cost = cost_factors.get(proposal.get("type"), 10)
            
            results.append({
                "proposal": proposal,
                "impact": impact,
                "benefit_score": benefit,
                "cost_score": cost,
                "cost_benefit_ratio": benefit / cost if cost > 0 else 0
            })
        
        return sorted(results, key=lambda r: r["cost_benefit_ratio"], reverse=True)


# Factory function
def create_world_model(device: Optional[str] = None) -> WorldModel:
    """Create a world model instance"""
    config = SimulationConfig()
    if device:
        config.device = device
    
    model = WorldModel(config)
    model = model.to(config.device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Created World Model: {num_params:,} parameters")
    
    return model


if __name__ == "__main__":
    # Test the world model
    model = create_world_model()
    
    # Create initial state
    config = SimulationConfig()
    state = TrafficState(config)
    
    # Random initialization
    state.vehicle_count = torch.rand(100, 100) * 30
    state.avg_speed = torch.rand(100, 100) * 50 + 10
    state.density = state.vehicle_count / (config.cell_size_meters / 1000)
    state.flow = state.density * state.avg_speed
    
    state = state.to(config.device)
    
    # Test simulation
    print("Running simulation...")
    states = model.simulate(state, num_steps=10)
    print(f"Simulated {len(states)} states")
    
    # Test intervention evaluation
    intervention = {
        "type": "add_flyover",
        "location": (50, 50),
        "magnitude": 1.0
    }
    
    print("\nEvaluating intervention...")
    impact = model.evaluate_intervention(state, intervention, horizon=10)
    print(f"Speed change: {impact['speed_change']:.2f} km/h")
    print(f"Flow change: {impact['flow_change']:.2f} veh/h")
