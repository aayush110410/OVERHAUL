"""
New Traffic God - Configuration Settings
=========================================
All configuration parameters for the Traffic Foundation Model
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import yaml


# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

@dataclass
class ModelConfig:
    """Foundation Model Configuration"""
    
    # Architecture
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    d_ff: int = 3072
    vocab_size: int = 50000
    max_seq_len: int = 2048
    dropout: float = 0.1
    
    # Model variants
    model_size: str = "base"  # small, base, large, xl
    
    # Pre-configured sizes
    SIZES = {
        "small": {"d_model": 512, "n_heads": 8, "n_layers": 6, "d_ff": 2048},
        "base": {"d_model": 768, "n_heads": 12, "n_layers": 12, "d_ff": 3072},
        "large": {"d_model": 1024, "n_heads": 16, "n_layers": 24, "d_ff": 4096},
        "xl": {"d_model": 1536, "n_heads": 24, "n_layers": 32, "d_ff": 6144}
    }
    
    @classmethod
    def from_size(cls, size: str) -> "ModelConfig":
        """Create config from predefined size"""
        if size not in cls.SIZES:
            raise ValueError(f"Unknown size: {size}. Choose from {list(cls.SIZES.keys())}")
        return cls(model_size=size, **cls.SIZES[size])


@dataclass
class TrainingConfig:
    """Training Configuration"""
    
    # Basic training
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_epochs: int = 100
    warmup_steps: int = 10000
    gradient_clip: float = 1.0
    
    # Data
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Optimizer
    optimizer: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # Scheduler
    scheduler: str = "cosine"  # cosine, linear, constant
    min_lr: float = 1e-6
    
    # Mixed precision
    fp16: bool = True
    bf16: bool = False
    
    # Checkpointing
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    
    # Distributed
    distributed: bool = False
    world_size: int = 1
    local_rank: int = -1


@dataclass  
class RLHFConfig:
    """RLHF Training Configuration"""
    
    # PPO
    ppo_epochs: int = 4
    clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    
    # Reward model
    reward_model_lr: float = 1e-5
    reward_batch_size: int = 16
    
    # KL penalty
    kl_coef: float = 0.1
    target_kl: float = 0.02
    
    # Generation
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass
class InferenceConfig:
    """Inference Configuration"""
    
    # Generation
    max_length: int = 1024
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    do_sample: bool = True
    num_beams: int = 1
    
    # Caching
    use_cache: bool = True
    cache_size: int = 1000
    
    # Batching
    batch_size: int = 1
    max_batch_tokens: int = 4096
    
    # Device
    device: str = "cuda"
    dtype: str = "float16"


@dataclass
class RAGConfig:
    """RAG System Configuration"""
    
    # Retrieval
    top_k: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 50
    
    # Embedding
    embedding_dim: int = 768
    embedding_model: str = "traffic-embedding-base"
    
    # Index
    index_type: str = "faiss"  # faiss, annoy, hnswlib
    index_nlist: int = 100
    index_nprobe: int = 10
    
    # Hybrid search
    use_bm25: bool = True
    bm25_weight: float = 0.3
    dense_weight: float = 0.7


@dataclass
class WorldModelConfig:
    """World Model / Simulator Configuration"""
    
    # Grid
    grid_size: int = 100
    cell_size: float = 0.1  # km
    
    # Simulation
    dt: float = 0.1  # seconds
    max_steps: int = 3600
    
    # Neural network
    state_dim: int = 256
    action_dim: int = 64
    hidden_dim: int = 512
    
    # Physics
    max_speed: float = 120.0  # km/h
    acceleration_factor: float = 0.1
    friction: float = 0.02


@dataclass
class DataConfig:
    """Data Configuration"""
    
    # Paths
    data_dir: str = "./data"
    cache_dir: str = "./cache"
    checkpoint_dir: str = "./checkpoints"
    
    # Noida specific
    region: str = "noida"
    include_regions: List[str] = field(default_factory=lambda: [
        "noida", "indirapuram", "greater_noida", 
        "ghaziabad", "delhi_ncr"
    ])
    
    # Data sources
    use_osm: bool = True
    use_google_traffic: bool = False
    use_historical: bool = True
    
    # Preprocessing
    normalize: bool = True
    augment: bool = True
    time_encoding: str = "sinusoidal"


@dataclass
class APIConfig:
    """API Configuration"""
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 4
    
    # Rate limiting
    rate_limit: int = 100  # requests per minute
    burst_limit: int = 20
    
    # Auth
    require_auth: bool = False
    api_key_header: str = "X-API-Key"
    
    # CORS
    allow_origins: List[str] = field(default_factory=lambda: ["*"])
    allow_methods: List[str] = field(default_factory=lambda: ["GET", "POST"])


# ============================================================================
# CONFIGURATION LOADER
# ============================================================================

class ConfigLoader:
    """Load and manage configurations"""
    
    DEFAULT_CONFIGS = {
        "model": ModelConfig,
        "training": TrainingConfig,
        "rlhf": RLHFConfig,
        "inference": InferenceConfig,
        "rag": RAGConfig,
        "world_model": WorldModelConfig,
        "data": DataConfig,
        "api": APIConfig
    }
    
    def __init__(self, config_path: Optional[str] = None):
        self.configs = {}
        
        # Load defaults
        for name, config_cls in self.DEFAULT_CONFIGS.items():
            self.configs[name] = config_cls()
        
        # Load from file if provided
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
    
    def load_from_file(self, path: str):
        """Load configs from YAML file"""
        with open(path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        for name, values in yaml_config.items():
            if name in self.DEFAULT_CONFIGS:
                config_cls = self.DEFAULT_CONFIGS[name]
                self.configs[name] = config_cls(**values)
    
    def save_to_file(self, path: str):
        """Save current configs to YAML file"""
        yaml_config = {}
        for name, config in self.configs.items():
            yaml_config[name] = config.__dict__
        
        with open(path, 'w') as f:
            yaml.dump(yaml_config, f, default_flow_style=False)
    
    def get(self, name: str):
        """Get a specific config"""
        return self.configs.get(name)
    
    def update(self, name: str, **kwargs):
        """Update a config with new values"""
        if name in self.configs:
            for key, value in kwargs.items():
                if hasattr(self.configs[name], key):
                    setattr(self.configs[name], key, value)


# ============================================================================
# ENVIRONMENT-SPECIFIC CONFIGS
# ============================================================================

DEVELOPMENT_CONFIG = {
    "model": {"model_size": "small", "d_model": 256, "n_layers": 4},
    "training": {"batch_size": 8, "max_epochs": 10},
    "inference": {"device": "cpu"},
    "api": {"workers": 1, "require_auth": False}
}

PRODUCTION_CONFIG = {
    "model": {"model_size": "large", "d_model": 1024, "n_layers": 24},
    "training": {"batch_size": 64, "fp16": True, "distributed": True},
    "inference": {"device": "cuda", "dtype": "float16"},
    "api": {"workers": 8, "require_auth": True, "rate_limit": 1000}
}

TESTING_CONFIG = {
    "model": {"model_size": "small", "d_model": 128, "n_layers": 2},
    "training": {"batch_size": 2, "max_epochs": 1},
    "inference": {"device": "cpu"},
    "api": {"workers": 1}
}


def get_config(env: str = "development") -> ConfigLoader:
    """Get configuration for environment"""
    loader = ConfigLoader()
    
    env_configs = {
        "development": DEVELOPMENT_CONFIG,
        "production": PRODUCTION_CONFIG,
        "testing": TESTING_CONFIG
    }
    
    if env in env_configs:
        for name, values in env_configs[env].items():
            loader.update(name, **values)
    
    return loader


# ============================================================================
# NOIDA-SPECIFIC SETTINGS
# ============================================================================

NOIDA_SETTINGS = {
    # Peak hours (24-hour format)
    "morning_peak_start": 8,
    "morning_peak_end": 11,
    "evening_peak_start": 17,
    "evening_peak_end": 21,
    
    # Traffic multipliers
    "peak_multiplier": 2.5,
    "weekend_multiplier": 0.6,
    "holiday_multiplier": 0.4,
    "rain_multiplier": 1.5,
    "fog_multiplier": 1.8,
    
    # Key corridors
    "major_corridors": [
        "DND Flyway",
        "Noida-Greater Noida Expressway",
        "NH-24",
        "Sector 62 - Ghaziabad Link",
        "Film City Road"
    ],
    
    # Metro stations
    "metro_influence_radius": 2.0,  # km
    "metro_capacity_factor": 0.85,
    
    # AQI thresholds
    "aqi_good": 50,
    "aqi_moderate": 100,
    "aqi_unhealthy": 150,
    "aqi_very_unhealthy": 200,
    "aqi_hazardous": 300,
    
    # Speed limits (km/h)
    "speed_limit_expressway": 100,
    "speed_limit_main_road": 60,
    "speed_limit_sector_road": 40,
    "speed_limit_residential": 30
}


# Quick access
def get_default_model_config() -> ModelConfig:
    return ModelConfig()

def get_default_training_config() -> TrainingConfig:
    return TrainingConfig()

def get_noida_settings() -> Dict:
    return NOIDA_SETTINGS.copy()
