# Imagen 3 Configuration
# Dedicated API key for Nano Banana Pro overlay generation

import os
from dataclasses import dataclass
from typing import Optional

# Imagen API Key — uses dedicated key, or falls back to main Gemini key
IMAGEN_API_KEY = os.getenv("IMAGEN_API_KEY") or os.getenv("GEMINI_API_KEY", "")

@dataclass
class ImagenConfig:
    """Configuration for Imagen 3 overlay generation"""
    api_key: str
    model: str = "imagen-3.0-generate-002"  # Nano Banana Pro model
    output_mime_type: str = "image/png"
    aspect_ratio: str = "16:9"  # Good for map overlays
    safety_filter_level: str = "BLOCK_ONLY_HIGH"
    person_generation: str = "DONT_ALLOW"  # No people in traffic overlays

def get_imagen_config() -> ImagenConfig:
    """Get Imagen configuration"""
    return ImagenConfig(api_key=IMAGEN_API_KEY)

def imagen_enabled() -> bool:
    """Check if Imagen API is configured"""
    return bool(IMAGEN_API_KEY and len(IMAGEN_API_KEY) > 10)
