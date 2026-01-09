# Imagen 3 (Nano Banana Pro) Overlay Generator
# Generates high-quality 3D traffic visualization overlays

import base64
import asyncio
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class OverlayType(Enum):
    """Types of 3D overlays we can generate"""
    TRAFFIC_HEATMAP = "traffic_heatmap"
    CONGESTION_3D = "congestion_3d"
    POLLUTION_CLOUD = "pollution_cloud"
    FLOW_ARROWS = "flow_arrows"
    INFRASTRUCTURE = "infrastructure"
    EMERGENCY_ROUTE = "emergency_route"
    EV_CHARGING = "ev_charging"
    TRANSIT_HUB = "transit_hub"

@dataclass
class OverlayResult:
    """Result from overlay generation"""
    success: bool
    image_base64: Optional[str] = None
    mime_type: str = "image/png"
    prompt_used: str = ""
    error: Optional[str] = None
    overlay_type: str = ""
    metadata: Optional[Dict[str, Any]] = None

class ImagenOverlayGenerator:
    """
    Generates high-quality 3D overlays using Google's Imagen 3 (Nano Banana Pro).
    
    These overlays are designed to be composited on top of Azure Maps tiles
    to provide rich traffic visualization.
    """
    
    # Overlay prompt templates optimized for traffic visualization
    OVERLAY_PROMPTS = {
        OverlayType.TRAFFIC_HEATMAP: """
            Hyper-realistic 3D visualization overlay, top-down isometric view,
            traffic density heatmap showing {intensity} congestion levels,
            glowing neon colors: red for heavy traffic, yellow for moderate, green for light,
            transparent background, smooth gradients, futuristic holographic style,
            city grid pattern, {location} urban area, cinematic lighting,
            8K ultra HD, professional visualization, no text or labels
        """,
        
        OverlayType.CONGESTION_3D: """
            3D extruded traffic congestion visualization, isometric aerial view,
            road segments as 3D bars, height represents traffic volume,
            color gradient from green (free flow) to red (gridlock),
            {location} city streets, transparent glass-like material,
            glowing edges, futuristic smart city aesthetic,
            clean minimal design, dark background with neon accents,
            professional data visualization style, 8K render
        """,
        
        OverlayType.POLLUTION_CLOUD: """
            Atmospheric pollution visualization, 3D volumetric clouds,
            air quality index overlay, {aqi_level} pollution levels,
            particle density visualization, orange and gray toxic haze,
            transparent layered effect, {location} cityscape silhouette,
            environmental monitoring aesthetic, scientific visualization,
            dramatic lighting, photorealistic render, transparent PNG
        """,
        
        OverlayType.FLOW_ARROWS: """
            Dynamic traffic flow visualization, 3D animated-style arrows,
            directional flow patterns showing vehicle movement,
            {direction} traffic direction, glowing cyan and magenta trails,
            speed indicated by arrow length and brightness,
            futuristic HUD overlay style, transparent background,
            clean vector-like aesthetic, smart city dashboard,
            professional infographic quality, 8K resolution
        """,
        
        OverlayType.INFRASTRUCTURE: """
            Smart city infrastructure overlay, 3D isometric view,
            {infrastructure_type} highlighted with glowing outlines,
            connected nodes and network visualization,
            futuristic urban planning aesthetic, neon blue accents,
            transparent background, holographic projection style,
            technical blueprint meets sci-fi visualization,
            clean professional render, 8K ultra detailed
        """,
        
        OverlayType.EMERGENCY_ROUTE: """
            Emergency response route visualization, 3D aerial view,
            highlighted optimal path with pulsing red glow,
            {route_type} emergency corridor, cleared traffic lanes,
            ambulance/fire route overlay, urgent visual indicators,
            transparent background, dramatic lighting,
            professional emergency services aesthetic, 8K render
        """,
        
        OverlayType.EV_CHARGING: """
            Electric vehicle charging network overlay, 3D map view,
            charging station locations as glowing green nodes,
            battery level indicators, range circles visualization,
            {coverage} charging coverage area, futuristic EV aesthetic,
            sustainable energy theme, leaf green and electric blue,
            transparent background, clean infographic style, 8K
        """,
        
        OverlayType.TRANSIT_HUB: """
            Public transit hub visualization, 3D isometric overlay,
            {transit_type} stations and routes highlighted,
            passenger flow visualization, connection nodes glowing,
            metro/bus network aesthetic, urban mobility theme,
            transparent background, professional transit map style,
            futuristic smart city design, 8K ultra detailed render
        """
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with API key"""
        from .config import get_imagen_config
        
        config = get_imagen_config()
        self.api_key = api_key or config.api_key
        self.model = config.model
        self.config = config
        self._client = None
    
    def _get_client(self):
        """Lazy-load the Google Generative AI client"""
        if self._client is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai
            except ImportError:
                raise ImportError("google-generativeai package required. Install with: pip install google-generativeai")
        return self._client
    
    def _build_prompt(self, overlay_type: OverlayType, context: Dict[str, Any]) -> str:
        """Build optimized prompt for overlay generation"""
        template = self.OVERLAY_PROMPTS.get(overlay_type, self.OVERLAY_PROMPTS[OverlayType.TRAFFIC_HEATMAP])
        
        # Default context values
        defaults = {
            "intensity": "moderate",
            "location": "Delhi NCR",
            "aqi_level": "moderate",
            "direction": "multi-directional",
            "infrastructure_type": "traffic signals and sensors",
            "route_type": "ambulance",
            "coverage": "comprehensive",
            "transit_type": "metro and bus"
        }
        
        # Merge with provided context
        merged = {**defaults, **context}
        
        # Format the template
        prompt = template.format(**merged).strip()
        
        # Add quality enhancers
        prompt += ", masterpiece quality, ultra-detailed, professional visualization"
        
        return prompt
    
    async def generate_overlay(
        self,
        overlay_type: OverlayType,
        context: Optional[Dict[str, Any]] = None,
        aspect_ratio: Optional[str] = None
    ) -> OverlayResult:
        """
        Generate a 3D overlay image using Imagen 3.
        
        Args:
            overlay_type: Type of overlay to generate
            context: Additional context for prompt customization
            aspect_ratio: Override aspect ratio (default: 16:9)
        
        Returns:
            OverlayResult with base64 encoded image
        """
        context = context or {}
        
        try:
            genai = self._get_client()
            
            # Build the prompt
            prompt = self._build_prompt(overlay_type, context)
            
            logger.info(f"Generating {overlay_type.value} overlay with Imagen 3")
            
            # Use Imagen 3 model
            imagen = genai.ImageGenerationModel(self.model)
            
            # Generate the image
            result = await asyncio.to_thread(
                imagen.generate_images,
                prompt=prompt,
                number_of_images=1,
                aspect_ratio=aspect_ratio or self.config.aspect_ratio,
                safety_filter_level=self.config.safety_filter_level,
                person_generation=self.config.person_generation
            )
            
            if result.images:
                # Get the first image
                image = result.images[0]
                
                # Convert to base64
                image_bytes = image._pil_image.tobytes() if hasattr(image, '_pil_image') else image.data
                
                # Try to get PNG bytes
                import io
                from PIL import Image
                
                if hasattr(image, '_pil_image'):
                    buffer = io.BytesIO()
                    image._pil_image.save(buffer, format='PNG')
                    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                else:
                    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                
                return OverlayResult(
                    success=True,
                    image_base64=image_base64,
                    mime_type="image/png",
                    prompt_used=prompt,
                    overlay_type=overlay_type.value,
                    metadata={
                        "model": self.model,
                        "aspect_ratio": aspect_ratio or self.config.aspect_ratio,
                        "context": context
                    }
                )
            else:
                return OverlayResult(
                    success=False,
                    error="No images generated",
                    prompt_used=prompt,
                    overlay_type=overlay_type.value
                )
                
        except Exception as e:
            logger.error(f"Imagen overlay generation failed: {e}")
            return OverlayResult(
                success=False,
                error=str(e),
                overlay_type=overlay_type.value
            )
    
    async def generate_traffic_overlay(
        self,
        location: str = "Delhi NCR",
        congestion_level: str = "moderate",
        include_pollution: bool = False
    ) -> OverlayResult:
        """Convenience method for traffic overlays"""
        context = {
            "location": location,
            "intensity": congestion_level
        }
        
        if include_pollution:
            # Generate combined traffic + pollution overlay
            context["aqi_level"] = "moderate to high"
            return await self.generate_overlay(OverlayType.POLLUTION_CLOUD, context)
        
        return await self.generate_overlay(OverlayType.TRAFFIC_HEATMAP, context)
    
    async def generate_congestion_3d(
        self,
        location: str = "Delhi NCR",
        hotspots: Optional[List[str]] = None
    ) -> OverlayResult:
        """Generate 3D congestion bars overlay"""
        context = {
            "location": location,
            "hotspots": ", ".join(hotspots) if hotspots else "major intersections"
        }
        return await self.generate_overlay(OverlayType.CONGESTION_3D, context)
    
    async def generate_emergency_route(
        self,
        route_type: str = "ambulance",
        location: str = "Delhi NCR"
    ) -> OverlayResult:
        """Generate emergency route overlay"""
        context = {
            "route_type": route_type,
            "location": location
        }
        return await self.generate_overlay(OverlayType.EMERGENCY_ROUTE, context)
    
    def generate_overlay_sync(
        self,
        overlay_type: OverlayType,
        context: Optional[Dict[str, Any]] = None
    ) -> OverlayResult:
        """Synchronous wrapper for overlay generation"""
        return asyncio.run(self.generate_overlay(overlay_type, context))


# Factory function for easy access
def create_overlay_generator(api_key: Optional[str] = None) -> ImagenOverlayGenerator:
    """Create an overlay generator instance"""
    return ImagenOverlayGenerator(api_key)
