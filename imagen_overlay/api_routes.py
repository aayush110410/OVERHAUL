# FastAPI routes for Imagen 3 overlay generation
# Endpoints for generating and serving 3D map overlays

import base64
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel, Field

from .generator import ImagenOverlayGenerator, OverlayType, create_overlay_generator
from .config import imagen_enabled

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/imagen", tags=["Imagen 3D Overlays"])

# Pydantic models for request/response
class OverlayRequest(BaseModel):
    """Request model for overlay generation"""
    overlay_type: str = Field(
        default="traffic_heatmap",
        description="Type of overlay: traffic_heatmap, congestion_3d, pollution_cloud, flow_arrows, infrastructure, emergency_route, ev_charging, transit_hub"
    )
    location: str = Field(default="Delhi NCR", description="Location context for the overlay")
    intensity: Optional[str] = Field(default="moderate", description="Intensity level: low, moderate, high, critical")
    aspect_ratio: Optional[str] = Field(default="16:9", description="Aspect ratio: 1:1, 16:9, 9:16, 4:3, 3:4")
    include_pollution: Optional[bool] = Field(default=False, description="Include pollution visualization")
    custom_context: Optional[dict] = Field(default=None, description="Additional context parameters")

class OverlayResponse(BaseModel):
    """Response model for overlay generation"""
    success: bool
    image_base64: Optional[str] = None
    image_url: Optional[str] = None
    mime_type: str = "image/png"
    overlay_type: str
    prompt_used: Optional[str] = None
    error: Optional[str] = None


@router.get("/status")
async def imagen_status():
    """Check Imagen API status"""
    return {
        "enabled": imagen_enabled(),
        "model": "imagen-3.0-generate-002",
        "name": "Nano Banana Pro",
        "capabilities": [
            "traffic_heatmap",
            "congestion_3d", 
            "pollution_cloud",
            "flow_arrows",
            "infrastructure",
            "emergency_route",
            "ev_charging",
            "transit_hub"
        ]
    }


@router.post("/generate", response_model=OverlayResponse)
async def generate_overlay(request: OverlayRequest):
    """
    Generate a 3D overlay image using Imagen 3 (Nano Banana Pro).
    
    This creates high-quality visualization overlays that can be
    composited on top of the Azure Maps base layer.
    """
    if not imagen_enabled():
        raise HTTPException(status_code=503, detail="Imagen API not configured")
    
    try:
        generator = create_overlay_generator()
        
        # Map string to enum
        try:
            overlay_enum = OverlayType(request.overlay_type)
        except ValueError:
            overlay_enum = OverlayType.TRAFFIC_HEATMAP
        
        # Build context
        context = {
            "location": request.location,
            "intensity": request.intensity or "moderate"
        }
        
        if request.custom_context:
            context.update(request.custom_context)
        
        # Generate overlay
        result = await generator.generate_overlay(
            overlay_type=overlay_enum,
            context=context,
            aspect_ratio=request.aspect_ratio
        )
        
        if result.success:
            return OverlayResponse(
                success=True,
                image_base64=result.image_base64,
                mime_type=result.mime_type,
                overlay_type=result.overlay_type,
                prompt_used=result.prompt_used
            )
        else:
            return OverlayResponse(
                success=False,
                overlay_type=request.overlay_type,
                error=result.error
            )
            
    except Exception as e:
        logger.error(f"Overlay generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/generate/traffic")
async def generate_traffic_overlay(
    location: str = Query(default="Delhi NCR", description="Location for the overlay"),
    congestion: str = Query(default="moderate", description="Congestion level: low, moderate, high, critical"),
    include_pollution: bool = Query(default=False, description="Include AQI visualization")
):
    """Quick endpoint for traffic heatmap overlays"""
    if not imagen_enabled():
        raise HTTPException(status_code=503, detail="Imagen API not configured")
    
    try:
        generator = create_overlay_generator()
        result = await generator.generate_traffic_overlay(
            location=location,
            congestion_level=congestion,
            include_pollution=include_pollution
        )
        
        if result.success and result.image_base64:
            # Return image directly
            image_bytes = base64.b64decode(result.image_base64)
            return Response(
                content=image_bytes,
                media_type="image/png",
                headers={
                    "X-Overlay-Type": result.overlay_type,
                    "X-Location": location,
                    "Cache-Control": "public, max-age=300"
                }
            )
        else:
            raise HTTPException(status_code=500, detail=result.error or "Generation failed")
            
    except Exception as e:
        logger.error(f"Traffic overlay failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/generate/congestion3d")
async def generate_congestion_3d(
    location: str = Query(default="Delhi NCR"),
    hotspots: Optional[str] = Query(default=None, description="Comma-separated hotspot names")
):
    """Generate 3D congestion bars visualization"""
    if not imagen_enabled():
        raise HTTPException(status_code=503, detail="Imagen API not configured")
    
    try:
        generator = create_overlay_generator()
        hotspot_list = hotspots.split(",") if hotspots else None
        
        result = await generator.generate_congestion_3d(
            location=location,
            hotspots=hotspot_list
        )
        
        if result.success and result.image_base64:
            image_bytes = base64.b64decode(result.image_base64)
            return Response(content=image_bytes, media_type="image/png")
        else:
            raise HTTPException(status_code=500, detail=result.error or "Generation failed")
            
    except Exception as e:
        logger.error(f"Congestion 3D overlay failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/generate/emergency")
async def generate_emergency_overlay(
    route_type: str = Query(default="ambulance", description="Emergency type: ambulance, fire, police"),
    location: str = Query(default="Delhi NCR")
):
    """Generate emergency route overlay"""
    if not imagen_enabled():
        raise HTTPException(status_code=503, detail="Imagen API not configured")
    
    try:
        generator = create_overlay_generator()
        result = await generator.generate_emergency_route(
            route_type=route_type,
            location=location
        )
        
        if result.success and result.image_base64:
            image_bytes = base64.b64decode(result.image_base64)
            return Response(content=image_bytes, media_type="image/png")
        else:
            raise HTTPException(status_code=500, detail=result.error or "Generation failed")
            
    except Exception as e:
        logger.error(f"Emergency overlay failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Cached overlays for common scenarios
CACHED_OVERLAYS = {}

@router.get("/cached/{overlay_key}")
async def get_cached_overlay(overlay_key: str):
    """
    Get a pre-generated cached overlay.
    
    Available keys:
    - delhi_traffic_moderate
    - delhi_traffic_heavy
    - delhi_pollution_high
    - delhi_metro_network
    """
    if overlay_key in CACHED_OVERLAYS:
        data = CACHED_OVERLAYS[overlay_key]
        return Response(
            content=base64.b64decode(data["image"]),
            media_type="image/png",
            headers={"X-Cache": "HIT"}
        )
    
    raise HTTPException(status_code=404, detail=f"Cached overlay '{overlay_key}' not found")


@router.post("/cache/generate")
async def generate_and_cache(overlay_key: str, request: OverlayRequest):
    """Generate an overlay and cache it for fast access"""
    if not imagen_enabled():
        raise HTTPException(status_code=503, detail="Imagen API not configured")
    
    try:
        generator = create_overlay_generator()
        overlay_enum = OverlayType(request.overlay_type)
        
        context = {
            "location": request.location,
            "intensity": request.intensity
        }
        
        result = await generator.generate_overlay(overlay_enum, context)
        
        if result.success and result.image_base64:
            CACHED_OVERLAYS[overlay_key] = {
                "image": result.image_base64,
                "type": result.overlay_type,
                "location": request.location
            }
            return {"success": True, "key": overlay_key, "cached": True}
        else:
            return {"success": False, "error": result.error}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
