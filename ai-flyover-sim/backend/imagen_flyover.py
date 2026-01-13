"""
Imagen 3 (Nano Banana Pro) integration for flyover visualization.

Generates:
1. Conceptual aerial renders of proposed flyovers
2. Perspective views showing flyover in urban context
3. Realistic textures for 3D models
"""

import os
import base64
from typing import Optional, Dict, Any
import google.generativeai as genai

# Configure Imagen
IMAGEN_API_KEY = os.getenv("IMAGEN_API_KEY") or os.getenv("GEMINI_API_KEY")
IMAGEN_MODEL = "imagen-3.0-generate-002"  # Nano Banana Pro

def is_imagen_available() -> bool:
    """Check if Imagen API is configured."""
    return bool(IMAGEN_API_KEY)

def generate_flyover_concept(
    location_name: str,
    flyover_lanes: int,
    flyover_width_m: float,
    flyover_height_m: float,
    road_type: str,
    view_type: str = "aerial"
) -> Optional[Dict[str, Any]]:
    """
    Generate a conceptual visualization of the proposed flyover.
    
    Args:
        location_name: Name of the location (e.g., "Connaught Place, Delhi")
        flyover_lanes: Number of lanes on the flyover
        flyover_width_m: Width of flyover in meters
        flyover_height_m: Clearance height in meters
        road_type: Type of road (highway, arterial, etc.)
        view_type: "aerial" for bird's eye, "perspective" for street-level view
    
    Returns:
        Dict with 'image_base64' and 'prompt_used'
    """
    if not is_imagen_available():
        return None
    
    try:
        genai.configure(api_key=IMAGEN_API_KEY)
        imagen = genai.ImageGenerationModel(IMAGEN_MODEL)
        
        # Craft detailed architectural prompt
        if view_type == "aerial":
            prompt = f"""Photorealistic aerial drone view of a modern concrete flyover/elevated highway in {location_name}, India. 
The flyover has {flyover_lanes} lanes, approximately {flyover_width_m:.0f} meters wide, elevated {flyover_height_m:.0f} meters above ground.
Features: smooth gray concrete deck with lane markings, sturdy cylindrical pillars spaced evenly, 
metal safety barriers on edges, street lighting poles.
Below: {road_type} road with traffic, urban buildings, trees.
Daytime, clear sky, sharp shadows, high-resolution architectural photography style.
Modern Indian urban infrastructure, clean design."""
        else:
            prompt = f"""Photorealistic street-level perspective view of a modern concrete flyover in {location_name}, India.
Looking up at a {flyover_lanes}-lane elevated highway, {flyover_height_m:.0f} meters clearance height.
Features: massive concrete pillars, smooth deck underside, modern LED street lighting,
clear lane markings visible on the elevated road above.
Ground level shows urban traffic, pedestrians, commercial buildings typical of Indian cities.
Golden hour lighting, professional architectural photography, 8K quality.
Modern Indian infrastructure showcase."""
        
        result = imagen.generate_images(
            prompt=prompt,
            number_of_images=1,
            aspect_ratio="16:9",
            safety_filter_level="block_only_high",
            person_generation="dont_allow"
        )
        
        if result.images and len(result.images) > 0:
            img = result.images[0]
            img_bytes = img._pil_image.tobytes() if hasattr(img, '_pil_image') else None
            
            # Convert to base64
            import io
            buffer = io.BytesIO()
            if hasattr(img, '_pil_image'):
                img._pil_image.save(buffer, format='PNG')
            else:
                # Fallback - try to get raw data
                buffer.write(img._image_bytes if hasattr(img, '_image_bytes') else b'')
            
            return {
                "image_base64": base64.b64encode(buffer.getvalue()).decode('utf-8'),
                "prompt_used": prompt,
                "view_type": view_type
            }
        
        return None
        
    except Exception as e:
        print(f"[Imagen Flyover] Error: {e}")
        return None


def generate_flyover_texture(texture_type: str = "concrete_deck") -> Optional[Dict[str, Any]]:
    """
    Generate realistic textures for 3D flyover models.
    
    Args:
        texture_type: "concrete_deck", "pillar_surface", "barrier_metal"
    
    Returns:
        Dict with 'image_base64' - seamless tileable texture
    """
    if not is_imagen_available():
        return None
    
    try:
        genai.configure(api_key=IMAGEN_API_KEY)
        imagen = genai.ImageGenerationModel(IMAGEN_MODEL)
        
        texture_prompts = {
            "concrete_deck": """Seamless tileable texture of modern concrete road surface.
Gray concrete with subtle aggregate texture, slight weathering, faint tire marks.
Professional architectural texture, high resolution, even lighting, no shadows.
Perfect for 3D model UV mapping. Top-down view.""",
            
            "pillar_surface": """Seamless tileable texture of smooth concrete pillar surface.
Light gray concrete with form-work marks, subtle imperfections, industrial look.
Clean architectural concrete finish, even diffuse lighting, no shadows.
Suitable for 3D cylindrical mapping.""",
            
            "barrier_metal": """Seamless tileable texture of highway crash barrier metal.
Galvanized steel with subtle reflections, industrial coating, slight weathering.
W-beam guardrail texture, professional quality, even lighting.
Tileable for 3D model application."""
        }
        
        prompt = texture_prompts.get(texture_type, texture_prompts["concrete_deck"])
        
        result = imagen.generate_images(
            prompt=prompt,
            number_of_images=1,
            aspect_ratio="1:1",
            safety_filter_level="block_only_high",
            person_generation="dont_allow"
        )
        
        if result.images and len(result.images) > 0:
            img = result.images[0]
            import io
            buffer = io.BytesIO()
            if hasattr(img, '_pil_image'):
                img._pil_image.save(buffer, format='PNG')
            else:
                buffer.write(img._image_bytes if hasattr(img, '_image_bytes') else b'')
            
            return {
                "image_base64": base64.b64encode(buffer.getvalue()).decode('utf-8'),
                "texture_type": texture_type,
                "tileable": True
            }
        
        return None
        
    except Exception as e:
        print(f"[Imagen Texture] Error: {e}")
        return None


def generate_before_after_view(
    location_name: str,
    flyover_spec: Dict[str, Any],
    view: str = "after"
) -> Optional[Dict[str, Any]]:
    """
    Generate before/after comparison views.
    
    Args:
        location_name: Location name
        flyover_spec: Dict with flyover specifications
        view: "before" (current state) or "after" (with flyover)
    
    Returns:
        Dict with visualization
    """
    if not is_imagen_available():
        return None
    
    try:
        genai.configure(api_key=IMAGEN_API_KEY)
        imagen = genai.ImageGenerationModel(IMAGEN_MODEL)
        
        if view == "before":
            prompt = f"""Photorealistic aerial view of {location_name}, India showing current traffic conditions.
Busy intersection with heavy vehicle traffic, congestion visible.
Mix of cars, auto-rickshaws, buses, two-wheelers typical of Indian cities.
Urban setting with commercial buildings, trees, pedestrian activity.
Daytime, clear weather, drone photography style."""
        else:
            prompt = f"""Photorealistic aerial view of {location_name}, India with new modern flyover.
Elevated {flyover_spec.get('flyover_lanes', 2)}-lane highway gracefully spanning over the intersection.
Traffic flowing smoothly both on the flyover and ground level.
Modern concrete construction with safety barriers, street lighting.
Dramatic improvement in traffic flow visible.
Daytime, clear weather, professional infrastructure photography."""
        
        result = imagen.generate_images(
            prompt=prompt,
            number_of_images=1,
            aspect_ratio="16:9",
            safety_filter_level="block_only_high",
            person_generation="dont_allow"
        )
        
        if result.images and len(result.images) > 0:
            img = result.images[0]
            import io
            buffer = io.BytesIO()
            if hasattr(img, '_pil_image'):
                img._pil_image.save(buffer, format='PNG')
            else:
                buffer.write(img._image_bytes if hasattr(img, '_image_bytes') else b'')
            
            return {
                "image_base64": base64.b64encode(buffer.getvalue()).decode('utf-8'),
                "view": view,
                "prompt_used": prompt
            }
        
        return None
        
    except Exception as e:
        print(f"[Imagen Before/After] Error: {e}")
        return None
