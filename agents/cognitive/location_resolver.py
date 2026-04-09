"""Location resolver agent - resolves geographic mentions to coordinates."""

import asyncio
from typing import Dict, Any
from agents.cognitive.roles import AgentContext
from llm.geocoding import geocode


async def resolve_locations(ctx: AgentContext) -> None:
    """Resolve geographic mentions in parsed intent to coordinates.

    Adds resolved_locations and center_point to ctx.parsed_intent.
    """
    # Extract location mentions from parsed intent
    location_mentions = ctx.parsed_intent.get("entities", {}).get("locations", [])

    if not location_mentions:
        # Fallback to city if no specific locations mentioned
        city = ctx.city or "delhi"
        location_mentions = [city]

    resolved_locations = []
    center_point = {"lat": 28.6139, "lon": 77.2090}  # Default to Delhi

    # Resolve each location mention
    for location_name in location_mentions:
        try:
            coords = await geocode(location_name)
            if coords:
                resolved_locations.append({
                    "name": location_name,
                    "lat": coords["lat"],
                    "lon": coords["lon"]
                })
                # Use first resolved location as center point
                if len(resolved_locations) == 1:
                    center_point = {"lat": coords["lat"], "lon": coords["lon"]}
        except Exception as e:
            ctx.log("location_resolver", f"Failed to resolve '{location_name}': {e}", 0)

    # Store in parsed intent for downstream use
    ctx.parsed_intent["resolved_locations"] = resolved_locations
    ctx.parsed_intent["center_point"] = center_point

    ctx.log("location_resolver", f"Resolved {len(resolved_locations)} locations", 0)