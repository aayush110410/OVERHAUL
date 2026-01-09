// Imagen 3D Overlay Integration for MapLibre
// Uses Google's Imagen 3 (Nano Banana Pro) for high-quality traffic visualization

import { API_BASE } from './config'

/**
 * Overlay types supported by Imagen 3
 */
export const OVERLAY_TYPES = {
  TRAFFIC_HEATMAP: 'traffic_heatmap',
  CONGESTION_3D: 'congestion_3d',
  POLLUTION_CLOUD: 'pollution_cloud',
  FLOW_ARROWS: 'flow_arrows',
  INFRASTRUCTURE: 'infrastructure',
  EMERGENCY_ROUTE: 'emergency_route',
  EV_CHARGING: 'ev_charging',
  TRANSIT_HUB: 'transit_hub'
}

/**
 * Intensity levels for overlays
 */
export const INTENSITY_LEVELS = {
  LOW: 'low',
  MODERATE: 'moderate',
  HIGH: 'high',
  CRITICAL: 'critical'
}

/**
 * Check if Imagen overlay API is available
 */
export async function checkImagenStatus() {
  try {
    const resp = await fetch(`${API_BASE}/imagen/status`)
    if (!resp.ok) return { enabled: false, error: 'API not available' }
    return await resp.json()
  } catch (e) {
    return { enabled: false, error: e.message }
  }
}

/**
 * Generate a 3D overlay image
 * @param {Object} options - Generation options
 * @param {string} options.overlayType - Type from OVERLAY_TYPES
 * @param {string} options.location - Location name (e.g., "Delhi NCR")
 * @param {string} options.intensity - Intensity from INTENSITY_LEVELS
 * @param {string} options.aspectRatio - Aspect ratio (default: "16:9")
 * @param {boolean} options.includePollution - Include pollution visualization
 * @returns {Promise<{success: boolean, imageUrl?: string, error?: string}>}
 */
export async function generateOverlay({
  overlayType = OVERLAY_TYPES.TRAFFIC_HEATMAP,
  location = 'Delhi NCR',
  intensity = INTENSITY_LEVELS.MODERATE,
  aspectRatio = '16:9',
  includePollution = false,
  customContext = null
}) {
  try {
    const resp = await fetch(`${API_BASE}/imagen/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        overlay_type: overlayType,
        location,
        intensity,
        aspect_ratio: aspectRatio,
        include_pollution: includePollution,
        custom_context: customContext
      })
    })

    if (!resp.ok) {
      const error = await resp.text()
      return { success: false, error }
    }

    const data = await resp.json()
    
    if (data.success && data.image_base64) {
      // Convert base64 to data URL
      const imageUrl = `data:${data.mime_type};base64,${data.image_base64}`
      return {
        success: true,
        imageUrl,
        overlayType: data.overlay_type,
        promptUsed: data.prompt_used
      }
    }
    
    return { success: false, error: data.error || 'Unknown error' }
  } catch (e) {
    return { success: false, error: e.message }
  }
}

/**
 * Quick traffic overlay generation
 */
export async function generateTrafficOverlay(location = 'Delhi NCR', congestion = 'moderate') {
  try {
    const resp = await fetch(
      `${API_BASE}/imagen/generate/traffic?location=${encodeURIComponent(location)}&congestion=${congestion}`
    )
    
    if (!resp.ok) {
      return { success: false, error: await resp.text() }
    }
    
    // Response is direct image
    const blob = await resp.blob()
    const imageUrl = URL.createObjectURL(blob)
    
    return { success: true, imageUrl, overlayType: 'traffic_heatmap' }
  } catch (e) {
    return { success: false, error: e.message }
  }
}

/**
 * Add an overlay image to a MapLibre map as an image layer
 * @param {maplibregl.Map} map - The MapLibre map instance
 * @param {string} imageUrl - Data URL or blob URL of the overlay image
 * @param {Object} bounds - Geographic bounds { north, south, east, west }
 * @param {string} layerId - Unique layer ID (default: 'imagen-overlay')
 */
export function addOverlayToMap(map, imageUrl, bounds, layerId = 'imagen-overlay') {
  // Remove existing overlay if present
  if (map.getLayer(layerId)) {
    map.removeLayer(layerId)
  }
  if (map.getSource(layerId)) {
    map.removeSource(layerId)
  }

  // Add image source with geographic coordinates
  map.addSource(layerId, {
    type: 'image',
    url: imageUrl,
    coordinates: [
      [bounds.west, bounds.north],  // top-left
      [bounds.east, bounds.north],  // top-right
      [bounds.east, bounds.south],  // bottom-right
      [bounds.west, bounds.south]   // bottom-left
    ]
  })

  // Add raster layer for the overlay
  map.addLayer({
    id: layerId,
    type: 'raster',
    source: layerId,
    paint: {
      'raster-opacity': 0.75,
      'raster-fade-duration': 500
    }
  })

  return layerId
}

/**
 * Update overlay opacity
 */
export function setOverlayOpacity(map, layerId, opacity) {
  if (map.getLayer(layerId)) {
    map.setPaintProperty(layerId, 'raster-opacity', opacity)
  }
}

/**
 * Remove overlay from map
 */
export function removeOverlay(map, layerId = 'imagen-overlay') {
  if (map.getLayer(layerId)) {
    map.removeLayer(layerId)
  }
  if (map.getSource(layerId)) {
    map.removeSource(layerId)
  }
}

/**
 * Delhi NCR bounds for overlay positioning
 */
export const DELHI_NCR_BOUNDS = {
  north: 28.85,
  south: 28.40,
  east: 77.55,
  west: 76.85
}

/**
 * Generate and display overlay on map (convenience function)
 */
export async function showTrafficOverlayOnMap(map, options = {}) {
  const {
    location = 'Delhi NCR',
    congestion = 'moderate',
    bounds = DELHI_NCR_BOUNDS,
    layerId = 'traffic-overlay'
  } = options

  // Generate the overlay
  const result = await generateTrafficOverlay(location, congestion)
  
  if (!result.success) {
    console.error('Failed to generate overlay:', result.error)
    return { success: false, error: result.error }
  }

  // Add to map
  addOverlayToMap(map, result.imageUrl, bounds, layerId)
  
  return { success: true, layerId }
}
