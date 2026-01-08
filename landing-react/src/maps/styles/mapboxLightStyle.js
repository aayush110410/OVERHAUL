/**
 * Mapbox Light Street v11 Style
 * Clean light basemap for OVERHAUL Live Demo
 */

export function makeMapboxLightStyle(token) {
  return `mapbox://styles/mapbox/light-v11?access_token=${token}`
}

export function makeMapboxStreetsStyle(token) {
  return `mapbox://styles/mapbox/streets-v12?access_token=${token}`
}

// Direct style URL for use with MapLibre GL (requires Mapbox token)
export const mapboxLightV11 = {
  version: 8,
  name: 'Mapbox Light v11',
  sources: {},
  layers: []
}
