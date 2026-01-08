// Azure Maps raster tiles (Microsoft) for the live demo.
// Uses the backend as a proxy so we don't expose keys client-side.
// Backend endpoint: GET /azure/maps/tile

export function makeAzureMapsLightStyle(apiBase) {
  const base = String(apiBase || '').replace(/\/$/, '')

  return {
    version: 8,
    sources: {
      'azure-base-light': {
        type: 'raster',
        tiles: [
          `${base}/azure/maps/tile?tilesetId=microsoft.base.road&zoom={z}&x={x}&y={y}&tileSize=256`,
        ],
        tileSize: 256,
        attribution: '&copy; Microsoft',
      },
    },
    layers: [
      {
        id: 'azure-base-light-layer',
        type: 'raster',
        source: 'azure-base-light',
        minzoom: 0,
        maxzoom: 22,
      },
    ],
  }
}
