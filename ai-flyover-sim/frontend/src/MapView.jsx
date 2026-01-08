import { useEffect, useRef } from 'react'
import mapboxgl from 'mapbox-gl'
import { useAppStore } from './store'
import { addFlyoverLayer } from './FlyoverLayer'
import './map.css'

mapboxgl.accessToken = import.meta.env.VITE_MAPBOX_TOKEN || ''

export default function MapView() {
  const mapRef = useRef(null)
  const mapLoaded = useRef(false)
  const flyover = useAppStore((s) => s.flyover)

  useEffect(() => {
    if (mapRef.current) return
    
    try {
      const map = new mapboxgl.Map({
        container: 'map-container',
        style: 'mapbox://styles/mapbox/streets-v12',
        center: [77.22, 28.63],
        zoom: 14,
        pitch: 45,
        bearing: -20,
      })

      map.on('load', () => {
        mapLoaded.current = true
        
        if (flyover) {
          addFlyoverLayer(map, flyover)
        }
      })

      map.on('error', (e) => {
        console.error('Map error:', e)
      })

      mapRef.current = map
    } catch (err) {
      console.error('Failed to create map:', err)
    }
  }, [])

  useEffect(() => {
    if (!mapRef.current || !flyover) return
    
    if (mapLoaded.current) {
      addFlyoverLayer(mapRef.current, flyover)
    } else {
      mapRef.current.on('load', () => {
        addFlyoverLayer(mapRef.current, flyover)
      })
    }
  }, [flyover])

  return <div id="map-container" className="map-container" />
}
