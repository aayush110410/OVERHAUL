
import { useState, useEffect, useRef, useCallback } from 'react'
import { Link, useLocation, useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import mapboxgl from 'mapbox-gl'
import 'mapbox-gl/dist/mapbox-gl.css'
import './App.css'
// Import the advanced 3D flyover agent
import { addFlyoverLayer, removeFlyoverLayer } from './FlyoverLayer'
import { apiFetchJson, API_BASE } from './api/config'
import { cartoLightOsmStyle } from './maps/styles/cartoLightOsmStyle'

const MAPBOX_TOKEN = (import.meta.env.VITE_MAPBOX_TOKEN || '').trim()
mapboxgl.accessToken = MAPBOX_TOKEN

// ============================================
// OV LOADER (Same style as Contact page - no bar)
// Zooms IN to transition - FAST VERSION
// ============================================
function OVLoader({ onComplete }) {
  const [phase, setPhase] = useState('zoomOut') // zoomOut -> hold -> zoomIn -> done
  
  useEffect(() => {
    // Phase 1: Zoom out from large scale
    const holdTimer = setTimeout(() => {
      setPhase('hold')
    }, 400)
    
    return () => clearTimeout(holdTimer)
  }, [])
  
  useEffect(() => {
    if (phase === 'hold') {
      // Phase 2: Brief hold at normal size
      const zoomInTimer = setTimeout(() => {
        setPhase('zoomIn')
      }, 250)
      return () => clearTimeout(zoomInTimer)
    }
  }, [phase])
  
  useEffect(() => {
    if (phase === 'zoomIn') {
      // Phase 3: Zoom in and fade out
      const completeTimer = setTimeout(() => {
        onComplete()
      }, 500)
      return () => clearTimeout(completeTimer)
    }
  }, [phase, onComplete])

  return (
    <motion.div 
      className="loader-ln"
      initial={{ opacity: 0 }}
      animate={{ opacity: phase === 'zoomIn' ? 0 : 1 }}
      transition={{ 
        duration: phase === 'zoomIn' ? 0.3 : 0.2, 
        ease: [0.4, 0, 0.2, 1],
        delay: phase === 'zoomIn' ? 0.25 : 0
      }}
    >
      <div className="loader-ln-content">
        <motion.div 
          className="loader-ln-logo"
          initial={{ scale: 50, opacity: 0 }}
          animate={{
            scale: phase === 'zoomOut' ? 1 : phase === 'hold' ? 1 : 50,
            opacity: 1
          }}
          transition={{
            duration: phase === 'zoomOut' ? 0.4 : phase === 'zoomIn' ? 0.5 : 0.1,
            ease: [0.4, 0, 0.2, 1]
          }}
        >
          <motion.span 
            className="loader-ln-text-o"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.2, delay: 0.15, ease: [0.4, 0, 0.2, 1] }}
          >
            O
          </motion.span>
          <motion.span 
            className="loader-ln-text-v"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.2, delay: 0.2, ease: [0.4, 0, 0.2, 1] }}
          >
            V
          </motion.span>
        </motion.div>
      </div>
    </motion.div>
  )
}

// ============================================
// EXIT LOADER - Zoom Out then Zoom In
// ============================================
function ExitLoader({ onComplete }) {
  const [phase, setPhase] = useState('zoomOut')
  
  useEffect(() => {
    const holdTimer = setTimeout(() => {
      setPhase('hold')
    }, 400)
    return () => clearTimeout(holdTimer)
  }, [])
  
  useEffect(() => {
    if (phase === 'hold') {
      const zoomInTimer = setTimeout(() => {
        setPhase('zoomIn')
      }, 200)
      return () => clearTimeout(zoomInTimer)
    }
  }, [phase])
  
  useEffect(() => {
    if (phase === 'zoomIn') {
      const completeTimer = setTimeout(() => {
        onComplete()
      }, 500)
      return () => clearTimeout(completeTimer)
    }
  }, [phase, onComplete])

  return (
    <motion.div 
      className="loader-ln"
      initial={{ opacity: 0 }}
      animate={{ opacity: phase === 'zoomIn' ? 0 : 1 }}
      transition={{ 
        duration: phase === 'zoomIn' ? 0.3 : 0.2, 
        ease: [0.76, 0, 0.24, 1],
        delay: phase === 'zoomIn' ? 0.25 : 0
      }}
    >
      <div className="loader-ln-content">
        <motion.div 
          className="loader-ln-logo"
          initial={{ scale: 50, opacity: 0 }}
          animate={{
            scale: phase === 'zoomOut' ? 1 : phase === 'hold' ? 1 : 50,
            opacity: 1
          }}
          transition={{
            duration: phase === 'zoomOut' ? 0.4 : phase === 'zoomIn' ? 0.5 : 0.1,
            ease: [0.76, 0, 0.24, 1]
          }}
        >
          <motion.span className="loader-ln-text-o" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.15, delay: 0.15 }}>O</motion.span>
          <motion.span className="loader-ln-text-v" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.15, delay: 0.2 }}>V</motion.span>
        </motion.div>
      </div>
    </motion.div>
  )
}

// Custom Cursor
function CustomCursor() {
  const cursorRef = useRef(null)
  const [isHovering, setIsHovering] = useState(false)
  const mousePos = useRef({ x: 0, y: 0 })
  const rafId = useRef(null)

  useEffect(() => {
    const updateCursor = () => {
      if (cursorRef.current) {
        cursorRef.current.style.transform = `translate3d(${mousePos.current.x}px, ${mousePos.current.y}px, 0) translate(-50%, -50%)`
      }
      rafId.current = requestAnimationFrame(updateCursor)
    }
    rafId.current = requestAnimationFrame(updateCursor)

    const moveCursor = (e) => { mousePos.current = { x: e.clientX, y: e.clientY } }
    const handleMouseOver = (e) => {
      if (e.target.tagName === 'A' || e.target.tagName === 'BUTTON' || e.target.closest('a') || e.target.closest('button') || e.target.closest('input')) {
        setIsHovering(true)
      }
    }
    const handleMouseOut = (e) => {
      if (e.target.tagName === 'A' || e.target.tagName === 'BUTTON' || e.target.closest('a') || e.target.closest('button') || e.target.closest('input')) {
        setIsHovering(false)
      }
    }

    document.addEventListener('mousemove', moveCursor, { passive: true })
    document.addEventListener('mouseover', handleMouseOver, { passive: true })
    document.addEventListener('mouseout', handleMouseOut, { passive: true })

    return () => {
      if (rafId.current) cancelAnimationFrame(rafId.current)
      document.removeEventListener('mousemove', moveCursor)
      document.removeEventListener('mouseover', handleMouseOver)
      document.removeEventListener('mouseout', handleMouseOut)
    }
  }, [])

  return <div ref={cursorRef} className={`cursor ${isHovering ? 'hovering' : ''}`} />
}

// Geocode place name to coordinates (with India bias)
async function geocode(place) {
  const raw = (place || '').trim()
  if (!raw) throw new Error('Location is required')

  // Prefer Nominatim via backend (no client-side key)
  try {
    const q = raw.toLowerCase().includes('india') ? raw : `${raw}, India`
    const data = await apiFetchJson(`/geocode?query=${encodeURIComponent(q)}&limit=1`)
    const results = Array.isArray(data?.results) ? data.results : []
    const r = results[0]
    if (r?.lat != null && r?.lon != null) {
      return {
        coords: [parseFloat(r.lon), parseFloat(r.lat)],
        name: r.display_name || raw
      }
    }
  } catch {
    // fall back
  }

  if (!MAPBOX_TOKEN) {
    throw new Error(`Geocoding unavailable. Configure VITE_MAPBOX_TOKEN or backend geocoding at ${API_BASE}.`)
  }

  const searchQuery = raw.toLowerCase().includes('india') ? raw : `${raw}, India`
  const url = `https://api.mapbox.com/geocoding/v5/mapbox.places/${encodeURIComponent(searchQuery)}.json?access_token=${MAPBOX_TOKEN}&limit=1&types=poi,address,place,locality,neighborhood`
  const res = await fetch(url)
  const data = await res.json()
  if (data.features && data.features.length > 0) {
    return {
      coords: data.features[0].center,
      name: data.features[0].place_name
    }
  }
  throw new Error(`Could not find location: ${raw}`)
}

// Reverse geocode coords to place name
async function reverseGeocode(lng, lat) {
  // Prefer Nominatim via backend
  try {
    const data = await apiFetchJson(`/reverse-geocode?lat=${encodeURIComponent(lat)}&lon=${encodeURIComponent(lng)}`)
    if (data?.display_name) return data.display_name
  } catch {
    // fall back
  }

  if (MAPBOX_TOKEN) {
    const url = `https://api.mapbox.com/geocoding/v5/mapbox.places/${lng},${lat}.json?access_token=${MAPBOX_TOKEN}&limit=1`
    const res = await fetch(url)
    const data = await res.json()
    if (data.features && data.features.length > 0) {
      return data.features[0].place_name
    }
  }
  return `${lat.toFixed(4)}, ${lng.toFixed(4)}`
}

// Get driving route
async function getRoute(start, end) {
  const url = `https://api.mapbox.com/directions/v5/mapbox/driving-traffic/${start[0]},${start[1]};${end[0]},${end[1]}?geometries=geojson&overview=full&access_token=${MAPBOX_TOKEN}`
  const res = await fetch(url)
  const data = await res.json()
  if (data.routes && data.routes.length > 0) {
    return {
      geometry: data.routes[0].geometry,
      distance: data.routes[0].distance,
      duration: data.routes[0].duration
    }
  }
  throw new Error('Could not find route')
}

// Generate flyover path points
function generateFlyoverPath(routeCoords, numPoints = 50) {
  const points = []
  const totalLength = routeCoords.length
  for (let i = 0; i < numPoints; i++) {
    const t = i / (numPoints - 1)
    const idx = t * (totalLength - 1)
    const lowIdx = Math.floor(idx)
    const highIdx = Math.min(lowIdx + 1, totalLength - 1)
    const frac = idx - lowIdx
    const lng = routeCoords[lowIdx][0] + frac * (routeCoords[highIdx][0] - routeCoords[lowIdx][0])
    const lat = routeCoords[lowIdx][1] + frac * (routeCoords[highIdx][1] - routeCoords[lowIdx][1])
    points.push([lng, lat])
  }
  return points
}

// Generate corridor polygon for 3D extrusion
function generateCorridorPolygon(path, widthMeters = 20) {
  const coords = []
  const widthDeg = widthMeters / 111000
  for (let i = 0; i < path.length; i++) {
    const [lng, lat] = path[i]
    let angle = 0
    if (i < path.length - 1) {
      angle = Math.atan2(path[i + 1][1] - lat, path[i + 1][0] - lng)
    } else if (i > 0) {
      angle = Math.atan2(lat - path[i - 1][1], lng - path[i - 1][0])
    }
    const perpAngle = angle + Math.PI / 2
    coords.push([lng + Math.cos(perpAngle) * widthDeg, lat + Math.sin(perpAngle) * widthDeg])
  }
  for (let i = path.length - 1; i >= 0; i--) {
    const [lng, lat] = path[i]
    let angle = 0
    if (i < path.length - 1) {
      angle = Math.atan2(path[i + 1][1] - lat, path[i + 1][0] - lng)
    } else if (i > 0) {
      angle = Math.atan2(lat - path[i - 1][1], lng - path[i - 1][0])
    }
    const perpAngle = angle - Math.PI / 2
    coords.push([lng + Math.cos(perpAngle) * widthDeg, lat + Math.sin(perpAngle) * widthDeg])
  }
  coords.push(coords[0])
  return coords
}

// Generate pillar positions
function generatePillars(path, spacing = 8) {
  const pillars = []
  for (let i = 0; i < path.length; i += spacing) {
    pillars.push({
      type: 'Feature',
      properties: { height: 12 },
      geometry: { type: 'Point', coordinates: path[i] }
    })
  }
  return pillars
}

export default function FlyoverSim() {
  const location = useLocation()
  const navigate = useNavigate()
  const skipLoader = location.state?.skipLoader || false
  
  const [pageLoading, setPageLoading] = useState(!skipLoader)
  const [exiting, setExiting] = useState(false)
  const [pendingNavigation, setPendingNavigation] = useState(null)

  const mapContainer = useRef(null)
  const mapRef = useRef(null)
  const markersRef = useRef([])
  const userMarkerRef = useRef(null)
  
  const [mapLoaded, setMapLoaded] = useState(false)
  const [fromLocation, setFromLocation] = useState('')
  const [toLocation, setToLocation] = useState('')
  const [searchQuery, setSearchQuery] = useState('')
  const [isProcessing, setIsProcessing] = useState(false)
  const [isLocating, setIsLocating] = useState(false)
  const [userCoords, setUserCoords] = useState(null)
  const [stats, setStats] = useState(null)
  const [analysisLog, setAnalysisLog] = useState([])

  const addLog = useCallback((message, type = 'info') => {
    setAnalysisLog(prev => [...prev, { message, type, time: new Date().toLocaleTimeString() }])
  }, [])

  // Handle browser back/forward buttons
  useEffect(() => {
    window.history.pushState({ skipLoader: true }, '', window.location.href)
    const handlePopState = () => {
      setExiting(true)
      setPendingNavigation('/demo')
    }
    window.addEventListener('popstate', handlePopState)
    return () => window.removeEventListener('popstate', handlePopState)
  }, [])

  useEffect(() => {
    document.title = 'OVERHAUL | 3D Flyover'
  }, [])

  const handleBackDemo = (e) => {
    e.preventDefault()
    setExiting(true)
    setPendingNavigation('/demo')
  }

  const handleBackHome = (e) => {
    e.preventDefault()
    setExiting(true)
    setPendingNavigation('/')
  }

  const handleExitComplete = () => {
    navigate(pendingNavigation || '/demo', { state: { skipLoader: true } })
  }

  if (!MAPBOX_TOKEN) {
    return (
      <>
        <CustomCursor />
        <AnimatePresence mode="wait">
          {pageLoading && <OVLoader key="entry-loader" onComplete={() => setPageLoading(false)} />}
        </AnimatePresence>
        <AnimatePresence mode="wait">
          {exiting && <ExitLoader key="exit-loader" onComplete={handleExitComplete} />}
        </AnimatePresence>
        {!pageLoading && !exiting && (
          <div className="contact-page">
            <div className="contact-nav">
              <a href="/" onClick={handleBackHome} className="nav-logo">OVERHAUL™</a>
              <a href="/demo" onClick={handleBackDemo} className="back-btn">← BACK</a>
            </div>
            <div className="contact-content">
              <div className="contact-header">
                <span className="contact-label">CONFIG REQUIRED</span>
                <h1 className="contact-title">Missing Mapbox Token</h1>
                <p className="contact-subtitle">
                  Set <span style={{ color: 'var(--orange)' }}>VITE_MAPBOX_TOKEN</span> in your <code>.env</code> to run the flyover map.
                </p>
              </div>
            </div>
          </div>
        )}
      </>
    )
  }

  // Initialize map
  useEffect(() => {
    if (pageLoading || mapRef.current || !mapContainer.current) return

    let cancelled = false
    let didFallback = false

    const attachOnLoad = () => {
      if (!mapRef.current) return

      mapRef.current.on('load', () => {
        if (!mapRef.current) return
        setMapLoaded(true)

        // Add empty sources for visualization
        mapRef.current.addSource('flyover-corridor', {
          type: 'geojson',
          data: { type: 'FeatureCollection', features: [] }
        })
        mapRef.current.addSource('ground-route', {
          type: 'geojson',
          data: { type: 'FeatureCollection', features: [] }
        })
        mapRef.current.addSource('flyover-pillars', {
          type: 'geojson',
          data: { type: 'FeatureCollection', features: [] }
        })

        // Ground route (dashed)
        mapRef.current.addLayer({
          id: 'ground-route-line',
          type: 'line',
          source: 'ground-route',
          paint: {
            'line-color': '#ff4d00',
            'line-width': 4,
            'line-opacity': 0.7,
            'line-dasharray': [2, 2]
          }
        })

        // Flyover corridor (3D)
        mapRef.current.addLayer({
          id: 'flyover-corridor-fill',
          type: 'fill-extrusion',
          source: 'flyover-corridor',
          paint: {
            'fill-extrusion-color': '#CCFF00',
            'fill-extrusion-height': 15,
            'fill-extrusion-base': 10,
            'fill-extrusion-opacity': 0.85
          }
        })

        // Pillars
        mapRef.current.addLayer({
          id: 'flyover-pillars-circles',
          type: 'circle',
          source: 'flyover-pillars',
          paint: {
            'circle-radius': 5,
            'circle-color': '#ff4d00',
            'circle-stroke-width': 2,
            'circle-stroke-color': '#CCFF00'
          }
        })

        addLog('L-DRAGO: System initialized', 'system')
        addLog('Map ready. Enter locations to simulate flyover.', 'info')
      })
    }

    const rebuildWithFallback = () => {
      if (didFallback || cancelled || !mapContainer.current) return
      didFallback = true

      try {
        mapRef.current?.remove()
      } catch {
        // ignore
      }
      mapRef.current = null

      mapRef.current = new mapboxgl.Map({
        container: mapContainer.current,
        style: cartoLightOsmStyle,
        center: [77.38, 28.62],
        zoom: 12,
        pitch: 45,
        bearing: -10
      })
      mapRef.current.addControl(new mapboxgl.NavigationControl(), 'top-right')
      attachOnLoad()
    }

    ;(async () => {
      // Preflight Mapbox token: if unauthorized/unreachable, fall back to Carto Light OSM
      let style = 'mapbox://styles/mapbox/light-v11'
      try {
        const controller = new AbortController()
        const timeout = setTimeout(() => controller.abort(), 3500)
        const resp = await fetch(
          `https://api.mapbox.com/styles/v1/mapbox/light-v11?access_token=${encodeURIComponent(MAPBOX_TOKEN)}`,
          { signal: controller.signal }
        )
        clearTimeout(timeout)
        if (!resp.ok) style = cartoLightOsmStyle
      } catch {
        style = cartoLightOsmStyle
      }

      if (cancelled || mapRef.current || !mapContainer.current) return

      mapRef.current = new mapboxgl.Map({
        container: mapContainer.current,
        style,
        center: [77.38, 28.62],
        zoom: 12,
        pitch: 45,
        bearing: -10
      })

      // If the container was animated/hidden during mount, Mapbox can render a blank canvas.
      // A resize on the next tick reliably fixes this.
      setTimeout(() => mapRef.current?.resize(), 0)
      mapRef.current.addControl(new mapboxgl.NavigationControl(), 'top-right')

      // If Mapbox style fails at runtime (e.g. 401), rebuild once with fallback.
      mapRef.current.on('error', rebuildWithFallback)

      attachOnLoad()
    })()

    return () => {
      cancelled = true
      try {
        mapRef.current?.remove()
      } catch {
        // ignore
      }
      mapRef.current = null
    }
  }, [addLog])

  // Get user's current location
  const getUserLocation = async () => {
    if (!navigator.geolocation) {
      addLog('Geolocation not supported', 'error')
      return
    }

    setIsLocating(true)
    addLog('Detecting your location...', 'info')

    navigator.geolocation.getCurrentPosition(
      async (position) => {
        const { longitude, latitude } = position.coords
        setUserCoords([longitude, latitude])
        
        // Reverse geocode
        try {
          const placeName = await reverseGeocode(longitude, latitude)
          setFromLocation(placeName.split(',')[0]) // Use short name
          addLog(`Located: ${placeName.split(',')[0]}`, 'success')
        } catch {
          setFromLocation(`${latitude.toFixed(4)}, ${longitude.toFixed(4)}`)
        }

        // Add/update user marker
        if (userMarkerRef.current) userMarkerRef.current.remove()
        
        const el = document.createElement('div')
        el.className = 'user-location-marker'
        el.innerHTML = '<div class="pulse"></div><div class="dot"></div>'
        
        userMarkerRef.current = new mapboxgl.Marker({ element: el })
          .setLngLat([longitude, latitude])
          .addTo(mapRef.current)

        mapRef.current.flyTo({ center: [longitude, latitude], zoom: 14, pitch: 45 })
        setIsLocating(false)
      },
      (error) => {
        addLog(`Location error: ${error.message}`, 'error')
        setIsLocating(false)
      },
      { enableHighAccuracy: true, timeout: 10000 }
    )
  }

  // Search and go to location
  const searchLocation = async () => {
    if (!searchQuery.trim()) return
    
    try {
      addLog(`Searching: ${searchQuery}`, 'info')
      const result = await geocode(searchQuery)
      mapRef.current.flyTo({ center: result.coords, zoom: 15, pitch: 45 })
      addLog(`Found: ${result.name}`, 'success')
      setSearchQuery('')
    } catch (err) {
      addLog(err.message, 'error')
    }
  }

  // Clear visualization
  const clearVisualization = () => {
    markersRef.current.forEach(m => m.remove())
    markersRef.current = []
    if (mapRef.current.getSource('flyover-corridor')) {
      mapRef.current.getSource('flyover-corridor').setData({ type: 'FeatureCollection', features: [] })
    }
    if (mapRef.current.getSource('ground-route')) {
      mapRef.current.getSource('ground-route').setData({ type: 'FeatureCollection', features: [] })
    }
    if (mapRef.current.getSource('flyover-pillars')) {
      mapRef.current.getSource('flyover-pillars').setData({ type: 'FeatureCollection', features: [] })
    }
  }

  // Simulate flyover
  const simulateFlyover = async () => {
    if (!fromLocation || !toLocation) {
      addLog('Please enter both origin and destination', 'error')
      return
    }

    setIsProcessing(true)
    setStats(null)
    setAnalysisLog([])
    clearVisualization()

    try {
      // Phase 1: Geocoding
      addLog('L-DRAGO: Initializing spatial analysis...', 'system')
      addLog(`Geocoding origin: ${fromLocation}`, 'info')
      
      const startResult = await geocode(fromLocation)
      addLog(`Origin resolved: ${startResult.name}`, 'success')
      
      addLog(`Geocoding destination: ${toLocation}`, 'info')
      const endResult = await geocode(toLocation)
      addLog(`Destination resolved: ${endResult.name}`, 'success')

      // Add markers
      const startMarker = new mapboxgl.Marker({ color: '#CCFF00' })
        .setLngLat(startResult.coords)
        .setPopup(new mapboxgl.Popup().setHTML(`<b>Origin</b><br>${startResult.name}`))
        .addTo(mapRef.current)
      
      const endMarker = new mapboxgl.Marker({ color: '#ff4d00' })
        .setLngLat(endResult.coords)
        .setPopup(new mapboxgl.Popup().setHTML(`<b>Destination</b><br>${endResult.name}`))
        .addTo(mapRef.current)
      
      markersRef.current.push(startMarker, endMarker)

      // Phase 2: Routing
      addLog('L-DRAGO: Computing optimal route...', 'system')
      await new Promise(r => setTimeout(r, 400))
      
      const route = await getRoute(startResult.coords, endResult.coords)
      const routeCoords = route.geometry.coordinates
      
      addLog(`Route: ${(route.distance / 1000).toFixed(2)} km, ${(route.duration / 60).toFixed(1)} min`, 'success')

      // Show ground route
      mapRef.current.getSource('ground-route').setData({
        type: 'Feature',
        geometry: { type: 'LineString', coordinates: routeCoords }
      })


      // Phase 3: Generate flyover (Three.js 3D agent)
      addLog('L-DRAGO: Generating advanced 3D flyover...', 'system')
      await new Promise(r => setTimeout(r, 500))

      // Remove any previous 3D flyover
      removeFlyoverLayer(mapRef.current)

      // Prepare flyover data for the agent
      const flyoverData = {
        flyover_path: routeCoords,
        flyover_width_m: 18,
        flyover_height_m: 15,
        pillars: [], // let the agent auto-place pillars
      }
      addFlyoverLayer(mapRef.current, flyoverData)
      addLog('3D flyover rendered with advanced geometry', 'success')

      // Fit bounds
      const bounds = new mapboxgl.LngLatBounds()
      routeCoords.forEach(c => bounds.extend(c))
      mapRef.current.fitBounds(bounds, { padding: 80, pitch: 50, bearing: -15 })

      // Phase 4: Impact stats
      addLog('L-DRAGO: Calculating traffic impact...', 'system')
      await new Promise(r => setTimeout(r, 400))

      const distKm = route.distance / 1000
      const originalTime = route.duration / 60
      const timeSaved = Math.round(originalTime * 0.35)
      const congestionReduction = Math.round(25 + Math.random() * 20)
      const vehiclesPerHour = Math.round(2500 + Math.random() * 1500)
      const co2Reduction = Math.round(distKm * 18)

      setStats({
        distance: distKm.toFixed(2),
        timeSaved,
        congestionReduction,
        vehiclesPerHour,
        pillars: pillars.length,
        co2Reduction
      })

      addLog('L-DRAGO: SIMULATION COMPLETE ✓', 'success')

    } catch (err) {
      addLog(`Error: ${err.message}`, 'error')
    } finally {
      setIsProcessing(false)
    }
  }

  return (
    <>
      <CustomCursor />
      
      <AnimatePresence mode="wait">
        {pageLoading && <OVLoader key="entry-loader" onComplete={() => setPageLoading(false)} />}
      </AnimatePresence>
      
      <AnimatePresence mode="wait">
        {exiting && <ExitLoader key="exit-loader" onComplete={handleExitComplete} />}
      </AnimatePresence>
      
      <AnimatePresence>
        {!pageLoading && !exiting && (
    <motion.div 
      className="flyover-page"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.5 }}
    >
      {/* Navbar */}
      <nav className="flyover-navbar">
        <a href="/" onClick={handleBackHome} className="flyover-logo">OVERHAUL</a>
        <div className="flyover-nav-center">
          <span className="flyover-nav-dot"></span>
          <span>3D FLYOVER SIMULATOR</span>
          <span className="flyover-nav-tag">L-DRAGO</span>
        </div>
        <a href="/demo" onClick={handleBackDemo} className="flyover-back-btn">← BACK TO DEMO</a>
      </nav>

      <main className="flyover-main">
        {/* Map Section */}
        <section className="flyover-map-section">
          <div className="flyover-map-card">
            <div className="flyover-card-header">
              <span>3D FLYOVER VISUALIZATION</span>
              {!mapLoaded && <span className="flyover-loading-tag">LOADING</span>}
            </div>
            
            <div className="flyover-map-wrapper">
              <div ref={mapContainer} className="flyover-map" />
              
              {/* Search Bar on Map */}
              <div className="map-search-bar">
                <input
                  type="text"
                  placeholder="Search location..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && searchLocation()}
                />
                <button onClick={searchLocation}>🔍</button>
              </div>

              {/* My Location Button */}
              <button 
                className="my-location-btn" 
                onClick={getUserLocation}
                disabled={isLocating}
                title="My Location"
              >
                {isLocating ? (
                  <span className="loc-spinner"></span>
                ) : (
                  <svg viewBox="0 0 24 24" fill="currentColor" width="20" height="20">
                    <path d="M12 8c-2.21 0-4 1.79-4 4s1.79 4 4 4 4-1.79 4-4-1.79-4-4-4zm8.94 3A8.994 8.994 0 0013 3.06V1h-2v2.06A8.994 8.994 0 003.06 11H1v2h2.06A8.994 8.994 0 0011 20.94V23h2v-2.06A8.994 8.994 0 0020.94 13H23v-2h-2.06zM12 19c-3.87 0-7-3.13-7-7s3.13-7 7-7 7 3.13 7 7-3.13 7-7 7z"/>
                  </svg>
                )}
              </button>
            </div>
          </div>
        </section>

        {/* Control Panel */}
        <aside className="flyover-control-panel">
          {/* Input Section */}
          <div className="flyover-input-section">
            <div className="flyover-card-header">
              <span>FLYOVER PLANNER</span>
              <span className="flyover-tag">INPUT</span>
            </div>

            <div className="flyover-inputs">
              <div className="flyover-input-group">
                <label>ORIGIN</label>
                <div className="input-with-btn">
                  <input
                    type="text"
                    placeholder="Enter starting point"
                    value={fromLocation}
                    onChange={(e) => setFromLocation(e.target.value)}
                  />
                  <button 
                    className="use-location-btn" 
                    onClick={getUserLocation}
                    disabled={isLocating}
                    title="Use my location"
                  >
                    📍
                  </button>
                </div>
              </div>

              <div className="flyover-input-group">
                <label>DESTINATION</label>
                <input
                  type="text"
                  placeholder="Enter destination"
                  value={toLocation}
                  onChange={(e) => setToLocation(e.target.value)}
                />
              </div>

              <button 
                className="simulate-btn"
                onClick={simulateFlyover}
                disabled={isProcessing || !fromLocation || !toLocation}
              >
                {isProcessing ? (
                  <span className="btn-spinner"></span>
                ) : (
                  'SIMULATE FLYOVER'
                )}
              </button>
            </div>
          </div>

          {/* Analysis Log */}
          <div className="flyover-log-section">
            <div className="flyover-card-header">
              <span>L-DRAGO ANALYSIS</span>
              {analysisLog.some(l => l.type === 'error') && <span className="flyover-tag error">ERROR</span>}
            </div>
            <div className="flyover-log-content">
              {analysisLog.length === 0 ? (
                <div className="log-empty">Awaiting simulation...</div>
              ) : (
                analysisLog.map((log, i) => (
                  <div key={i} className={`log-entry log-${log.type}`}>
                    <span className="log-time">{log.time}</span>
                    <span className="log-msg">{log.message}</span>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Stats */}
          {stats && (
            <motion.div 
              className="flyover-stats-section"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <div className="flyover-card-header">
                <span>TRAFFIC IMPACT</span>
                <span className="flyover-tag success">COMPLETE</span>
              </div>
              <div className="stats-grid">
                <div className="stat-item highlight">
                  <span className="stat-value">-{stats.timeSaved}m</span>
                  <span className="stat-label">TIME SAVED</span>
                </div>
                <div className="stat-item">
                  <span className="stat-value">{stats.congestionReduction}%</span>
                  <span className="stat-label">LESS TRAFFIC</span>
                </div>
                <div className="stat-item">
                  <span className="stat-value">{stats.vehiclesPerHour}</span>
                  <span className="stat-label">VEH/HOUR</span>
                </div>
                <div className="stat-item">
                  <span className="stat-value">{stats.pillars}</span>
                  <span className="stat-label">PILLARS</span>
                </div>
              </div>
              <div className="eco-badge">
                🌿 ~{stats.co2Reduction} kg CO₂ reduction/day
              </div>
            </motion.div>
          )}
        </aside>
      </main>

      <style>{`
        .flyover-page {
          min-height: 100vh;
          background: #0a0a0a;
          color: white;
        }
        .flyover-navbar {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 15px 30px;
          border-bottom: 1px solid rgba(204,255,0,0.15);
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          z-index: 100;
          background: rgba(10,10,10,0.95);
          backdrop-filter: blur(10px);
        }
        .flyover-logo {
          font-family: 'Bebas Neue', sans-serif;
          font-size: 1.5rem;
          color: #CCFF00;
          text-decoration: none;
          letter-spacing: 2px;
        }
        .flyover-nav-center {
          display: flex;
          align-items: center;
          gap: 10px;
          font-family: 'Space Mono', monospace;
          font-size: 0.8rem;
          letter-spacing: 0.1em;
        }
        .flyover-nav-dot {
          width: 8px;
          height: 8px;
          background: #ff4d00;
          border-radius: 50%;
          animation: pulse 2s infinite;
        }
        .flyover-nav-tag {
          background: rgba(204,255,0,0.15);
          color: #CCFF00;
          padding: 4px 10px;
          border-radius: 2px;
          font-size: 0.7rem;
        }
        .flyover-back-btn {
          font-family: 'Space Mono', monospace;
          font-size: 0.75rem;
          color: rgba(255,255,255,0.7);
          text-decoration: none;
          padding: 8px 15px;
          border: 1px solid rgba(255,255,255,0.2);
          border-radius: 2px;
          transition: all 0.3s;
        }
        .flyover-back-btn:hover {
          border-color: #CCFF00;
          color: #CCFF00;
        }
        .flyover-main {
          display: grid;
          grid-template-columns: 1fr 400px;
          gap: 20px;
          padding: 80px 20px 20px;
          height: 100vh;
          max-width: 1800px;
          margin: 0 auto;
        }
        .flyover-map-section {
          height: calc(100vh - 100px);
        }
        .flyover-map-card {
          height: 100%;
          background: rgba(255,255,255,0.02);
          border: 1px solid rgba(204,255,0,0.15);
          border-radius: 4px;
          overflow: hidden;
          display: flex;
          flex-direction: column;
        }
        .flyover-card-header {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 12px 15px;
          border-bottom: 1px solid rgba(204,255,0,0.1);
          font-family: 'Space Mono', monospace;
          font-size: 0.75rem;
          letter-spacing: 0.1em;
          color: rgba(255,255,255,0.8);
        }
        .flyover-loading-tag, .flyover-tag {
          background: rgba(204,255,0,0.2);
          color: #CCFF00;
          padding: 3px 8px;
          border-radius: 2px;
          font-size: 0.65rem;
        }
        .flyover-tag.error {
          background: rgba(255,77,0,0.2);
          color: #ff4d00;
        }
        .flyover-tag.success {
          background: rgba(0,255,100,0.15);
          color: #66ff99;
        }
        .flyover-map-wrapper {
          flex: 1;
          position: relative;
        }
        .flyover-map {
          width: 100%;
          height: 100%;
        }
        
        /* Search Bar on Map */
        .map-search-bar {
          position: absolute;
          top: 15px;
          left: 15px;
          display: flex;
          gap: 5px;
          z-index: 10;
        }
        .map-search-bar input {
          width: 220px;
          padding: 10px 12px;
          background: rgba(10,10,10,0.9);
          border: 1px solid rgba(204,255,0,0.3);
          border-radius: 2px;
          color: white;
          font-family: 'Space Mono', monospace;
          font-size: 0.8rem;
          outline: none;
        }
        .map-search-bar input:focus {
          border-color: #CCFF00;
        }
        .map-search-bar button {
          padding: 10px 14px;
          background: rgba(204,255,0,0.2);
          border: 1px solid rgba(204,255,0,0.3);
          border-radius: 2px;
          color: #CCFF00;
          cursor: pointer;
          font-size: 1rem;
        }
        .map-search-bar button:hover {
          background: rgba(204,255,0,0.3);
        }

        /* My Location Button */
        .my-location-btn {
          position: absolute;
          bottom: 100px;
          right: 10px;
          width: 40px;
          height: 40px;
          background: rgba(10,10,10,0.9);
          border: 1px solid rgba(255,255,255,0.3);
          border-radius: 4px;
          color: white;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          transition: all 0.3s;
          z-index: 10;
        }
        .my-location-btn:hover {
          background: rgba(204,255,0,0.2);
          border-color: #CCFF00;
          color: #CCFF00;
        }
        .my-location-btn:disabled {
          opacity: 0.7;
        }
        .loc-spinner {
          width: 16px;
          height: 16px;
          border: 2px solid rgba(255,255,255,0.3);
          border-top-color: #CCFF00;
          border-radius: 50%;
          animation: spin 1s linear infinite;
        }

        /* User Location Marker */
        .user-location-marker {
          position: relative;
        }
        .user-location-marker .dot {
          width: 14px;
          height: 14px;
          background: #4285F4;
          border: 3px solid white;
          border-radius: 50%;
          box-shadow: 0 2px 6px rgba(0,0,0,0.3);
        }
        .user-location-marker .pulse {
          position: absolute;
          width: 40px;
          height: 40px;
          background: rgba(66,133,244,0.3);
          border-radius: 50%;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          animation: pulse-ring 2s infinite;
        }

        /* Control Panel */
        .flyover-control-panel {
          display: flex;
          flex-direction: column;
          gap: 15px;
          height: calc(100vh - 100px);
          overflow-y: auto;
        }
        .flyover-input-section, .flyover-log-section, .flyover-stats-section {
          background: rgba(255,255,255,0.02);
          border: 1px solid rgba(204,255,0,0.15);
          border-radius: 4px;
        }
        .flyover-inputs {
          padding: 15px;
          display: flex;
          flex-direction: column;
          gap: 12px;
        }
        .flyover-input-group {
          display: flex;
          flex-direction: column;
          gap: 6px;
        }
        .flyover-input-group label {
          font-family: 'Space Mono', monospace;
          font-size: 0.65rem;
          letter-spacing: 0.15em;
          color: rgba(255,255,255,0.5);
        }
        .flyover-input-group input {
          padding: 12px;
          background: rgba(0,0,0,0.4);
          border: 1px solid rgba(255,255,255,0.15);
          border-radius: 2px;
          color: white;
          font-family: 'Space Mono', monospace;
          font-size: 0.85rem;
          outline: none;
        }
        .flyover-input-group input:focus {
          border-color: #CCFF00;
        }
        .input-with-btn {
          display: flex;
          gap: 8px;
        }
        .input-with-btn input {
          flex: 1;
        }
        .use-location-btn {
          padding: 12px 14px;
          background: rgba(204,255,0,0.15);
          border: 1px solid rgba(204,255,0,0.3);
          border-radius: 2px;
          cursor: pointer;
          font-size: 1rem;
        }
        .use-location-btn:hover {
          background: rgba(204,255,0,0.25);
        }
        .simulate-btn {
          padding: 16px;
          background: #CCFF00;
          border: none;
          border-radius: 2px;
          color: #0a0a0a;
          font-family: 'Space Mono', monospace;
          font-size: 0.9rem;
          font-weight: bold;
          letter-spacing: 0.15em;
          cursor: pointer;
          transition: all 0.3s;
          display: flex;
          align-items: center;
          justify-content: center;
          margin-top: 5px;
        }
        .simulate-btn:hover:not(:disabled) {
          box-shadow: 0 8px 30px rgba(204,255,0,0.4);
        }
        .simulate-btn:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }
        .btn-spinner {
          width: 20px;
          height: 20px;
          border: 3px solid rgba(10,10,10,0.3);
          border-top-color: #0a0a0a;
          border-radius: 50%;
          animation: spin 1s linear infinite;
        }

        /* Log */
        .flyover-log-content {
          max-height: 180px;
          overflow-y: auto;
          padding: 10px;
        }
        .log-empty {
          padding: 20px;
          text-align: center;
          color: rgba(255,255,255,0.4);
          font-family: 'Space Mono', monospace;
          font-size: 0.75rem;
        }
        .log-entry {
          display: flex;
          gap: 10px;
          padding: 6px 8px;
          margin-bottom: 4px;
          border-radius: 2px;
          font-family: 'Space Mono', monospace;
          font-size: 0.7rem;
        }
        .log-time {
          color: rgba(255,255,255,0.4);
          flex-shrink: 0;
        }
        .log-msg {
          color: rgba(255,255,255,0.8);
        }
        .log-info { background: rgba(255,255,255,0.03); }
        .log-success { background: rgba(204,255,0,0.08); }
        .log-success .log-msg { color: #CCFF00; }
        .log-error { background: rgba(255,77,0,0.1); }
        .log-error .log-msg { color: #ff4d00; }
        .log-system { background: rgba(255,77,0,0.05); border-left: 2px solid #ff4d00; }
        .log-system .log-msg { color: #ff9966; }

        /* Stats */
        .stats-grid {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 1px;
          background: rgba(204,255,0,0.1);
          padding: 1px;
        }
        .stat-item {
          background: rgba(10,10,10,0.9);
          padding: 15px 10px;
          text-align: center;
        }
        .stat-item.highlight {
          background: rgba(204,255,0,0.1);
        }
        .stat-value {
          display: block;
          font-family: 'Bebas Neue', sans-serif;
          font-size: 1.5rem;
          color: #CCFF00;
        }
        .stat-item.highlight .stat-value {
          text-shadow: 0 0 20px rgba(204,255,0,0.5);
        }
        .stat-label {
          font-family: 'Space Mono', monospace;
          font-size: 0.6rem;
          letter-spacing: 0.1em;
          color: rgba(255,255,255,0.5);
        }
        .eco-badge {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 8px;
          padding: 12px;
          background: rgba(0,255,100,0.08);
          border-top: 1px solid rgba(0,255,100,0.1);
          font-family: 'Space Mono', monospace;
          font-size: 0.75rem;
          color: #66ff99;
        }

        @keyframes spin {
          to { transform: rotate(360deg); }
        }
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
        @keyframes pulse-ring {
          0% { transform: translate(-50%, -50%) scale(0.5); opacity: 1; }
          100% { transform: translate(-50%, -50%) scale(1.5); opacity: 0; }
        }

        @media (max-width: 1100px) {
          .flyover-main {
            grid-template-columns: 1fr;
            height: auto;
          }
          .flyover-map-section {
            height: 50vh;
          }
          .flyover-control-panel {
            height: auto;
          }
        }
      `}</style>
    </motion.div>
        )}
      </AnimatePresence>
    </>
  )
}
