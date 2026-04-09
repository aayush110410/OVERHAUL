/**
 * DEMO 2 — OVERHAUL COMMAND CENTER v3
 *
 * Includes: Region-Aware Intelligence Engine (LDRAGO)
 *  - Click-to-select route on map
 *  - Congestion-colored route segments
 *  - Animated particle flow along route
 *  - Real AQI + traffic data with deterministic fallback
 *  - AI orchestration via ldragoController
 *  - IntelligencePanel with travel time, congestion, AQI, PM2.5
 */

import { useState, useEffect, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useNavigate, useLocation, Link } from 'react-router-dom'
import mapboxgl from 'mapbox-gl'
import 'mapbox-gl/dist/mapbox-gl.css'

const MAPBOX_TOKEN = (import.meta.env.VITE_MAPBOX_TOKEN || '').trim()
mapboxgl.accessToken = MAPBOX_TOKEN
import ReactMarkdown from 'react-markdown'
import { API_BASE } from '../api/config'
import ScenarioTemplates from '../components/ScenarioTemplates'
import ScenarioBuilder from '../components/ScenarioBuilder'
import ScenarioComparison from '../components/ScenarioComparison'
import EngineResults from '../components/EngineResults'
import RouteSelector from '../components/RouteSelector'
import IntelligencePanel from '../components/IntelligencePanel'
import TrafficLegend from '../components/TrafficLegend'
import { orchestrate, buildImpactCards, extractLocationsFromPrompt } from '../services/ldragoController'
import { reverseGeocode } from '../services/routeService'
import { appendMessage, buildContextualPrompt, clearMemory, getRecentHistory } from '../services/conversationMemory'
import { extractLocationsFromText, parseLocationsList, paintHalos, clearHalos } from '../services/regionHaloService'
import './Demo2.css'

const MAPBOX_STYLE = 'mapbox://styles/mapbox/dark-v11'

// ─────────────────────────────────────────────
// BOOT LOADER
// ─────────────────────────────────────────────
function CommandCenterLoader({ onComplete }) {
  const [progress, setProgress] = useState(0)
  const [phase, setPhase] = useState('boot')

  useEffect(() => {
    const t1 = setTimeout(() => setPhase('loading'), 600)
    return () => clearTimeout(t1)
  }, [])

  useEffect(() => {
    if (phase !== 'loading') return
    const interval = setInterval(() => {
      setProgress(p => {
        if (p >= 100) {
          clearInterval(interval)
          setTimeout(() => { setPhase('exit'); setTimeout(onComplete, 600) }, 300)
          return 100
        }
        return p + 2.5
      })
    }, 25)
    return () => clearInterval(interval)
  }, [phase, onComplete])

  const statusLabel =
    progress < 20 ? 'INITIALISING SYSTEM' :
    progress < 45 ? 'LOADING NCR DATASETS' :
    progress < 70 ? 'CALIBRATING ENGINES' :
    progress < 90 ? 'CONNECTING LIVE FEEDS' : 'READY'

  return (
    <motion.div
      className="ov-loader"
      initial={{ opacity: 1 }}
      animate={phase === 'exit' ? { opacity: 0 } : { opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <div className="ov-loader-inner">
        <motion.div
          className="ov-loader-logo"
          initial={{ scale: 0.6, opacity: 0 }}
          animate={phase === 'exit' ? { scale: 2.5, opacity: 0 } : { scale: 1, opacity: 1 }}
          transition={{ duration: phase === 'exit' ? 0.5 : 0.5, ease: [0.16, 1, 0.3, 1] }}
        >
          <span className="ov-logo-o">O</span><span className="ov-logo-v">V</span>
        </motion.div>
        <div className="ov-loader-subtitle">COMMAND CENTER</div>
        <div className="ov-loader-bar-wrap">
          <motion.div className="ov-loader-bar-fill" initial={{ width: 0 }} animate={{ width: `${progress}%` }} transition={{ duration: 0.1 }} />
        </div>
        <div className="ov-loader-status">
          <span>{statusLabel}</span>
          <span className="ov-loader-pct">{Math.round(progress)}%</span>
        </div>
      </div>
    </motion.div>
  )
}

// ─────────────────────────────────────────────
// METRIC CARD
// ─────────────────────────────────────────────
function MetricCard({ icon, label, value, delta, source, delay = 0 }) {
  const isGood = delta && delta.startsWith('-')
  const isBad  = delta && delta.startsWith('+') && !label.toLowerCase().includes('travel')
  return (
    <motion.div className="ov-metric" initial={{ opacity: 0, x: -16 }} animate={{ opacity: 1, x: 0 }} transition={{ delay, duration: 0.5, ease: [0.16, 1, 0.3, 1] }}>
      <div className="ov-metric-accent" />
      <div className="ov-metric-icon">{icon}</div>
      <div className="ov-metric-body">
        <div className="ov-metric-label">{label}</div>
        <div className="ov-metric-value">{value}</div>
        {delta && <div className={`ov-metric-delta ${isGood ? 'good' : isBad ? 'bad' : ''}`}>{delta}</div>}
        {source && <div className="ov-metric-source">{source}</div>}
      </div>
    </motion.div>
  )
}

// ─────────────────────────────────────────────
// SCENARIO PILL
// ─────────────────────────────────────────────
function ScenarioPill({ label, icon, onClick, active }) {
  return (
    <button className={`ov-pill ${active ? 'active' : ''}`} onClick={onClick}>
      <span className="ov-pill-icon">{icon}</span>
      <span className="ov-pill-label">{label}</span>
    </button>
  )
}

// ─────────────────────────────────────────────
// PARTICLE ENGINE
// ─────────────────────────────────────────────
function useParticleEngine(canvasRef, mapRef, segments, isActive) {
  const animFrameRef = useRef(null)
  const particlesRef = useRef([])

  useEffect(() => {
    if (!isActive || !segments?.length || !canvasRef.current || !mapRef.current) {
      cancelAnimationFrame(animFrameRef.current)
      return
    }

    const canvas = canvasRef.current
    const ctx    = canvas.getContext('2d')
    const map    = mapRef.current

    const NUM_PARTICLES = Math.min(60, segments.length * 2)

    // Init particles spread along segments
    particlesRef.current = Array.from({ length: NUM_PARTICLES }, (_, i) => ({
      t: i / NUM_PARTICLES,   // position along route [0,1]
      speed: 0.0008 + Math.random() * 0.0012,
      alpha: 0.6 + Math.random() * 0.4,
      size:  2 + Math.random() * 2,
    }))

    function getSegmentAtT(t) {
      const idx = Math.floor(t * (segments.length - 1))
      return segments[Math.min(idx, segments.length - 1)]
    }

    function project(coord) {
      const pt = map.project(coord)
      return { x: pt.x, y: pt.y }
    }

    function draw() {
      const { width, height } = canvas
      ctx.clearRect(0, 0, width, height)

      particlesRef.current.forEach(p => {
        p.t += p.speed
        if (p.t > 1) p.t = 0

        const seg = getSegmentAtT(p.t)
        if (!seg) return

        try {
          const pt = project(seg.coord)
          ctx.beginPath()
          ctx.arc(pt.x, pt.y, p.size, 0, Math.PI * 2)
          ctx.fillStyle = seg.color + Math.round(p.alpha * 255).toString(16).padStart(2, '0')
          ctx.shadowColor = seg.color
          ctx.shadowBlur  = 8
          ctx.fill()
        } catch {}
      })

      animFrameRef.current = requestAnimationFrame(draw)
    }

    function resize() {
      canvas.width  = window.innerWidth
      canvas.height = window.innerHeight
    }

    resize()
    window.addEventListener('resize', resize)
    map.on('move', () => { /* re-render on next frame naturally */ })

    animFrameRef.current = requestAnimationFrame(draw)
    return () => {
      cancelAnimationFrame(animFrameRef.current)
      window.removeEventListener('resize', resize)
    }
  }, [isActive, segments])
}

// ─────────────────────────────────────────────
// MARKER HELPERS
// ─────────────────────────────────────────────
function makeMarkerEl(type) {
  const el = document.createElement('div')
  el.className = `ov-map-marker ${type}`
  el.innerHTML = type === 'start'
    ? `<div class="ov-marker-pin green"><div class="ov-marker-ring"></div><span>A</span></div>`
    : `<div class="ov-marker-pin orange"><div class="ov-marker-ring"></div><span>B</span></div>`
  return el
}

// ─────────────────────────────────────────────
// MAIN COMPONENT
// ─────────────────────────────────────────────
export default function Demo2() {
  const location  = useLocation()
  const skipLoader = location.state?.skipLoader

  // UI state
  const [loading, setLoading]         = useState(!skipLoader)
  const [currentTime, setCurrentTime] = useState('')
  const [sidebarOpen, setSidebarOpen] = useState(true)

  // Map
  const mapContainer = useRef(null)
  const mapRef       = useRef(null)
  const [mapReady, setMapReady] = useState(false)

  // Particle canvas
  const particleCanvas = useRef(null)

  // Route / intelligence state
  const [clickMode, setClickMode]       = useState(null)  // 'start' | 'end' | null
  const [startCoords, setStartCoords]   = useState(null)  // [lng, lat]
  const [endCoords, setEndCoords]       = useState(null)
  const [startName, setStartName]       = useState('')
  const [endName, setEndName]           = useState('')
  const startMarkerRef = useRef(null)
  const endMarkerRef   = useRef(null)
  const [routeActive, setRouteActive]        = useState(false)
  const [routeSegments, setRouteSegments]    = useState([])
  const [intelligenceReport, setIntelligenceReport] = useState(null)
  const [routeLoading, setRouteLoading]      = useState(false)
  const [routeError, setRouteError]          = useState('')

  // Standard analysis state (chat)
  const [prompt, setPrompt]                 = useState('')
  const [mode, setMode]                     = useState('fast')
  const [isAnalyzing, setIsAnalyzing]       = useState(false)
  const [analysisProgress, setAnalysisProgress] = useState(0)
  const [progressMessage, setProgressMessage]   = useState('')
  const [errorMessage, setErrorMessage]         = useState('')
  const [chatHistory, setChatHistory]       = useState([])
  const [stats, setStats] = useState({
    travelTime: { value: '--', source: '', delta: null },
    pm25:       { value: '--', source: '', delta: null },
    vkt:        { value: '--', source: '', delta: null },
    co2:        { value: '--', source: '', delta: null },
  })
  const [liveSources, setLiveSources]           = useState([])
  const [analysisComplete, setAnalysisComplete] = useState(false)

  // Tabs & engine data
  const [controlTab, setControlTab]                     = useState('chat')
  const [engineDomains, setEngineDomains]               = useState(null)
  const [engineRecommendations, setEngineRecommendations] = useState([])
  const [engineWarnings, setEngineWarnings]             = useState([])
  const [impactCards, setImpactCards]                   = useState([])

  // Quick scenarios
  const [activeScenario, setActiveScenario] = useState(null)
  const QUICK_SCENARIOS = [
    { id: 'green', icon: '🌿', label: 'Green Delhi 2030', prompt: 'What if Delhi bans all private cars in Connaught Place and adds 500 electric buses on Ring Road?' },
    { id: 'metro', icon: '🚇', label: 'Metro Expansion',  prompt: 'Simulate extending Delhi Metro Phase 4 with 25 new stations covering Dwarka, Najafgarh, and Aerocity by 2028' },
    { id: 'ev',    icon: '⚡', label: 'EV Revolution',    prompt: 'What happens if 40% of Delhi NCR vehicles switch to electric by 2027 with 2000 fast charging stations?' },
  ]

  // Particle animation
  useParticleEngine(particleCanvas, mapRef, routeSegments, routeActive)

  // Clock
  useEffect(() => {
    const tick = () => {
      const now = new Date()
      setCurrentTime(now.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false }))
    }
    tick()
    const id = setInterval(tick, 1000)
    return () => clearInterval(id)
  }, [])

  // ── MAP INITIALIZATION ──────────────────────────────────────────────────
  useEffect(() => {
    if (loading || !mapContainer.current || mapRef.current) return

    const map = new mapboxgl.Map({
      container: mapContainer.current,
      style: MAPBOX_STYLE,
      center: [77.21, 28.61],
      zoom: 11.5,
      pitch: 45,
      bearing: -12,
      maxPitch: 75,
      antialias: true,
    })

    map.addControl(new mapboxgl.NavigationControl({ showCompass: true, showZoom: false }), 'bottom-right')
    map.on('error', e => console.error('[Map]', e.error?.message || e))

    map.on('load', () => {
      // Terrain
      map.addSource('mapbox-dem', { type: 'raster-dem', url: 'mapbox://mapbox.mapbox-terrain-dem-v1', tileSize: 512, maxzoom: 14 })
      map.setTerrain({ source: 'mapbox-dem', exaggeration: 1.2 })

      // 3D buildings
      const layers = map.getStyle().layers
      const labelLayerId = layers.find(l => l.type === 'symbol' && l.layout?.['text-field'])?.id
      map.addLayer({
        id: '3d-buildings', source: 'composite', 'source-layer': 'building',
        filter: ['==', 'extrude', 'true'], type: 'fill-extrusion', minzoom: 12,
        paint: { 'fill-extrusion-color': '#0f1923', 'fill-extrusion-height': ['get', 'height'], 'fill-extrusion-base': ['get', 'min_height'], 'fill-extrusion-opacity': 0.7 },
      }, labelLayerId)

      // Sky
      map.addLayer({ id: 'sky', type: 'sky', paint: { 'sky-type': 'atmosphere', 'sky-atmosphere-sun': [0.0, 0.0], 'sky-atmosphere-sun-intensity': 5 } })

      // Legacy sources (for Live route + AQI)
      map.addSource('edges',     { type: 'geojson', data: { type: 'FeatureCollection', features: [] } })
      map.addSource('pollution', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } })
      map.addSource('liveRoute', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } })

      // Live AQI pollution circles
      map.addLayer({
        id: 'pollution-heat', type: 'circle', source: 'pollution',
        paint: {
          'circle-radius': ['interpolate', ['linear'], ['get', 'intensity'], 0, 0, 1, 50],
          'circle-color': 'rgba(255, 77, 0, 0.25)', 'circle-stroke-color': 'rgba(255, 77, 0, 0.6)',
          'circle-stroke-width': 1, 'circle-blur': 0.6, 'circle-opacity': 0.6,
        },
      })

      // Live edges
      map.addLayer({
        id: 'edges-layer', type: 'line', source: 'edges',
        paint: {
          'line-color': ['interpolate', ['linear'], ['get', 'ev_share'], 0, '#1a2030', 25, '#CCFF00', 60, '#00ff88', 90, '#ffff00'],
          'line-width': ['case', ['get', 'primary'], 6, 3], 'line-opacity': 0.7, 'line-blur': 0.5,
        },
      })

      // ── INTELLIGENT ROUTE LAYERS ──────────────────
      map.addSource('traffic-route', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } })

      // Route glow (wide blur)
      map.addLayer({
        id: 'traffic-route-glow', type: 'line', source: 'traffic-route',
        paint: {
          'line-color': ['get', 'color'],
          'line-width': 18,
          'line-opacity': 0.18,
          'line-blur': 14,
        },
      })

      // Route casing (border)
      map.addLayer({
        id: 'traffic-route-casing', type: 'line', source: 'traffic-route',
        paint: {
          'line-color': 'rgba(0,0,0,0.4)',
          'line-width': 10,
          'line-gap-width': 0,
        },
      })

      // Route fill (color-coded congestion)
      map.addLayer({
        id: 'traffic-route-fill', type: 'line', source: 'traffic-route',
        paint: {
          'line-color': ['get', 'color'],
          'line-width': 6,
          'line-opacity': 0.95,
        },
      })

      setMapReady(true)
      fetchLiveData(map)
    })

    // ── MAP CLICK HANDLER ──────────────────────────
    map.on('click', (e) => {
      handleMapClick(e.lngLat.lng, e.lngLat.lat)
    })

    mapRef.current = map
    setTimeout(() => map.resize(), 50)
    return () => { map.remove(); mapRef.current = null; setMapReady(false) }
  }, [loading])

  // ── MAP CLICK HANDLER (ref to current clickMode) ──
  const clickModeRef = useRef(clickMode)
  useEffect(() => { clickModeRef.current = clickMode }, [clickMode])

  const handleMapClick = useCallback(async (lng, lat) => {
    const mode = clickModeRef.current
    if (!mode) return

    const geo = await reverseGeocode(lng, lat)
    const name = geo.shortName || `${lat.toFixed(4)}, ${lng.toFixed(4)}`
    const coords = [lng, lat]

    if (mode === 'start') {
      setStartCoords(coords)
      setStartName(name)
      setClickMode('end')  // auto-advance to picking end
    } else {
      setEndCoords(coords)
      setEndName(name)
      setClickMode(null)
    }
  }, [])

  // ── MARKER MANAGEMENT ──────────────────────────────
  useEffect(() => {
    if (!mapRef.current || !mapReady) return
    if (startMarkerRef.current) startMarkerRef.current.remove()
    if (startCoords) {
      const el = makeMarkerEl('start')
      startMarkerRef.current = new mapboxgl.Marker({ element: el })
        .setLngLat(startCoords)
        .addTo(mapRef.current)
    }
  }, [startCoords, mapReady])

  useEffect(() => {
    if (!mapRef.current || !mapReady) return
    if (endMarkerRef.current) endMarkerRef.current.remove()
    if (endCoords) {
      const el = makeMarkerEl('end')
      endMarkerRef.current = new mapboxgl.Marker({ element: el })
        .setLngLat(endCoords)
        .addTo(mapRef.current)
    }
  }, [endCoords, mapReady])

  // ── ROUTE ANALYSIS (LDRAGO) ────────────────────────
  const runRouteAnalysis = useCallback(async () => {
    if (!startCoords || !endCoords || routeLoading) return
    setRouteLoading(true)
    setRouteError('')
    setControlTab('route')

    try {
      const report = await orchestrate({
        startCoords,
        endCoords,
        prompt: `Analyze route from ${startName} to ${endName}`,
      })

      setIntelligenceReport(report)
      setRouteActive(true)
      setRouteSegments(report.traffic.segments)

      // Update map traffic route layer
      if (mapRef.current?.getSource('traffic-route')) {
        mapRef.current.getSource('traffic-route').setData(report.trafficGeoJSON)
      }

      // Fit map to route bounds
      const coords = report.route.geometry.coordinates
      if (coords.length > 1) {
        const bounds = coords.reduce((b, c) => b.extend(c), new mapboxgl.LngLatBounds(coords[0], coords[0]))
        mapRef.current?.fitBounds(bounds, { padding: { top: 80, bottom: 100, left: 260, right: 460 }, maxZoom: 14, duration: 1200 })
      }

      // Update stat metrics from report
      const s = report.traffic.summary
      const e = report.env
      setStats(prev => ({
        ...prev,
        travelTime: { value: `${s.predictedDuration_min} min`, source: `Live • ${s.dataSource}`, delta: s.delayPercent > 0 ? `+${s.delayPercent}% delay` : null },
        pm25:       { value: `${e.pm25} µg/m³`, source: e.source, delta: null },
      }))

      // Add result to chat
      setImpactCards(buildImpactCards(report))
      setChatHistory(prev => [
        ...prev,
        { role: 'user', content: `Analyze route from ${startName} to ${endName}` },
        { role: 'assistant', content: report.summaryText, meta: { mode: 'LDRAGO', runtime: '—', engineCount: 3 } },
      ])
    } catch (err) {
      console.error('[RouteAnalysis]', err)
      setRouteError(err.message)
    } finally {
      setRouteLoading(false)
    }
  }, [startCoords, endCoords, startName, endName, routeLoading])

  // Auto-analyze when both points are set
  useEffect(() => {
    if (startCoords && endCoords && !routeActive) {
      runRouteAnalysis()
    }
  }, [startCoords, endCoords])

  // ── CLEAR ROUTE ─────────────────────────────────────
  const clearRoute = useCallback(() => {
    setStartCoords(null); setEndCoords(null)
    setStartName(''); setEndName('')
    setRouteActive(false); setIntelligenceReport(null); setRouteSegments([])
    setClickMode(null); setRouteError('')
    if (startMarkerRef.current) { startMarkerRef.current.remove(); startMarkerRef.current = null }
    if (endMarkerRef.current)   { endMarkerRef.current.remove();   endMarkerRef.current = null }
    if (mapRef.current?.getSource('traffic-route')) {
      mapRef.current.getSource('traffic-route').setData({ type: 'FeatureCollection', features: [] })
    }
  }, [])

  // ── LIVE GENERAL DATA ───────────────────────────────
  const fetchWithRetry = async (url, options = {}, retries = 2, timeout = 30000) => {
    for (let i = 0; i <= retries; i++) {
      try {
        const ctrl = new AbortController()
        const tid  = setTimeout(() => ctrl.abort(), timeout)
        const resp = await fetch(url, { ...options, signal: ctrl.signal })
        clearTimeout(tid)
        return resp
      } catch (err) {
        if (i === retries) throw err
        await new Promise(r => setTimeout(r, 1500))
      }
    }
  }

  const fetchLiveData = async (map) => {
    try {
      const resp = await fetchWithRetry(`${API_BASE}/live/route`)
      if (resp.ok) {
        const geojson = await resp.json()
        if (map.getSource('liveRoute')) map.getSource('liveRoute').setData(geojson)
        const src = geojson?.features?.[0]?.properties?.source || 'Route API'
        setLiveSources(prev => prev.find(s => s.name === src) ? prev : [...prev, { name: src, detail: 'Live corridor' }])
      }
    } catch (e) { console.warn('[Live] Route:', e.message) }

    try {
      const resp = await fetchWithRetry(`${API_BASE}/live/aqi?lat=28.62&lon=77.35`)
      if (resp.ok) {
        const data   = await resp.json()
        const series = Array.isArray(data.series) ? data.series.filter(pt => pt.datetime && pt.pm25 != null) : []
        const latest = series.length ? series[series.length - 1] : null
        if (latest) setStats(s => ({ ...s, pm25: { value: `${latest.pm25.toFixed(1)} µg/m³`, source: `Live • ${data.source || 'AQI'}`, delta: null } }))
        const aqiSrc = data.source ? String(data.source) : 'AQI Feed'
        setLiveSources(prev => prev.find(s => s.name === aqiSrc) ? prev : [...prev, { name: aqiSrc, detail: 'Air quality' }])
      }
    } catch (e) { console.warn('[Live] AQI:', e.message) }
  }

  // ── MAP UPDATE (for standard analysis) ─────────────
  const updateMap = useCallback((edges, pollution) => {
    const map = mapRef.current; if (!map) return
    if (edges && map.getSource('edges')) {
      const gj = edges.type === 'FeatureCollection' ? edges : { type: 'FeatureCollection', features: Array.isArray(edges) ? edges : [] }
      map.getSource('edges').setData(gj)
      if (gj.features.length) {
        const bounds = gj.features.reduce((b, f) => { if (f.geometry?.coordinates) f.geometry.coordinates.forEach(c => b.extend(c)); return b }, new mapboxgl.LngLatBounds())
        if (!bounds.isEmpty()) map.fitBounds(bounds, { padding: 80, maxZoom: 14 })
      }
    }
    if (pollution && map.getSource('pollution')) {
      const gj = pollution.type === 'FeatureCollection' ? pollution : { type: 'FeatureCollection', features: Array.isArray(pollution) ? pollution : [] }
      map.getSource('pollution').setData(gj)
    }
  }, [])

  // ── STANDARD CHAT ANALYSIS (LDRAGOv2) ──────────────
  const runAnalysis = useCallback(async (overridePrompt) => {
    const text = overridePrompt || prompt
    if (!text.trim() || isAnalyzing) return

    // Tip if user explicitly types an A→B route but hasn't selected map points
    const { found, startLabel, endLabel } = extractLocationsFromPrompt(text)
    const hasExplicitRoute = found && startLabel && endLabel
    if (hasExplicitRoute && !startCoords && !endCoords) {
      setChatHistory(prev => [...prev,
        { role: 'user', content: text },
        { role: 'assistant', content: `**Tip:** I detected a route query (${startLabel} → ${endLabel}).\n\nUse the **ROUTE** tab to pin exact start & end points on the map for real-time traffic data, colour-coded congestion segments, and live AQI.\n\nRunning a general analysis now...` }
      ])
    }

    setIsAnalyzing(true)
    setAnalysisProgress(0)
    setProgressMessage('Initialising LDRAGO v2...')
    setErrorMessage('')
    setPrompt('')
    setChatHistory(prev => [...prev, { role: 'user', content: text }])

    // Append user message to conversation memory
    appendMessage('user', text, routeActive && intelligenceReport ? {
      route: intelligenceReport?.startName + ' → ' + intelligenceReport?.endName,
      congestion: intelligenceReport?.traffic?.summary?.avgCongestion,
      aqi: intelligenceReport?.env?.aqi,
    } : {})

    const start = Date.now()
    const progressStages = [
      [0,   12,  '📡 Agent 1 — Parsing intent & extracting entities...'],
      [12,  26,  '📍 Agent 2 — Resolving NCR locations & landmarks...'],
      [26,  40,  '🗺️ Agent 3 — Planning analysis strategy...'],
      [40,  58,  '🔬 Agent 4 — Research + BPR simulation engines...'],
      [58,  74,  '🧠 Agent 5 — Deep reasoning + critique pass...'],
      [74,  90,  '✍️ Agent 6 — Synthesising final intelligence report...'],
      [90,  97,  '🗺 Generating map visualisations...'],
    ]
    const totalMs = 20000
    const progressInterval = setInterval(() => {
      const elapsed = Date.now() - start
      const raw = Math.min(95, (elapsed / totalMs) * 100)
      setAnalysisProgress(Math.round(raw))
      const stage = progressStages.find(([lo, hi]) => raw >= lo && raw < hi)
      if (stage) setProgressMessage(stage[2])
    }, 150)

    // Build enriched prompt with conversation memory + live route telemetry
    let enrichedPrompt = buildContextualPrompt(text, routeActive ? intelligenceReport : null)

    // Extra: if a route IS active but memory context not available, still inject telemetry directly
    if (routeActive && intelligenceReport && !enrichedPrompt.includes('TELEMETRY')) {
      const r = intelligenceReport
      const s = r.traffic?.summary || {}
      const e = r.env || {}
      const now = new Date().toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', timeZone: 'Asia/Kolkata' })
      enrichedPrompt = [
        `[LIVE ROUTE TELEMETRY — ${now} IST]`,
        `Route: ${r.startName} → ${r.endName}`,
        `Distance: ${r.route?.distance_km} km | Predicted: ${s.predictedDuration_min} min | Congestion: ${s.avgCongestion}/100`,
        `Speed: ${s.avgSpeed_kmh} km/h | Density: ${s.avgVehicleDensity} veh/km | Delay: +${s.delayPercent}%`,
        `AQI: ${e.aqi} (${e.category}) | PM2.5: ${e.pm25} µg/m³ | Health risk: ${e.healthRisk}`,
        `Weather: ${e.temperature_c}°C | Humidity: ${e.humidity_pct}% | Wind: ${e.wind_kmh} km/h`,
        ``, `User query: ${text}`,
      ].join('\n')
    }

    try {
      // Primary: /chat/v2 — 6-agent LDRAGOv2 pipeline with 90s timeout
      let data = null
      let lastError = null

      for (let attempt = 1; attempt <= 2; attempt++) {
        try {
          const controller = new AbortController()
          const timeoutId = setTimeout(() => controller.abort(), 90000)
          const resp = await fetch(`${API_BASE}/chat/v2`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              prompt: enrichedPrompt,
              city: 'delhi',
              mode: mode === 'fast' ? 'fast' : 'full',
            }),
            signal: controller.signal,
          })
          clearTimeout(timeoutId)
          if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${await resp.text()}`)
          data = await resp.json()
          break // success
        } catch (fetchErr) {
          lastError = fetchErr
          if (attempt === 1 && (fetchErr.name === 'AbortError' || fetchErr.message?.includes('fetch'))) {
            // Fallback to /chat on network failure
            setProgressMessage('Switching to base engine...')
            try {
              const r2 = await fetch(`${API_BASE}/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text, context: {} }),
              })
              if (r2.ok) {
                const d2 = await r2.json()
                data = { summary: d2.response || d2.message || 'Analysis complete.', outputs: {}, manifest: { mode: 'chat_fallback' } }
                break
              }
            } catch {}
          }
        }
      }

      if (!data) throw new Error(lastError?.message || 'Backend unreachable — is the server running?')

      clearInterval(progressInterval)
      setAnalysisProgress(100); setProgressMessage('Complete')

      // ── Extract metadata
      const modelsUsed  = data.manifest?.models || data.outputs?.brainInsights?.models_used || {}
      const agentTrace  = data.outputs?.brainInsights?.agent_trace || []
      const engineCount = data.outputs?.domains ? Object.keys(data.outputs.domains).length : 0
      const runtime     = data.manifest?.runtime_s ?? ((Date.now() - start) / 1000).toFixed(1)
      const modelLabel  = Object.values(modelsUsed).filter(Boolean).join(' → ') || 'LDRAGOv2'

      const mainResponse = data.summary || data.outputs?.tldr || 'Analysis complete.'

      // ── Paint region halos on the map
      const responseLocations = [
        ...parseLocationsList(data.locations || []),
        ...extractLocationsFromText(mainResponse),
      ].slice(0, 6) // Max 6 halos at once

      if (responseLocations.length > 0 && mapRef.current) {
        clearHalos(mapRef.current)
        paintHalos(mapRef.current, responseLocations, {
          color: '#CCFF00',
          opacity: 0.18,
          radiusKm: 1.0,
        })
        // Auto-dismiss halos after 18 seconds
        setTimeout(() => { clearHalos(mapRef.current) }, 18000)
      }

      // ── Save assistant response to conversation memory
      appendMessage('assistant', mainResponse.slice(0, 800), {
        model: modelLabel,
        runtime: String(runtime),
        locationCount: responseLocations.length,
      })

      setChatHistory(prev => [...prev, {
        role: 'assistant',
        content: mainResponse,
        meta: {
          mode: data.manifest?.mode || 'ldrago_v2',
          model: modelLabel,
          engineCount,
          runtime: String(runtime),
          agentTrace,
          critique: data.outputs?.critique,
          locationCount: responseLocations.length,
        }
      }])

      // Update left-panel metric stats if baseline returned
      if (data.baseline) {
        const live = data.live || {}
        const getDelta = (theme) => {
          if (!data.outputs?.impactCards) return null
          const card = data.outputs.impactCards.find(c => c.theme?.toLowerCase().includes(theme))
          if (card?.delta) { const m = card.delta.match(/([+-]?\d+\.?\d*)/); return m ? parseFloat(m[1]) : null }
          return null
        }
        setStats({
          travelTime: { value: (live?.travel?.travel_time_min ?? data.baseline.avg_travel_time_min) ? `${(live?.travel?.travel_time_min ?? data.baseline.avg_travel_time_min).toFixed(1)} min` : '--', source: live?.travel ? `Live • ${live.travel.source || 'OSRM'}` : 'LDRAGOv2', delta: getDelta('travel') ? `${getDelta('travel') > 0 ? '+' : ''}${getDelta('travel').toFixed(1)}%` : null },
          pm25:       { value: (live?.aqi?.latest_pm25 ?? data.baseline.pm25) ? `${(live?.aqi?.latest_pm25 ?? data.baseline.pm25).toFixed(1)} µg/m³` : '--', source: live?.aqi ? `Live • ${live.aqi.source || 'OpenAQ'}` : 'LDRAGOv2', delta: getDelta('air') ? `${getDelta('air') > 0 ? '+' : ''}${getDelta('air').toFixed(1)}%` : null },
        })
      }

      if (data.live?.sources) {
        setLiveSources(prev => { const next = [...prev]; data.live.sources.forEach(src => { if (!next.find(s => s.name === src.name)) next.push(src) }); return next })
      }

      if (data.geojson) updateMap(data.geojson.edges, data.geojson.pollution)
      if (data.outputs?.domains) { setEngineDomains(data.outputs.domains); setTimeout(() => setControlTab('results'), 300) }
      if (data.outputs?.engineRecommendations) setEngineRecommendations(data.outputs.engineRecommendations)
      if (data.outputs?.engineWarnings) setEngineWarnings(data.outputs.engineWarnings)
      if (data.outputs?.impactCards) setImpactCards(data.outputs.impactCards)
      setAnalysisComplete(true)
    } catch (err) {
      clearInterval(progressInterval)
      console.error('[LDRAGO v2]', err.name, err.message)
      const friendly = err.name === 'AbortError'
        ? 'Request timed out (90s). The AI engines may be overloaded — try again or use Fast mode.'
        : err.message?.includes('fetch') || err.message?.includes('network') || err.message?.includes('unreachable')
          ? 'Cannot reach backend (localhost:8000). Please ensure the Python server is running.'
          : err.message
      setErrorMessage(friendly)
      setChatHistory(prev => [...prev, {
        role: 'assistant',
        content: `⚠ **${friendly}**\n\n*Tip: Make sure the backend is running — \`cd OVERHAUL-main && python app.py\`*`,
        isError: true,
      }])
    } finally {
      setIsAnalyzing(false)
    }
  }, [prompt, mode, isAnalyzing, updateMap, startCoords, endCoords, routeActive, intelligenceReport, mapRef])

  // Scenario handlers
  const handleScenario = useCallback((scenario) => {
    setActiveScenario(scenario.id); setControlTab('chat'); setPrompt(scenario.prompt); runAnalysis(scenario.prompt)
  }, [runAnalysis])

  const handleSelectTemplate = useCallback((template) => { setPrompt(template.description || template.title); setControlTab('chat') }, [])

  const handleRunTemplate = useCallback((result) => {
    const r = result?.results || result?.outputs || {}
    if (r.domains) setEngineDomains(r.domains)
    if (r.recommendations) setEngineRecommendations(r.recommendations)
    if (r.ranked)  setEngineRecommendations(r.ranked)
    if (r.warnings) setEngineWarnings(r.warnings)
    if (r.impactCards) setImpactCards(r.impactCards)
    const title = result?.scenario || result?.template || 'Scenario'
    let responseText = result?.description || ''
    if (r.recommendations?.length) responseText += '\n\n**Key Recommendations:**\n' + r.recommendations.map((rec, i) => `${i + 1}. ${typeof rec === 'string' ? rec : rec.text || JSON.stringify(rec)}`).join('\n')
    if (r.impactCards?.length) responseText += '\n\n**Impact Summary:**\n' + r.impactCards.map(c => `- **${c.metric}**: ${c.value} (${c.delta || ''})`).join('\n')
    setChatHistory(prev => [...prev, { role: 'user', content: `[Template] ${title}` }, { role: 'assistant', content: responseText || 'Analysis complete.' }])
    if (r.domains) setTimeout(() => setControlTab('results'), 300)
    else setControlTab('chat')
    setAnalysisComplete(true)
  }, [])

  const handleBuildScenario = useCallback((scenario) => {
    setPrompt(scenario.prompt || scenario); setControlTab('chat')
    setTimeout(() => runAnalysis(scenario.prompt || scenario), 100)
  }, [runAnalysis])

  useEffect(() => {
    const handler = e => { if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') { e.preventDefault(); runAnalysis() } }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [runAnalysis])

  const TABS = [
    { id: 'route',     label: 'ROUTE',     icon: '🗺️' },
    { id: 'chat',      label: 'CHAT',      icon: '💬' },
    { id: 'results',   label: 'RESULTS',   icon: '📊' },
    { id: 'templates', label: 'TEMPLATES', icon: '📋' },
    { id: 'builder',   label: 'BUILDER',   icon: '🔧' },
    { id: 'compare',   label: 'COMPARE',   icon: '⚖️' },
  ]

  // ─────────────────────────────────────────
  // RENDER
  // ─────────────────────────────────────────
  return (
    <>
      <AnimatePresence>
        {loading && <CommandCenterLoader onComplete={() => setLoading(false)} />}
      </AnimatePresence>

      {!loading && (
        <div className={`ov-cc ${clickMode ? `cursor-mode-${clickMode}` : ''}`}>
          {/* ── FULL-BLEED MAP ── */}
          <div ref={mapContainer} className="ov-map" />

          {/* ── PARTICLE CANVAS ── */}
          <canvas ref={particleCanvas} className="ov-particle-canvas" />

          {/* ── VIGNETTE ── */}
          <div className="ov-vignette" />

          {/* ── CLICK MODE OVERLAY HINT ── */}
          {clickMode && (
            <div className="ov-click-overlay-hint">
              <span className={`ov-click-dot ${clickMode}`} />
              {clickMode === 'start' ? 'Click map to set START point (A)' : 'Click map to set END point (B)'}
              <button className="ov-click-cancel" onClick={() => setClickMode(null)}>✕ Cancel</button>
            </div>
          )}

          {/* ── NAV ── */}
          <nav className="ov-nav">
            <div className="ov-nav-left">
              <Link to="/" className="ov-nav-logo">
                <span className="ov-nav-o">O</span><span className="ov-nav-v">V</span>
              </Link>
              <div className="ov-nav-sep" />
              <div className="ov-live-badge">
                <span className="ov-live-dot" />
                <span>LIVE</span>
              </div>
              {routeActive && (
                <>
                  <div className="ov-nav-sep" />
                  <span className="ov-nav-route-badge">
                    {startName} → {endName}
                  </span>
                </>
              )}
              {!routeActive && (
                <>
                  <div className="ov-nav-sep" />
                  <span className="ov-nav-region">Delhi NCR</span>
                </>
              )}
            </div>

            <div className="ov-nav-center">
              <span className="ov-nav-title">COMMAND CENTER</span>
            </div>

            <div className="ov-nav-right">
              <span className="ov-nav-clock">{currentTime}</span>
              {liveSources.length > 0 && (
                <>
                  <div className="ov-nav-sep" />
                  <span className="ov-nav-feeds">{liveSources.length}× feeds</span>
                </>
              )}
              <div className="ov-nav-sep" />
              <Link to="/demo" className="ov-nav-link">V1 DEMO</Link>
              <Link to="/" className="ov-nav-link">HOME</Link>
            </div>
          </nav>

          {/* ── LEFT PANEL ── */}
          <div className="ov-left-panel">
            {/* Metrics */}
            <div className="ov-metrics">
              {/* Metrics header with live pulse */}
              <div className="ov-metrics-head">
                <span className="ov-panel-label">LIVE METRICS</span>
                <div className="ov-metrics-head-right">
                  {liveSources.length > 0 && (
                    <span className="ov-metrics-feeds-badge">{liveSources.length}× FEEDS</span>
                  )}
                  <div className="ov-metrics-pulse" />
                </div>
              </div>

              {/* Only real-time, meaningful metrics */}
              <MetricCard icon="⏱" label="Travel Time" value={stats.travelTime.value} delta={stats.travelTime.delta} source={stats.travelTime.source} delay={0.1} />
              <MetricCard icon="🌬" label="PM2.5" value={stats.pm25.value} delta={stats.pm25.delta} source={stats.pm25.source} delay={0.2} />
            </div>

            {/* Intelligence panel (shows when route is active) */}
            <AnimatePresence>
              {intelligenceReport && (
                <IntelligencePanel
                  report={intelligenceReport}
                  onClose={() => { setIntelligenceReport(null) }}
                />
              )}
            </AnimatePresence>
          </div>

          {/* ── RIGHT: CONTROL PANEL ── */}
          <motion.div
            className={`ov-console ${sidebarOpen ? 'open' : 'closed'}`}
            initial={{ x: 24, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ delay: 0.4, duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
          >
            <button className="ov-console-toggle" onClick={() => setSidebarOpen(s => !s)} title={sidebarOpen ? 'Collapse' : 'Expand'}>
              {sidebarOpen ? '›' : '‹'}
            </button>

            {sidebarOpen && (
              <>
                {/* Header */}
                <div className="ov-console-head">
                  <div className="ov-console-title">
                    <span className="ov-panel-label">CONTROL PANEL</span>
                    {routeActive && <span className="ov-console-route-active">● ROUTE ACTIVE</span>}
                  </div>
                  <div className="ov-mode-switch">
                    <button className={`ov-mode-btn ${mode === 'fast' ? 'active' : ''}`} onClick={() => setMode('fast')}><span>⚡</span> FAST</button>
                    <button className={`ov-mode-btn ${mode === 'deep' ? 'active' : ''}`} onClick={() => setMode('deep')}><span>🧠</span> DEEP</button>
                  </div>
                </div>

                {/* Tabs */}
                <div className="ov-tabs">
                  {TABS.map(tab => (
                    <button key={tab.id} className={`ov-tab ${controlTab === tab.id ? 'active' : ''} ${tab.id === 'route' ? 'route-tab' : ''}`} onClick={() => setControlTab(tab.id)}>
                      <span className="ov-tab-icon">{tab.icon}</span>
                      <span className="ov-tab-label">{tab.label}</span>
                      {tab.id === 'route' && routeActive && <span className="ov-tab-badge" />}
                    </button>
                  ))}
                </div>

                {/* Tab Content */}
                <div className="ov-tab-content">

                  {/* ── ROUTE TAB ── */}
                  {controlTab === 'route' && (
                    <div className="ov-tab-scroll">
                      {routeError && (
                        <div className="ov-error" style={{ marginBottom: '12px' }}>
                          <span>⚠ {routeError}</span>
                          <button onClick={() => setRouteError('')}>✕</button>
                        </div>
                      )}
                      <RouteSelector
                        clickMode={clickMode}
                        startName={startName}
                        endName={endName}
                        startCoords={startCoords}
                        endCoords={endCoords}
                        isLoading={routeLoading}
                        onStartSearch={(coords, name) => { setStartCoords(coords); setStartName(name) }}
                        onEndSearch={(coords, name) => { setEndCoords(coords); setEndName(name) }}
                        onSetClickMode={setClickMode}
                        onAnalyze={runRouteAnalysis}
                        onClear={clearRoute}
                      />
                      {routeActive && intelligenceReport && (
                        <div className="ov-route-summary-inline">
                          <div className="ov-rs-summary-title">LDRAGO REPORT</div>
                          <div className="ov-rs-summary-chips">
                            <span className="ov-intel-chip">{intelligenceReport.route.distance_km} km</span>
                            <span className="ov-intel-chip lime">{intelligenceReport.traffic.summary.predictedDuration_min} min</span>
                            <span className="ov-intel-chip" style={{ color: intelligenceReport.env.categoryColor }}>AQI {intelligenceReport.env.aqi}</span>
                          </div>
                          <div className="ov-rs-summary-congestion">
                            Congestion: <strong style={{ color: intelligenceReport.traffic.summary.avgCongestion > 60 ? '#FF4D00' : '#CCFF00' }}>
                              {intelligenceReport.prediction.congestionLabel} ({intelligenceReport.traffic.summary.avgCongestion}/100)
                            </strong>
                          </div>
                          {intelligenceReport.prediction.suggestions.slice(0, 2).map((s, i) => (
                            <div key={i} className="ov-rs-suggestion">→ {s}</div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}

                  {/* ── CHAT ── */}
                  {controlTab === 'chat' && (
                    <>
                      <div className="ov-chat">
                        {chatHistory.length === 0 && (
                          <div className="ov-chat-empty">
                            <div className="ov-chat-empty-orb">🌐</div>
                            <p>Ask anything about Delhi NCR — traffic, pollution, AQI, routes, economics.</p>
                            <p className="ov-chat-hint">Use the <strong>ROUTE</strong> tab to pick start &amp; end on the map for live route intelligence.</p>
                            <div className="ov-chat-capabilities">
                              <span className="ov-cap-chip">🧠 LDRAGOv2 6-Agent Pipeline</span>
                              <span className="ov-cap-chip">🔬 BPR Physics Engine</span>
                              <span className="ov-cap-chip">📡 Live AQI (Open-Meteo)</span>
                              <span className="ov-cap-chip">🗺️ Mapbox Traffic API</span>
                              <span className="ov-cap-chip">🌐 Auto Map Halos</span>
                            </div>
                          </div>
                        )}

                        {chatHistory.map((msg, i) => (
                          <div key={i} className={`ov-msg ${msg.role} ${msg.isError ? 'error' : ''}`}>
                            <div className="ov-msg-head">
                              <span className="ov-msg-role">
                                {msg.role === 'user' ? '◉ YOU' : '◈ LDRAGO'}
                              </span>
                              {msg.meta && (
                                <div className="ov-msg-badges">
                                  {msg.meta.model && msg.meta.model !== 'LDRAGOv2' && (
                                    <span className="ov-badge-model" title={msg.meta.model}>
                                      {msg.meta.model.length > 28 ? msg.meta.model.slice(0, 26) + '…' : msg.meta.model}
                                    </span>
                                  )}
                                  {msg.meta.runtime && <span className="ov-badge-time">{msg.meta.runtime}s</span>}
                                  {msg.meta.engineCount > 0 && <span className="ov-badge-engines">{msg.meta.engineCount} engines</span>}
                                  {msg.meta.locationCount > 0 && (
                                    <span className="ov-badge-halos" title="Locations highlighted on map">
                                      🌐 {msg.meta.locationCount} mapped
                                    </span>
                                  )}
                                </div>
                              )}
                            </div>
                            <div className="ov-msg-body">
                              {msg.role === 'assistant'
                                ? <ReactMarkdown>{msg.content}</ReactMarkdown>
                                : <p>{msg.content}</p>}
                            </div>
                            {msg.meta?.agentTrace?.length > 0 && (
                              <details className="ov-agent-trace">
                                <summary>Agent trace ({msg.meta.agentTrace.length} steps)</summary>
                                <div className="ov-trace-steps">
                                  {msg.meta.agentTrace.map((step, j) => (
                                    <div key={j} className="ov-trace-step">
                                      <span className="ov-trace-idx">{j + 1}</span>
                                      <span className="ov-trace-txt">
                                        {typeof step === 'string' ? step : step.agent || step.step || JSON.stringify(step)}
                                      </span>
                                    </div>
                                  ))}
                                </div>
                              </details>
                            )}
                            {msg.meta?.critique && (
                              <div className="ov-msg-critique">
                                <span className="ov-critique-label">🔍 Critique</span>
                                <span className="ov-critique-text">
                                  {typeof msg.meta.critique === 'string'
                                    ? msg.meta.critique
                                    : msg.meta.critique.verdict || JSON.stringify(msg.meta.critique)}
                                </span>
                              </div>
                            )}
                          </div>
                        ))}

                        {isAnalyzing && (
                          <div className="ov-msg assistant analyzing">
                            <div className="ov-msg-head">
                              <span className="ov-msg-role">◈ LDRAGO</span>
                              <span className="ov-msg-meta analyzing-label">{progressMessage}</span>
                            </div>
                            <div className="ov-progress-bar">
                              <div className="ov-progress-fill" style={{ width: `${analysisProgress}%` }} />
                            </div>
                            <div className="ov-progress-pct">{analysisProgress}%</div>
                            <div className="ov-progress-dots"><span /><span /><span /></div>
                          </div>
                        )}
                      </div>

                      {errorMessage && (
                        <div className="ov-error">
                          <span>⚠ {errorMessage}</span>
                          <button onClick={() => setErrorMessage('')}>✕</button>
                        </div>
                      )}

                      <div className="ov-input-area">
                        <textarea
                          className="ov-input"
                          value={prompt}
                          onChange={e => setPrompt(e.target.value)}
                          placeholder="Ask about traffic, AQI, route cost, metro vs car, best departure time..."
                          rows={2}
                          disabled={isAnalyzing}
                          onKeyDown={e => { if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') { e.preventDefault(); runAnalysis() } }}
                        />
                        <button className="ov-send" onClick={() => runAnalysis()} disabled={isAnalyzing || !prompt.trim()}>
                          {isAnalyzing ? <span className="ov-spinner" /> : <span>→</span>}
                        </button>
                      </div>
                      <div className="ov-input-hint">⌘ + Enter · {mode === 'fast' ? '⚡ Fast ~10s' : '🧠 Deep ~30s'} · 💬 Context-aware · 🗺 Auto halos</div>
                    </>
                  )}

                  {/* ── RESULTS ── */}
                  {controlTab === 'results' && (
                    <div className="ov-tab-scroll">
                      {engineDomains ? (
                        <EngineResults domains={engineDomains} recommendations={engineRecommendations} warnings={engineWarnings} impactCards={impactCards} />
                      ) : (
                        <div className="ov-chat-empty">
                          <div className="ov-chat-empty-orb">📊</div>
                          <p>Run a simulation or template first to view engine results here.</p>
                        </div>
                      )}
                    </div>
                  )}

                  {/* ── TEMPLATES ── */}
                  {controlTab === 'templates' && (
                    <div className="ov-tab-scroll">
                      <ScenarioTemplates onSelectTemplate={handleSelectTemplate} onRunTemplate={handleRunTemplate} />
                    </div>
                  )}

                  {/* ── BUILDER ── */}
                  {controlTab === 'builder' && (
                    <div className="ov-tab-scroll">
                      <ScenarioBuilder onBuildScenario={handleBuildScenario} />
                    </div>
                  )}

                  {/* ── COMPARE ── */}
                  {controlTab === 'compare' && (
                    <div className="ov-tab-scroll">
                      <ScenarioComparison />
                    </div>
                  )}
                </div>
              </>
            )}
          </motion.div>

          {/* ── BOTTOM: QUICK SCENARIOS ── */}
          <div className="ov-scenarios">
            <span className="ov-scenarios-label">QUICK SCENARIOS</span>
            <div className="ov-scenarios-pills">
              {QUICK_SCENARIOS.map(s => (
                <ScenarioPill key={s.id} icon={s.icon} label={s.label} active={activeScenario === s.id} onClick={() => handleScenario(s)} />
              ))}
            </div>
          </div>

          {/* ── TRAFFIC LEGEND / MAP LEGEND (bottom left) ── */}
          {mapReady && (
            <motion.div className="ov-legend-container" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 1 }}>
              <TrafficLegend summary={intelligenceReport?.traffic?.summary || null} />
              {!routeActive && (
                <div className="ov-legend" style={{ marginTop: '6px' }}>
                  <div className="ov-panel-label" style={{ marginBottom: '8px' }}>MAP LAYERS</div>
                  <div className="ov-legend-row"><span className="ov-legend-line" style={{ background: '#CCFF00' }} /><span className="ov-legend-row-label">EV Corridor</span></div>
                  <div className="ov-legend-row"><span className="ov-legend-dot" style={{ background: 'rgba(255,77,0,0.7)' }} /><span className="ov-legend-row-label">Pollution Zone</span></div>
                </div>
              )}
            </motion.div>
          )}
        </div>
      )}
    </>
  )
}
