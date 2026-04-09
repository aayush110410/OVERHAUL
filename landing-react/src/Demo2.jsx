/**
 * DEMO 2 — OVERHAUL COMMAND CENTER
 * 
 * Design Philosophy: "Mission Control meets Luxury"
 * 
 * A single-screen command center with a full-bleed map as the canvas,
 * floating glass panels for controls and results, and a conversational
 * AI sidebar. Everything is visible at once — no scrolling, no tabs
 * hidden behind other tabs. The map IS the page.
 * 
 * Layout: Full-viewport map with floating overlays
 *   ┌─────────────────────────────────────────────────────┐
 *   │ ╔═══NAV════════════════════════════════════════════╗ │
 *   │ ║  OV   ●LIVE    Delhi NCR    ⏱ 14:32   ← Home   ║ │
 *   │ ╚═════════════════════════════════════════════════ ╝ │
 *   │                                                     │
 *   │  ┌──────────┐                    ┌────────────────┐ │
 *   │  │ METRICS  │      M A P        │  AI CONSOLE    │ │
 *   │  │ (glass)  │                    │  (glass)       │ │
 *   │  │  4 cards │                    │  chat + input  │ │
 *   │  └──────────┘                    │  mode toggle   │ │
 *   │                                  │  results       │ │
 *   │  ┌──────────┐                    └────────────────┘ │
 *   │  │ LAYERS   │                                       │
 *   │  └──────────┘                                       │
 *   │                                                     │
 *   │  ┌──────────────────────────────────────────────┐   │
 *   │  │  QUICK SCENARIOS  (3 pill buttons)            │   │
 *   │  └──────────────────────────────────────────────┘   │
 *   └─────────────────────────────────────────────────────┘
 */

import { useState, useEffect, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useNavigate, useLocation, Link } from 'react-router-dom'
import mapboxgl from 'mapbox-gl'
import 'mapbox-gl/dist/mapbox-gl.css'

const MAPBOX_TOKEN = (import.meta.env.VITE_MAPBOX_TOKEN || '').trim()
mapboxgl.accessToken = MAPBOX_TOKEN
import ReactMarkdown from 'react-markdown'
import { API_BASE } from './api/config'
import ScenarioTemplates from './components/ScenarioTemplates'
import ScenarioBuilder from './components/ScenarioBuilder'
import ScenarioComparison from './components/ScenarioComparison'
import EngineResults from './components/EngineResults'
import './Demo2.css'

// ============================================
// MAP STYLE — Mapbox Dark
// ============================================
const MAPBOX_STYLE = 'mapbox://styles/mapbox/dark-v11'

// ============================================
// LOADING SCREEN
// ============================================
function CommandCenterLoader({ onComplete }) {
  const [progress, setProgress] = useState(0)
  const [phase, setPhase] = useState('boot') // boot → loading → ready → exit

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
          setTimeout(() => {
            setPhase('exit')
            setTimeout(onComplete, 600)
          }, 300)
          return 100
        }
        return p + 3
      })
    }, 25)
    return () => clearInterval(interval)
  }, [phase, onComplete])

  return (
    <motion.div
      className="d2-loader"
      initial={{ opacity: 1 }}
      animate={phase === 'exit' ? { opacity: 0 } : { opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <div className="d2-loader-inner">
        <motion.div
          className="d2-loader-logo"
          initial={{ scale: 0.7, opacity: 0 }}
          animate={phase === 'exit' ? { scale: 3, opacity: 0 } : { scale: 1, opacity: 1 }}
          transition={{ duration: phase === 'exit' ? 0.5 : 0.4, ease: [0.16, 1, 0.3, 1] }}
        >
          <span className="d2-logo-o">O</span>
          <span className="d2-logo-v">V</span>
        </motion.div>

        <div className="d2-loader-label">COMMAND CENTER</div>

        <div className="d2-loader-bar">
          <motion.div
            className="d2-loader-fill"
            initial={{ width: 0 }}
            animate={{ width: `${progress}%` }}
            transition={{ duration: 0.1 }}
          />
        </div>
        <div className="d2-loader-pct">{progress}%</div>
      </div>
    </motion.div>
  )
}

// ============================================
// METRIC CARD (floating glass)
// ============================================
function MetricCard({ icon, label, value, delta, source, delay = 0 }) {
  const isPositive = delta && !delta.startsWith('+') && delta.includes('-')
  return (
    <motion.div
      className="d2-metric"
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
    >
      <div className="d2-metric-icon">{icon}</div>
      <div className="d2-metric-body">
        <div className="d2-metric-label">{label}</div>
        <div className="d2-metric-value">{value}</div>
        {delta && (
          <div className={`d2-metric-delta ${isPositive ? 'pos' : 'neg'}`}>{delta}</div>
        )}
      </div>
      {source && <div className="d2-metric-source">{source}</div>}
    </motion.div>
  )
}

// ============================================
// QUICK SCENARIO PILL
// ============================================
function ScenarioPill({ label, icon, onClick, active }) {
  return (
    <button
      className={`d2-pill ${active ? 'active' : ''}`}
      onClick={onClick}
    >
      <span className="d2-pill-icon">{icon}</span>
      {label}
    </button>
  )
}

// ============================================
// CUSTOM CURSOR
// ============================================
function CustomCursor() {
  const cursorRef = useRef(null)

  useEffect(() => {
    const el = cursorRef.current
    if (!el) return

    const moveCursor = (e) => {
      el.style.transform = `translate3d(${e.clientX}px, ${e.clientY}px, 0) translate(-50%, -50%)`
    }

    const handleOver = (e) => {
      if (e.target.tagName === 'A' || e.target.tagName === 'BUTTON' ||
          e.target.closest('a') || e.target.closest('button')) {
        el.classList.add('hovering')
      }
    }

    const handleOut = (e) => {
      if (e.target.tagName === 'A' || e.target.tagName === 'BUTTON' ||
          e.target.closest('a') || e.target.closest('button')) {
        el.classList.remove('hovering')
      }
    }

    document.addEventListener('mousemove', moveCursor, { passive: true })
    document.addEventListener('mouseover', handleOver, { passive: true })
    document.addEventListener('mouseout', handleOut, { passive: true })

    return () => {
      document.removeEventListener('mousemove', moveCursor)
      document.removeEventListener('mouseover', handleOver)
      document.removeEventListener('mouseout', handleOut)
    }
  }, [])

  return <div ref={cursorRef} className="cursor" />
}

// ============================================
// MAIN COMPONENT
// ============================================
export default function Demo2() {
  const navigate = useNavigate()
  const location = useLocation()
  const skipLoader = location.state?.skipLoader

  // UI state
  const [loading, setLoading] = useState(!skipLoader)
  const [currentTime, setCurrentTime] = useState('')
  const [sidebarOpen, setSidebarOpen] = useState(true)

  // Map
  const mapContainer = useRef(null)
  const mapRef = useRef(null)
  const [mapReady, setMapReady] = useState(false)

  // Analysis
  const [prompt, setPrompt] = useState('')
  const [mode, setMode] = useState('fast')
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysisProgress, setAnalysisProgress] = useState(0)
  const [progressMessage, setProgressMessage] = useState('')
  const [errorMessage, setErrorMessage] = useState('')

  // Results
  const [chatHistory, setChatHistory] = useState([])
  const [stats, setStats] = useState({
    travelTime: { value: '--', source: '', delta: null },
    pm25:       { value: '--', source: '', delta: null },
    vkt:        { value: '--', source: '', delta: null },
    co2:        { value: '--', source: '', delta: null },
  })
  const [liveSources, setLiveSources] = useState([])
  const [analysisComplete, setAnalysisComplete] = useState(false)

  // Control panel tabs
  const [controlTab, setControlTab] = useState('chat') // chat | templates | builder | compare | temporal
  // Engine results
  const [engineDomains, setEngineDomains] = useState(null)
  const [engineRecommendations, setEngineRecommendations] = useState([])
  const [engineWarnings, setEngineWarnings] = useState([])
  const [impactCards, setImpactCards] = useState([])

  // Quick scenarios
  const [activeScenario, setActiveScenario] = useState(null)
  const QUICK_SCENARIOS = [
    { id: 'green', icon: '🌿', label: 'Green Delhi 2030', prompt: 'What if Delhi bans all private cars in Connaught Place and adds 500 electric buses on Ring Road?' },
    { id: 'metro', icon: '🚇', label: 'Metro Expansion', prompt: 'Simulate extending Delhi Metro Phase 4 with 25 new stations covering Dwarka, Najafgarh, and Aerocity by 2028' },
    { id: 'ev', icon: '⚡', label: 'EV Revolution', prompt: 'What happens if 40% of Delhi NCR vehicles switch to electric by 2027 with 2000 fast charging stations?' },
  ]

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

  // ==========================================
  // MAP INITIALIZATION
  // ==========================================
  useEffect(() => {
    if (loading || !mapContainer.current || mapRef.current) return

    const map = new mapboxgl.Map({
      container: mapContainer.current,
      style: MAPBOX_STYLE,
      center: [77.21, 28.61],
      zoom: 11.5,
      pitch: 55,
      bearing: -15,
      maxPitch: 75,
      antialias: true,
    })

    map.addControl(new mapboxgl.NavigationControl({ showCompass: true, showZoom: false }), 'bottom-right')

    map.on('error', (e) => console.error('[Map]', e.error?.message || e))

    map.on('load', () => {
      // ── 3D Terrain ──
      map.addSource('mapbox-dem', {
        type: 'raster-dem',
        url: 'mapbox://mapbox.mapbox-terrain-dem-v1',
        tileSize: 512,
        maxzoom: 14
      })
      map.setTerrain({ source: 'mapbox-dem', exaggeration: 1.3 })

      // ── 3D Buildings ──
      const layers = map.getStyle().layers
      const labelLayerId = layers.find(l => l.type === 'symbol' && l.layout?.['text-field'])?.id
      map.addLayer({
        id: '3d-buildings',
        source: 'composite',
        'source-layer': 'building',
        filter: ['==', 'extrude', 'true'],
        type: 'fill-extrusion',
        minzoom: 12,
        paint: {
          'fill-extrusion-color': '#0f1923',
          'fill-extrusion-height': ['get', 'height'],
          'fill-extrusion-base': ['get', 'min_height'],
          'fill-extrusion-opacity': 0.7
        }
      }, labelLayerId)

      // ── Sky atmosphere ──
      map.addLayer({
        id: 'sky',
        type: 'sky',
        paint: {
          'sky-type': 'atmosphere',
          'sky-atmosphere-sun': [0.0, 0.0],
          'sky-atmosphere-sun-intensity': 5
        }
      })

      // Sources
      map.addSource('edges', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } })
      map.addSource('pollution', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } })
      map.addSource('route', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } })

      // Pollution heatmap
      map.addLayer({
        id: 'pollution-heat',
        type: 'circle',
        source: 'pollution',
        paint: {
          'circle-radius': ['interpolate', ['linear'], ['get', 'intensity'], 0, 0, 1, 50],
          'circle-color': 'rgba(255, 77, 0, 0.25)',
          'circle-stroke-color': 'rgba(255, 77, 0, 0.6)',
          'circle-stroke-width': 1,
          'circle-blur': 0.6,
          'circle-opacity': 0.6
        }
      })

      // Traffic edges
      map.addLayer({
        id: 'edges-layer',
        type: 'line',
        source: 'edges',
        paint: {
          'line-color': ['interpolate', ['linear'], ['get', 'ev_share'], 0, '#334155', 25, '#CCFF00', 60, '#00ff88', 90, '#ffff00'],
          'line-width': ['case', ['get', 'primary'], 6, 3],
          'line-opacity': 0.85,
          'line-blur': 0.5
        }
      })

      // Route glow
      map.addLayer({
        id: 'route-glow',
        type: 'line',
        source: 'route',
        paint: { 'line-color': '#CCFF00', 'line-width': 14, 'line-opacity': 0.2, 'line-blur': 10 }
      })
      map.addLayer({
        id: 'route-line',
        type: 'line',
        source: 'route',
        paint: { 'line-color': '#CCFF00', 'line-width': 3, 'line-opacity': 0.9 }
      })

      setMapReady(true)
      fetchLiveData(map)
    })

    mapRef.current = map
    setTimeout(() => map.resize(), 50)

    return () => { map.remove(); mapRef.current = null; setMapReady(false) }
  }, [loading])

  // ==========================================
  // LIVE DATA FETCHING
  // ==========================================
  const fetchWithRetry = async (url, options = {}, retries = 2, timeout = 30000) => {
    for (let i = 0; i <= retries; i++) {
      try {
        const ctrl = new AbortController()
        const tid = setTimeout(() => ctrl.abort(), timeout)
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
    // Route
    try {
      const resp = await fetchWithRetry(`${API_BASE}/live/route`)
      if (resp.ok) {
        const geojson = await resp.json()
        if (map.getSource('route')) map.getSource('route').setData(geojson)
        const src = geojson?.features?.[0]?.properties?.source || 'Route API'
        setLiveSources(prev => prev.find(s => s.name === src) ? prev : [...prev, { name: src, detail: 'Live corridor' }])
      }
    } catch (e) { console.warn('[Live] Route fetch failed:', e.message) }

    // AQI
    try {
      const resp = await fetchWithRetry(`${API_BASE}/live/aqi?lat=28.62&lon=77.35`)
      if (resp.ok) {
        const data = await resp.json()
        const series = Array.isArray(data.series) ? data.series.filter(pt => pt.datetime && pt.pm25 != null) : []
        const latest = series.length ? series[series.length - 1] : null
        if (latest) {
          setStats(s => ({
            ...s,
            pm25: { value: `${latest.pm25.toFixed(1)} µg/m³`, source: `Live • ${data.source || 'AQI'}`, delta: null }
          }))
        }
        const aqiSrc = data.source ? String(data.source) : 'AQI Feed'
        setLiveSources(prev => prev.find(s => s.name === aqiSrc) ? prev : [...prev, { name: aqiSrc, detail: 'Air quality' }])
      }
    } catch (e) { console.warn('[Live] AQI fetch failed:', e.message) }
  }

  // ==========================================
  // MAP UPDATE
  // ==========================================
  const updateMap = useCallback((edges, pollution) => {
    const map = mapRef.current
    if (!map) return

    if (edges && map.getSource('edges')) {
      const gj = edges.type === 'FeatureCollection' ? edges : { type: 'FeatureCollection', features: Array.isArray(edges) ? edges : [] }
      map.getSource('edges').setData(gj)
      if (gj.features.length) {
        const bounds = gj.features.reduce((b, f) => {
          if (f.geometry?.coordinates) f.geometry.coordinates.forEach(c => b.extend(c))
          return b
        }, new mapboxgl.LngLatBounds())
        if (!bounds.isEmpty()) map.fitBounds(bounds, { padding: 80, maxZoom: 14 })
      }
    }

    if (pollution && map.getSource('pollution')) {
      const gj = pollution.type === 'FeatureCollection' ? pollution : { type: 'FeatureCollection', features: Array.isArray(pollution) ? pollution : [] }
      map.getSource('pollution').setData(gj)
    }
  }, [])

  // ==========================================
  // RUN ANALYSIS
  // ==========================================
  const runAnalysis = useCallback(async (overridePrompt) => {
    const text = overridePrompt || prompt
    if (!text.trim() || isAnalyzing) return

    setIsAnalyzing(true)
    setAnalysisProgress(0)
    setProgressMessage('Connecting to AI...')
    setErrorMessage('')
    setPrompt('')

    // Add user message to chat
    setChatHistory(prev => [...prev, { role: 'user', content: text }])

    const start = Date.now()
    const progressInterval = setInterval(() => {
      const elapsed = Date.now() - start
      const raw = Math.min(92, (elapsed / 18000) * 100)
      setAnalysisProgress(Math.round(raw))
      if (raw < 15) setProgressMessage('Connecting to AI...')
      else if (raw < 35) setProgressMessage('Loading NCR datasets...')
      else if (raw < 55) setProgressMessage('Analyzing traffic patterns...')
      else if (raw < 75) setProgressMessage('Processing air quality data...')
      else setProgressMessage('Generating insights...')
    }, 200)

    try {
      const resp = await fetchWithRetry(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: text,
          mode,
          scenario: {
            title: 'Prompt-driven analysis',
            region: 'Delhi NCR',
            intervention: 'prompt_only',
            horizon_months: 12,
            parameters: {}
          }
        })
      }, 3, 60000)

      clearInterval(progressInterval)
      setAnalysisProgress(100)
      setProgressMessage('Complete')

      if (!resp.ok) {
        const errText = await resp.text()
        throw new Error(errText || 'Analysis failed')
      }

      const data = await resp.json()

      // Extract main response
      const mainResponse = data.outputs?.response || data.summary || 'Analysis complete.'
      setChatHistory(prev => [
        ...prev,
        {
          role: 'assistant',
          content: mainResponse,
          meta: {
            mode: data.mode || mode,
            model: data.outputs?.model,
            engineCount: data.outputs?.domains ? Object.keys(data.outputs.domains).length : 0,
            runtime: ((Date.now() - start) / 1000).toFixed(1)
          }
        }
      ])

      // Update stats
      const getDelta = (theme) => {
        if (!data.outputs?.impactCards) return null
        const card = data.outputs.impactCards.find(c => c.theme?.toLowerCase().includes(theme))
        if (card?.delta) {
          const m = card.delta.match(/([-+]?\d+\.?\d*)/)
          return m ? parseFloat(m[1]) : null
        }
        return null
      }

      if (data.baseline) {
        const live = data.live || {}
        const travelVal = live?.travel?.travel_time_min ?? data.baseline.avg_travel_time_min
        const pmVal = live?.aqi?.latest_pm25 ?? data.baseline.pm25

        setStats({
          travelTime: {
            value: travelVal ? `${travelVal.toFixed(1)} min` : '--',
            source: live?.travel ? `Live • ${live.travel.source || 'OSRM'}` : 'Model',
            delta: getDelta('travel') ? `${getDelta('travel') > 0 ? '+' : ''}${getDelta('travel').toFixed(1)}%` : null
          },
          pm25: {
            value: pmVal ? `${pmVal.toFixed(1)} µg/m³` : '--',
            source: live?.aqi ? `Live • ${live.aqi.source || 'AQI'}` : 'Model',
            delta: getDelta('air') ? `${getDelta('air') > 0 ? '+' : ''}${getDelta('air').toFixed(1)}%` : null
          },
          vkt: {
            value: data.baseline.total_vkt ? `${Math.round(data.baseline.total_vkt)} km` : '--',
            source: 'Model',
            delta: getDelta('vkt') ? `${getDelta('vkt') > 0 ? '+' : ''}${getDelta('vkt').toFixed(1)}%` : null
          },
          co2: {
            value: data.baseline.co2_kg ? `${Math.round(data.baseline.co2_kg)} kg` : '--',
            source: 'Model',
            delta: getDelta('co2') ? `${getDelta('co2') > 0 ? '+' : ''}${getDelta('co2').toFixed(1)}%` : null
          }
        })
      }

      // Update live sources
      if (data.live?.sources) {
        setLiveSources(prev => {
          const next = [...prev]
          data.live.sources.forEach(src => {
            if (!next.find(s => s.name === src.name)) next.push(src)
          })
          return next
        })
      }

      // Update map
      if (data.geojson) updateMap(data.geojson.edges, data.geojson.pollution)

      // Store engine domain data
      if (data.outputs?.domains) {
        setEngineDomains(data.outputs.domains)
        setTimeout(() => setControlTab('results'), 300)
      }
      if (data.outputs?.ranked) setEngineRecommendations(data.outputs.ranked)
      if (data.outputs?.warnings) setEngineWarnings(data.outputs.warnings)
      if (data.outputs?.impactCards) setImpactCards(data.outputs.impactCards)

      setAnalysisComplete(true)
    } catch (err) {
      clearInterval(progressInterval)
      console.error('[Analysis]', err)
      setErrorMessage(err.message)
      setChatHistory(prev => [...prev, { role: 'assistant', content: `Error: ${err.message}`, isError: true }])
    } finally {
      setIsAnalyzing(false)
    }
  }, [prompt, mode, isAnalyzing, updateMap])

  // Quick scenario handler
  const handleScenario = useCallback((scenario) => {
    setActiveScenario(scenario.id)
    setControlTab('chat')
    setPrompt(scenario.prompt)
    runAnalysis(scenario.prompt)
  }, [runAnalysis])

  // Template handlers (from ScenarioTemplates component)
  const handleSelectTemplate = useCallback((template) => {
    setPrompt(template.description || template.title)
    setControlTab('chat')
  }, [])

  const handleRunTemplate = useCallback((result) => {
    // result shape: { template, scenario, description, results: { domains, recommendations, warnings, impactCards, ... } }
    const r = result?.results || result?.outputs || {}
    if (r.domains) setEngineDomains(r.domains)
    if (r.recommendations) setEngineRecommendations(r.recommendations)
    if (r.ranked) setEngineRecommendations(r.ranked)
    if (r.warnings) setEngineWarnings(r.warnings)
    if (r.impactCards) setImpactCards(r.impactCards)

    // Build a chat message from the scenario results
    const title = result?.scenario || result?.template || 'Scenario'
    let responseText = result?.description || ''
    if (r.recommendations?.length) {
      responseText += '\n\n**Key Recommendations:**\n' + r.recommendations.map((rec, i) => `${i + 1}. ${typeof rec === 'string' ? rec : rec.text || rec.recommendation || JSON.stringify(rec)}`).join('\n')
    }
    if (r.impactCards?.length) {
      responseText += '\n\n**Impact Summary:**\n' + r.impactCards.map(c => `- **${c.metric}**: ${c.value} (${c.delta || ''})`).join('\n')
    }
    setChatHistory(prev => [
      ...prev,
      { role: 'user', content: `[Template] ${title}` },
      { role: 'assistant', content: responseText || 'Analysis complete.' }
    ])
    if (r.domains) setTimeout(() => setControlTab('results'), 300)
    else setControlTab('chat')
    setAnalysisComplete(true)
  }, [])

  // Builder handler
  const handleBuildScenario = useCallback((scenario) => {
    setPrompt(scenario.prompt || scenario)
    setControlTab('chat')
    setTimeout(() => runAnalysis(scenario.prompt || scenario), 100)
  }, [runAnalysis])

  // Follow-up from chat
  const handleFollowUp = useCallback((text) => {
    setPrompt(text)
    setControlTab('chat')
    setTimeout(() => runAnalysis(text), 100)
  }, [runAnalysis])

  // Keyboard shortcut
  useEffect(() => {
    const handler = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
        e.preventDefault()
        runAnalysis()
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [runAnalysis])

  // ==========================================
  // RENDER
  // ==========================================
  return (
    <>
      {/* LOADER */}
      <AnimatePresence>
        {loading && <CommandCenterLoader onComplete={() => setLoading(false)} />}
      </AnimatePresence>

      {!loading && (
        <div className="d2">
          {/* ——— FULL-BLEED MAP ——— */}
          <div ref={mapContainer} className="d2-map" />

          {/* ——— TOP NAV BAR ——— */}
          <nav className="d2-nav">
            <div className="d2-nav-left">
              <Link to="/" className="d2-nav-logo">
                <span className="d2-nav-o">O</span>
                <span className="d2-nav-v">V</span>
              </Link>
              <div className="d2-nav-divider" />
              <div className="d2-nav-status">
                <span className="d2-status-dot" />
                <span>LIVE</span>
              </div>
              <div className="d2-nav-divider" />
              <span className="d2-nav-region">Delhi NCR</span>
            </div>

            <div className="d2-nav-center">
              <span className="d2-nav-title">COMMAND CENTER</span>
            </div>

            <div className="d2-nav-right">
              <span className="d2-nav-clock">{currentTime}</span>
              <div className="d2-nav-divider" />
              {liveSources.length > 0 && (
                <span className="d2-nav-feeds">{liveSources.length} feed{liveSources.length > 1 ? 's' : ''}</span>
              )}
              <div className="d2-nav-divider" />
              <Link to="/flyover" className="d2-nav-link">3D Flyover</Link>
              <Link to="/demo" className="d2-nav-link">V1 Demo</Link>
              <Link to="/" className="d2-nav-link">Home</Link>
            </div>
          </nav>

          {/* ——— LEFT: METRICS PANEL ——— */}
          <div className="d2-metrics-panel">
            <MetricCard icon="🚗" label="Travel Time" value={stats.travelTime.value} delta={stats.travelTime.delta} source={stats.travelTime.source} delay={0.1} />
            <MetricCard icon="🌬️" label="PM2.5" value={stats.pm25.value} delta={stats.pm25.delta} source={stats.pm25.source} delay={0.2} />
            <MetricCard icon="📏" label="VKT" value={stats.vkt.value} delta={stats.vkt.delta} source={stats.vkt.source} delay={0.3} />
            <MetricCard icon="🏭" label="CO₂" value={stats.co2.value} delta={stats.co2.delta} source={stats.co2.source} delay={0.4} />
          </div>

          {/* ——— RIGHT: CONTROL PANEL ——— */}
          <motion.div
            className={`d2-console ${sidebarOpen ? 'open' : 'closed'}`}
            initial={{ x: 20, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ delay: 0.3, duration: 0.5 }}
          >
            {/* Toggle */}
            <button className="d2-console-toggle" onClick={() => setSidebarOpen(s => !s)}>
              {sidebarOpen ? '▶' : '◀'}
            </button>

            {sidebarOpen && (
              <>
                {/* Header with mode switch */}
                <div className="d2-console-header">
                  <span className="d2-console-title">Control Panel</span>
                  <div className="d2-mode-switch">
                    <button className={`d2-mode-btn ${mode === 'fast' ? 'active' : ''}`} onClick={() => setMode('fast')}>
                      ⚡ Fast
                    </button>
                    <button className={`d2-mode-btn ${mode === 'deep' ? 'active' : ''}`} onClick={() => setMode('deep')}>
                      🧠 Deep
                    </button>
                  </div>
                </div>

                {/* Tab Navigation */}
                <div className="d2-tabs">
                  {[
                    { id: 'chat', label: 'CHAT', icon: '💬' },
                    { id: 'results', label: 'RESULTS', icon: '📊' },
                    { id: 'templates', label: 'TEMPLATES', icon: '📋' },
                    { id: 'builder', label: 'BUILDER', icon: '🔧' },
                    { id: 'compare', label: 'COMPARE', icon: '⚖️' },
                  ].map(tab => (
                    <button
                      key={tab.id}
                      className={`d2-tab ${controlTab === tab.id ? 'active' : ''}`}
                      onClick={() => setControlTab(tab.id)}
                    >
                      <span className="d2-tab-icon">{tab.icon}</span>
                      <span className="d2-tab-label">{tab.label}</span>
                    </button>
                  ))}
                </div>

                {/* Tab Content */}
                <div className="d2-tab-content">
                  {/* ——— CHAT TAB ——— */}
                  {controlTab === 'chat' && (
                    <>
                      <div className="d2-chat">
                        {chatHistory.length === 0 && (
                          <div className="d2-chat-empty">
                            <div className="d2-chat-empty-icon">🌐</div>
                            <p>Ask anything about Delhi NCR traffic, pollution, or urban planning.</p>
                            <p className="d2-chat-hint">Try a quick scenario below, or type your own.</p>
                          </div>
                        )}

                        {chatHistory.map((msg, i) => (
                          <div key={i} className={`d2-chat-msg ${msg.role} ${msg.isError ? 'error' : ''}`}>
                            <div className="d2-chat-msg-header">
                              <span className="d2-chat-msg-role">
                                {msg.role === 'user' ? '👤 You' : '🤖 Overhaul AI'}
                              </span>
                              {msg.meta && (
                                <span className="d2-chat-msg-meta">
                                  {msg.meta.runtime}s • {msg.meta.mode} • {msg.meta.engineCount} engines
                                </span>
                              )}
                            </div>
                            <div className="d2-chat-msg-body">
                              {msg.role === 'assistant' ? (
                                <ReactMarkdown>{msg.content}</ReactMarkdown>
                              ) : (
                                <p>{msg.content}</p>
                              )}
                            </div>
                          </div>
                        ))}

                        {isAnalyzing && (
                          <div className="d2-chat-msg assistant analyzing">
                            <div className="d2-analyzing-bar">
                              <div className="d2-analyzing-fill" style={{ width: `${analysisProgress}%` }} />
                            </div>
                            <span className="d2-analyzing-text">{progressMessage}</span>
                          </div>
                        )}
                      </div>

                      {/* Error */}
                      {errorMessage && (
                        <div className="d2-error">
                          <span>⚠️</span> {errorMessage}
                          <button onClick={() => setErrorMessage('')}>✕</button>
                        </div>
                      )}

                      {/* Input */}
                      <div className="d2-input-area">
                        <textarea
                          className="d2-input"
                          value={prompt}
                          onChange={e => setPrompt(e.target.value)}
                          placeholder="What if Delhi bans diesel vehicles in 2027..."
                          rows={2}
                          disabled={isAnalyzing}
                          onKeyDown={e => {
                            if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
                              e.preventDefault()
                              runAnalysis()
                            }
                          }}
                        />
                        <button
                          className="d2-send"
                          onClick={() => runAnalysis()}
                          disabled={isAnalyzing || !prompt.trim()}
                        >
                          {isAnalyzing ? <span className="d2-send-spinner" /> : '→'}
                        </button>
                      </div>
                      <div className="d2-input-hint">⌘+Enter to send • {mode === 'fast' ? 'Fast mode ~10s' : 'Deep mode ~30s'}</div>
                    </>
                  )}

                  {/* ——— RESULTS TAB ——— */}
                  {controlTab === 'results' && (
                    <div className="d2-tab-scroll">
                      {engineDomains ? (
                        <EngineResults
                          domains={engineDomains}
                          recommendations={engineRecommendations}
                          warnings={engineWarnings}
                          impactCards={impactCards}
                        />
                      ) : (
                        <div className="d2-chat-empty">
                          <div className="d2-chat-empty-icon">📊</div>
                          <p>Run a simulation or template first to see engine results here.</p>
                        </div>
                      )}
                    </div>
                  )}

                  {/* ——— TEMPLATES TAB ——— */}
                  {controlTab === 'templates' && (
                    <div className="d2-tab-scroll">
                      <ScenarioTemplates
                        onSelectTemplate={handleSelectTemplate}
                        onRunTemplate={handleRunTemplate}
                      />
                    </div>
                  )}

                  {/* ——— BUILDER TAB ——— */}
                  {controlTab === 'builder' && (
                    <div className="d2-tab-scroll">
                      <ScenarioBuilder
                        onBuildScenario={handleBuildScenario}
                      />
                    </div>
                  )}

                  {/* ——— COMPARE TAB ——— */}
                  {controlTab === 'compare' && (
                    <div className="d2-tab-scroll">
                      <ScenarioComparison />
                    </div>
                  )}
                </div>
              </>
            )}
          </motion.div>

          {/* ——— BOTTOM: QUICK SCENARIOS ——— */}
          <div className="d2-scenarios">
            {QUICK_SCENARIOS.map(s => (
              <ScenarioPill
                key={s.id}
                icon={s.icon}
                label={s.label}
                active={activeScenario === s.id}
                onClick={() => handleScenario(s)}
              />
            ))}
          </div>

          {/* ——— BOTTOM LEFT: DATA SOURCES BADGE ——— */}
          {liveSources.length > 0 && (
            <div className="d2-sources">
              <span className="d2-sources-label">DATA FEEDS</span>
              {liveSources.map((s, i) => (
                <span key={i} className="d2-source-tag">{s.name}</span>
              ))}
            </div>
          )}

          {/* Custom Cursor */}
          <CustomCursor />
        </div>
      )}
    </>
  )
}
