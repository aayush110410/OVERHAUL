import { useState, useEffect, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useNavigate, useLocation } from 'react-router-dom'
import maplibregl from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Filler, Tooltip } from 'chart.js'
import { Line } from 'react-chartjs-2'
import './App.css'

// Register Chart.js components
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Filler, Tooltip)

// API Base URL
const API_BASE = typeof window !== 'undefined' && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')
  ? 'http://localhost:8000'
  : 'https://overhaul-1.onrender.com'

// ============================================
// OV LOADER (Same style as Contact page)
// ============================================
function OVLoader({ onComplete }) {
  const [phase, setPhase] = useState('zoomOut') // zoomOut -> hold -> zoomIn -> done
  const [count, setCount] = useState(0)
  
  useEffect(() => {
    // Count up animation
    const countInterval = setInterval(() => {
      setCount(prev => {
        if (prev >= 100) {
          clearInterval(countInterval)
          return 100
        }
        return prev + 2
      })
    }, 25)

    // Phase 1: Zoom out from large scale
    const holdTimer = setTimeout(() => {
      setPhase('hold')
    }, 600)
    
    return () => {
      clearTimeout(holdTimer)
      clearInterval(countInterval)
    }
  }, [])
  
  useEffect(() => {
    if (phase === 'hold') {
      // Phase 2: Hold at normal size while loading completes
      const zoomInTimer = setTimeout(() => {
        setPhase('zoomIn')
      }, 1200)
      return () => clearTimeout(zoomInTimer)
    }
  }, [phase])
  
  useEffect(() => {
    if (phase === 'zoomIn') {
      // Phase 3: Zoom in and fade out
      const completeTimer = setTimeout(() => {
        onComplete()
      }, 600)
      return () => clearTimeout(completeTimer)
    }
  }, [phase, onComplete])

  return (
    <motion.div 
      className="loader-ln"
      initial={{ opacity: 0 }}
      animate={{ opacity: phase === 'zoomIn' ? 0 : 1 }}
      transition={{ 
        duration: phase === 'zoomIn' ? 0.4 : 0.3, 
        ease: [0.4, 0, 0.2, 1],
        delay: phase === 'zoomIn' ? 0.3 : 0
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
            duration: phase === 'zoomOut' ? 0.6 : phase === 'zoomIn' ? 0.6 : 0.1,
            ease: [0.4, 0, 0.2, 1]
          }}
        >
          <motion.span 
            className="loader-ln-text-o"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.3, delay: 0.25, ease: [0.4, 0, 0.2, 1] }}
          >
            O
          </motion.span>
          <motion.span 
            className="loader-ln-text-v"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.3, delay: 0.3, ease: [0.4, 0, 0.2, 1] }}
          >
            V
          </motion.span>
        </motion.div>
        
        {/* Loading Bar */}
        <motion.div 
          className="loader-ln-bar"
          initial={{ scaleX: 0, opacity: 0 }}
          animate={{ scaleX: 1, opacity: phase === 'zoomIn' ? 0 : 1 }}
          transition={{ duration: 0.4, delay: 0.4, ease: [0.4, 0, 0.2, 1] }}
        >
          <motion.div 
            className="loader-ln-fill"
            initial={{ width: '0%' }}
            animate={{ width: `${count}%` }}
            transition={{ duration: 0.1, ease: 'linear' }}
          />
        </motion.div>
        
        <motion.div 
          className="loader-ln-status"
          initial={{ opacity: 0 }}
          animate={{ opacity: phase === 'zoomIn' ? 0 : 1 }}
          transition={{ delay: 0.5, duration: 0.3, ease: [0.4, 0, 0.2, 1] }}
        >
          <span>INITIALIZING DEMO</span>
          <span>{count}%</span>
        </motion.div>
      </div>
    </motion.div>
  )
}

// ============================================
// EXIT LOADER - Zoom Out then Zoom In
// ============================================
function ExitLoader({ onComplete }) {
  const [phase, setPhase] = useState('zoomOut') // zoomOut -> hold -> zoomIn -> done
  
  useEffect(() => {
    // Phase 1: Zoom out (logo appears from large scale)
    const holdTimer = setTimeout(() => {
      setPhase('hold')
    }, 500)
    
    return () => clearTimeout(holdTimer)
  }, [])
  
  useEffect(() => {
    if (phase === 'hold') {
      // Phase 2: Brief hold at normal size
      const zoomInTimer = setTimeout(() => {
        setPhase('zoomIn')
      }, 300)
      return () => clearTimeout(zoomInTimer)
    }
  }, [phase])
  
  useEffect(() => {
    if (phase === 'zoomIn') {
      // Phase 3: Zoom in and fade out
      const completeTimer = setTimeout(() => {
        onComplete()
      }, 600)
      return () => clearTimeout(completeTimer)
    }
  }, [phase, onComplete])

  return (
    <motion.div 
      className="loader-ln"
      initial={{ opacity: 0 }}
      animate={{ opacity: phase === 'zoomIn' ? 0 : 1 }}
      transition={{ 
        duration: phase === 'zoomIn' ? 0.4 : 0.3, 
        ease: [0.76, 0, 0.24, 1],
        delay: phase === 'zoomIn' ? 0.3 : 0
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
            duration: phase === 'zoomOut' ? 0.5 : phase === 'zoomIn' ? 0.6 : 0.1,
            ease: [0.76, 0, 0.24, 1]
          }}
        >
          <motion.span 
            className="loader-ln-text-o"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.2, delay: 0.2 }}
          >
            O
          </motion.span>
          <motion.span 
            className="loader-ln-text-v"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.2, delay: 0.25 }}
          >
            V
          </motion.span>
        </motion.div>
      </div>
    </motion.div>
  )
}

// ============================================
// STAT CARD COMPONENT
// ============================================
function StatCard({ icon, value, label, source, delta, delay = 0 }) {
  return (
    <motion.div 
      className="demo-stat-card"
      initial={{ opacity: 0, y: 40 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
      whileHover={{ x: 8, borderColor: 'rgba(204, 255, 0, 0.5)', transition: { duration: 0.4, ease: [0.16, 1, 0.3, 1] } }}
      style={{ willChange: 'transform, opacity' }}
    >
      <div className="demo-stat-icon">{icon}</div>
      <div className="demo-stat-value">{value}</div>
      <div className="demo-stat-label">{label}</div>
      {source && <div className="demo-stat-source">{source}</div>}
      {delta && (
        <div className={`demo-stat-delta ${parseFloat(delta) < 0 ? 'positive' : parseFloat(delta) > 0 ? 'negative' : ''}`}>
          {delta}
        </div>
      )}
    </motion.div>
  )
}

// ============================================
// RECOMMENDATION CARD
// ============================================
function RecommendationCard({ title, desc, metrics, index }) {
  return (
    <motion.div 
      className="demo-rec-card"
      initial={{ opacity: 0, x: -40 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: 0.12 * index, duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
      whileHover={{ x: 10, borderColor: 'rgba(204, 255, 0, 0.5)', transition: { duration: 0.4, ease: [0.16, 1, 0.3, 1] } }}
      style={{ willChange: 'transform, opacity' }}
    >
      <div className="demo-rec-title">{title}</div>
      <div className="demo-rec-desc">{desc}</div>
      <div className="demo-rec-metrics">
        {metrics.map((m, i) => (
          <div key={i} className="demo-rec-metric">
            <span className="demo-rec-metric-label">{m.label}</span>
            <span className="demo-rec-metric-value">{m.value}</span>
          </div>
        ))}
      </div>
    </motion.div>
  )
}

// ============================================
// MAIN DEMO PAGE
// ============================================
function Demo() {
  const location = useLocation()
  const skipLoader = location.state?.skipLoader || false
  const [loading, setLoading] = useState(!skipLoader)
  const [exiting, setExiting] = useState(false)
  const [hovering, setHovering] = useState(false)
  const [pendingNavigation, setPendingNavigation] = useState(null)
  const navigate = useNavigate()
  
  // Cursor refs for lag-free tracking
  const cursorRef = useRef(null)
  const mousePos = useRef({ x: 0, y: 0 })
  const rafId = useRef(null)

  // Handle browser back/forward buttons
  useEffect(() => {
    window.history.pushState({ skipLoader: true }, '', window.location.href)
    
    const handlePopState = (e) => {
      setExiting(true)
      setPendingNavigation('/')
    }
    
    window.addEventListener('popstate', handlePopState)
    
    return () => {
      window.removeEventListener('popstate', handlePopState)
    }
  }, [])

  // Set page title
  useEffect(() => {
    document.title = 'OVERHAUL | Simulator'
  }, [])
  
  // Demo State
  const [prompt, setPrompt] = useState('Analyze impact of increased EV adoption on corridor traffic flow and air quality')
  const [mode, setMode] = useState('fast')
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysisComplete, setAnalysisComplete] = useState(false)
  const [systemStatus, setSystemStatus] = useState('System Online')
  const [currentTime, setCurrentTime] = useState('')
  const [errorMessage, setErrorMessage] = useState('')
  const [confidenceLevel, setConfidenceLevel] = useState('ready')
  
  // Results State
  const [stats, setStats] = useState({
    travelTime: { value: '--', source: '', delta: null },
    pm25: { value: '--', source: '', delta: null },
    vkt: { value: '--', source: '', delta: null },
    co2: { value: '--', source: '', delta: null }
  })
  const [summary, setSummary] = useState('')
  const [summaryHtml, setSummaryHtml] = useState('Configure your traffic scenario using the control panel. The system will analyze impacts on congestion, air quality, and emissions using advanced SUMO simulation and RAG-enhanced intelligence.')
  const [recommendations, setRecommendations] = useState([])
  const [travelChartData, setTravelChartData] = useState(null)
  const [aqiChartData, setAqiChartData] = useState(null)
  const [liveSources, setLiveSources] = useState([])
  const [manifest, setManifest] = useState(null)
  const [liveContext, setLiveContext] = useState({ sources: [] })
  
  // Map ref
  const mapContainer = useRef(null)
  const mapRef = useRef(null)

  // Merge live context helper
  const mergeLiveContext = useCallback((live) => {
    if (!live) return liveContext
    
    setLiveContext(prev => {
      const currentSources = prev.sources || []
      const incomingSources = (live.sources || []).map(src => ({
        name: src?.name || 'Live Feed',
        detail: src?.detail || ''
      }))

      const dedupedSources = [...currentSources]
      incomingSources.forEach(src => {
        const idx = dedupedSources.findIndex(existing => existing.name === src.name && existing.detail === src.detail)
        if (idx >= 0) {
          dedupedSources[idx] = { ...dedupedSources[idx], ...src }
        } else {
          dedupedSources.push(src)
        }
      })

      const mergedTravel = { ...(prev.travel || {}) }
      if (live.travel) Object.assign(mergedTravel, live.travel)
      const mergedAqi = { ...(prev.aqi || {}) }
      if (live.aqi) Object.assign(mergedAqi, live.aqi)

      return {
        ...prev,
        ...live,
        travel: Object.keys(mergedTravel).length ? mergedTravel : prev.travel,
        aqi: Object.keys(mergedAqi).length ? mergedAqi : prev.aqi,
        sources: dedupedSources
      }
    })
  }, [])

  // Handle back navigation
  const handleBackHome = (e) => {
    e.preventDefault()
    setExiting(true)
    setPendingNavigation('/')
  }

  const handleExitComplete = () => {
    navigate(pendingNavigation || '/', { state: { skipLoader: true } })
  }

  // Clock
  useEffect(() => {
    const updateClock = () => {
      const now = new Date()
      setCurrentTime(now.toTimeString().split(' ')[0])
    }
    updateClock()
    const interval = setInterval(updateClock, 1000)
    return () => clearInterval(interval)
  }, [])

  // Cursor tracking - lag-free with requestAnimationFrame
  useEffect(() => {
    const updateCursor = () => {
      if (cursorRef.current) {
        cursorRef.current.style.transform = `translate3d(${mousePos.current.x}px, ${mousePos.current.y}px, 0) translate(-50%, -50%)`
      }
      rafId.current = requestAnimationFrame(updateCursor)
    }
    
    rafId.current = requestAnimationFrame(updateCursor)
    
    const handleMouseMove = (e) => {
      mousePos.current = { x: e.clientX, y: e.clientY }
    }
    
    window.addEventListener('mousemove', handleMouseMove, { passive: true })
    
    return () => {
      cancelAnimationFrame(rafId.current)
      window.removeEventListener('mousemove', handleMouseMove)
    }
  }, [])

  // Hover detection
  useEffect(() => {
    const handleHover = () => {
      const hoverable = document.querySelectorAll('a, button, .magnetic-btn, input, textarea, .demo-mode-btn')
      hoverable.forEach(el => {
        el.addEventListener('mouseenter', () => setHovering(true))
        el.addEventListener('mouseleave', () => setHovering(false))
      })
    }
    if (!loading && !exiting) {
      setTimeout(handleHover, 100)
    }
  }, [loading, exiting])

  // Keyboard shortcut for running analysis
  useEffect(() => {
    const handleKeyDown = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
        e.preventDefault()
        runAnalysis()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [prompt, mode, isAnalyzing])

  // Initialize Map
  useEffect(() => {
    if (!loading && mapContainer.current && !mapRef.current) {
      mapRef.current = new maplibregl.Map({
        container: mapContainer.current,
        style: {
          version: 8,
          sources: {
            'carto-dark': {
              type: 'raster',
              tiles: [
                'https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png',
                'https://b.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png',
                'https://c.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png'
              ],
              tileSize: 256,
              attribution: '&copy; OpenStreetMap contributors &copy; CARTO'
            }
          },
          layers: [
            {
              id: 'carto-dark-layer',
              type: 'raster',
              source: 'carto-dark',
              minzoom: 0,
              maxzoom: 22
            }
          ]
        },
        center: [77.31, 28.60],
        zoom: 12.8,
        pitch: 50,
        bearing: -15
      })

      mapRef.current.addControl(new maplibregl.NavigationControl(), 'top-right')

      mapRef.current.on('load', () => {
        // Add edges source for traffic visualization
        mapRef.current.addSource('edges', {
          type: 'geojson',
          data: { type: 'FeatureCollection', features: [] }
        })

        // Add pollution source
        mapRef.current.addSource('pollution', {
          type: 'geojson',
          data: { type: 'FeatureCollection', features: [] }
        })

        // Add route source
        mapRef.current.addSource('route', {
          type: 'geojson',
          data: { type: 'FeatureCollection', features: [] }
        })

        // Pollution layer
        mapRef.current.addLayer({
          id: 'pollution-layer',
          type: 'circle',
          source: 'pollution',
          paint: {
            'circle-radius': ['interpolate', ['linear'], ['get', 'intensity'], 0, 0, 1, 40],
            'circle-color': 'rgba(255, 77, 0, 0.35)',
            'circle-stroke-color': 'rgba(255, 77, 0, 0.8)',
            'circle-stroke-width': 1,
            'circle-opacity': 0.7
          }
        })

        // Edges layer (traffic lines)
        mapRef.current.addLayer({
          id: 'edges-layer',
          type: 'line',
          source: 'edges',
          paint: {
            'line-color': [
              'interpolate',
              ['linear'],
              ['get', 'ev_share'],
              0, '#475569',
              25, '#CCFF00',
              60, '#00ff88',
              90, '#ffff00'
            ],
            'line-width': ['case', ['get', 'primary'], 8, 4],
            'line-opacity': 0.9,
            'line-blur': 0.8
          }
        })

        // Route layer with glow effect
        mapRef.current.addLayer({
          id: 'route-glow',
          type: 'line',
          source: 'route',
          paint: {
            'line-color': '#CCFF00',
            'line-width': 12,
            'line-opacity': 0.3,
            'line-blur': 8
          }
        })

        mapRef.current.addLayer({
          id: 'route-line',
          type: 'line',
          source: 'route',
          paint: {
            'line-color': '#CCFF00',
            'line-width': 4,
            'line-opacity': 0.9
          }
        })

        // Fetch initial live data
        fetchLiveRoute()
        fetchLiveAQI()
      })
    }

    return () => {
      if (mapRef.current) {
        mapRef.current.remove()
        mapRef.current = null
      }
    }
  }, [loading])

  // Helper function for fetch with timeout and retry
  const fetchWithRetry = async (url, options = {}, retries = 3, timeout = 30000) => {
    for (let i = 0; i < retries; i++) {
      try {
        const controller = new AbortController()
        const timeoutId = setTimeout(() => controller.abort(), timeout)
        const resp = await fetch(url, { ...options, signal: controller.signal })
        clearTimeout(timeoutId)
        return resp
      } catch (err) {
        if (i === retries - 1) throw err
        // Wait before retry (backend might be waking up)
        await new Promise(r => setTimeout(r, 2000))
      }
    }
  }

  // Fetch live route
  const fetchLiveRoute = async () => {
    if (!mapRef.current) return
    try {
      const resp = await fetchWithRetry(`${API_BASE}/live/route`)
      if (!resp.ok) throw new Error('Route API failed')
      const geojson = await resp.json()
      
      if (mapRef.current.getSource('route')) {
        mapRef.current.getSource('route').setData(geojson)
      }
      
      const sourceName = geojson?.features?.[0]?.properties?.source || 'Live Route Service'
      mergeLiveContext({
        travel: { geojson, source: sourceName },
        sources: [{ name: sourceName, detail: 'Corridor route overlay (live endpoint)' }]
      })
      setLiveSources(prev => {
        const exists = prev.find(s => s.name === sourceName)
        if (exists) return prev
        return [...prev, { name: sourceName, detail: 'Corridor route overlay' }]
      })
    } catch (err) {
      console.error('Live route error', err)
    }
  }

  // Fetch live AQI
  const fetchLiveAQI = async () => {
    try {
      const resp = await fetchWithRetry(`${API_BASE}/live/aqi?lat=28.62&lon=77.35`)
      if (!resp.ok) throw new Error('AQI API failed')
      const data = await resp.json()
      
      const series = Array.isArray(data.series) ? data.series.filter(pt => pt.datetime && pt.pm25 != null) : []
      const latest = series.length ? series[series.length - 1] : null
      
      mergeLiveContext({
        aqi: {
          series: series.slice(-48),
          latest_pm25: latest ? latest.pm25 : null,
          latest_timestamp: latest ? latest.datetime : null
        },
        sources: [{ name: 'OpenAQ', detail: 'Direct corridor sensor bubble (live endpoint)' }]
      })
      
      setLiveSources(prev => {
        const exists = prev.find(s => s.name === 'OpenAQ')
        if (exists) return prev
        return [...prev, { name: 'OpenAQ', detail: 'Air quality data' }]
      })

      // Update AQI chart with live data
      if (series.length > 0) {
        const aqiLabels = series.slice(-12).map(pt => {
          const date = new Date(pt.datetime)
          return `${date.getHours()}:00`
        })
        const aqiData = series.slice(-12).map(pt => pt.pm25)
        
        setAqiChartData({
          labels: aqiLabels,
          datasets: [{
            data: aqiData,
            borderColor: '#FF4D00',
            backgroundColor: 'rgba(255, 77, 0, 0.1)',
            borderWidth: 2,
            fill: true,
            tension: 0.4,
            pointRadius: 4,
            pointBackgroundColor: '#FF4D00'
          }]
        })
      }
    } catch (err) {
      console.error('Live AQI error', err)
    }
  }

  // Update map with data
  const updateMap = (edgesGeojson, pollutionGeojson) => {
    if (!mapRef.current) return
    
    // Update edges
    if (edgesGeojson && mapRef.current.getSource('edges')) {
      const validGeojson = edgesGeojson.type === 'FeatureCollection' ? edgesGeojson : {
        type: 'FeatureCollection',
        features: Array.isArray(edgesGeojson) ? edgesGeojson : []
      }
      mapRef.current.getSource('edges').setData(validGeojson)
      
      // Fit bounds to show all features
      if (validGeojson.features && validGeojson.features.length > 0) {
        const bounds = validGeojson.features.reduce((bounds, feature) => {
          if (feature.geometry && feature.geometry.coordinates) {
            feature.geometry.coordinates.forEach(coord => {
              bounds.extend(coord)
            })
          }
          return bounds
        }, new maplibregl.LngLatBounds())
        
        if (!bounds.isEmpty()) {
          mapRef.current.fitBounds(bounds, { padding: 50, maxZoom: 14 })
        }
      }
    }
    
    // Update pollution
    if (pollutionGeojson && mapRef.current.getSource('pollution')) {
      const validPollution = pollutionGeojson.type === 'FeatureCollection' ? pollutionGeojson : {
        type: 'FeatureCollection',
        features: Array.isArray(pollutionGeojson) ? pollutionGeojson : []
      }
      mapRef.current.getSource('pollution').setData(validPollution)
    }
  }

  // Run Analysis
  const runAnalysis = async () => {
    if (isAnalyzing) return

    setIsAnalyzing(true)
    setSystemStatus(mode === 'deep' ? 'Deep Analysis Running' : 'Fast Analysis Running')
    setErrorMessage('')

    try {
      const response = await fetchWithRetry(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt,
          mode,
          scenario: {
            title: 'Prompt-driven analysis',
            region: 'Sector-78 to Vasundhara',
            intervention: 'prompt_only',
            horizon_months: 12,
            parameters: {}
          }
        })
      }, 3, 60000) // 60 second timeout for chat

      if (!response.ok) {
        const text = await response.text()
        throw new Error(text || 'Analysis failed')
      }

      const data = await response.json()
      console.log('Backend response:', data)

      // Merge live context
      mergeLiveContext(data.live || {})

      // Update live sources
      if (data.live?.sources) {
        setLiveSources(prev => {
          const newSources = [...prev]
          data.live.sources.forEach(src => {
            if (!newSources.find(s => s.name === src.name)) {
              newSources.push(src)
            }
          })
          return newSources
        })
      }

      // Helper to extract delta from impact cards
      const getDelta = (theme) => {
        if (!data.outputs?.impactCards) return null
        const card = data.outputs.impactCards.find(c => c.theme && c.theme.toLowerCase().includes(theme))
        if (card && card.delta) {
          const match = card.delta.match(/([-+]?\d+\.?\d*)/)
          return match ? parseFloat(match[1]) : null
        }
        return null
      }

      // Update stats with proper deltas
      const sourceLabel = (isLive, provider) => isLive ? `Live • ${provider}` : 'Scenario Model'
      
      if (data.baseline) {
        const live = data.live || liveContext
        const travelVal = live?.travel?.travel_time_min ?? data.baseline.avg_travel_time_min
        const pmVal = live?.aqi?.latest_pm25 ?? data.baseline.pm25
        
        const travelDelta = getDelta('travel')
        const airDelta = getDelta('air')
        const vktDelta = getDelta('vkt')
        const co2Delta = getDelta('co2')

        setStats({
          travelTime: {
            value: travelVal ? `${travelVal.toFixed(1)} min` : '--',
            source: sourceLabel(!!live?.travel, live?.travel?.source || 'OSRM'),
            delta: travelDelta ? `${travelDelta > 0 ? '+' : ''}${travelDelta.toFixed(1)}%` : null
          },
          pm25: {
            value: pmVal ? `${pmVal.toFixed(1)} µg/m³` : '--',
            source: sourceLabel(!!live?.aqi, 'OpenAQ'),
            delta: airDelta ? `${airDelta > 0 ? '+' : ''}${airDelta.toFixed(1)}%` : null
          },
          vkt: {
            value: data.baseline.total_vkt ? `${Math.round(data.baseline.total_vkt)} km` : '--',
            source: 'Scenario Model',
            delta: vktDelta ? `${vktDelta > 0 ? '+' : ''}${vktDelta.toFixed(1)}%` : null
          },
          co2: {
            value: data.baseline.co2_kg ? `${Math.round(data.baseline.co2_kg)} kg` : '--',
            source: 'Scenario Model',
            delta: co2Delta ? `${co2Delta > 0 ? '+' : ''}${co2Delta.toFixed(1)}%` : null
          }
        })
      }

      // Build summary HTML
      const tldr = data.outputs?.tldr || data.summary || 'Analysis completed successfully'
      const narrative = Array.isArray(data.outputs?.narrative) ? data.outputs.narrative : []
      const explanation = Array.isArray(data.outputs?.explanation) ? data.outputs.explanation : []
      const deepFacts = Array.isArray(data.outputs?.deepFacts) ? data.outputs.deepFacts : []
      const brainInsights = data.outputs?.brainInsights || null
      const metricsSummary = data.outputs?.metricsSummary || null

      let html = ''
      
      // Clean, readable executive summary
      html += `<div class="analysis-response">
        <div class="response-main">
          <p class="response-summary">${tldr}</p>
        </div>`

      // Show metrics summary cards if available
      if (metricsSummary) {
        html += `<div class="response-metrics-summary">
          <div class="metrics-grid">`
        
        if (metricsSummary.aqi) {
          const aqiChange = metricsSummary.aqi.change || '0%'
          const isImprovement = aqiChange.includes('-')
          html += `<div class="metric-card ${isImprovement ? 'positive' : ''}">
            <span class="metric-icon">🌬️</span>
            <div class="metric-info">
              <span class="metric-label">Air Quality</span>
              <span class="metric-value">${metricsSummary.aqi.current?.toFixed(0) || '--'} → ${metricsSummary.aqi.projected?.toFixed(0) || '--'}</span>
              <span class="metric-change ${isImprovement ? 'positive' : ''}">${aqiChange}</span>
            </div>
          </div>`
        }
        
        if (metricsSummary.traffic) {
          const trafficChange = metricsSummary.traffic.change || '0%'
          const isImprovement = trafficChange.includes('+')
          html += `<div class="metric-card ${isImprovement ? 'positive' : ''}">
            <span class="metric-icon">🚗</span>
            <div class="metric-info">
              <span class="metric-label">Traffic Flow</span>
              <span class="metric-value">${trafficChange}</span>
              <span class="metric-change ${isImprovement ? 'positive' : ''}">${isImprovement ? 'Improved' : 'Impact'}</span>
            </div>
          </div>`
        }
        
        html += `</div></div>`
      }

      // Show brain insights if available (key findings)
      if (brainInsights && brainInsights.keyFindings && brainInsights.keyFindings.length > 0) {
        html += `<div class="response-findings">
          <h4>📊 Key Findings</h4>
          <ul>`
        brainInsights.keyFindings.slice(0, 4).forEach(f => {
          const confidence = f.confidence ? Math.round(f.confidence * 100) : null
          const source = f.source || ''
          html += `<li>
            <span class="finding-text">${f.finding}</span>
            <div class="finding-meta">
              ${confidence ? `<span class="finding-confidence">${confidence}% confidence</span>` : ''}
              ${source ? `<span class="finding-source">${source}</span>` : ''}
            </div>
          </li>`
        })
        html += `</ul></div>`
      }

      // Show detailed analysis sections if available
      if (brainInsights && brainInsights.detailedAnalysis) {
        const analysis = brainInsights.detailedAnalysis
        html += `<div class="response-details">
          <h4>📋 Detailed Analysis</h4>`
        
        // Primary answer (most important)
        if (analysis.primary_answer && analysis.primary_answer !== 'No specific intervention detected - showing current baselines.') {
          html += `<div class="detail-section detail-primary">
            <div class="detail-content">
              <p>${analysis.primary_answer}</p>
            </div>
          </div>`
        }
        
        // Supporting evidence
        if (analysis.supporting_evidence) {
          html += `<div class="detail-section">
            <span class="detail-icon">📈</span>
            <div class="detail-content">
              <strong>Data Sources</strong>
              <p>${analysis.supporting_evidence}</p>
            </div>
          </div>`
        }
        
        // Caveats
        if (analysis.caveats) {
          html += `<div class="detail-section detail-caveat">
            <span class="detail-icon">⚠️</span>
            <div class="detail-content">
              <strong>Assumptions</strong>
              <p>${analysis.caveats}</p>
            </div>
          </div>`
        }
        
        html += `</div>`
      }

      // Real world context
      if (brainInsights && brainInsights.realWorldContext) {
        const context = brainInsights.realWorldContext
        html += `<div class="response-context">
          <h4>🌍 Real-World Context</h4>`
        
        if (context.similar_cities && context.similar_cities.length > 0) {
          html += `<div class="context-section">
            <strong>Similar Cities:</strong>
            <span class="context-cities">${context.similar_cities.join(', ')}</span>
          </div>`
        }
        
        if (context.lessons && context.lessons.length > 0) {
          html += `<div class="context-section">
            <strong>Key Lessons:</strong>
            <ul>`
          context.lessons.forEach(lesson => {
            html += `<li>${lesson}</li>`
          })
          html += `</ul></div>`
        }
        
        html += `</div>`
      }

      // Show recommendations if available
      if (brainInsights && brainInsights.recommendations && brainInsights.recommendations.length > 0) {
        const recs = brainInsights.recommendations
        html += `<div class="response-recommendations">
          <h4>💡 Recommendations</h4>
          <ul>`
        recs.slice(0, 4).forEach(rec => {
          if (typeof rec === 'object') {
            const priority = rec.priority || 'medium'
            const timeline = rec.timeline || ''
            html += `<li class="rec-item priority-${priority}">
              <span class="rec-text">${rec.recommendation}</span>
              <div class="rec-meta">
                <span class="rec-priority">${priority}</span>
                ${timeline ? `<span class="rec-timeline">${timeline}</span>` : ''}
              </div>
            </li>`
          } else {
            html += `<li>${rec}</li>`
          }
        })
        html += `</ul></div>`
      }

      // Data transparency
      if (brainInsights && brainInsights.dataTransparency) {
        const transparency = brainInsights.dataTransparency
        html += `<div class="response-transparency">
          <div class="transparency-badge">
            <span class="transparency-icon">✓</span>
            <span class="transparency-text">
              Analysis: <strong>${brainInsights.calculationType || 'physics-based'}</strong> | 
              Confidence: <strong>${transparency.confidence_level || 'high'}</strong>
            </span>
          </div>
        </div>`
      }

      // Fallback to old narrative format if no brain insights
      if (!brainInsights && narrative.length > 0) {
        html += `<div class="response-narrative">
          <ul>`
        narrative.forEach(line => {
          html += `<li>${line}</li>`
        })
        html += `</ul></div>`
      }

      // Follow-up suggestions
      if (brainInsights && brainInsights.followUpSuggestions && brainInsights.followUpSuggestions.length > 0) {
        html += `<div class="response-followup">
          <p class="followup-label">You might also want to ask:</p>
          <div class="followup-chips">`
        brainInsights.followUpSuggestions.slice(0, 2).forEach(q => {
          html += `<span class="followup-chip">${q}</span>`
        })
        html += `</div></div>`
      }

      html += `</div>`

      setSummaryHtml(html)

      // Update recommendations
      if (data.ranked && data.ranked.length > 0) {
        setRecommendations(data.ranked.slice(0, 5).map((item, i) => ({
          title: item.name || `Intervention ${i + 1}`,
          desc: item.desc || 'Strategic intervention to optimize traffic flow',
          metrics: [
            { label: 'Time Saved', value: `${item.pred_delta_tt_min > 0 ? '-' : ''}${Math.abs(item.pred_delta_tt_min || 0).toFixed(1)} min` },
            { label: 'Impact', value: (item.pred_delta_tt_min || 0) > 5 ? 'High' : (item.pred_delta_tt_min || 0) > 2 ? 'Medium' : 'Low' },
            { label: 'Cost', value: item.cost || 'Medium' }
          ]
        })))
      }

      // Update charts
      const hours = ['0h', '2h', '4h', '6h', '8h', '10h', '12h']
      const live = data.live || liveContext
      const baselineTravel = (live?.travel?.travel_time_min ?? data.baseline?.avg_travel_time_min) || 25
      const candidateTravel = data.ranked?.[0]?.avg_travel_time_min || baselineTravel * 0.85
      
      const travelData = hours.map((_, i) => {
        const progress = i / (hours.length - 1)
        return baselineTravel - (baselineTravel - candidateTravel) * progress
      })

      setTravelChartData({
        labels: hours,
        datasets: [{
          data: travelData,
          borderColor: '#CCFF00',
          backgroundColor: 'rgba(204, 255, 0, 0.1)',
          borderWidth: 2,
          fill: true,
          tension: 0.4,
          pointRadius: 4,
          pointBackgroundColor: '#CCFF00'
        }]
      })

      // AQI Chart
      if (live?.aqi?.series?.length) {
        const aqiLabels = live.aqi.series.map(pt => {
          const date = new Date(pt.datetime)
          return `${date.getHours()}:00`
        })
        const aqiData = live.aqi.series.map(pt => pt.pm25)
        
        setAqiChartData({
          labels: aqiLabels,
          datasets: [{
            data: aqiData,
            borderColor: '#FF4D00',
            backgroundColor: 'rgba(255, 77, 0, 0.1)',
            borderWidth: 2,
            fill: true,
            tension: 0.4,
            pointRadius: 4,
            pointBackgroundColor: '#FF4D00'
          }]
        })
      } else {
        const baselinePM = data.baseline?.pm25 || 150
        const candidatePM = data.ranked?.[0]?.pm25 || baselinePM * 0.9
        const aqiData = hours.map((_, i) => {
          const progress = i / (hours.length - 1)
          return baselinePM - (baselinePM - candidatePM) * progress
        })

        setAqiChartData({
          labels: hours,
          datasets: [{
            data: aqiData,
            borderColor: '#FF4D00',
            backgroundColor: 'rgba(255, 77, 0, 0.1)',
            borderWidth: 2,
            fill: true,
            tension: 0.4,
            pointRadius: 4,
            pointBackgroundColor: '#FF4D00'
          }]
        })
      }

      // Update map
      updateMap(data.edges_geojson, data.pollution_hotspots)
      
      // Update route layer with live data
      if (data.live?.travel?.geojson && mapRef.current?.getSource('route')) {
        mapRef.current.getSource('route').setData(data.live.travel.geojson)
      }

      // Update manifest
      if (data.manifest) {
        setManifest(data.manifest)
      }

      // Update confidence level
      if (data.outputs?.confidenceLevel) {
        setConfidenceLevel(data.outputs.confidenceLevel)
      }

      setAnalysisComplete(true)

    } catch (error) {
      console.error('Analysis error:', error)
      setErrorMessage(error.message || 'Failed to connect to backend. Ensure FastAPI is running on port 8000.')
    } finally {
      setIsAnalyzing(false)
      setSystemStatus('System Online')
    }
  }

  // Export Report
  const exportReport = () => {
    if (!manifest) {
      setErrorMessage('No data to export. Run analysis first.')
      setTimeout(() => setErrorMessage(''), 3000)
      return
    }

    const report = {
      timestamp: new Date().toISOString(),
      summary: summaryHtml.replace(/<[^>]*>/g, ''),
      scenario: {
        title: 'Prompt-driven analysis',
        region: 'Sector-78 to Vasundhara',
        intervention: 'prompt_only',
        horizon_months: 12,
        parameters: {}
      },
      manifest
    }

    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `overhaul-report-${Date.now()}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  // Copy Manifest
  const copyManifest = async () => {
    if (!manifest) {
      setErrorMessage('No manifest to copy. Run analysis first.')
      setTimeout(() => setErrorMessage(''), 3000)
      return
    }

    try {
      await navigator.clipboard.writeText(JSON.stringify(manifest, null, 2))
      // Show success briefly
      const btn = document.querySelector('.demo-copy-btn')
      if (btn) {
        const original = btn.textContent
        btn.textContent = '✓ Copied!'
        setTimeout(() => { btn.textContent = original }, 2000)
      }
    } catch (error) {
      setErrorMessage('Failed to copy to clipboard')
      setTimeout(() => setErrorMessage(''), 3000)
    }
  }

  // Chart options
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: 'rgba(10, 10, 10, 0.95)',
        titleColor: '#CCFF00',
        bodyColor: '#f5f5f5',
        borderColor: '#CCFF00',
        borderWidth: 1,
        titleFont: { family: 'Space Mono' },
        bodyFont: { family: 'Space Mono' }
      }
    },
    scales: {
      x: {
        grid: { color: 'rgba(204, 255, 0, 0.1)' },
        ticks: { color: 'rgba(255, 255, 255, 0.5)', font: { family: 'Space Mono', size: 10 } }
      },
      y: {
        grid: { color: 'rgba(204, 255, 0, 0.1)' },
        ticks: { color: 'rgba(255, 255, 255, 0.5)', font: { family: 'Space Mono', size: 10 } }
      }
    }
  }

  const aqiChartOptions = {
    ...chartOptions,
    plugins: {
      ...chartOptions.plugins,
      tooltip: {
        ...chartOptions.plugins.tooltip,
        titleColor: '#FF4D00',
        borderColor: '#FF4D00'
      }
    }
  }

  return (
    <>
      {/* Custom Cursor */}
      <div 
        ref={cursorRef}
        className={`cursor ${hovering ? 'hovering' : ''}`}
      />

      {/* Moving Background */}
      <div className="moving-bg">
        <svg className="moving-bg-svg" viewBox="0 0 1000 1000" preserveAspectRatio="none">
          <motion.path
            d="M0,500 Q250,400 500,500 T1000,500"
            stroke="rgba(204, 255, 0, 0.1)"
            strokeWidth="2"
            fill="none"
            animate={{ d: [
              "M0,500 Q250,400 500,500 T1000,500",
              "M0,500 Q250,600 500,500 T1000,500",
              "M0,500 Q250,400 500,500 T1000,500"
            ]}}
            transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
          />
          <motion.path
            d="M0,300 Q250,200 500,300 T1000,300"
            stroke="rgba(255, 77, 0, 0.08)"
            strokeWidth="1"
            fill="none"
            animate={{ d: [
              "M0,300 Q250,200 500,300 T1000,300",
              "M0,300 Q250,400 500,300 T1000,300",
              "M0,300 Q250,200 500,300 T1000,300"
            ]}}
            transition={{ duration: 10, repeat: Infinity, ease: "easeInOut", delay: 1 }}
          />
          <motion.path
            d="M0,700 Q250,600 500,700 T1000,700"
            stroke="rgba(255, 0, 128, 0.06)"
            strokeWidth="1"
            fill="none"
            animate={{ d: [
              "M0,700 Q250,600 500,700 T1000,700",
              "M0,700 Q250,800 500,700 T1000,700",
              "M0,700 Q250,600 500,700 T1000,700"
            ]}}
            transition={{ duration: 12, repeat: Infinity, ease: "easeInOut", delay: 2 }}
          />
        </svg>
      </div>

      {/* Entry Loader */}
      <AnimatePresence mode="wait">
        {loading && (
          <OVLoader key="entry-loader" onComplete={() => setLoading(false)} />
        )}
      </AnimatePresence>

      {/* Exit Loader */}
      <AnimatePresence mode="wait">
        {exiting && (
          <ExitLoader key="exit-loader" onComplete={handleExitComplete} />
        )}
      </AnimatePresence>

      {/* Page Content */}
      <AnimatePresence>
        {!loading && !exiting && (
          <motion.div 
            className="demo-page"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.5 }}
          >
            {/* Navigation */}
            <nav className="demo-nav">
              <a href="/" onClick={handleBackHome} className="demo-nav-logo">
                <span className="logo-o">O</span>
                <span className="logo-v">V</span>
                <span className="logo-text">ERHAUL</span>
              </a>
              
              <div className="demo-nav-center">
                <div className="demo-status">
                  <span className={`demo-status-dot ${isAnalyzing ? 'analyzing' : ''}`} />
                  <span>{systemStatus}</span>
                </div>
                <div className="demo-clock">{currentTime}</div>
              </div>

              <a href="/" onClick={handleBackHome} className="demo-back-btn">
                ← BACK TO HOME
              </a>
            </nav>

            {/* Main Content */}
            <div className="demo-container">
              <div className="demo-grid">
                {/* Left Column */}
                <div className="demo-left">
                  {/* Hero Section */}
                  <motion.div 
                    className="demo-hero"
                    initial={{ opacity: 0, y: 30 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1, duration: 0.6 }}
                  >
                    <div className="demo-hero-tag">
                      <span className="tag-dot" />
                      <span>EARTH INTELLIGENCE PLATFORM</span>
                    </div>
                    <h1 className="demo-hero-title">
                      LIVE<br/>
                      <span className="text-outline">DEMO</span>
                    </h1>
                    <p className="demo-hero-subtitle">
                      Sector-78 → Vasundhara • Delhi NCR Region
                    </p>
                  </motion.div>

                  {/* Map Section */}
                  <motion.div 
                    className="demo-card demo-map-card"
                    initial={{ opacity: 0, y: 30 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2, duration: 0.6 }}
                  >
                    <div className="demo-card-header">
                      <span className="demo-card-title">LIVE CORRIDOR VISUALIZATION</span>
                      <span className="demo-card-tag">REAL-TIME</span>
                    </div>
                    <div ref={mapContainer} className="demo-map" />
                  </motion.div>

                  {/* Stats Grid */}
                  <div className="demo-stats-grid">
                    <StatCard 
                      icon="🕒" 
                      value={stats.travelTime.value} 
                      label="AVG TRAVEL TIME" 
                      source={stats.travelTime.source}
                      delta={stats.travelTime.delta}
                      delay={0.3} 
                    />
                    <StatCard 
                      icon="☁️" 
                      value={stats.pm25.value} 
                      label="PM₂.₅ LEVEL" 
                      source={stats.pm25.source}
                      delta={stats.pm25.delta}
                      delay={0.35} 
                    />
                    <StatCard 
                      icon="🚗" 
                      value={stats.vkt.value} 
                      label="VEHICLE KM" 
                      source={stats.vkt.source}
                      delta={stats.vkt.delta}
                      delay={0.4} 
                    />
                    <StatCard 
                      icon="🔥" 
                      value={stats.co2.value} 
                      label="CO₂ EMISSIONS" 
                      source={stats.co2.source}
                      delta={stats.co2.delta}
                      delay={0.45} 
                    />
                  </div>

                  {/* AI Analysis */}
                  <motion.div 
                    className="demo-card"
                    initial={{ opacity: 0, y: 30 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.5, duration: 0.6 }}
                  >
                    <div className="demo-card-header">
                      <span className="demo-card-title">AI-GENERATED ANALYSIS</span>
                      <span className={`demo-card-badge ${confidenceLevel}`}>
                        {analysisComplete ? `${confidenceLevel.toUpperCase()} CONFIDENCE` : 'READY'}
                      </span>
                    </div>
                    <div className="demo-summary">
                      <div dangerouslySetInnerHTML={{ __html: summaryHtml }} />
                    </div>
                    {liveSources.length > 0 && (
                      <div className="demo-live-sources">
                        {liveSources.map((src, i) => (
                          <span key={i} className="demo-live-chip" title={src.detail}>{src.name}</span>
                        ))}
                      </div>
                    )}
                  </motion.div>

                  {/* Charts */}
                  <div className="demo-charts-grid">
                    <motion.div 
                      className="demo-card"
                      initial={{ opacity: 0, y: 30 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.6, duration: 0.6 }}
                    >
                      <div className="demo-card-header">
                        <span className="demo-card-title">TRAVEL TIME TRENDS</span>
                      </div>
                      <div className="demo-chart">
                        {travelChartData ? (
                          <Line data={travelChartData} options={chartOptions} />
                        ) : (
                          <div className="demo-chart-placeholder">
                            Run analysis to view trends
                          </div>
                        )}
                      </div>
                    </motion.div>

                    <motion.div 
                      className="demo-card"
                      initial={{ opacity: 0, y: 30 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.65, duration: 0.6 }}
                    >
                      <div className="demo-card-header">
                        <span className="demo-card-title">AIR QUALITY INDEX</span>
                      </div>
                      <div className="demo-chart">
                        {aqiChartData ? (
                          <Line data={aqiChartData} options={aqiChartOptions} />
                        ) : (
                          <div className="demo-chart-placeholder">
                            Run analysis to view AQI
                          </div>
                        )}
                      </div>
                    </motion.div>
                  </div>

                  {/* Recommendations */}
                  <motion.div 
                    className="demo-card"
                    initial={{ opacity: 0, y: 30 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.7, duration: 0.6 }}
                  >
                    <div className="demo-card-header">
                      <span className="demo-card-title">RECOMMENDED INTERVENTIONS</span>
                      <span className="demo-card-subtitle">AI-ranked by effectiveness</span>
                    </div>
                    <div className="demo-recommendations">
                      {recommendations.length > 0 ? (
                        recommendations.map((rec, i) => (
                          <RecommendationCard key={i} index={i} {...rec} />
                        ))
                      ) : (
                        <div className="demo-rec-placeholder">
                          Run analysis to view recommendations
                        </div>
                      )}
                    </div>
                  </motion.div>
                </div>

                {/* Right Column - Control Panel */}
                <div className="demo-right">
                  <motion.div 
                    className="demo-control-panel"
                    initial={{ opacity: 0, x: 30 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.3, duration: 0.6 }}
                  >
                    <div className="demo-panel-header">
                      <span className="demo-panel-title">CONTROL PANEL</span>
                      <span className="demo-panel-subtitle">Configure Traffic Scenario</span>
                    </div>

                    {/* Prompt Input */}
                    <div className="demo-input-group">
                      <label className="demo-input-label">SCENARIO DESCRIPTION</label>
                      <textarea
                        className="demo-textarea"
                        value={prompt}
                        onChange={(e) => setPrompt(e.target.value)}
                        placeholder="Describe your traffic scenario..."
                        rows={4}
                      />
                    </div>

                    {/* Mode Toggle */}
                    <div className="demo-input-group">
                      <label className="demo-input-label">ANALYSIS MODE</label>
                      <div className="demo-mode-toggle">
                        <button 
                          className={`demo-mode-btn ${mode === 'fast' ? 'active' : ''}`}
                          onClick={() => setMode('fast')}
                        >
                          FAST
                        </button>
                        <button 
                          className={`demo-mode-btn ${mode === 'deep' ? 'active' : ''}`}
                          onClick={() => setMode('deep')}
                        >
                          DEEP
                        </button>
                      </div>
                    </div>

                    {/* Run Button */}
                    <motion.button 
                      className="demo-run-btn"
                      onClick={runAnalysis}
                      disabled={isAnalyzing}
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                    >
                      {isAnalyzing ? (
                        <span className="demo-spinner" />
                      ) : (
                        'RUN ANALYSIS'
                      )}
                    </motion.button>

                    {/* Error Message */}
                    {errorMessage && (
                      <motion.div 
                        className="demo-error"
                        initial={{ opacity: 0, y: -10 }}
                        animate={{ opacity: 1, y: 0 }}
                      >
                        {errorMessage}
                      </motion.div>
                    )}

                    {/* Quick Actions */}
                    <div className="demo-actions">
                      <button className="demo-action-btn" onClick={exportReport}>
                        📥 EXPORT REPORT
                      </button>
                    </div>

                    {/* Keyboard Shortcut Hint */}
                    <div className="demo-hint">
                      Press <kbd>⌘</kbd> + <kbd>Enter</kbd> to run analysis
                    </div>
                  </motion.div>
                </div>
              </div>
            </div>

            {/* Footer */}
            <footer className="demo-footer">
              <span>© 2025 OVERHAUL. ALL RIGHTS RESERVED.</span>
              <span className="demo-footer-tag">TRAFFIC INTELLIGENCE PLATFORM</span>
            </footer>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
}

export default Demo
