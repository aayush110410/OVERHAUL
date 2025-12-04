import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useNavigate, useLocation } from 'react-router-dom'
import './App.css'

// ============================================
// OV LOADER (Same style as main page)
// ============================================
function OVLoader({ onComplete }) {
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
      }, 250)
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
// EXIT LOADER
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
          <motion.span 
            className="loader-ln-text-o"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.15, delay: 0.15 }}
          >
            O
          </motion.span>
          <motion.span 
            className="loader-ln-text-v"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.15, delay: 0.2 }}
          >
            V
          </motion.span>
        </motion.div>
      </div>
    </motion.div>
  )
}

// ============================================
// DOCS PAGE
// ============================================
function Docs() {
  const location = useLocation()
  const skipLoader = location.state?.skipLoader || false
  const [loading, setLoading] = useState(!skipLoader)
  const [exiting, setExiting] = useState(false)
  const [hovering, setHovering] = useState(false)
  const [pendingNavigation, setPendingNavigation] = useState(null)
  const navigate = useNavigate()
  
  const cursorRef = useRef(null)
  const mousePos = useRef({ x: 0, y: 0 })
  const rafId = useRef(null)

  useEffect(() => {
    window.history.pushState({ skipLoader: true }, '', window.location.href)
    
    const handlePopState = (e) => {
      setExiting(true)
      setPendingNavigation('/')
    }
    
    window.addEventListener('popstate', handlePopState)
    return () => window.removeEventListener('popstate', handlePopState)
  }, [])

  useEffect(() => {
    document.title = 'OVERHAUL | Documentation'
  }, [])

  const handleBackHome = (e) => {
    e.preventDefault()
    setExiting(true)
    setPendingNavigation('/')
  }

  const handleExitComplete = () => {
    navigate(pendingNavigation || '/', { state: { skipLoader: true } })
  }

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

  useEffect(() => {
    const handleHover = () => {
      const hoverable = document.querySelectorAll('a, button, .magnetic-btn')
      hoverable.forEach(el => {
        el.addEventListener('mouseenter', () => setHovering(true))
        el.addEventListener('mouseleave', () => setHovering(false))
      })
    }
    if (!loading && !exiting) handleHover()
  }, [loading, exiting])

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
            className="contact-page"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.4 }}
          >
            {/* Navigation */}
            <nav className="contact-nav">
              <a href="/" onClick={handleBackHome} className="nav-logo">OVERHAUL™</a>
              <a href="/" onClick={handleBackHome} className="back-btn">
                ← BACK TO HOME
              </a>
            </nav>

            {/* Main Content */}
            <div className="contact-content">
              <motion.div 
                className="contact-header"
                initial={{ opacity: 0, y: 50 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1, duration: 0.6 }}
              >
                <span className="contact-label">TECHNICAL OVERVIEW</span>
                <h1 className="contact-title">
                  DOCUMENT<span className="text-outline">ATION</span>
                </h1>
              </motion.div>

              {/* Docs Grid - Same as contact cards */}
              <div className="contact-grid docs-grid">
                {/* Architecture Card */}
                <motion.div 
                  className="contact-card"
                  initial={{ opacity: 0, x: -50 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.2, duration: 0.6 }}
                >
                  <div className="card-icon">⚙</div>
                  <h2 className="card-title">ARCHITECTURE</h2>
                  <p className="card-desc">
                    Multi-model orchestration system where LDRAGO acts as the 
                    central coordinator. When a user query arrives, LDRAGO 
                    analyzes the prompt, determines which specialized models 
                    need activation, distributes tasks, and synthesizes their 
                    collective outputs into a unified, visualized response.
                  </p>
                  <div className="card-extras">
                    <span>6 Models • Collaborative Intelligence</span>
                  </div>
                </motion.div>

                {/* Model Communication Card */}
                <motion.div 
                  className="contact-card"
                  initial={{ opacity: 0, x: 50 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.3, duration: 0.6 }}
                >
                  <div className="card-icon">🔄</div>
                  <h2 className="card-title">MODEL COMMUNICATION</h2>
                  <p className="card-desc">
                    Our models don't work in isolation—they discuss like a team. 
                    Each model shares its findings with others, cross-references 
                    insights, and collectively arrives at the best answer. LDRAGO 
                    then validates this against the original query before 
                    generating the final user-friendly response.
                  </p>
                  <div className="card-extras">
                    <span>Group Discussion • Cross-Validation</span>
                  </div>
                </motion.div>
              </div>

              {/* Second Row */}
              <div className="contact-grid docs-grid" style={{ marginTop: '30px' }}>
                {/* LDRAGO Card */}
                <motion.div 
                  className="contact-card"
                  initial={{ opacity: 0, x: -50 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.35, duration: 0.6 }}
                >
                  <div className="card-icon">🐉</div>
                  <h2 className="card-title">LDRAGO - ORCHESTRATOR</h2>
                  <p className="card-desc">
                    High-parameter custom language model built on curated datasets 
                    and current affairs. LDRAGO reads user prompts, understands 
                    intent, activates relevant models, assigns tasks, receives 
                    collective outputs, validates against requirements, and 
                    generates engaging visualized responses for the user.
                  </p>
                  <div className="card-extras">
                    <span>Prompt Analysis • Task Distribution • Response Generation</span>
                  </div>
                </motion.div>

                {/* TrafficSim Card */}
                <motion.div 
                  className="contact-card"
                  initial={{ opacity: 0, x: 50 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.4, duration: 0.6 }}
                >
                  <div className="card-icon">🚗</div>
                  <h2 className="card-title">TRAFFIC MODEL</h2>
                  <p className="card-desc">
                    Analyzes routes and traffic patterns—shortest paths, longest 
                    alternatives, heavy traffic zones, light traffic windows. 
                    Considers time-of-day variations to suggest optimal travel 
                    routes. Integrates with other models to factor in weather 
                    impacts and event-based diversions.
                  </p>
                  <div className="card-extras">
                    <span>Route Analysis • Time-Based Patterns</span>
                  </div>
                </motion.div>
              </div>

              {/* Third Row */}
              <div className="contact-grid docs-grid" style={{ marginTop: '30px' }}>
                {/* AQI/Pollution Card */}
                <motion.div 
                  className="contact-card"
                  initial={{ opacity: 0, x: -50 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.45, duration: 0.6 }}
                >
                  <div className="card-icon">🌿</div>
                  <h2 className="card-title">POLLUTION & AQI MODEL</h2>
                  <p className="card-desc">
                    Monitors air quality at destinations and along routes. 
                    Works with the Weather Model to predict AQI changes based 
                    on wind patterns, temperature, and atmospheric conditions. 
                    Provides health-conscious recommendations for travel timing 
                    and outdoor activities.
                  </p>
                  <div className="card-extras">
                    <span>AQI Monitoring • Weather-Integrated Predictions</span>
                  </div>
                </motion.div>

                {/* Weather Card */}
                <motion.div 
                  className="contact-card"
                  initial={{ opacity: 0, x: 50 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.5, duration: 0.6 }}
                >
                  <div className="card-icon">🌦️</div>
                  <h2 className="card-title">WEATHER MODEL</h2>
                  <p className="card-desc">
                    Predicts temperatures, seasonal patterns, rainfall, humidity, 
                    and wind conditions. Goes beyond standard forecasts by 
                    identifying hyperlocal variations—those small but significant 
                    details often missed by conventional weather services. Factors 
                    into pollution dispersion and traffic recommendations.
                  </p>
                  <div className="card-extras">
                    <span>Hyperlocal • Seasonal • Micro-Climate Aware</span>
                  </div>
                </motion.div>
              </div>

              {/* Fourth Row */}
              <div className="contact-grid docs-grid" style={{ marginTop: '30px' }}>
                {/* Policy & Economics Card */}
                <motion.div 
                  className="contact-card"
                  initial={{ opacity: 0, x: -50 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.55, duration: 0.6 }}
                >
                  <div className="card-icon">📜</div>
                  <h2 className="card-title">POLICY & ECONOMICS MODEL</h2>
                  <p className="card-desc">
                    Tracks new regulations, special notices, VIP movement alerts, 
                    route diversions, and temporary restrictions. Monitors market 
                    closures, economic events, and local administrative decisions. 
                    Ensures users aren't caught off-guard by sudden policy changes 
                    or road blocks during their journey.
                  </p>
                  <div className="card-extras">
                    <span>Regulations • Diversions • Market Status</span>
                  </div>
                </motion.div>

                {/* Behavior Card */}
                <motion.div 
                  className="contact-card"
                  initial={{ opacity: 0, x: 50 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.6, duration: 0.6 }}
                >
                  <div className="card-icon">👥</div>
                  <h2 className="card-title">BEHAVIORAL MODEL</h2>
                  <p className="card-desc">
                    Understands human patterns—weekends attract outings, holidays 
                    create crowds at popular spots, festivals change movement 
                    dynamics. Predicts crowd density at destinations based on 
                    day, time, and cultural context. Helps users plan around 
                    peak hours and busy periods.
                  </p>
                  <div className="card-extras">
                    <span>Crowd Prediction • Cultural Context</span>
                  </div>
                </motion.div>
              </div>

              {/* Fifth Row - Workflow Example */}
              <div className="contact-grid docs-grid" style={{ marginTop: '30px' }}>
                {/* Workflow Card - Full Width */}
                <motion.div 
                  className="contact-card"
                  style={{ gridColumn: '1 / -1' }}
                  initial={{ opacity: 0, y: 30 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.65, duration: 0.6 }}
                >
                  <div className="card-icon">💡</div>
                  <h2 className="card-title">HOW IT WORKS — EXAMPLE</h2>
                  <p className="card-desc" style={{ maxWidth: '100%' }}>
                    <strong style={{ color: 'var(--lime)' }}>User asks:</strong> "Planning a trip to Delhi this Saturday—what should I expect?"
                    <br/><br/>
                    <strong style={{ color: 'var(--orange)' }}>LDRAGO activates all 5 models:</strong>
                    <br/>• <strong>Traffic Model</strong> → Evaluates routes: shortest, least congested, time-optimized options
                    <br/>• <strong>Pollution Model</strong> → Checks AQI at destination and along routes
                    <br/>• <strong>Weather Model</strong> → Forecasts Saturday's conditions—temperature, rain probability, wind
                    <br/>• <strong>Policy Model</strong> → Scans for VIP movements, route diversions, special restrictions
                    <br/>• <strong>Behavioral Model</strong> → Notes it's a weekend—expect crowds at popular spots
                    <br/><br/>
                    <strong style={{ color: 'var(--lime)' }}>Models discuss:</strong> Traffic suggests Route A, but Weather warns of rain affecting that area. 
                    Pollution Model notes AQI is better on Route B. Policy Model confirms no diversions on Route B. 
                    Behavioral Model adds that a popular market on Route A will be crowded Saturday afternoon.
                    <br/><br/>
                    <strong style={{ color: 'var(--orange)' }}>Collective recommendation:</strong> Take Route B, depart by 9 AM to avoid afternoon crowds, 
                    carry rain protection. LDRAGO validates this against the original query, then presents 
                    an interactive, visualized response to the user.
                  </p>
                </motion.div>
              </div>

              {/* Sixth Row */}
              <div className="contact-grid docs-grid" style={{ marginTop: '30px' }}>
                {/* Data Sources Card */}
                <motion.div 
                  className="contact-card"
                  initial={{ opacity: 0, x: -50 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.7, duration: 0.6 }}
                >
                  <div className="card-icon">📊</div>
                  <h2 className="card-title">DATA FOUNDATION</h2>
                  <p className="card-desc">
                    Built on curated datasets from public sources including 
                    OpenStreetMap for road networks, government portals for 
                    policies, and open environmental databases. Continuously 
                    updated with current affairs to keep models relevant and 
                    contextually aware.
                  </p>
                  <div className="card-extras">
                    <span>OSM • Public Data • Current Affairs</span>
                  </div>
                </motion.div>

                {/* Visualization Card */}
                <motion.div 
                  className="contact-card"
                  initial={{ opacity: 0, x: 50 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.75, duration: 0.6 }}
                >
                  <div className="card-icon">📈</div>
                  <h2 className="card-title">INTERACTIVE RESPONSES</h2>
                  <p className="card-desc">
                    Answers aren't just text—they're visual experiences. 
                    Route suggestions come with maps, weather with visual 
                    forecasts, crowds with density indicators. Users interact 
                    with their answers, exploring alternatives and drilling 
                    down into specific details they care about.
                  </p>
                  <div className="card-extras">
                    <span>Visual • Interactive • Explorable</span>
                  </div>
                </motion.div>
              </div>

              {/* Additional Info - same as contact page */}
              <motion.div 
                className="contact-additional"
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.85, duration: 0.6 }}
              >
                <div className="additional-item">
                  <span className="additional-label">VERSION</span>
                  <span className="additional-value">v2.4.1</span>
                </div>
                <div className="additional-divider" />
                <div className="additional-item">
                  <span className="additional-label">API STATUS</span>
                  <span className="additional-value">COMING SOON</span>
                </div>
                <div className="additional-divider" />
                <div className="additional-item">
                  <span className="additional-label">RESOURCES</span>
                  <div className="social-links">
                    <a href="https://github.com/aayush110410/OVERHAUL" target="_blank" rel="noopener noreferrer" className="social-link">GITHUB</a>
                    <a href="/features" className="social-link">FEATURES</a>
                    <a href="/contact" className="social-link">CONTACT</a>
                  </div>
                </div>
              </motion.div>

              {/* Decorative Elements */}
              <motion.div 
                className="contact-decoration"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.6, duration: 1 }}
              >
                <div className="deco-line deco-line-1" />
                <div className="deco-line deco-line-2" />
                <div className="deco-circle" />
              </motion.div>
            </div>

            {/* Footer */}
            <footer className="contact-footer">
              <span>© 2025 OVERHAUL. ALL RIGHTS RESERVED.</span>
            </footer>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
}

export default Docs
