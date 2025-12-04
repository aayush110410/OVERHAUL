import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence, useInView } from 'framer-motion'
import { useNavigate, useLocation, Link } from 'react-router-dom'
import './App.css'

// ============================================
// OV LOADER (Same as other pages)
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
          animate={{ scale: phase === 'zoomOut' ? 1 : phase === 'hold' ? 1 : 50, opacity: 1 }}
          transition={{ duration: phase === 'zoomOut' ? 0.4 : phase === 'zoomIn' ? 0.5 : 0.1, ease: [0.76, 0, 0.24, 1] }}
        >
          <motion.span className="loader-ln-text-o" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.15, delay: 0.15 }}>O</motion.span>
          <motion.span className="loader-ln-text-v" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.15, delay: 0.2 }}>V</motion.span>
        </motion.div>
      </div>
    </motion.div>
  )
}

// ============================================
// EXIT LOADER (Same as other pages)
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
      }, 450)
      return () => clearTimeout(completeTimer)
    }
  }, [phase, onComplete])

  return (
    <motion.div 
      className="loader-ln"
      initial={{ opacity: 0 }}
      animate={{ opacity: phase === 'zoomIn' ? 0 : 1 }}
      transition={{ duration: 0.2 }}
    >
      <div className="loader-ln-content">
        <motion.div 
          className="loader-ln-logo"
          initial={{ scale: 50, opacity: 0 }}
          animate={{ scale: phase === 'zoomOut' ? 1 : phase === 'hold' ? 1 : 50, opacity: 1 }}
          transition={{ duration: phase === 'zoomOut' ? 0.4 : phase === 'zoomIn' ? 0.45 : 0.1, ease: [0.76, 0, 0.24, 1] }}
        >
          <motion.span className="loader-ln-text-o">O</motion.span>
          <motion.span className="loader-ln-text-v">V</motion.span>
        </motion.div>
      </div>
    </motion.div>
  )
}

// ============================================
// ANIMATED SECTION
// ============================================
function AnimatedSection({ children, className, delay = 0 }) {
  const ref = useRef(null)
  const isInView = useInView(ref, { once: true, margin: "-100px" })
  
  return (
    <motion.div
      ref={ref}
      className={className}
      initial={{ opacity: 0, y: 50 }}
      animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 50 }}
      transition={{ duration: 0.6, delay, ease: [0.16, 1, 0.3, 1] }}
    >
      {children}
    </motion.div>
  )
}

// ============================================
// FEATURES PAGE
// ============================================
function Features() {
  const location = useLocation()
  const skipLoader = location.state?.skipLoader || false
  const [loading, setLoading] = useState(!skipLoader)
  const [exiting, setExiting] = useState(false)
  const [hovering, setHovering] = useState(false)
  const [pendingNavigation, setPendingNavigation] = useState(null)
  const navigate = useNavigate()
  
  // Cursor refs
  const cursorRef = useRef(null)
  const mousePos = useRef({ x: 0, y: 0 })
  const rafId = useRef(null)

  useEffect(() => {
    document.title = 'OVERHAUL | Features'
  }, [])

  // Handle browser back/forward
  useEffect(() => {
    window.history.pushState({ skipLoader: true }, '', window.location.href)
    const handlePopState = () => {
      setExiting(true)
      setPendingNavigation('/')
    }
    window.addEventListener('popstate', handlePopState)
    return () => window.removeEventListener('popstate', handlePopState)
  }, [])

  const handleBackHome = (e) => {
    e.preventDefault()
    setExiting(true)
    setPendingNavigation('/')
  }

  const handleNavigation = (e, path) => {
    e.preventDefault()
    setExiting(true)
    setPendingNavigation(path)
  }

  const handleExitComplete = () => {
    navigate(pendingNavigation || '/', { state: { skipLoader: true } })
  }

  // Cursor tracking
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

  // Hover states
  useEffect(() => {
    const handleHover = () => {
      const hoverable = document.querySelectorAll('a, button, .feature-card, .engine-card')
      hoverable.forEach(el => {
        el.addEventListener('mouseenter', () => setHovering(true))
        el.addEventListener('mouseleave', () => setHovering(false))
      })
    }
    if (!loading && !exiting) handleHover()
  }, [loading, exiting])

  // Engine data - 5 specialized models + LDRAGO
  const engines = [
    {
      icon: '🚗',
      name: 'Traffic Model',
      tagline: 'Route Intelligence',
      description: 'Analyzes routes and traffic patterns—shortest paths, congested zones, light traffic windows based on time-of-day variations.',
      outputs: ['Route Options', 'Congestion Levels', 'Time-Based Patterns', 'Alternative Paths'],
      color: '#ccff00'
    },
    {
      icon: '🌿',
      name: 'Pollution & AQI Model',
      tagline: 'Air Quality Intelligence',
      description: 'Monitors air quality at destinations and along routes. Works with Weather Model for integrated AQI predictions.',
      outputs: ['AQI Levels', 'Route Air Quality', 'Health Recommendations', 'Pollution Trends'],
      color: '#00d4ff'
    },
    {
      icon: '🌦️',
      name: 'Weather Model',
      tagline: 'Hyperlocal Forecasting',
      description: 'Predicts temperatures, rainfall, wind, and seasonal patterns. Identifies hyperlocal variations often missed by standard forecasts.',
      outputs: ['Temperature', 'Rainfall Probability', 'Wind Conditions', 'Micro-Climate Data'],
      color: '#60a5fa'
    },
    {
      icon: '📜',
      name: 'Policy & Economics Model',
      tagline: 'Regulatory Intelligence',
      description: 'Tracks regulations, VIP movements, route diversions, market closures, and administrative decisions affecting travel.',
      outputs: ['Route Restrictions', 'Diversions', 'Market Status', 'Special Notices'],
      color: '#ff6b35'
    },
    {
      icon: '👥',
      name: 'Behavioral Model',
      tagline: 'Human Pattern Recognition',
      description: 'Understands human patterns—weekend crowds, holiday impacts, festival dynamics. Predicts crowd density based on context.',
      outputs: ['Crowd Predictions', 'Peak Hours', 'Cultural Events', 'Destination Popularity'],
      color: '#a855f7'
    }
  ]

  const capabilities = [
    { icon: '💬', title: 'Natural Language Queries', desc: 'Ask complex questions in plain English' },
    { icon: '🔄', title: 'Model Collaboration', desc: 'Models discuss and cross-validate answers' },
    { icon: '🎯', title: 'Unified Responses', desc: 'Collective intelligence, single answer' },
    { icon: '📊', title: 'Interactive Visuals', desc: 'Maps, charts, and explorable responses' },
    { icon: '🌐', title: 'Hyperlocal Insights', desc: 'Details often missed by others' },
    { icon: '⚡', title: 'Current Affairs', desc: 'Always updated with latest data' },
  ]

  return (
    <>
      {/* Custom Cursor */}
      <div 
        ref={cursorRef}
        className={`cursor ${hovering ? 'hovering' : ''}`}
      />

      {/* Moving Background - Same as Contact page */}
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
            className="contact-page features-page"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.4 }}
          >
            {/* Navigation - Same as Contact */}
            <nav className="contact-nav">
              <a href="/" onClick={handleBackHome} className="nav-logo">OVERHAUL™</a>
              <a href="/" onClick={handleBackHome} className="back-btn">
                ← BACK TO HOME
              </a>
            </nav>

            {/* Hero Section */}
            <div className="contact-content features-content">
              <motion.div 
                className="contact-header"
                initial={{ opacity: 0, y: 50 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1, duration: 0.6 }}
              >
                <span className="contact-label">PLATFORM CAPABILITIES</span>
                <h1 className="contact-title">
                  FEATURES<br/>
                  <span className="text-outline">& ENGINES</span>
                </h1>
              </motion.div>

              {/* LDRAGO Section */}
              <AnimatedSection className="features-ldrago-section" delay={0.2}>
                <div className="ldrago-grid">
                  <div className="ldrago-visual">
                    <div className="brain-container">
                      <motion.div 
                        className="brain-ring ring-outer"
                        animate={{ rotate: 360 }}
                        transition={{ duration: 30, repeat: Infinity, ease: "linear" }}
                      />
                      <motion.div 
                        className="brain-ring ring-middle"
                        animate={{ rotate: -360 }}
                        transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
                      />
                      <motion.div 
                        className="brain-ring ring-inner"
                        animate={{ rotate: 360 }}
                        transition={{ duration: 15, repeat: Infinity, ease: "linear" }}
                      />
                      <motion.div 
                        className="brain-core"
                        animate={{ scale: [1, 1.1, 1] }}
                        transition={{ duration: 2, repeat: Infinity }}
                      >
                        🧠
                      </motion.div>
                    </div>
                  </div>
                  
                  <div className="ldrago-content">
                    <span className="features-badge">THE ORCHESTRATOR</span>
                    <h2 className="ldrago-title">
                      <span className="gradient-text">LDRAGO</span>
                      <span className="ldrago-subtitle">Master Intelligence</span>
                    </h2>
                    <p className="ldrago-fullform">
                      Large-scale Dynamic Resource Allocation & Governance Orchestrator
                    </p>
                    <p className="ldrago-desc">
                      LDRAGO is the central brain of OVERHAUL. When you ask a question, LDRAGO analyzes your 
                      prompt, activates the relevant specialized models, distributes tasks, facilitates their 
                      group discussion, validates collective outputs, and delivers a unified visualized response.
                    </p>
                    <div className="ldrago-pills">
                      {['Prompt Analysis', 'Model Activation', 'Task Distribution', 'Output Validation', 'Response Generation'].map((pill, i) => (
                        <motion.span 
                          key={i}
                          className="ldrago-pill"
                          whileHover={{ scale: 1.05, borderColor: 'var(--lime)' }}
                        >
                          {pill}
                        </motion.span>
                      ))}
                    </div>
                  </div>
                </div>
              </AnimatedSection>

              {/* Stats */}
              <AnimatedSection className="features-stats" delay={0.3}>
                <div className="stats-row">
                  <div className="stat-item">
                    <span className="stat-value">6</span>
                    <span className="stat-label">Intelligent Models</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-value">1</span>
                    <span className="stat-label">Unified Answer</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-value">5</span>
                    <span className="stat-label">Specialized Engines</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-value">∞</span>
                    <span className="stat-label">Collaborative Insights</span>
                  </div>
                </div>
              </AnimatedSection>

              {/* Engines */}
              <AnimatedSection className="features-engines" delay={0.4}>
                <h3 className="section-title">
                  <span className="features-badge">SPECIALIZED MODELS</span>
                  5 Expert Models
                </h3>
                <div className="engines-grid">
                  {engines.map((engine, i) => (
                    <motion.div
                      key={i}
                      className="engine-card"
                      style={{ '--engine-color': engine.color }}
                      initial={{ opacity: 0, y: 30 }}
                      whileInView={{ opacity: 1, y: 0 }}
                      viewport={{ once: true }}
                      transition={{ delay: 0.1 * i, duration: 0.5 }}
                      whileHover={{ y: -8, borderColor: engine.color }}
                    >
                      <div className="engine-icon">{engine.icon}</div>
                      <h4 className="engine-name">{engine.name}</h4>
                      <span className="engine-tagline">{engine.tagline}</span>
                      <p className="engine-desc">{engine.description}</p>
                      <div className="engine-outputs">
                        <span className="outputs-label">OUTPUTS</span>
                        <ul>
                          {engine.outputs.map((output, j) => (
                            <li key={j}>{output}</li>
                          ))}
                        </ul>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </AnimatedSection>

              {/* Conversational AI Demo */}
              <AnimatedSection className="features-conversation" delay={0.3}>
                <h3 className="section-title">
                  <span className="features-badge">COLLABORATIVE INTELLIGENCE</span>
                  Ask Anything, Models Discuss
                </h3>
                <div className="conversation-demo">
                  <motion.div 
                    className="query-bubble"
                    initial={{ opacity: 0, x: -30 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true }}
                    transition={{ delay: 0.2 }}
                  >
                    <span className="bubble-icon">💬</span>
                    <p>"Planning a trip to Delhi this Saturday—what should I expect?"</p>
                  </motion.div>
                  
                  <motion.div 
                    className="response-bubble"
                    initial={{ opacity: 0, x: 30 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true }}
                    transition={{ delay: 0.4 }}
                  >
                    <div className="response-header">
                      <span className="response-icon">🐉</span>
                      <span className="response-label">LDRAGO + 5 Models</span>
                    </div>
                    <p>
                      <span className="highlight">Traffic:</span> Route B recommended (less congested). 
                      <span className="highlight"> Weather:</span> Rain expected afternoon—carry umbrella. 
                      <span className="highlight"> AQI:</span> Better on Route B. 
                      <span className="highlight"> Crowds:</span> Weekend rush at popular spots after 2 PM. 
                      <span className="highlight"> Policy:</span> No diversions reported.
                      <br/><br/>
                      <strong>Recommendation:</strong> Depart by 9 AM via Route B for best experience.
                    </p>
                  </motion.div>
                </div>
              </AnimatedSection>

              {/* Capabilities Grid */}
              <AnimatedSection className="features-capabilities" delay={0.3}>
                <h3 className="section-title">
                  <span className="features-badge">CAPABILITIES</span>
                  Platform Features
                </h3>
                <div className="capabilities-grid">
                  {capabilities.map((cap, i) => (
                    <motion.div
                      key={i}
                      className="capability-card"
                      initial={{ opacity: 0, y: 20 }}
                      whileInView={{ opacity: 1, y: 0 }}
                      viewport={{ once: true }}
                      transition={{ delay: 0.05 * i, duration: 0.4 }}
                      whileHover={{ borderColor: 'rgba(204, 255, 0, 0.3)' }}
                    >
                      <span className="capability-icon">{cap.icon}</span>
                      <h4>{cap.title}</h4>
                      <p>{cap.desc}</p>
                    </motion.div>
                  ))}
                </div>
              </AnimatedSection>

              {/* CTA */}
              <AnimatedSection className="features-cta" delay={0.3}>
                <h3 className="cta-title">Ready to Transform Your City?</h3>
                <p className="cta-desc">Experience the power of earth-scale simulation</p>
                <div className="cta-buttons">
                  <Link 
                    to="/demo" 
                    onClick={(e) => handleNavigation(e, '/demo')}
                    className="cta-btn primary"
                  >
                    TRY DEMO →
                  </Link>
                  <Link 
                    to="/contact" 
                    onClick={(e) => handleNavigation(e, '/contact')}
                    className="cta-btn secondary"
                  >
                    CONTACT US
                  </Link>
                </div>
              </AnimatedSection>

              {/* Footer */}
              <footer className="features-footer">
                <div className="footer-logo">OVERHAUL™</div>
                <p className="footer-tagline">Simulating tomorrow, today.</p>
                <div className="footer-links">
                  <Link to="/" onClick={handleBackHome}>Home</Link>
                  <Link to="/demo" onClick={(e) => handleNavigation(e, '/demo')}>Demo</Link>
                  <Link to="/contact" onClick={(e) => handleNavigation(e, '/contact')}>Contact</Link>
                  <Link to="/privacy">Privacy</Link>
                </div>
                <p className="footer-copy">© 2025 OVERHAUL. All rights reserved.</p>
              </footer>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
}

export default Features
