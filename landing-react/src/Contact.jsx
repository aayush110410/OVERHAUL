import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useNavigate, useLocation } from 'react-router-dom'
import './App.css'

// ============================================
// OV LOADER (Same style as main page - no bar)
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
  const [phase, setPhase] = useState('zoomOut') // zoomOut -> hold -> zoomIn -> done
  
  useEffect(() => {
    // Phase 1: Zoom out (logo appears from large scale)
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
      }, 200)
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
// CONTACT PAGE
// ============================================
function Contact() {
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
    // Push a state so we can intercept the back button
    window.history.pushState({ skipLoader: true }, '', window.location.href)
    
    const handlePopState = (e) => {
      // Trigger exit animation then navigate
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
    document.title = 'OVERHAUL | Contact'
  }, [])

  const handleBackHome = (e) => {
    e.preventDefault()
    setExiting(true)
    setPendingNavigation('/')
  }

  // Handle exit - show loader then navigate
  const handleExitComplete = () => {
    navigate(pendingNavigation || '/', { state: { skipLoader: true } })
  }

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

      {/* Entry Loader - OV appears, then zooms in */}
      <AnimatePresence mode="wait">
        {loading && (
          <OVLoader key="entry-loader" onComplete={() => setLoading(false)} />
        )}
      </AnimatePresence>

      {/* Exit Loader - OV zooms out then zooms in to go back home */}
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
                <span className="contact-label">GET IN TOUCH</span>
                <h1 className="contact-title">
                  CONTACT<br/>
                  <span className="text-outline">US</span>
                </h1>
              </motion.div>

              <div className="contact-grid">
                {/* Developer Contact */}
                <motion.div 
                  className="contact-card"
                  initial={{ opacity: 0, x: -50 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.2, duration: 0.6 }}
                >
                  <div className="card-icon">{'</>'}</div>
                  <h2 className="card-title">FOR DEVELOPERS</h2>
                  <p className="card-desc">
                    Technical inquiries, API access, integration support, and developer partnerships.
                  </p>
                  <a href="mailto:founders@overhaul.co.in" className="card-email">
                    <span className="email-label">EMAIL</span>
                    <span className="email-address">founders@overhaul.co.in</span>
                  </a>
                  <div className="card-extras">
                    <span>Response time: 24-48 hours</span>
                  </div>
                </motion.div>

                {/* General Query Contact */}
                <motion.div 
                  className="contact-card"
                  initial={{ opacity: 0, x: 50 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.3, duration: 0.6 }}
                >
                  <div className="card-icon">✉</div>
                  <h2 className="card-title">GENERAL QUERIES</h2>
                  <p className="card-desc">
                    Product questions, feedback, press inquiries, and partnership opportunities.
                  </p>
                  <a href="mailto:query.overhaul@gmail.com" className="card-email">
                    <span className="email-label">EMAIL</span>
                    <span className="email-address">query.overhaul@gmail.com</span>
                  </a>
                  <div className="card-extras">
                    <span>Response time: 1-3 business days</span>
                  </div>
                </motion.div>
              </div>

              {/* Additional Info */}
              <motion.div 
                className="contact-additional"
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4, duration: 0.6 }}
              >
                <div className="additional-item">
                  <span className="additional-label">LOCATION</span>
                  <span className="additional-value">GLOBAL / REMOTE</span>
                </div>
                <div className="additional-divider" />
                <div className="additional-item">
                  <span className="additional-label">WORKING HOURS</span>
                  <span className="additional-value">24/7 ASYNC</span>
                </div>
                <div className="additional-divider" />
                <div className="additional-item">
                  <span className="additional-label">SOCIALS</span>
                  <div className="social-links">
                    <a href="#" className="social-link">TWITTER</a>
                    <a href="#" className="social-link">LINKEDIN</a>
                    <a href="#" className="social-link">GITHUB</a>
                  </div>
                </div>
              </motion.div>

              {/* Decorative Elements */}
              <motion.div 
                className="contact-decoration"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.5, duration: 1 }}
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

export default Contact
