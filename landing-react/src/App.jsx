import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence, useScroll, useSpring, useInView } from 'framer-motion'
import { Link, useLocation } from 'react-router-dom'
import { supabase } from './supabase'
import './App.css'

// ============================================
// JOIN US FORM - INLINE DROPDOWN
// ============================================
function JoinUsForm({ isOpen, onClose }) {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    occupation: '',
    useCase: '',
    suggestions: ''
  })
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [submitStatus, setSubmitStatus] = useState(null)

  const handleChange = (e) => {
    setFormData(prev => ({
      ...prev,
      [e.target.name]: e.target.value
    }))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setIsSubmitting(true)
    setSubmitStatus(null)

    try {
      if (!supabase) {
        // Demo mode - just show success
        console.log('Demo mode - form data:', formData)
        setSubmitStatus('success')
        setFormData({ name: '', email: '', occupation: '', useCase: '', suggestions: '' })
        setTimeout(() => {
          setSubmitStatus(null)
        }, 3000)
        setIsSubmitting(false)
        return
      }

      const { error } = await supabase
        .from('waitlist')
        .insert([
          {
            name: formData.name,
            email: formData.email,
            occupation: formData.occupation,
            use_case: formData.useCase,
            suggestions: formData.suggestions,
            created_at: new Date().toISOString()
          }
        ])

      if (error) throw error

      setSubmitStatus('success')
      setFormData({ name: '', email: '', occupation: '', useCase: '', suggestions: '' })
      
      setTimeout(() => {
        setSubmitStatus(null)
      }, 3000)
    } catch (error) {
      console.error('Error submitting form:', error)
      setSubmitStatus('error')
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          className="join-form-dropdown"
          initial={{ height: 0, opacity: 0 }}
          animate={{ height: 'auto', opacity: 1 }}
          exit={{ height: 0, opacity: 0 }}
          transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
        >
          <motion.div
            className="join-form-inner"
            initial={{ y: -20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: -20, opacity: 0 }}
            transition={{ duration: 0.4, delay: 0.1 }}
          >
            {submitStatus === 'success' ? (
              <motion.div 
                className="form-success-inline"
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
              >
                <div className="success-check">✓</div>
                <h3>WELCOME TO THE REVOLUTION</h3>
                <p>We'll be in touch soon. Get ready to transform the future.</p>
              </motion.div>
            ) : (
              <>
                <div className="form-header-inline">
                  <span className="form-tag">JOIN THE WAITLIST</span>
                  <button className="form-close-inline" onClick={onClose}>
                    CLOSE ×
                  </button>
                </div>
                
                <form onSubmit={handleSubmit} className="join-form-grid">
                  <div className="form-row">
                    <div className="form-field-inline">
                      <label>NAME *</label>
                      <input
                        type="text"
                        name="name"
                        value={formData.name}
                        onChange={handleChange}
                        required
                        placeholder="Your full name"
                      />
                    </div>
                    <div className="form-field-inline">
                      <label>EMAIL *</label>
                      <input
                        type="email"
                        name="email"
                        value={formData.email}
                        onChange={handleChange}
                        required
                        placeholder="your@email.com"
                      />
                    </div>
                  </div>

                  <div className="form-row">
                    <div className="form-field-inline">
                      <label>OCCUPATION *</label>
                      <input
                        type="text"
                        name="occupation"
                        value={formData.occupation}
                        onChange={handleChange}
                        required
                        placeholder="e.g. Software Engineer"
                      />
                    </div>
                    <div className="form-field-inline full-width">
                      <label>WHAT WILL YOU USE IT FOR? *</label>
                      <textarea
                        name="useCase"
                        value={formData.useCase}
                        onChange={handleChange}
                        required
                        placeholder="Describe your use case..."
                        rows={2}
                      />
                    </div>
                  </div>

                  <div className="form-row">
                    <div className="form-field-inline full-width">
                      <label>FEATURE SUGGESTIONS</label>
                      <textarea
                        name="suggestions"
                        value={formData.suggestions}
                        onChange={handleChange}
                        placeholder="Any features you'd like to see?"
                        rows={2}
                      />
                    </div>
                  </div>

                  {submitStatus === 'error' && (
                    <motion.div 
                      className="form-error-inline"
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                    >
                      Something went wrong. Please try again.
                    </motion.div>
                  )}

                  <div className="form-actions">
                    <button 
                      type="submit" 
                      className="btn-submit"
                      disabled={isSubmitting}
                    >
                      <span>{isSubmitting ? 'SUBMITTING...' : 'SUBMIT APPLICATION'}</span>
                      <span className="btn-arrow">→</span>
                    </button>
                  </div>
                </form>
              </>
            )}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}

// ============================================
// ANIMATED COUNTER - JACKPOT STYLE
// ============================================
function JackpotCounter({ value, suffix = '', prefix = '' }) {
  const ref = useRef(null)
  const isInView = useInView(ref, { once: true })
  const [displayValue, setDisplayValue] = useState(0)
  
  useEffect(() => {
    if (!isInView) return
    
    const numericValue = parseFloat(value.replace(/[^0-9.]/g, ''))
    const duration = 2000
    const steps = 60
    const increment = numericValue / steps
    let current = 0
    
    const timer = setInterval(() => {
      current += increment
      if (current >= numericValue) {
        setDisplayValue(numericValue)
        clearInterval(timer)
      } else {
        setDisplayValue(Math.floor(current))
      }
    }, duration / steps)
    
    return () => clearInterval(timer)
  }, [isInView, value])
  
  return (
    <span ref={ref} className="jackpot-counter">
      {prefix}{displayValue.toLocaleString()}{suffix}
    </span>
  )
}

// ============================================
// CUSTOM CURSOR - SINGLE ELEMENT
// ============================================
function CustomCursor() {
  const cursorRef = useRef(null)
  const [isHovering, setIsHovering] = useState(false)
  const mousePos = useRef({ x: 0, y: 0 })
  const cursorPos = useRef({ x: 0, y: 0 })
  const rafId = useRef(null)

  useEffect(() => {
    const updateCursor = () => {
      if (cursorRef.current) {
        cursorRef.current.style.transform = `translate3d(${mousePos.current.x}px, ${mousePos.current.y}px, 0) translate(-50%, -50%)`
      }
      
      rafId.current = requestAnimationFrame(updateCursor)
    }
    
    rafId.current = requestAnimationFrame(updateCursor)
    
    const moveCursor = (e) => {
      mousePos.current = { x: e.clientX, y: e.clientY }
    }

    const handleMouseOver = (e) => {
      const target = e.target
      if (target.tagName === 'A' || target.tagName === 'BUTTON' || 
          target.closest('a') || target.closest('button')) {
        setIsHovering(true)
      }
    }

    const handleMouseOut = (e) => {
      const target = e.target
      if (target.tagName === 'A' || target.tagName === 'BUTTON' ||
          target.closest('a') || target.closest('button')) {
        setIsHovering(false)
      }
    }

    document.addEventListener('mousemove', moveCursor, { passive: true })
    document.addEventListener('mouseover', handleMouseOver, { passive: true })
    document.addEventListener('mouseout', handleMouseOut, { passive: true })

    return () => {
      cancelAnimationFrame(rafId.current)
      document.removeEventListener('mousemove', moveCursor)
      document.removeEventListener('mouseover', handleMouseOver)
      document.removeEventListener('mouseout', handleMouseOut)
    }
  }, [])

  return <div ref={cursorRef} className={`cursor ${isHovering ? 'hovering' : ''}`} />
}

// ============================================
// MOVING BACKGROUND LINES - LN STYLE
// ============================================
function MovingBackground() {
  return (
    <div className="moving-bg">
      <svg className="moving-bg-svg" viewBox="0 0 1920 1080" preserveAspectRatio="none">
        <defs>
          <linearGradient id="lineGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="rgba(204, 255, 0, 0)" />
            <stop offset="30%" stopColor="rgba(204, 255, 0, 0.15)" />
            <stop offset="50%" stopColor="rgba(204, 255, 0, 0.25)" />
            <stop offset="70%" stopColor="rgba(204, 255, 0, 0.15)" />
            <stop offset="100%" stopColor="rgba(204, 255, 0, 0)" />
          </linearGradient>
        </defs>
        {[...Array(10)].map((_, i) => (
          <motion.path
            key={i}
            d={`M${-300 + i * 100},${80 + i * 80} Q${400 + i * 50},${250 + Math.sin(i) * 200} ${960},${540} T${2200 + i * 100},${950 + i * 20}`}
            fill="none"
            stroke="url(#lineGrad)"
            strokeWidth={i % 3 === 0 ? "2" : "1"}
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{ 
              pathLength: [0, 1],
              pathOffset: [0, 1],
              opacity: [0, 0.8, 0]
            }}
            transition={{
              duration: 12 + i * 2,
              repeat: Infinity,
              ease: "linear",
              delay: i * 1
            }}
          />
        ))}
      </svg>
      
      {/* Organic blob shapes like LN site */}
      <div className="organic-shapes">
        {[...Array(5)].map((_, i) => (
          <motion.div
            key={i}
            className="organic-shape"
            style={{
              left: `${10 + i * 20}%`,
              top: `${15 + (i % 3) * 30}%`,
            }}
            animate={{
              x: [0, 40, -30, 0],
              y: [0, -50, 30, 0],
              scale: [1, 1.1, 0.95, 1],
            }}
            transition={{
              duration: 25 + i * 5,
              repeat: Infinity,
              ease: "easeInOut",
            }}
          />
        ))}
      </div>
    </div>
  )
}

// ============================================
// LOADER - OV ZOOM TRANSITION (LN STYLE)
// ============================================
function Loader({ onComplete }) {
  const [phase, setPhase] = useState('intro')
  const [count, setCount] = useState(0)
  
  useEffect(() => {
    // Start with intro animation
    const introTimer = setTimeout(() => {
      setPhase('loading')
    }, 800)
    
    return () => clearTimeout(introTimer)
  }, [])
  
  useEffect(() => {
    if (phase !== 'loading') return
    
    const interval = setInterval(() => {
      setCount(prev => {
        if (prev >= 100) {
          clearInterval(interval)
          setTimeout(() => {
            setPhase('zooming')
            setTimeout(() => {
              onComplete()
            }, 1200)
          }, 400)
          return 100
        }
        return prev + 2
      })
    }, 30)
    return () => clearInterval(interval)
  }, [onComplete, phase])

  return (
    <motion.div 
      className="loader-ln"
      initial={{ opacity: 1 }}
      animate={phase === 'zooming' ? { opacity: 0 } : { opacity: 1 }}
      transition={{ duration: 0.5, ease: [0.76, 0, 0.24, 1], delay: phase === 'zooming' ? 0.8 : 0 }}
    >
      <div className="loader-ln-content">
        <motion.div 
          className="loader-ln-logo"
          initial={{ scale: 0.8, opacity: 0 }}
          animate={phase === 'zooming' 
            ? { scale: 50, opacity: 1 } 
            : { scale: 1, opacity: 1 }
          }
          transition={phase === 'zooming' 
            ? { duration: 1, ease: [0.76, 0, 0.24, 1] }
            : { duration: 0.5, ease: [0.16, 1, 0.3, 1] }
          }
        >
          <motion.span 
            className="loader-ln-text-o"
            initial={{ y: 40, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.1, ease: [0.16, 1, 0.3, 1] }}
          >
            O
          </motion.span>
          <motion.span 
            className="loader-ln-text-v"
            initial={{ y: 40, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.2, ease: [0.16, 1, 0.3, 1] }}
          >
            V
          </motion.span>
        </motion.div>
        
        <motion.div 
          className="loader-ln-bar"
          initial={{ opacity: 0, scaleX: 0 }}
          animate={{ opacity: 1, scaleX: 1 }}
          transition={{ duration: 0.4, delay: 0.4, ease: [0.16, 1, 0.3, 1] }}
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
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, delay: 0.5, ease: [0.16, 1, 0.3, 1] }}
        >
          <span>INITIALIZING OVERHAUL</span>
          <span>{count}%</span>
        </motion.div>
      </div>
    </motion.div>
  )
}

// ============================================
// MARQUEE COMPONENT
// ============================================
function Marquee({ children, speed = 20, reverse = false }) {
  return (
    <div className="marquee-container">
      <motion.div
        className="marquee-content"
        animate={{ x: reverse ? ['0%', '-50%'] : ['-50%', '0%'] }}
        transition={{ duration: speed, repeat: Infinity, ease: 'linear' }}
      >
        {children}
        {children}
      </motion.div>
    </div>
  )
}

// ============================================
// MAGNETIC BUTTON
// ============================================
function MagneticButton({ children, className, href, to, onClick, external }) {
  const ref = useRef(null)
  const [position, setPosition] = useState({ x: 0, y: 0 })

  const handleMouse = (e) => {
    const { clientX, clientY } = e
    const { left, top, width, height } = ref.current.getBoundingClientRect()
    const x = (clientX - left - width / 2) * 0.3
    const y = (clientY - top - height / 2) * 0.3
    setPosition({ x, y })
  }

  const handleLeave = () => setPosition({ x: 0, y: 0 })

  // Use Link for internal navigation, anchor for external, button otherwise
  if (to) {
    return (
      <motion.div
        ref={ref}
        onMouseMove={handleMouse}
        onMouseLeave={handleLeave}
        animate={{ x: position.x, y: position.y }}
        transition={{ type: 'spring', stiffness: 100, damping: 20, mass: 0.8 }}
        whileTap={{ scale: 0.95 }}
        style={{ display: 'inline-block', willChange: 'transform' }}
      >
        <Link to={to} className={className} onClick={onClick}>
          {children}
        </Link>
      </motion.div>
    )
  }

  // External link with target blank
  if (href && external) {
    return (
      <motion.a
        ref={ref}
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        className={className}
        onClick={onClick}
        onMouseMove={handleMouse}
        onMouseLeave={handleLeave}
        animate={{ x: position.x, y: position.y }}
        transition={{ type: 'spring', stiffness: 100, damping: 20, mass: 0.8 }}
        whileTap={{ scale: 0.95 }}
        style={{ willChange: 'transform' }}
      >
        {children}
      </motion.a>
    )
  }

  const Component = href ? motion.a : motion.button

  return (
    <Component
      ref={ref}
      href={href}
      className={className}
      onClick={onClick}
      onMouseMove={handleMouse}
      onMouseLeave={handleLeave}
      animate={{ x: position.x, y: position.y }}
      transition={{ type: 'spring', stiffness: 100, damping: 20, mass: 0.8 }}
      whileTap={{ scale: 0.95 }}
      style={{ willChange: 'transform' }}
    >
      {children}
    </Component>
  )
}

// ============================================
// REVEAL TEXT ANIMATION
// ============================================
function RevealText({ children, className = '' }) {
  const ref = useRef(null)
  const isInView = useInView(ref, { once: true, margin: "-100px" })
  
  return (
    <motion.div
      ref={ref}
      className={`reveal-text ${className}`}
      initial={{ opacity: 0, y: 80 }}
      animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 80 }}
      transition={{ duration: 1, ease: [0.16, 1, 0.3, 1] }}
      style={{ willChange: 'transform, opacity' }}
    >
      {children}
    </motion.div>
  )
}

// ============================================
// MAIN APP
// ============================================
function App() {
  const location = useLocation()
  const skipLoader = location.state?.skipLoader || false
  const [loading, setLoading] = useState(!skipLoader)
  const [showJoinForm, setShowJoinForm] = useState(false)
  const containerRef = useRef(null)
  
  const { scrollYProgress } = useScroll()
  const smoothProgress = useSpring(scrollYProgress, { stiffness: 80, damping: 40, mass: 0.5 })

  // Set page title
  useEffect(() => {
    document.title = 'OVERHAUL | Home'
  }, [])

  return (
    <>
      {/* Custom Cursor */}
      <CustomCursor />
      
      {/* Moving Background */}
      <MovingBackground />
      
      {/* Progress bar */}
      <motion.div className="progress-bar" style={{ scaleX: smoothProgress }} />

      <AnimatePresence mode="wait">
        {loading && <Loader key="loader" onComplete={() => setLoading(false)} />}
      </AnimatePresence>

      <div ref={containerRef} className="main-content">
        {/* ============================================ */}
        {/* HERO SECTION */}
        {/* ============================================ */}
        <section className="hero">
          <nav className="nav">
            <div className="nav-logo">OVERHAUL™</div>
            <div className="nav-links">
              <a href="#about">ABOUT</a>
              <a href="#features">FEATURES</a>
              <a href="#journey">ROADMAP</a>
              <Link to="/contact">CONTACT</Link>
              <Link to="/support" className="nav-link-special">SUPPORT US</Link>
            </div>
            <MagneticButton 
              className="nav-cta" 
              onClick={() => {
                document.getElementById('cta-section')?.scrollIntoView({ behavior: 'smooth' })
                setTimeout(() => setShowJoinForm(true), 500)
              }}
            >
              JOIN US
            </MagneticButton>
          </nav>

          <div className="hero-content">
            <motion.div 
              className="hero-tag"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
            >
              <span className="tag-dot" />
              TRAFFIC SIMULATION PLATFORM
            </motion.div>
            
            <div className="hero-title-container">
              <motion.h1 
                className="hero-title"
                initial={{ y: 100, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ delay: 0.1, duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
              >
                OVER<span className="title-haul">HAUL</span>
              </motion.h1>
              <motion.p 
                className="hero-subtitle"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3, duration: 0.6 }}
              >
                THE WORLD REIMAGINED
              </motion.p>
            </div>

            <motion.p 
              className="hero-desc"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.4, duration: 0.5 }}
            >
              Pushing the boundaries of Artificial Intelligence.<br/>
              <span className="highlight-lime">Simulating tomorrow, today.</span>
            </motion.p>

            <motion.div 
              className="hero-ctas"
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5, duration: 0.5 }}
            >
              <MagneticButton className="btn-brutal" to="/demo">
                LAUNCH SIMULATOR
              </MagneticButton>
              <MagneticButton className="btn-outline" to="/demo">
                <span className="btn-icon">▶</span>
                WATCH DEMO
              </MagneticButton>
            </motion.div>

            <motion.div 
              className="hero-scroll-indicator"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 1 }}
            >
              <motion.div
                className="scroll-content"
                animate={{ y: [0, 8, 0] }}
                transition={{ duration: 1.5, repeat: Infinity }}
              >
                <span className="scroll-text">SCROLL TO EXPLORE</span>
                <span className="scroll-arrow">↓</span>
              </motion.div>
            </motion.div>
          </div>
        </section>

        {/* ============================================ */}
        {/* MARQUEE SECTION */}
        {/* ============================================ */}
        <div className="marquee-section">
          <Marquee speed={25}>
            <span>SIMULATE • ANALYZE • OPTIMIZE • DEPLOY • REDEFINING LIMITS • </span>
          </Marquee>
        </div>

        {/* ============================================ */}
        {/* ABOUT / INTRO SECTION */}
        {/* ============================================ */}
        <section className="about" id="about">
          <div className="about-intro">
            <RevealText>
              <span className="section-label">SINCE 2025</span>
            </RevealText>
            <RevealText className="about-headline">
              <h2>
                REDEFINING HOW CITIES<br/>
                <span className="text-outline">MOVE & BREATHE</span>
              </h2>
            </RevealText>
            <RevealText>
              <p className="about-text">
                Overhaul is the next generation Earth simulation platform, 
                leveraging AI and real-time data to transform the new world.
                We don't just predict World — We shape it.
              </p>
            </RevealText>
          </div>

          <div className="about-features">
            {[
              { num: '01', title: 'REAL-TIME\nSIMULATION', desc: 'Watch traffic flow in real-time with millisecond precision' },
              { num: '02', title: 'AI-POWERED\nPREDICTIONS', desc: 'Machine learning models trained on millions of scenarios' },
              { num: '03', title: 'IMPACT\nANALYSIS', desc: 'Understand the ripple effects of every decision' }
            ].map((feature, i) => (
              <motion.div 
                key={i}
                className="feature-block"
                initial={{ opacity: 0, y: 50, rotateX: -15 }}
                whileInView={{ opacity: 1, y: 0, rotateX: 0 }}
                viewport={{ once: true, margin: "-50px" }}
                transition={{ 
                  delay: i * 0.15,
                  duration: 0.6,
                  ease: [0.16, 1, 0.3, 1]
                }}
              >
                <motion.span 
                  className="feature-num"
                  initial={{ scale: 0, opacity: 0 }}
                  whileInView={{ scale: 1, opacity: 1 }}
                  viewport={{ once: true }}
                  transition={{ 
                    delay: i * 0.15 + 0.2,
                    type: "spring",
                    stiffness: 200
                  }}
                >
                  {feature.num}
                </motion.span>
                <motion.h3
                  initial={{ opacity: 0, x: -20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: i * 0.15 + 0.3 }}
                  style={{ whiteSpace: 'pre-line' }}
                >
                  {feature.title}
                </motion.h3>
                <motion.p
                  initial={{ opacity: 0 }}
                  whileInView={{ opacity: 1 }}
                  viewport={{ once: true }}
                  transition={{ delay: i * 0.15 + 0.4 }}
                >
                  {feature.desc}
                </motion.p>
              </motion.div>
            ))}
          </div>
        </section>

        {/* ============================================ */}
        {/* STATS SECTION - WITH JACKPOT COUNTERS */}
        {/* ============================================ */}
        <section className="stats" id="stats">
          <div className="stats-header">
            <RevealText>
              <span className="section-label">THE NUMBERS</span>
            </RevealText>
            <RevealText>
              <h2 className="stats-title">PLATFORM<br/>STATISTICS</h2>
            </RevealText>
          </div>
          
          <div className="stats-grid">
            <motion.div 
              className="stat-card"
              initial={{ opacity: 0, y: 50, scale: 0.9 }}
              whileInView={{ opacity: 1, y: 0, scale: 1 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ 
                duration: 0.5,
                ease: [0.16, 1, 0.3, 1]
              }}
            >
              <motion.span 
                className="stat-label"
                initial={{ opacity: 0 }}
                whileInView={{ opacity: 1 }}
                viewport={{ once: true }}
                transition={{ delay: 0.2 }}
              >
                SIMULATIONS RUN
              </motion.span>
              <span className="stat-value">
                <JackpotCounter value="10000" suffix="+" />
              </span>
            </motion.div>
            
            <motion.div 
              className="stat-card"
              initial={{ opacity: 0, y: 50, scale: 0.9 }}
              whileInView={{ opacity: 1, y: 0, scale: 1 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ 
                delay: 0.1,
                duration: 0.5,
                ease: [0.16, 1, 0.3, 1]
              }}
            >
              <motion.span 
                className="stat-label"
                initial={{ opacity: 0 }}
                whileInView={{ opacity: 1 }}
                viewport={{ once: true }}
                transition={{ delay: 0.3 }}
              >
                EFFICIENCY GAIN
              </motion.span>
              <span className="stat-value">
                <JackpotCounter value="47" suffix="%" />
              </span>
            </motion.div>
            
            <motion.div 
              className="stat-card"
              initial={{ opacity: 0, y: 50, scale: 0.9 }}
              whileInView={{ opacity: 1, y: 0, scale: 1 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ 
                delay: 0.2,
                duration: 0.5,
                ease: [0.16, 1, 0.3, 1]
              }}
            >
              <motion.span 
                className="stat-label"
                initial={{ opacity: 0 }}
                whileInView={{ opacity: 1 }}
                viewport={{ once: true }}
                transition={{ delay: 0.4 }}
              >
                WE CAN PLAN UPTO
              </motion.span>
              <span className="stat-value">
                <JackpotCounter value="150" suffix="+CITIES" />
              </span>
            </motion.div>
            
            <motion.div 
              className="stat-card stat-card-large"
              initial={{ opacity: 0, y: 50, scale: 0.9 }}
              whileInView={{ opacity: 1, y: 0, scale: 1 }}
              viewport={{ once: true, margin: "-50px" }}
              transition={{ 
                delay: 0.3,
                duration: 0.5,
                ease: [0.16, 1, 0.3, 1]
              }}
            >
              <motion.span 
                className="stat-label"
                initial={{ opacity: 0 }}
                whileInView={{ opacity: 1 }}
                viewport={{ once: true }}
                transition={{ delay: 0.5 }}
              >
                UPTO DATA POINTS CAN BE PROCESSED DAILY
              </motion.span>
              <span className="stat-value">
                <JackpotCounter value="2300000" />
              </span>
            </motion.div>
          </div>
        </section>

        {/* ============================================ */}
        {/* FEATURES SECTION */}
        {/* ============================================ */}
        <section className="features" id="features">
          <div className="features-header">
            <RevealText>
              <span className="section-label">CAPABILITIES</span>
            </RevealText>
            <RevealText>
              <h2 className="features-title">THE<br/>ARSENAL</h2>
            </RevealText>
          </div>

          <div className="features-list">
            {[
              { 
                title: 'REAL-TIME SIMULATION', 
                desc: 'Watch traffic patterns, Policies, Weather, Digital Population emerge and evolve in real-time. Every vehicle, every human, every intersection, every decision.',
                tag: 'CORE'
              },
              { 
                title: 'AI PREDICTION ENGINE', 
                desc: 'Our neural networks have learned from millions of scenarios. They know what happens next.',
                tag: 'AI'
              },
              { 
                title: 'MULTI-SCENARIO TESTING', 
                desc: 'Run hundreds of what-if scenarios simultaneously. Compare outcomes side by side.',
                tag: 'ANALYSIS'
              },
              { 
                title: 'IMPACT VISUALIZATION', 
                desc: 'See the ripple effects of every change. Understand dependencies and bottlenecks.',
                tag: 'VISUAL'
              },
              { 
                title: 'EXPORT & DEPLOY', 
                desc: 'From simulation to implementation in one click. Export reports, data, and recommendations.',
                tag: 'OUTPUT'
              },
              { 
                title: 'COLLABORATIVE WORKSPACE', 
                desc: 'Work together in real-time. Share scenarios, annotations, and insights with your team.',
                tag: 'TEAM'
              }
            ].map((feature, i) => (
              <motion.div 
                key={i}
                className="feature-row"
                initial={{ opacity: 0, x: -80 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true, margin: "-50px" }}
                transition={{ 
                  delay: i * 0.08,
                  duration: 0.6,
                  ease: [0.16, 1, 0.3, 1]
                }}
                whileHover={{ x: 20 }}
              >
                <motion.span 
                  className="feature-index"
                  initial={{ scale: 0, rotate: -180 }}
                  whileInView={{ scale: 1, rotate: 0 }}
                  viewport={{ once: true }}
                  transition={{ 
                    delay: i * 0.08 + 0.2,
                    type: "spring",
                    stiffness: 200
                  }}
                >
                  {String(i + 1).padStart(2, '0')}
                </motion.span>
                <div className="feature-content">
                  <div className="feature-title-row">
                    <motion.h3
                      initial={{ opacity: 0, y: 20 }}
                      whileInView={{ opacity: 1, y: 0 }}
                      viewport={{ once: true }}
                      transition={{ delay: i * 0.08 + 0.15 }}
                    >
                      {feature.title}
                    </motion.h3>
                    <motion.span 
                      className="feature-tag"
                      initial={{ opacity: 0, scale: 0.5 }}
                      whileInView={{ opacity: 1, scale: 1 }}
                      viewport={{ once: true }}
                      transition={{ delay: i * 0.08 + 0.25 }}
                    >
                      {feature.tag}
                    </motion.span>
                  </div>
                  <motion.p
                    initial={{ opacity: 0 }}
                    whileInView={{ opacity: 1 }}
                    viewport={{ once: true }}
                    transition={{ delay: i * 0.08 + 0.3 }}
                  >
                    {feature.desc}
                  </motion.p>
                </div>
                <motion.span 
                  className="feature-arrow"
                  initial={{ opacity: 0, x: -20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: i * 0.08 + 0.35 }}
                  whileHover={{ x: 10 }}
                >
                  →
                </motion.span>
              </motion.div>
            ))}
          </div>
        </section>

        {/* ============================================ */}
        {/* QUOTE SECTION - EPIC ANIMATED */}
        {/* ============================================ */}
        <section className="quote-section">
          <div className="quote-container">
            {/* Animated background lines */}
            <div className="quote-bg-lines">
              {[...Array(5)].map((_, i) => (
                <motion.div
                  key={i}
                  className="quote-line-bg"
                  initial={{ scaleX: 0 }}
                  whileInView={{ scaleX: 1 }}
                  viewport={{ once: true }}
                  transition={{ 
                    duration: 1.5, 
                    delay: i * 0.1,
                    ease: [0.16, 1, 0.3, 1]
                  }}
                />
              ))}
            </div>
            
            {/* Opening quote mark */}
            <motion.span
              className="quote-mark quote-mark-open"
              initial={{ opacity: 0, scale: 0, rotate: -90 }}
              whileInView={{ opacity: 0.15, scale: 1, rotate: 0 }}
              viewport={{ once: true }}
              transition={{ 
                duration: 0.8,
                type: "spring",
                stiffness: 100
              }}
            >
              "
            </motion.span>
            
            <motion.blockquote 
              className="big-quote"
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: true }}
            >
              {/* First line with word-by-word reveal */}
              <div className="quote-line-wrapper">
                {"It doesn't matter where you start,".split(' ').map((word, i) => (
                  <motion.span
                    key={i}
                    className="quote-word"
                    initial={{ 
                      opacity: 0, 
                      y: 60,
                      filter: 'blur(10px)'
                    }}
                    whileInView={{ 
                      opacity: 1, 
                      y: 0,
                      filter: 'blur(0px)'
                    }}
                    viewport={{ once: true, margin: "-100px" }}
                    transition={{ 
                      duration: 0.5,
                      delay: 0.2 + i * 0.06,
                      ease: [0.16, 1, 0.3, 1]
                    }}
                  >
                    {word}
                  </motion.span>
                ))}
              </div>
              
              {/* Second line - the highlight with dramatic entrance */}
              <div className="quote-line-wrapper quote-highlight-wrapper">
                {"it's how you progress from there.".split(' ').map((word, i) => (
                  <motion.span
                    key={i}
                    className="quote-word quote-highlight"
                    initial={{ 
                      opacity: 0, 
                      y: 60,
                      filter: 'blur(10px)'
                    }}
                    whileInView={{ 
                      opacity: 1, 
                      y: 0,
                      filter: 'blur(0px)'
                    }}
                    viewport={{ once: true, margin: "-100px" }}
                    transition={{ 
                      duration: 0.5,
                      delay: 0.6 + i * 0.07,
                      ease: [0.16, 1, 0.3, 1]
                    }}
                  >
                    {word}
                  </motion.span>
                ))}
              </div>
            </motion.blockquote>
            
            {/* Closing quote mark */}
            <motion.span
              className="quote-mark quote-mark-close"
              initial={{ opacity: 0, scale: 0, rotate: 90 }}
              whileInView={{ opacity: 0.15, scale: 1, rotate: 0 }}
              viewport={{ once: true }}
              transition={{ 
                duration: 0.8,
                delay: 0.8,
                type: "spring",
                stiffness: 100
              }}
            >
              "
            </motion.span>
            
            {/* Animated underline */}
            <motion.div
              className="quote-underline"
              initial={{ scaleX: 0 }}
              whileInView={{ scaleX: 1 }}
              viewport={{ once: true }}
              transition={{ 
                duration: 1,
                delay: 1.2,
                ease: [0.16, 1, 0.3, 1]
              }}
            />
            
            <motion.cite 
              className="quote-cite"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.6, delay: 1.2 }}
            >
              <motion.span
                initial={{ width: 0 }}
                whileInView={{ width: 40 }}
                viewport={{ once: true }}
                transition={{ duration: 0.4, delay: 1.4 }}
                className="cite-line"
              />
              OVERHAUL PHILOSOPHY
            </motion.cite>
          </div>
        </section>

        {/* ============================================ */}
        {/* TIMELINE SECTION */}
        {/* ============================================ */}
        <section className="timeline" id="journey">
          <div className="timeline-header">
            <RevealText>
              <span className="section-label">MILESTONES</span>
            </RevealText>
            <RevealText>
              <h2 className="timeline-title">OUR<br/>JOURNEY</h2>
            </RevealText>
          </div>
          
          <div className="timeline-items">
            {[
              { year: 'Q1 2025', title: 'FOUNDED', desc: 'Started with a vision to revolutionize World simulation' },
              { year: 'Q2 2025', title: 'RESEARCH ', desc: 'First phase of research and data analysis initiated' },
              { year: 'Q3 2025', title: 'DESIGNING & DEVELOPMENT', desc: 'Designed the custom simulation engine and Devoloped in house from Scratch' },
              { year: 'Q4 2025', title: 'AI INTEGRATION', desc: 'Integrating the in house Neural network prediction engine' },
            ].map((item, i) => (
              <motion.div 
                key={i}
                className="timeline-item"
                initial={{ opacity: 0, y: 50 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: "-50px" }}
                transition={{ 
                  duration: 0.6, 
                  delay: i * 0.15,
                  ease: [0.16, 1, 0.3, 1]
                }}
              >
                <motion.span 
                  className="timeline-year"
                  initial={{ scale: 0.5, opacity: 0 }}
                  whileInView={{ scale: 1, opacity: 1 }}
                  viewport={{ once: true }}
                  transition={{ 
                    duration: 0.5, 
                    delay: i * 0.15 + 0.2,
                    type: "spring",
                    stiffness: 200
                  }}
                >
                  {item.year}
                </motion.span>
                <div className="timeline-content">
                  <motion.h3
                    initial={{ opacity: 0, x: -20 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.4, delay: i * 0.15 + 0.3 }}
                  >
                    {item.title}
                  </motion.h3>
                  <motion.p
                    initial={{ opacity: 0 }}
                    whileInView={{ opacity: 1 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.4, delay: i * 0.15 + 0.4 }}
                  >
                    {item.desc}
                  </motion.p>
                </div>
              </motion.div>
            ))}
          </div>
        </section>

        {/* ============================================ */}
        {/* CTA SECTION */}
        {/* ============================================ */}
        <section className="cta" id="cta-section">
          <motion.div 
            className="cta-content"
            initial={{ opacity: 0, scale: 0.9 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
          >
            <motion.span 
              className="cta-label"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.2 }}
            >
              READY TO START?
            </motion.span>
            <motion.h2 
              className="cta-title"
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.3, duration: 0.6 }}
            >
              LET'S BUILD<br/>
              <span className="text-outline">THE FUTURE</span>
            </motion.h2>
            <motion.p 
              className="cta-desc"
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: true }}
              transition={{ delay: 0.4 }}
            >
            Join the Next Big Thing in World of Artifical Intelligence. <br/>
            Overhaul today and transform your vision into reality.
            </motion.p>
            <motion.div 
              className="cta-buttons"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: 0.5, duration: 0.5 }}
            >
              <div onClick={() => setShowJoinForm(!showJoinForm)}>
                <MagneticButton className={`btn-brutal btn-large ${showJoinForm ? 'active' : ''}`}>
                  {showJoinForm ? 'CLOSE ×' : 'JOIN US →'}
                </MagneticButton>
              </div>
              <Link to="/support">
                <MagneticButton className="btn-outline">
                  SUPPORT US
                </MagneticButton>
              </Link>
            </motion.div>
            
            {/* Join Form Dropdown */}
            <JoinUsForm isOpen={showJoinForm} onClose={() => setShowJoinForm(false)} />
          </motion.div>
        </section>

        {/* ============================================ */}
        {/* FOOTER */}
        {/* ============================================ */}
        <footer className="footer">
          <div className="footer-main">
            <div className="footer-brand">
              <div className="footer-logo">OVERHAUL™</div>
              <p className="footer-tagline">Simulating tomorrow, today.</p>
            </div>
            
            <div className="footer-links-grid">
              <div className="footer-col">
                <h4>PLATFORM</h4>
                <a href="#features">Features</a>
                <Link to="/support">Join The Revolution</Link>
                <a href="#journey">Roadmap</a>
                <a href="#">Changelog</a>
              </div>
              <div className="footer-col">
                <h4>COMPANY</h4>
                <a href="#about">About</a>
                <a href="#">Careers</a>
                <a href="#">Press</a>
                <Link to="/contact">Contact</Link>
              </div>
              <div className="footer-col">
                <h4>LEGAL</h4>
                <Link to="/privacy">Privacy Policy</Link>
                <Link to="/terms">Terms & Conditions</Link>
                <Link to="/refunds">Cancellation & Refunds</Link>
                <Link to="/shipping">Shipping</Link>
              </div>
              <div className="footer-col">
                <h4>FOLLOW</h4>
                <a href="#">Twitter</a>
                <a href="#">LinkedIn</a>
                <a href="#">GitHub</a>
                <a href="#">Discord</a>
              </div>
            </div>
          </div>
          
          <div className="footer-bottom">
            <span>© 2025 OVERHAUL. ALL RIGHTS RESERVED.</span>
            <div className="footer-legal">
              <Link to="/privacy">Privacy Policy</Link>
              <Link to="/terms">Terms of Service</Link>
            </div>
          </div>
        </footer>
      </div>
    </>
  )
}

export default App
