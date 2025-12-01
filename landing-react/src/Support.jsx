import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useNavigate, Link } from 'react-router-dom'
import './App.css'

// ============================================
// OV LOADER
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
// FLOATING PARTICLES
// ============================================
function FloatingParticles() {
  const particles = Array.from({ length: 30 }, (_, i) => ({
    id: i,
    size: Math.random() * 4 + 2,
    x: Math.random() * 100,
    y: Math.random() * 100,
    duration: Math.random() * 20 + 15,
    delay: Math.random() * 5
  }))

  return (
    <div className="floating-particles">
      {particles.map(p => (
        <motion.div
          key={p.id}
          className="particle"
          style={{
            width: p.size,
            height: p.size,
            left: `${p.x}%`,
            top: `${p.y}%`,
          }}
          animate={{
            y: [0, -30, 0],
            x: [0, Math.random() * 20 - 10, 0],
            opacity: [0.2, 0.6, 0.2],
            scale: [1, 1.2, 1]
          }}
          transition={{
            duration: p.duration,
            delay: p.delay,
            repeat: Infinity,
            ease: "easeInOut"
          }}
        />
      ))}
    </div>
  )
}

// ============================================
// QUICK AMOUNT BUTTON
// ============================================
function QuickAmountBtn({ amount, selected, onClick }) {
  return (
    <motion.button
      className={`quick-amount-btn ${selected ? 'selected' : ''}`}
      onClick={() => onClick(amount)}
      whileHover={{ scale: 1.05, y: -2 }}
      whileTap={{ scale: 0.98 }}
      transition={{ type: "spring", stiffness: 400, damping: 25 }}
    >
      ₹{amount.toLocaleString()}
    </motion.button>
  )
}

// ============================================
// SUPPORT PAGE
// ============================================
function Support() {
  const [loading, setLoading] = useState(true)
  const [exiting, setExiting] = useState(false)
  const [hovering, setHovering] = useState(false)
  const [amount, setAmount] = useState('')
  const [paymentStatus, setPaymentStatus] = useState(null)
  const [isFocused, setIsFocused] = useState(false)
  const navigate = useNavigate()
  
  const cursorRef = useRef(null)
  const mousePos = useRef({ x: 0, y: 0 })
  const rafId = useRef(null)
  const inputRef = useRef(null)

  const quickAmounts = [100, 500, 1000, 2500, 5000]
  const minAmount = 10

  // Set page title
  useEffect(() => {
    document.title = 'OVERHAUL | Support Us'
  }, [])

  // Load Razorpay script
  useEffect(() => {
    const script = document.createElement('script')
    script.src = 'https://checkout.razorpay.com/v1/checkout.js'
    script.async = true
    document.body.appendChild(script)
    return () => {
      if (document.body.contains(script)) {
        document.body.removeChild(script)
      }
    }
  }, [])

  const handleBackHome = (e) => {
    e.preventDefault()
    setExiting(true)
  }

  const handleExitComplete = () => {
    navigate('/', { state: { skipLoader: true } })
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

  useEffect(() => {
    const handleHover = () => {
      const hoverable = document.querySelectorAll('a, button, .magnetic-btn, .quick-amount-btn, input')
      hoverable.forEach(el => {
        el.addEventListener('mouseenter', () => setHovering(true))
        el.addEventListener('mouseleave', () => setHovering(false))
      })
    }
    if (!loading && !exiting) handleHover()
  }, [loading, exiting])

  const handlePayment = () => {
    const numAmount = parseInt(amount)
    
    if (!numAmount || numAmount < minAmount) {
      return
    }

    if (!window.Razorpay) {
      alert('Payment system is loading. Please try again.')
      return
    }

    const options = {
      key: 'rzp_test_RmHYwuBZOYY3QD',
      amount: numAmount * 100,
      currency: 'INR',
      name: 'OVERHAUL',
      description: 'Support the Future of AI & Simulations',
      image: '/favicon.png',
      handler: function (response) {
        setPaymentStatus('success')
        setAmount('')
      },
      prefill: {
        name: '',
        email: '',
        contact: ''
      },
      notes: {
        purpose: 'Support OVERHAUL Development'
      },
      theme: {
        color: '#CCFF00'
      },
      modal: {
        ondismiss: function() {
          console.log('Payment modal closed')
        }
      }
    }

    const rzp = new window.Razorpay(options)
    
    rzp.on('payment.failed', function (response) {
      setPaymentStatus('failed')
    })

    rzp.open()
  }

  const handleQuickAmount = (amt) => {
    setAmount(amt.toString())
    if (inputRef.current) {
      inputRef.current.focus()
    }
  }

  const handleAmountChange = (e) => {
    const val = e.target.value.replace(/[^0-9]/g, '')
    setAmount(val)
  }

  const isValidAmount = amount && parseInt(amount) >= minAmount

  return (
    <>
      {/* Custom Cursor */}
      <div 
        ref={cursorRef}
        className={`cursor ${hovering ? 'hovering' : ''}`}
      />

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
            className="support-page-new"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.5 }}
          >
            {/* Floating Particles */}
            <FloatingParticles />

            {/* Gradient Orbs */}
            <div className="gradient-orbs">
              <motion.div 
                className="gradient-orb orb-1"
                animate={{ 
                  x: [0, 50, 0],
                  y: [0, -30, 0],
                  scale: [1, 1.1, 1]
                }}
                transition={{ duration: 15, repeat: Infinity, ease: "easeInOut" }}
              />
              <motion.div 
                className="gradient-orb orb-2"
                animate={{ 
                  x: [0, -40, 0],
                  y: [0, 40, 0],
                  scale: [1, 1.15, 1]
                }}
                transition={{ duration: 18, repeat: Infinity, ease: "easeInOut" }}
              />
              <motion.div 
                className="gradient-orb orb-3"
                animate={{ 
                  x: [0, 30, 0],
                  y: [0, 50, 0],
                  scale: [1, 1.2, 1]
                }}
                transition={{ duration: 20, repeat: Infinity, ease: "easeInOut" }}
              />
            </div>

            {/* Navigation */}
            <motion.nav 
              className="support-nav"
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2, duration: 0.6 }}
            >
              <a href="/" onClick={handleBackHome} className="nav-logo">OVERHAUL™</a>
              <a href="/" onClick={handleBackHome} className="back-btn-support">
                ← BACK
              </a>
            </motion.nav>

            {/* Main Content */}
            <div className="support-main">
              {/* Left Side - Message */}
              <motion.div 
                className="support-message"
                initial={{ opacity: 0, x: -50 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.3, duration: 0.8, ease: [0.25, 0.1, 0.25, 1] }}
              >
                <motion.span 
                  className="support-label"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.4, duration: 0.5 }}
                >
                  HELP US BUILD THE FUTURE
                </motion.span>
                
                <motion.h1 
                  className="support-headline"
                  initial={{ opacity: 0, y: 30 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.5, duration: 0.7 }}
                >
                  Support the next<br/>
                  <span className="text-gradient">age of AI</span>
                </motion.h1>
                
                <motion.p 
                  className="support-description"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.6, duration: 0.6 }}
                >
                  We're building intelligent simulations that will transform how cities move, 
                  how systems think, and how the world operates. Your support—no matter the 
                  size—fuels this revolution.
                </motion.p>

                <motion.div 
                  className="support-features"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.7, duration: 0.6 }}
                >
                  <div className="support-feature">
                    <span className="feature-icon">◈</span>
                    <span>AI-Powered Urban Intelligence</span>
                  </div>
                  <div className="support-feature">
                    <span className="feature-icon">◈</span>
                    <span>Real-Time Traffic Simulations</span>
                  </div>
                  <div className="support-feature">
                    <span className="feature-icon">◈</span>
                    <span>Open Research & Development</span>
                  </div>
                </motion.div>
              </motion.div>

              {/* Right Side - Payment Card */}
              <motion.div 
                className="support-card"
                initial={{ opacity: 0, x: 50, rotateY: -10 }}
                animate={{ opacity: 1, x: 0, rotateY: 0 }}
                transition={{ delay: 0.4, duration: 0.8, ease: [0.25, 0.1, 0.25, 1] }}
              >
                {/* Success Message */}
                <AnimatePresence mode="wait">
                  {paymentStatus === 'success' ? (
                    <motion.div 
                      className="success-state"
                      initial={{ opacity: 0, scale: 0.9 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.9 }}
                      transition={{ duration: 0.4 }}
                    >
                      <motion.div 
                        className="success-check"
                        initial={{ scale: 0 }}
                        animate={{ scale: 1 }}
                        transition={{ delay: 0.2, type: "spring", stiffness: 200 }}
                      >
                        ✓
                      </motion.div>
                      <h3>Thank You!</h3>
                      <p>Your support means the world to us. Together, we're building something extraordinary.</p>
                      <motion.button 
                        className="support-again-btn"
                        onClick={() => setPaymentStatus(null)}
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                      >
                        Support Again
                      </motion.button>
                    </motion.div>
                  ) : paymentStatus === 'failed' ? (
                    <motion.div 
                      className="error-state"
                      initial={{ opacity: 0, scale: 0.9 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.9 }}
                      transition={{ duration: 0.4 }}
                    >
                      <div className="error-icon">✕</div>
                      <h3>Payment Failed</h3>
                      <p>Something went wrong. Please try again.</p>
                      <motion.button 
                        className="try-again-btn"
                        onClick={() => setPaymentStatus(null)}
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                      >
                        Try Again
                      </motion.button>
                    </motion.div>
                  ) : (
                    <motion.div 
                      className="payment-form"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                    >
                      <h2 className="card-title">Enter any amount</h2>
                      <p className="card-subtitle">Minimum ₹10 • Every bit helps</p>

                      {/* Amount Input */}
                      <div className={`amount-input-wrapper ${isFocused ? 'focused' : ''} ${isValidAmount ? 'valid' : ''}`}>
                        <span className="rupee-symbol">₹</span>
                        <input 
                          ref={inputRef}
                          type="text"
                          inputMode="numeric"
                          pattern="[0-9]*"
                          placeholder="0"
                          value={amount}
                          onChange={handleAmountChange}
                          onFocus={() => setIsFocused(true)}
                          onBlur={() => setIsFocused(false)}
                          className="amount-input"
                        />
                      </div>

                      {/* Quick Amount Buttons */}
                      <div className="quick-amounts">
                        {quickAmounts.map(amt => (
                          <QuickAmountBtn 
                            key={amt}
                            amount={amt}
                            selected={amount === amt.toString()}
                            onClick={handleQuickAmount}
                          />
                        ))}
                      </div>

                      {/* Pay Button */}
                      <motion.button 
                        className={`pay-btn ${isValidAmount ? 'active' : ''}`}
                        onClick={handlePayment}
                        disabled={!isValidAmount}
                        whileHover={isValidAmount ? { scale: 1.02, y: -2 } : {}}
                        whileTap={isValidAmount ? { scale: 0.98 } : {}}
                        transition={{ type: "spring", stiffness: 400, damping: 25 }}
                      >
                        {isValidAmount ? (
                          <>Support with ₹{parseInt(amount).toLocaleString()}</>
                        ) : (
                          <>Enter amount to continue</>
                        )}
                      </motion.button>

                      {/* Security Note */}
                      <div className="security-note">
                        <span className="lock-icon">🔒</span>
                        <span>Secure payment via Razorpay • UPI, Cards & More</span>
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </motion.div>
            </div>

            {/* Bottom Quote */}
            <motion.div 
              className="support-quote"
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.9, duration: 0.6 }}
            >
              <p>"The future belongs to those who build it."</p>
            </motion.div>

            {/* Footer */}
            <motion.footer 
              className="support-footer"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 1, duration: 0.5 }}
            >
              <div className="footer-links">
                <Link to="/privacy">Privacy</Link>
                <Link to="/terms">Terms</Link>
                <Link to="/refunds">Refunds</Link>
              </div>
              <span>© 2025 OVERHAUL</span>
            </motion.footer>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
}

export default Support
