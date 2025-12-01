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
// SUPPORT TIERS
// ============================================
const supportTiers = [
  {
    id: 'pioneer',
    name: 'PIONEER',
    amount: 499,
    amountDisplay: '₹499',
    description: 'Be among the first to shape the future',
    perks: [
      'Early access to new features',
      'Name in credits',
      'Exclusive Discord role'
    ],
    popular: false
  },
  {
    id: 'visionary',
    name: 'VISIONARY',
    amount: 1999,
    amountDisplay: '₹1,999',
    description: 'For those who see beyond tomorrow',
    perks: [
      'Everything in Pioneer',
      'Priority feature requests',
      'Monthly insider updates',
      'Beta tester access'
    ],
    popular: true
  },
  {
    id: 'revolutionary',
    name: 'REVOLUTIONARY',
    amount: 4999,
    amountDisplay: '₹4,999',
    description: 'Lead the charge into the future',
    perks: [
      'Everything in Visionary',
      '1-on-1 call with founders',
      'Custom feature consideration',
      'Lifetime early access',
      'Exclusive merchandise'
    ],
    popular: false
  }
]

// ============================================
// SUPPORT PAGE
// ============================================
function Support() {
  const [loading, setLoading] = useState(true)
  const [exiting, setExiting] = useState(false)
  const [hovering, setHovering] = useState(false)
  const [selectedTier, setSelectedTier] = useState(null)
  const [customAmount, setCustomAmount] = useState('')
  const [paymentStatus, setPaymentStatus] = useState(null)
  const navigate = useNavigate()
  
  const cursorRef = useRef(null)
  const mousePos = useRef({ x: 0, y: 0 })
  const rafId = useRef(null)

  // Set page title
  useEffect(() => {
    document.title = 'OVERHAUL | Join The Revolution'
  }, [])

  // Load Razorpay script
  useEffect(() => {
    const script = document.createElement('script')
    script.src = 'https://checkout.razorpay.com/v1/checkout.js'
    script.async = true
    document.body.appendChild(script)
    return () => {
      document.body.removeChild(script)
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
      const hoverable = document.querySelectorAll('a, button, .magnetic-btn, .support-tier')
      hoverable.forEach(el => {
        el.addEventListener('mouseenter', () => setHovering(true))
        el.addEventListener('mouseleave', () => setHovering(false))
      })
    }
    if (!loading && !exiting) handleHover()
  }, [loading, exiting])

  const handlePayment = (amount, tierName = 'Custom') => {
    if (!window.Razorpay) {
      alert('Payment system is loading. Please try again.')
      return
    }

    const options = {
      key: 'rzp_live_fquBPILlNv57Vz', // Replace with your Razorpay Key ID
      amount: amount * 100, // Amount in paise
      currency: 'INR',
      name: 'OVERHAUL',
      description: `${tierName} - Join The Revolution`,
      image: '/favicon.png',
      handler: function (response) {
        setPaymentStatus('success')
        console.log('Payment successful:', response)
        // You can send this to your backend for verification
      },
      prefill: {
        name: '',
        email: '',
        contact: ''
      },
      notes: {
        tier: tierName,
        purpose: 'Support OVERHAUL'
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
      console.error('Payment failed:', response.error)
    })

    rzp.open()
  }

  const handleTierSelect = (tier) => {
    setSelectedTier(tier.id)
    handlePayment(tier.amount, tier.name)
  }

  const handleCustomPayment = () => {
    const amount = parseInt(customAmount)
    if (amount && amount >= 100) {
      handlePayment(amount, 'Custom Contribution')
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
            className="support-page"
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
            <div className="support-content">
              {/* Header */}
              <motion.div 
                className="support-header"
                initial={{ opacity: 0, y: 50 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1, duration: 0.6 }}
              >
                <span className="contact-label">BE PART OF SOMETHING BIGGER</span>
                <h1 className="support-title">
                  JOIN THE<br/>
                  <span className="text-lime">REVOLUTION</span>
                </h1>
                <p className="support-subtitle">
                  We're building the future of urban intelligence. Your involvement accelerates 
                  our mission to transform how cities think, breathe, and move.
                </p>
              </motion.div>

              {/* Success/Error Message */}
              <AnimatePresence>
                {paymentStatus === 'success' && (
                  <motion.div 
                    className="payment-message success"
                    initial={{ opacity: 0, y: -20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                  >
                    <span className="payment-icon">✓</span>
                    <div>
                      <h3>Welcome to the Revolution!</h3>
                      <p>Thank you for joining us. You're now part of something extraordinary.</p>
                    </div>
                  </motion.div>
                )}
                {paymentStatus === 'failed' && (
                  <motion.div 
                    className="payment-message error"
                    initial={{ opacity: 0, y: -20 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0 }}
                  >
                    <span className="payment-icon">✕</span>
                    <div>
                      <h3>Payment Unsuccessful</h3>
                      <p>Something went wrong. Please try again or contact us.</p>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Tiers */}
              <motion.div 
                className="support-tiers"
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2, duration: 0.6 }}
              >
                {supportTiers.map((tier, index) => (
                  <motion.div 
                    key={tier.id}
                    className={`support-tier ${tier.popular ? 'popular' : ''}`}
                    initial={{ opacity: 0, y: 30 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2 + index * 0.1, duration: 0.5 }}
                    onClick={() => handleTierSelect(tier)}
                  >
                    {tier.popular && <div className="tier-badge">MOST POPULAR</div>}
                    <h3 className="tier-name">{tier.name}</h3>
                    <div className="tier-amount">{tier.amountDisplay}</div>
                    <p className="tier-description">{tier.description}</p>
                    <ul className="tier-perks">
                      {tier.perks.map((perk, i) => (
                        <li key={i}>
                          <span className="perk-check">✓</span>
                          {perk}
                        </li>
                      ))}
                    </ul>
                    <button className="tier-button">
                      JOIN AS {tier.name}
                    </button>
                  </motion.div>
                ))}
              </motion.div>

              {/* Custom Amount */}
              <motion.div 
                className="custom-amount"
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5, duration: 0.6 }}
              >
                <h3>OR CHOOSE YOUR OWN AMOUNT</h3>
                <div className="custom-input-group">
                  <span className="currency-symbol">₹</span>
                  <input 
                    type="number" 
                    placeholder="Enter amount (min ₹100)"
                    value={customAmount}
                    onChange={(e) => setCustomAmount(e.target.value)}
                    min="100"
                  />
                  <button 
                    className="custom-pay-btn"
                    onClick={handleCustomPayment}
                    disabled={!customAmount || parseInt(customAmount) < 100}
                  >
                    PROCEED
                  </button>
                </div>
              </motion.div>

              {/* Trust Badges */}
              <motion.div 
                className="trust-section"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.6, duration: 0.6 }}
              >
                <div className="trust-badges">
                  <div className="trust-badge">
                    <span className="trust-icon">🔒</span>
                    <span>Secure Payments via Razorpay</span>
                  </div>
                  <div className="trust-badge">
                    <span className="trust-icon">⚡</span>
                    <span>Instant Confirmation</span>
                  </div>
                  <div className="trust-badge">
                    <span className="trust-icon">💳</span>
                    <span>All Cards & UPI Accepted</span>
                  </div>
                </div>
              </motion.div>

              {/* FAQ */}
              <motion.div 
                className="support-faq"
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.7, duration: 0.6 }}
              >
                <h3>FREQUENTLY ASKED</h3>
                <div className="faq-grid">
                  <div className="faq-item">
                    <h4>What happens after I join?</h4>
                    <p>You'll receive an email confirmation with details about your perks and how to access exclusive features.</p>
                  </div>
                  <div className="faq-item">
                    <h4>Is this a subscription?</h4>
                    <p>No, this is a one-time contribution. You won't be charged again unless you choose to.</p>
                  </div>
                  <div className="faq-item">
                    <h4>Can I get a refund?</h4>
                    <p>Yes, we offer refunds within 7 days. See our <Link to="/refunds">refund policy</Link> for details.</p>
                  </div>
                  <div className="faq-item">
                    <h4>How will my contribution be used?</h4>
                    <p>Your contribution directly funds development, research, and infrastructure to build better urban AI.</p>
                  </div>
                </div>
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

export default Support
