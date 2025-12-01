import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useNavigate, useLocation, Link } from 'react-router-dom'
import './App.css'

// ============================================
// OV LOADER (Same style as Contact page)
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
// POLICY CONTENT DATA
// ============================================
const policyContent = {
  privacy: {
    title: 'PRIVACY POLICY',
    subtitle: 'Your privacy matters to us',
    lastUpdated: 'December 1, 2025',
    sections: [
      {
        heading: 'Information We Collect',
        content: `We collect information you provide directly to us, such as when you create an account, use our services, or contact us for support. This may include:
        
• Name and email address
• Usage data and analytics
• Device and browser information
• Location data (with your consent)
• Communication preferences`
      },
      {
        heading: 'How We Use Your Information',
        content: `We use the information we collect to:

• Provide, maintain, and improve our services
• Process transactions and send related information
• Send technical notices and support messages
• Respond to your comments and questions
• Analyze usage patterns to improve user experience
• Protect against fraudulent or illegal activity`
      },
      {
        heading: 'Data Security',
        content: `We implement appropriate technical and organizational measures to protect your personal information against unauthorized access, alteration, disclosure, or destruction. This includes encryption, secure servers, and regular security audits.`
      },
      {
        heading: 'Your Rights',
        content: `You have the right to:

• Access your personal data
• Correct inaccurate data
• Request deletion of your data
• Opt-out of marketing communications
• Export your data in a portable format`
      },
      {
        heading: 'Contact Us',
        content: `For any privacy-related questions or concerns, please contact us at:

Email: founders@overhaul.co.in`
      }
    ]
  },
  terms: {
    title: 'TERMS & CONDITIONS',
    subtitle: 'Please read these terms carefully',
    lastUpdated: 'December 1, 2025',
    sections: [
      {
        heading: 'Acceptance of Terms',
        content: `By accessing or using OVERHAUL's services, you agree to be bound by these Terms and Conditions. If you do not agree to these terms, please do not use our services.`
      },
      {
        heading: 'Use of Services',
        content: `You agree to use our services only for lawful purposes and in accordance with these Terms. You are responsible for:

• Maintaining the confidentiality of your account
• All activities that occur under your account
• Ensuring your use complies with applicable laws
• Not interfering with the proper functioning of the service`
      },
      {
        heading: 'Intellectual Property',
        content: `All content, features, and functionality of OVERHAUL are owned by us and are protected by international copyright, trademark, and other intellectual property laws. You may not reproduce, distribute, or create derivative works without our express permission.`
      },
      {
        heading: 'Limitation of Liability',
        content: `OVERHAUL shall not be liable for any indirect, incidental, special, consequential, or punitive damages resulting from your use of or inability to use the service. Our total liability shall not exceed the amount paid by you in the past 12 months.`
      },
      {
        heading: 'Modifications',
        content: `We reserve the right to modify these terms at any time. We will notify users of significant changes via email or through our platform. Continued use after changes constitutes acceptance of the new terms.`
      },
      {
        heading: 'Governing Law',
        content: `These Terms shall be governed by and construed in accordance with the laws of India, without regard to its conflict of law provisions.`
      }
    ]
  },
  refunds: {
    title: 'CANCELLATION & REFUNDS',
    subtitle: 'Our refund policy',
    lastUpdated: 'December 1, 2025',
    sections: [
      {
        heading: 'Cancellation Policy',
        content: `You may cancel your subscription at any time through your account settings or by contacting our support team. Upon cancellation:

• Your subscription will remain active until the end of the current billing period
• You will not be charged for subsequent billing periods
• Access to premium features will end when your current period expires`
      },
      {
        heading: 'Refund Eligibility',
        content: `We offer refunds under the following conditions:

• Full refund within 7 days of initial purchase if service is unused
• Prorated refund for annual subscriptions within 30 days
• Full refund if service is unavailable for more than 72 consecutive hours
• Case-by-case consideration for exceptional circumstances`
      },
      {
        heading: 'Non-Refundable Items',
        content: `The following are non-refundable:

• Consumed API credits or usage
• Custom development or consultation services
• Subscriptions cancelled after the refund period
• Accounts terminated due to policy violations`
      },
      {
        heading: 'Refund Process',
        content: `To request a refund:

1. Contact us at founders@overhaul.co.in
2. Provide your account email and reason for refund
3. We will review your request within 3-5 business days
4. Approved refunds will be processed within 7-10 business days
5. Refunds will be credited to the original payment method`
      },
      {
        heading: 'Disputes',
        content: `If you have any disputes regarding charges or refunds, please contact our support team before initiating a chargeback with your payment provider. We are committed to resolving issues amicably.`
      }
    ]
  },
  shipping: {
    title: 'SHIPPING POLICY',
    subtitle: 'Digital delivery information',
    lastUpdated: 'December 1, 2025',
    sections: [
      {
        heading: 'Digital Products',
        content: `OVERHAUL is a digital platform providing software-as-a-service (SaaS) solutions. As such:

• All products and services are delivered digitally
• No physical shipping is required
• Access is granted immediately upon successful payment
• Login credentials are sent to your registered email`
      },
      {
        heading: 'Access Delivery',
        content: `Upon successful subscription or purchase:

• You will receive a confirmation email within minutes
• Access to the platform is immediate
• API keys (if applicable) are generated instantly
• Documentation and guides are available online`
      },
      {
        heading: 'Delivery Issues',
        content: `If you experience any issues accessing your purchase:

• Check your spam/junk folder for confirmation emails
• Ensure you're using the correct login credentials
• Clear your browser cache and try again
• Contact support at founders@overhaul.co.in`
      },
      {
        heading: 'International Access',
        content: `OVERHAUL services are available globally. However:

• Some features may vary by region due to regulatory requirements
• Payment processing may vary based on your location
• Support response times may vary based on time zones`
      }
    ]
  }
}

// ============================================
// POLICY PAGE COMPONENT
// ============================================
function PolicyPage({ type }) {
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

  const policy = policyContent[type]
  const pageTitle = type === 'privacy' ? 'Privacy Policy' 
    : type === 'terms' ? 'Terms & Conditions'
    : type === 'refunds' ? 'Cancellation & Refunds'
    : 'Shipping Policy'

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
    document.title = `OVERHAUL | ${pageTitle}`
  }, [pageTitle])

  const handleBackHome = (e) => {
    e.preventDefault()
    setExiting(true)
    setPendingNavigation('/')
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
            className="policy-page"
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
            <div className="policy-content">
              <motion.div 
                className="policy-header"
                initial={{ opacity: 0, y: 50 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1, duration: 0.6 }}
              >
                <span className="contact-label">LEGAL</span>
                <h1 className="policy-title">{policy.title}</h1>
                <p className="policy-subtitle">{policy.subtitle}</p>
                <p className="policy-date">Last updated: {policy.lastUpdated}</p>
              </motion.div>

              <motion.div 
                className="policy-sections"
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2, duration: 0.6 }}
              >
                {policy.sections.map((section, index) => (
                  <div key={index} className="policy-section">
                    <h2 className="policy-section-heading">{section.heading}</h2>
                    <div className="policy-section-content">
                      {section.content.split('\n').map((paragraph, pIndex) => (
                        <p key={pIndex}>{paragraph}</p>
                      ))}
                    </div>
                  </div>
                ))}
              </motion.div>

              {/* Quick Links */}
              <motion.div 
                className="policy-links"
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3, duration: 0.6 }}
              >
                <h3>Other Policies</h3>
                <div className="policy-links-grid">
                  {type !== 'privacy' && <Link to="/privacy">Privacy Policy</Link>}
                  {type !== 'terms' && <Link to="/terms">Terms & Conditions</Link>}
                  {type !== 'refunds' && <Link to="/refunds">Cancellation & Refunds</Link>}
                  {type !== 'shipping' && <Link to="/shipping">Shipping Policy</Link>}
                  <Link to="/contact">Contact Us</Link>
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

// Export individual page components
export function PrivacyPolicy() {
  return <PolicyPage type="privacy" />
}

export function TermsConditions() {
  return <PolicyPage type="terms" />
}

export function RefundsPolicy() {
  return <PolicyPage type="refunds" />
}

export function ShippingPolicy() {
  return <PolicyPage type="shipping" />
}
