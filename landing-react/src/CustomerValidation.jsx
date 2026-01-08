import { useEffect, useMemo, useRef, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import './App.css'
import { apiFetchJson } from './api/config'

// ============================================
// CUSTOM CURSOR
// ============================================
function CustomCursor() {
  const cursorRef = useRef(null)
  const [isHovering, setIsHovering] = useState(false)
  const mousePos = useRef({ x: 0, y: 0 })
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
          target.closest('a') || target.closest('button') ||
          target.closest('.vf-option') || target.closest('.vf-chip') ||
          target.closest('input') || target.closest('textarea') ||
          target.closest('select')) {
        setIsHovering(true)
      }
    }

    const handleMouseOut = (e) => {
      const target = e.target
      if (target.tagName === 'A' || target.tagName === 'BUTTON' || 
          target.closest('a') || target.closest('button') ||
          target.closest('.vf-option') || target.closest('.vf-chip') ||
          target.closest('input') || target.closest('textarea') ||
          target.closest('select')) {
        setIsHovering(false)
      }
    }

    document.addEventListener('mousemove', moveCursor, { passive: true })
    document.addEventListener('mouseover', handleMouseOver, { passive: true })
    document.addEventListener('mouseout', handleMouseOut, { passive: true })

    return () => {
      if (rafId.current) cancelAnimationFrame(rafId.current)
      document.removeEventListener('mousemove', moveCursor)
      document.removeEventListener('mouseover', handleMouseOver)
      document.removeEventListener('mouseout', handleMouseOut)
    }
  }, [])

  return <div ref={cursorRef} className={`cursor ${isHovering ? 'hovering' : ''}`} />
}

// ============================================
// OV LOADER
// ============================================
function OVLoader({ onComplete }) {
  const [phase, setPhase] = useState('zoomOut')

  useEffect(() => {
    const holdTimer = setTimeout(() => setPhase('hold'), 400)
    return () => clearTimeout(holdTimer)
  }, [])

  useEffect(() => {
    if (phase === 'hold') {
      const zoomInTimer = setTimeout(() => setPhase('zoomIn'), 250)
      return () => clearTimeout(zoomInTimer)
    }
  }, [phase])

  useEffect(() => {
    if (phase === 'zoomIn') {
      const completeTimer = setTimeout(() => onComplete(), 500)
      return () => clearTimeout(completeTimer)
    }
  }, [phase, onComplete])

  return (
    <motion.div
      className="loader-ln"
      initial={{ opacity: 0 }}
      animate={{ opacity: phase === 'zoomIn' ? 0 : 1 }}
      transition={{ duration: phase === 'zoomIn' ? 0.3 : 0.2, ease: [0.4, 0, 0.2, 1], delay: phase === 'zoomIn' ? 0.25 : 0 }}
    >
      <div className="loader-ln-content">
        <motion.div
          className="loader-ln-logo"
          initial={{ scale: 50, opacity: 0 }}
          animate={{ scale: phase === 'zoomOut' ? 1 : phase === 'hold' ? 1 : 50, opacity: 1 }}
          transition={{ duration: phase === 'zoomOut' ? 0.4 : phase === 'zoomIn' ? 0.5 : 0.1, ease: [0.4, 0, 0.2, 1] }}
        >
          <motion.span className="loader-ln-text-o" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.2, delay: 0.15 }}>O</motion.span>
          <motion.span className="loader-ln-text-v" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ duration: 0.2, delay: 0.2 }}>V</motion.span>
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
    const holdTimer = setTimeout(() => setPhase('hold'), 400)
    return () => clearTimeout(holdTimer)
  }, [])

  useEffect(() => {
    if (phase === 'hold') {
      const zoomInTimer = setTimeout(() => setPhase('zoomIn'), 200)
      return () => clearTimeout(zoomInTimer)
    }
  }, [phase])

  useEffect(() => {
    if (phase === 'zoomIn') {
      const completeTimer = setTimeout(() => onComplete(), 500)
      return () => clearTimeout(completeTimer)
    }
  }, [phase, onComplete])

  return (
    <motion.div
      className="loader-ln"
      initial={{ opacity: 0 }}
      animate={{ opacity: phase === 'zoomIn' ? 0 : 1 }}
      transition={{ duration: phase === 'zoomIn' ? 0.3 : 0.2, ease: [0.76, 0, 0.24, 1], delay: phase === 'zoomIn' ? 0.25 : 0 }}
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
// FLOATING PARTICLES
// ============================================
function FloatingParticles() {
  const particles = Array.from({ length: 25 }, (_, i) => ({
    id: i,
    size: Math.random() * 4 + 2,
    x: Math.random() * 100,
    y: Math.random() * 100,
    duration: Math.random() * 20 + 15,
    delay: Math.random() * 5
  }))

  return (
    <div className="vf-particles">
      {particles.map(p => (
        <motion.div
          key={p.id}
          className="vf-particle"
          style={{ width: p.size, height: p.size, left: `${p.x}%`, top: `${p.y}%` }}
          animate={{ y: [0, -30, 0], x: [0, Math.random() * 20 - 10, 0], opacity: [0.2, 0.5, 0.2] }}
          transition={{ duration: p.duration, delay: p.delay, repeat: Infinity, ease: "easeInOut" }}
        />
      ))}
    </div>
  )
}

// ============================================
// LIVE FEED ENTRY
// ============================================
function LiveFeedEntry({ entry, index }) {
  const timeAgo = (dateStr) => {
    if (!dateStr) return 'just now'
    const date = new Date(dateStr)
    const now = new Date()
    const diff = Math.floor((now - date) / 1000)
    if (diff < 60) return 'just now'
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`
    return `${Math.floor(diff / 86400)}d ago`
  }

  return (
    <motion.div
      className="vf-live-entry"
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.4, delay: index * 0.08 }}
    >
      <div className="vf-live-avatar">{(entry.name || 'A')[0].toUpperCase()}</div>
      <div className="vf-live-info">
        <div className="vf-live-name">{entry.name || 'Anonymous'}</div>
        <div className="vf-live-meta">
          {entry.display_email} • {timeAgo(entry.created_at)}
        </div>
      </div>
      <div className="vf-live-relevance">
        <span className="vf-relevance-dot" style={{ '--relevance': entry.problem_relevance || 5 }} />
        {entry.problem_relevance || 5}/5
      </div>
    </motion.div>
  )
}

// ============================================
// RELEVANCE SCALE
// ============================================
function RelevanceScale({ value, onChange }) {
  return (
    <div className="vf-relevance-scale">
      {[1, 2, 3, 4, 5].map(n => (
        <motion.button
          key={n}
          type="button"
          className={`vf-relevance-btn ${value === n ? 'active' : ''}`}
          onClick={() => onChange(n)}
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.95 }}
        >
          {n}
        </motion.button>
      ))}
    </div>
  )
}

// ============================================
// MULTI SELECT CHIPS
// ============================================
function MultiSelectChips({ options, selected, onChange }) {
  const toggle = (opt) => {
    if (selected.includes(opt)) {
      onChange(selected.filter(s => s !== opt))
    } else {
      onChange([...selected, opt])
    }
  }

  return (
    <div className="vf-chips">
      {options.map(opt => (
        <motion.button
          key={opt}
          type="button"
          className={`vf-chip ${selected.includes(opt) ? 'active' : ''}`}
          onClick={() => toggle(opt)}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          {opt}
          {selected.includes(opt) && <span className="vf-chip-check">✓</span>}
        </motion.button>
      ))}
    </div>
  )
}

// ============================================
// YES/NO/MAYBE SELECTOR
// ============================================
function OptionSelector({ options, value, onChange }) {
  return (
    <div className="vf-options">
      {options.map(opt => (
        <motion.button
          key={opt.value}
          type="button"
          className={`vf-option ${value === opt.value ? 'active' : ''}`}
          onClick={() => onChange(opt.value)}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          {opt.label}
        </motion.button>
      ))}
    </div>
  )
}

// ============================================
// MAIN COMPONENT
// ============================================
export default function CustomerValidation() {
  const location = useLocation()
  const navigate = useNavigate()
  const skipLoader = location.state?.skipLoader || false

  const [pageLoading, setPageLoading] = useState(!skipLoader)
  const [exiting, setExiting] = useState(false)
  const [pendingNavigation, setPendingNavigation] = useState(null)

  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')
  const [submittedMessage, setSubmittedMessage] = useState('')
  const [recentEntries, setRecentEntries] = useState([])
  const [totalCount, setTotalCount] = useState(0)
  const [currentStep, setCurrentStep] = useState(1)
  const totalSteps = 3

  // Form state
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [role, setRole] = useState('')
  const [organization, setOrganization] = useState('')
  const [problemRelevance, setProblemRelevance] = useState(5)
  const [hasExperience, setHasExperience] = useState(null)
  const [toolsShortcoming, setToolsShortcoming] = useState('')
  const [usageContexts, setUsageContexts] = useState([])
  const [learningToolValue, setLearningToolValue] = useState('')
  const [interestingAspect, setInterestingAspect] = useState('')
  const [suggestedImprovement, setSuggestedImprovement] = useState('')
  const [interestInTrying, setInterestInTrying] = useState('')

  const ROLE_OPTIONS = ['Student', 'Faculty / Researcher', 'Professional', 'Other']
  const CONTEXT_OPTIONS = ['Learning / education', 'Research', 'Policy or planning', 'Exploration / experimentation', 'Other']
  const ASPECT_OPTIONS = ['Simulations & what-if analysis', 'Natural language interaction', 'Understanding trade-offs', 'Visual maps & insights', 'Other']

  // Validation per step
  const canProceedStep1 = useMemo(() => {
    return name.trim() && email.trim() && role
  }, [name, email, role])

  const canProceedStep2 = useMemo(() => {
    return hasExperience !== null && usageContexts.length > 0 && learningToolValue
  }, [hasExperience, usageContexts, learningToolValue])

  const canSubmit = canProceedStep1 && canProceedStep2

  // Load entries and stats
  async function loadData() {
    try {
      const [entriesData, statsData] = await Promise.all([
        apiFetchJson('/validation/entries?page=1&page_size=8'),
        apiFetchJson('/validation/stats').catch(() => ({ total_entries: 0 }))
      ])
      setRecentEntries(Array.isArray(entriesData.items) ? entriesData.items : [])
      setTotalCount(statsData.total_entries || 0)
    } catch (e) {
      console.error('Failed to load validation data:', e)
    }
  }

  useEffect(() => {
    loadData()
    // Poll every 30 seconds for live updates
    const interval = setInterval(loadData, 30000)
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    window.history.pushState({ skipLoader: true }, '', window.location.href)
    const handlePopState = () => {
      setExiting(true)
      setPendingNavigation('/')
    }
    window.addEventListener('popstate', handlePopState)
    return () => window.removeEventListener('popstate', handlePopState)
  }, [])

  useEffect(() => {
    document.title = 'OVERHAUL | Validation'
  }, [])

  const handleBackHome = (e) => {
    e.preventDefault()
    setExiting(true)
    setPendingNavigation('/')
  }

  const handleExitComplete = () => {
    navigate(pendingNavigation || '/', { state: { skipLoader: true } })
  }

  async function onSubmit(e) {
    e.preventDefault()
    if (!canSubmit) return

    setSubmitting(true)
    setError('')
    setSubmittedMessage('')

    try {
      const payload = {
        name,
        email,
        role,
        organization,
        problem_relevance: problemRelevance,
        has_experience: hasExperience,
        tools_shortcoming: toolsShortcoming,
        usage_contexts: usageContexts,
        learning_tool_value: learningToolValue,
        interesting_aspect: interestingAspect,
        suggested_improvement: suggestedImprovement,
        interest_in_trying: interestInTrying,
      }

      const resp = await apiFetchJson('/validation/entries', {
        method: 'POST',
        body: JSON.stringify(payload)
      })

      setSubmittedMessage(resp?.is_approved 
        ? 'Thanks for your feedback! Your response is now visible.' 
        : 'Thanks! Your response was submitted for review.')

      // Reset form
      setName('')
      setEmail('')
      setRole('')
      setOrganization('')
      setProblemRelevance(5)
      setHasExperience(null)
      setToolsShortcoming('')
      setUsageContexts([])
      setLearningToolValue('')
      setInterestingAspect('')
      setSuggestedImprovement('')
      setInterestInTrying('')
      setCurrentStep(1)

      await loadData()
    } catch (e2) {
      setError(e2?.message || 'Submission failed. Please try again.')
    } finally {
      setSubmitting(false)
    }
  }

  const nextStep = () => {
    if (currentStep < totalSteps) setCurrentStep(currentStep + 1)
  }

  const prevStep = () => {
    if (currentStep > 1) setCurrentStep(currentStep - 1)
  }

  return (
    <>
      <CustomCursor />

      <AnimatePresence mode="wait">
        {pageLoading && <OVLoader key="entry-loader" onComplete={() => setPageLoading(false)} />}
      </AnimatePresence>

      <AnimatePresence mode="wait">
        {exiting && <ExitLoader key="exit-loader" onComplete={handleExitComplete} />}
      </AnimatePresence>

      <AnimatePresence>
        {!pageLoading && !exiting && (
          <motion.div
            className="vf-page"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.5 }}
          >
            <FloatingParticles />

            {/* Gradient orbs */}
            <div className="vf-orbs">
              <motion.div className="vf-orb vf-orb-1" animate={{ x: [0, 50, 0], y: [0, -30, 0] }} transition={{ duration: 15, repeat: Infinity, ease: "easeInOut" }} />
              <motion.div className="vf-orb vf-orb-2" animate={{ x: [0, -40, 0], y: [0, 40, 0] }} transition={{ duration: 18, repeat: Infinity, ease: "easeInOut" }} />
            </div>

            {/* Navigation */}
            <motion.nav
              className="vf-nav"
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2, duration: 0.6 }}
            >
              <a href="/" onClick={handleBackHome} className="vf-logo">OVERHAUL™</a>
              <a href="/" onClick={handleBackHome} className="vf-back">← BACK</a>
            </motion.nav>

            <div className="vf-container">
              {/* Left: Form */}
              <motion.div
                className="vf-form-section"
                initial={{ opacity: 0, x: -50 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.3, duration: 0.8 }}
              >
                <div className="vf-header">
                  <motion.span className="vf-label" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.4 }}>
                    CUSTOMER VALIDATION
                  </motion.span>
                  <motion.h1 className="vf-title" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5 }}>
                    Help us shape<br /><span className="vf-gradient">the future</span>
                  </motion.h1>
                  <motion.p className="vf-subtitle" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.6 }}>
                    Your insights drive our innovation. Every response matters.
                  </motion.p>
                </div>

                {/* Progress */}
                <div className="vf-progress">
                  <div className="vf-progress-bar">
                    <motion.div 
                      className="vf-progress-fill" 
                      animate={{ width: `${(currentStep / totalSteps) * 100}%` }}
                      transition={{ duration: 0.3 }}
                    />
                  </div>
                  <span className="vf-progress-text">Step {currentStep} of {totalSteps}</span>
                </div>

                <form onSubmit={onSubmit} className="vf-form">
                  {/* Step 1: Basic Info */}
                  {currentStep === 1 && (
                    <motion.div
                      className="vf-step"
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: -20 }}
                    >
                      <h2 className="vf-step-title">About You</h2>

                      <div className="vf-field">
                        <label className="vf-label-field">Name <span className="vf-required">*</span></label>
                        <input
                          type="text"
                          value={name}
                          onChange={e => setName(e.target.value)}
                          placeholder="Your name"
                          className="vf-input"
                          maxLength={100}
                          required
                        />
                      </div>

                      <div className="vf-field">
                        <label className="vf-label-field">
                          Email <span className="vf-required">*</span>
                          <span className="vf-hint">won't be shared</span>
                        </label>
                        <input
                          type="email"
                          value={email}
                          onChange={e => setEmail(e.target.value)}
                          placeholder="you@example.com"
                          className="vf-input"
                          maxLength={254}
                          required
                        />
                      </div>

                      <div className="vf-field">
                        <label className="vf-label-field">Role <span className="vf-required">*</span></label>
                        <div className="vf-options vf-options-grid">
                          {ROLE_OPTIONS.map(opt => (
                            <motion.button
                              key={opt}
                              type="button"
                              className={`vf-option ${role === opt ? 'active' : ''}`}
                              onClick={() => setRole(opt)}
                              whileHover={{ scale: 1.02 }}
                              whileTap={{ scale: 0.98 }}
                            >
                              {opt}
                            </motion.button>
                          ))}
                        </div>
                      </div>

                      <div className="vf-field">
                        <label className="vf-label-field">Institution / Organization <span className="vf-optional">(optional but recommended)</span></label>
                        <input
                          type="text"
                          value={organization}
                          onChange={e => setOrganization(e.target.value)}
                          placeholder="Where you work or study"
                          className="vf-input"
                          maxLength={120}
                        />
                      </div>

                      <div className="vf-actions">
                        <motion.button
                          type="button"
                          className="vf-btn-primary"
                          onClick={nextStep}
                          disabled={!canProceedStep1}
                          whileHover={{ scale: canProceedStep1 ? 1.02 : 1 }}
                          whileTap={{ scale: canProceedStep1 ? 0.98 : 1 }}
                        >
                          Continue →
                        </motion.button>
                      </div>
                    </motion.div>
                  )}

                  {/* Step 2: Core Questions */}
                  {currentStep === 2 && (
                    <motion.div
                      className="vf-step"
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: -20 }}
                    >
                      <h2 className="vf-step-title">Your Perspective</h2>

                      <div className="vf-field">
                        <label className="vf-label-field">How relevant is the problem OVERHAUL is trying to solve? <span className="vf-required">*</span></label>
                        <RelevanceScale value={problemRelevance} onChange={setProblemRelevance} />
                        <div className="vf-scale-labels">
                          <span>Not relevant</span>
                          <span>Extremely relevant</span>
                        </div>
                      </div>

                      <div className="vf-field">
                        <label className="vf-label-field">
                          Have you personally experienced or studied challenges related to traffic, infrastructure planning, environmental impact, or policy decision-making? <span className="vf-required">*</span>
                        </label>
                        <OptionSelector
                          options={[{ value: 'yes', label: 'Yes' }, { value: 'no', label: 'No' }]}
                          value={hasExperience === true ? 'yes' : hasExperience === false ? 'no' : ''}
                          onChange={v => setHasExperience(v === 'yes')}
                        />
                      </div>

                      <div className="vf-field">
                        <label className="vf-label-field">Where do you think current tools fall short? <span className="vf-optional">(optional)</span></label>
                        <textarea
                          value={toolsShortcoming}
                          onChange={e => setToolsShortcoming(e.target.value)}
                          placeholder="Share your thoughts..."
                          className="vf-textarea"
                          rows={3}
                          maxLength={2000}
                        />
                      </div>

                      <div className="vf-field">
                        <label className="vf-label-field">In which context do you see OVERHAUL being most useful? <span className="vf-required">*</span></label>
                        <MultiSelectChips
                          options={CONTEXT_OPTIONS}
                          selected={usageContexts}
                          onChange={setUsageContexts}
                        />
                      </div>

                      <div className="vf-field">
                        <label className="vf-label-field">Do you think a platform like OVERHAUL would be valuable as a learning or training tool? <span className="vf-required">*</span></label>
                        <OptionSelector
                          options={[{ value: 'yes', label: 'Yes' }, { value: 'maybe', label: 'Maybe' }, { value: 'no', label: 'No' }]}
                          value={learningToolValue}
                          onChange={setLearningToolValue}
                        />
                      </div>

                      <div className="vf-actions">
                        <motion.button
                          type="button"
                          className="vf-btn-secondary"
                          onClick={prevStep}
                          whileHover={{ scale: 1.02 }}
                          whileTap={{ scale: 0.98 }}
                        >
                          ← Back
                        </motion.button>
                        <motion.button
                          type="button"
                          className="vf-btn-primary"
                          onClick={nextStep}
                          disabled={!canProceedStep2}
                          whileHover={{ scale: canProceedStep2 ? 1.02 : 1 }}
                          whileTap={{ scale: canProceedStep2 ? 0.98 : 1 }}
                        >
                          Continue →
                        </motion.button>
                      </div>
                    </motion.div>
                  )}

                  {/* Step 3: Optional & Submit */}
                  {currentStep === 3 && (
                    <motion.div
                      className="vf-step"
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: -20 }}
                    >
                      <h2 className="vf-step-title">Final Thoughts</h2>

                      <div className="vf-field">
                        <label className="vf-label-field">Which aspect of OVERHAUL interests you the most? <span className="vf-optional">(optional)</span></label>
                        <div className="vf-chips">
                          {ASPECT_OPTIONS.map(opt => (
                            <motion.button
                              key={opt}
                              type="button"
                              className={`vf-chip ${interestingAspect === opt ? 'active' : ''}`}
                              onClick={() => setInterestingAspect(interestingAspect === opt ? '' : opt)}
                              whileHover={{ scale: 1.02 }}
                              whileTap={{ scale: 0.98 }}
                            >
                              {opt}
                            </motion.button>
                          ))}
                        </div>
                      </div>

                      <div className="vf-field">
                        <label className="vf-label-field">What is one improvement or feature you would suggest? <span className="vf-optional">(optional)</span></label>
                        <textarea
                          value={suggestedImprovement}
                          onChange={e => setSuggestedImprovement(e.target.value)}
                          placeholder="Your suggestion..."
                          className="vf-textarea"
                          rows={3}
                          maxLength={2000}
                        />
                      </div>

                      <div className="vf-field">
                        <label className="vf-label-field">Would you be interested in trying or using OVERHAUL in the future? <span className="vf-optional">(optional)</span></label>
                        <OptionSelector
                          options={[{ value: 'yes', label: 'Yes' }, { value: 'maybe', label: 'Maybe' }, { value: 'no', label: 'No' }]}
                          value={interestInTrying}
                          onChange={setInterestInTrying}
                        />
                      </div>

                      {submittedMessage && (
                        <motion.div 
                          className="vf-success"
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                        >
                          {submittedMessage}
                        </motion.div>
                      )}

                      {error && (
                        <motion.div 
                          className="vf-error"
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                        >
                          {error}
                        </motion.div>
                      )}

                      <div className="vf-actions">
                        <motion.button
                          type="button"
                          className="vf-btn-secondary"
                          onClick={prevStep}
                          whileHover={{ scale: 1.02 }}
                          whileTap={{ scale: 0.98 }}
                        >
                          ← Back
                        </motion.button>
                        <motion.button
                          type="submit"
                          className="vf-btn-submit"
                          disabled={!canSubmit || submitting}
                          whileHover={{ scale: canSubmit && !submitting ? 1.02 : 1 }}
                          whileTap={{ scale: canSubmit && !submitting ? 0.98 : 1 }}
                        >
                          {submitting ? (
                            <>
                              <span className="vf-spinner" />
                              Submitting...
                            </>
                          ) : (
                            'Submit Validation'
                          )}
                        </motion.button>
                      </div>
                    </motion.div>
                  )}
                </form>
              </motion.div>

              {/* Right: Live Feed */}
              <motion.div
                className="vf-feed-section"
                initial={{ opacity: 0, x: 50 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.4, duration: 0.8 }}
              >
                <div className="vf-feed-header">
                  <h3 className="vf-feed-title">
                    <span className="vf-live-dot" />
                    Live Responses
                  </h3>
                  <div className="vf-feed-counter">
                    <motion.span
                      className="vf-counter-number"
                      key={totalCount}
                      initial={{ scale: 1.2, opacity: 0 }}
                      animate={{ scale: 1, opacity: 1 }}
                    >
                      {totalCount}
                    </motion.span>
                    <span className="vf-counter-label">total responses</span>
                  </div>
                </div>

                <div className="vf-feed-list">
                  {recentEntries.length === 0 ? (
                    <div className="vf-feed-empty">
                      <span>Be the first to share your feedback!</span>
                    </div>
                  ) : (
                    recentEntries.map((entry, idx) => (
                      <LiveFeedEntry key={entry.id} entry={entry} index={idx} />
                    ))
                  )}
                </div>

                <div className="vf-feed-footer">
                  <span>Updated in real-time</span>
                </div>
              </motion.div>
            </div>

            {/* Footer */}
            <motion.footer
              className="vf-footer"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.8 }}
            >
              <span>© 2024-2026 OVERHAUL™. All rights reserved.</span>
            </motion.footer>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  )
}
