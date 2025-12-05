import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence, useScroll, useTransform, useSpring, useInView } from 'framer-motion'
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
// REVEAL TEXT
// ============================================
function RevealText({ children, className = '', delay = 0 }) {
  const ref = useRef(null)
  const isInView = useInView(ref, { once: true, margin: "-100px" })
  
  return (
    <motion.div
      ref={ref}
      className={`reveal-text ${className}`}
      initial={{ opacity: 0, y: 80 }}
      animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 80 }}
      transition={{ duration: 1, delay, ease: [0.16, 1, 0.3, 1] }}
    >
      {children}
    </motion.div>
  )
}

// ============================================
// MARQUEE
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
// GLITCH TEXT
// ============================================
function GlitchText({ children }) {
  return (
    <span className="glitch-text" data-text={children}>
      {children}
    </span>
  )
}

// ============================================
// FLOATING PARTICLES
// ============================================
function FloatingParticles() {
  return (
    <div className="floating-particles">
      {[...Array(20)].map((_, i) => (
        <motion.div
          key={i}
          className="particle"
          style={{
            left: `${Math.random() * 100}%`,
            top: `${Math.random() * 100}%`,
            width: `${Math.random() * 4 + 2}px`,
            height: `${Math.random() * 4 + 2}px`,
          }}
          animate={{
            y: [0, -30, 0],
            opacity: [0.2, 0.8, 0.2],
            scale: [1, 1.5, 1],
          }}
          transition={{
            duration: Math.random() * 3 + 2,
            repeat: Infinity,
            delay: Math.random() * 2,
          }}
        />
      ))}
    </div>
  )
}

// ============================================
// SKILL BAR
// ============================================
function SkillBar({ skill, level, delay = 0 }) {
  const ref = useRef(null)
  const isInView = useInView(ref, { once: true, margin: "-50px" })
  
  return (
    <div ref={ref} className="skill-bar">
      <div className="skill-info">
        <span className="skill-name">{skill}</span>
        <span className="skill-level">{level}%</span>
      </div>
      <div className="skill-track">
        <motion.div 
          className="skill-fill"
          initial={{ width: 0 }}
          animate={isInView ? { width: `${level}%` } : { width: 0 }}
          transition={{ duration: 1.5, delay, ease: [0.16, 1, 0.3, 1] }}
        />
      </div>
    </div>
  )
}

// ============================================
// FOUNDERS PAGE
// ============================================
function Founders() {
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
  const heroRef = useRef(null)
  
  const { scrollYProgress } = useScroll()
  const smoothProgress = useSpring(scrollYProgress, { stiffness: 80, damping: 40, mass: 0.5 })
  const heroOpacity = useTransform(smoothProgress, [0, 0.3], [1, 0])
  const heroScale = useTransform(smoothProgress, [0, 0.3], [1, 0.9])

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
    document.title = 'OVERHAUL | Founders'
  }, [])

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

  useEffect(() => {
    const handleHover = () => {
      const hoverable = document.querySelectorAll('a, button, .founder-epic-card, .social-btn-epic, .tech-tag')
      hoverable.forEach(el => {
        el.addEventListener('mouseenter', () => setHovering(true))
        el.addEventListener('mouseleave', () => setHovering(false))
      })
    }
    if (!loading && !exiting) handleHover()
  }, [loading, exiting])

  const founders = [
    {
      id: 'aayush',
      name: 'AAYUSH',
      lastName: 'SHARMA',
      role: 'Founder & Chief Architect',
      tagline: 'THE BRAIN',
      icon: '◈',
      color: '#ccff00',
      degree: 'B.Tech, Computer Science & Engineering (AI)',
      shortBio: 'The mind behind OVERHAUL\'s core AI architecture.',
      fullBio: `Aayush is the visionary architect behind OVERHAUL's revolutionary AI systems. He doesn't just write code — he builds intelligence. With mastery in C++, Python, and advanced system design, he has pioneered groundbreaking work in agentic AI, real-time city simulations, and autonomous decision-making systems.

His approach is singular: create AI that doesn't just process data, but truly understands the world it operates in. Every algorithm he designs is built to reason, adapt, and evolve.`,
      quote: '"I don\'t just want AI to assist — I want it to understand and reason. OVERHAUL is the beginning of that revolution."',
      achievements: [
        'Architected SafeSphere — intelligent geo-fencing for tourist safety',
        'Built RepairIQ — AI that diagnoses appliances like a master technician',
        'Smart India Hackathon 2025 Finalist',
        'Led multiple national innovation competitions',
        'Pioneered multi-agent simulation systems'
      ],
      skills: [
        { name: 'AI/ML Architecture', level: 95 },
        { name: 'System Design', level: 92 },
        { name: 'Python/C++', level: 94 },
        { name: 'Real-time Systems', level: 88 },
      ],
      techStack: ['Python', 'C++', 'TensorFlow', 'PyTorch', 'SUMO', 'LangChain', 'React', 'FastAPI'],
      linkedin: 'https://www.linkedin.com/in/aayush-sharma-1314r/',
      github: 'https://github.com/aayush110410',
    },
    {
      id: 'deepansh',
      name: 'DEEPANSH',
      lastName: 'MISHRA',
      role: 'Co-Founder & Chief Systems Engineer',
      tagline: 'THE ENGINE',
      icon: '⬡',
      color: '#ff6b35',
      degree: 'B.Tech, Computer Science & Data Science',
      shortBio: 'The powerhouse driving OVERHAUL\'s data systems.',
      fullBio: `Deepansh is the engine that powers OVERHAUL's complex data infrastructure. Where others see chaos, he sees patterns. His expertise spans data science, structured reasoning, and urban analytics — transforming raw city data into actionable intelligence.

He leads the multi-agent integration framework, ensuring that every AI component in OVERHAUL communicates seamlessly. His work brings order to complexity and precision to prediction.`,
      quote: '"Our cities don\'t need more data — they need intelligence that understands cause and effect. OVERHAUL does exactly that."',
      achievements: [
        'Mastered Python, SQL, and applied Machine Learning',
        'Co-developed SafeSphere, RepairIQ, and OVERHAUL',
        'President of E-Cell — leading entrepreneurship initiatives',
        'Mentored dozens of student founders',
        'Designed multi-agent communication protocols'
      ],
      skills: [
        { name: 'Data Engineering', level: 94 },
        { name: 'Machine Learning', level: 91 },
        { name: 'System Integration', level: 93 },
        { name: 'Analytics', level: 90 },
      ],
      techStack: ['Python', 'SQL', 'Pandas', 'Scikit-learn', 'Docker', 'PostgreSQL', 'Redis', 'Kafka'],
      linkedin: 'https://www.linkedin.com/in/deepansh-mishra-a1b323319/',
      github: 'https://github.com/picklefibre',
    }
  ]

  return (
    <>
      {/* Custom Cursor */}
      <div 
        ref={cursorRef}
        className={`cursor ${hovering ? 'hovering' : ''}`}
      />

      {/* Scroll Progress */}
      <motion.div 
        className="scroll-progress-founders"
        style={{ scaleX: smoothProgress }}
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
          <div className="founders-epic-page">
            {/* Floating Particles Background */}
            <FloatingParticles />

            {/* Grid Overlay */}
            <div className="founders-grid-bg" />

            {/* Navigation */}
            <nav className="founders-epic-nav">
              <a href="/" onClick={(e) => handleNavigation(e, '/')} className="nav-logo">
                <span className="logo-o">O</span>
                <span className="logo-v">V</span>
                <span className="logo-text">ERHAUL™</span>
              </a>
              <a href="/" onClick={(e) => handleNavigation(e, '/')} className="back-btn-epic">
                <span className="back-arrow">←</span>
                <span className="back-text">BACK TO HOME</span>
              </a>
            </nav>

            {/* ============================================ */}
            {/* HERO SECTION - FULL SCREEN */}
            {/* ============================================ */}
            <section 
              ref={heroRef}
              className="founders-hero-section"
            >
              {/* Animated Background Lines */}
              <div className="hero-bg-lines">
                {[...Array(5)].map((_, i) => (
                  <motion.div
                    key={i}
                    className="bg-line"
                    initial={{ scaleX: 0 }}
                    animate={{ scaleX: 1 }}
                    transition={{ duration: 0.4, delay: i * 0.02, ease: [0.16, 1, 0.3, 1] }}
                  />
                ))}
              </div>

              <div className="founders-hero-content">
                <motion.div 
                  className="hero-tag-founders"
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.25, ease: "easeOut" }}
                >
                  <span className="tag-dot" />
                  THE VISIONARIES BEHIND OVERHAUL
                </motion.div>
                
                <motion.h1 
                  className="founders-mega-title"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
                >
                  <span className="title-line-1">MEET THE</span>
                  <span className="title-line-2">
                    <GlitchText>FOUND</GlitchText>
                    <span className="title-outline">ERS</span>
                  </span>
                </motion.h1>

                <motion.p 
                  className="founders-hero-tagline"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ duration: 0.25, delay: 0.1 }}
                >
                  Two minds. One vision. Building the world's first<br/>
                  <span className="highlight-lime">self-improving, multi-agent AI ecosystem.</span>
                </motion.p>
              </div>

              <motion.div 
                className="founders-scroll-indicator"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.3 }}
              >
                <motion.div
                  animate={{ y: [0, 10, 0] }}
                  transition={{ duration: 1.5, repeat: Infinity }}
                >
                  <span className="scroll-arrow">↓</span>
                </motion.div>
              </motion.div>
            </section>

            {/* ============================================ */}
            {/* MARQUEE */}
            {/* ============================================ */}
            <div className="founders-marquee-section">
              <Marquee speed={30}>
                <span>INNOVATORS • ARCHITECTS • ENGINEERS • VISIONARIES • BUILDERS • DREAMERS • </span>
              </Marquee>
            </div>

            {/* ============================================ */}
            {/* FOUNDERS SHOWCASE SECTION */}
            {/* ============================================ */}
            <section className="founders-showcase-section">
              {founders.map((founder, index) => (
                <div key={founder.id} className="founder-showcase-block">
                  {/* Section Header */}
                  <RevealText>
                    <div className="showcase-header">
                      <span className="showcase-number">0{index + 1}</span>
                      <span className="showcase-divider">/</span>
                      <span className="showcase-total">02</span>
                    </div>
                  </RevealText>

                  {/* Main Founder Card */}
                  <div className={`founder-showcase-grid ${index % 2 === 1 ? 'reverse' : ''}`}>
                    {/* Left: Big Name + Quick Info */}
                    <motion.div 
                      className="founder-identity"
                      initial={{ opacity: 0, x: index % 2 === 0 ? -100 : 100 }}
                      whileInView={{ opacity: 1, x: 0 }}
                      viewport={{ once: true, margin: "-100px" }}
                      transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
                    >
                      <div className="identity-badge" style={{ borderColor: founder.color }}>
                        <span className="badge-icon" style={{ color: founder.color }}>{founder.icon}</span>
                        <span className="badge-text">{founder.tagline}</span>
                      </div>
                      
                      <h2 className="founder-big-name">
                        <span className="name-first">{founder.name}</span>
                        <span className="name-last" style={{ WebkitTextStrokeColor: founder.color }}>{founder.lastName}</span>
                      </h2>
                      
                      <div className="founder-role-epic" style={{ color: founder.color }}>
                        {founder.role}
                      </div>

                      <p className="founder-short-bio">{founder.shortBio}</p>

                      <div className="founder-socials-epic">
                        <a href={founder.linkedin} target="_blank" rel="noopener noreferrer" className="social-btn-epic">
                          <svg viewBox="0 0 24 24" fill="currentColor"><path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/></svg>
                          LINKEDIN
                        </a>
                        <a href={founder.github} target="_blank" rel="noopener noreferrer" className="social-btn-epic">
                          <svg viewBox="0 0 24 24" fill="currentColor"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
                          GITHUB
                        </a>
                      </div>
                    </motion.div>

                    {/* Right: Detailed Info */}
                    <motion.div 
                      className="founder-details"
                      initial={{ opacity: 0, x: index % 2 === 0 ? 100 : -100 }}
                      whileInView={{ opacity: 1, x: 0 }}
                      viewport={{ once: true, margin: "-100px" }}
                      transition={{ duration: 0.8, delay: 0.2, ease: [0.16, 1, 0.3, 1] }}
                    >
                      {/* Bio */}
                      <div className="detail-block">
                        <h3 className="detail-label">THE STORY</h3>
                        <p className="detail-bio">{founder.fullBio}</p>
                      </div>

                      {/* Quote */}
                      <motion.blockquote 
                        className="founder-quote-epic"
                        style={{ borderColor: founder.color }}
                        whileHover={{ x: 10 }}
                        transition={{ duration: 0.3 }}
                      >
                        {founder.quote}
                      </motion.blockquote>

                      {/* Skills */}
                      <div className="detail-block">
                        <h3 className="detail-label">EXPERTISE</h3>
                        <div className="skills-grid">
                          {founder.skills.map((skill, i) => (
                            <SkillBar 
                              key={skill.name} 
                              skill={skill.name} 
                              level={skill.level}
                              delay={i * 0.1}
                            />
                          ))}
                        </div>
                      </div>

                      {/* Tech Stack */}
                      <div className="detail-block">
                        <h3 className="detail-label">TECH ARSENAL</h3>
                        <div className="tech-tags">
                          {founder.techStack.map((tech, i) => (
                            <motion.span 
                              key={tech} 
                              className="tech-tag"
                              initial={{ opacity: 0, scale: 0.8 }}
                              whileInView={{ opacity: 1, scale: 1 }}
                              viewport={{ once: true }}
                              transition={{ delay: i * 0.05 }}
                              whileHover={{ scale: 1.1, backgroundColor: founder.color, color: '#0a0a0a' }}
                            >
                              {tech}
                            </motion.span>
                          ))}
                        </div>
                      </div>

                      {/* Achievements */}
                      <div className="detail-block">
                        <h3 className="detail-label">ACHIEVEMENTS</h3>
                        <ul className="achievements-epic">
                          {founder.achievements.map((achievement, i) => (
                            <motion.li 
                              key={i}
                              initial={{ opacity: 0, x: -20 }}
                              whileInView={{ opacity: 1, x: 0 }}
                              viewport={{ once: true }}
                              transition={{ delay: i * 0.1 }}
                            >
                              <span className="achievement-bullet" style={{ color: founder.color }}>→</span>
                              {achievement}
                            </motion.li>
                          ))}
                        </ul>
                      </div>
                    </motion.div>
                  </div>

                  {/* Decorative Divider */}
                  {index < founders.length - 1 && (
                    <motion.div 
                      className="founder-divider-epic"
                      initial={{ scaleX: 0 }}
                      whileInView={{ scaleX: 1 }}
                      viewport={{ once: true }}
                      transition={{ duration: 1 }}
                    />
                  )}
                </div>
              ))}
            </section>

            {/* ============================================ */}
            {/* VISION SECTION - EPIC */}
            {/* ============================================ */}
            <section className="founders-vision-epic">
              <div className="vision-bg-glow" />
              
              <RevealText>
                <span className="section-label">THE SHARED VISION</span>
              </RevealText>
              
              <RevealText>
                <h2 className="vision-mega-title">
                  BUILDING<br/>
                  <span className="text-outline">TOMORROW</span>
                </h2>
              </RevealText>

              <RevealText>
                <div className="vision-content-epic">
                  <p>
                    Together, Aayush and Deepansh aren't just building software — they're architecting 
                    the future of how humanity interacts with complex systems. OVERHAUL represents their 
                    unified vision: a world where AI doesn't just analyze data, but truly understands 
                    cause and effect.
                  </p>
                  <p>
                    Their mission is audacious: create the world's first self-improving, multi-agent AI 
                    ecosystem that models reality — cities, systems, and behaviors — with unprecedented 
                    precision. An intelligence that learns, adapts, and evolves.
                  </p>
                </div>
              </RevealText>

              <motion.div 
                className="vision-quote-block"
                initial={{ opacity: 0, y: 50 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.8 }}
              >
                <div className="quote-marks">"</div>
                <blockquote>
                  We're not coding another product. We're building an intelligence that learns 
                  how the world works — and how it can work better.
                </blockquote>
                <div className="quote-attribution">— THE FOUNDERS</div>
              </motion.div>

              <motion.div 
                className="vision-stats"
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6 }}
              >
                {[
                  { value: '2', label: 'FOUNDERS' },
                  { value: '1', label: 'VISION' },
                  { value: '∞', label: 'POSSIBILITIES' },
                ].map((stat, i) => (
                  <motion.div 
                    key={i}
                    className="vision-stat"
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ delay: i * 0.1 }}
                  >
                    <span className="stat-value-epic">{stat.value}</span>
                    <span className="stat-label-epic">{stat.label}</span>
                  </motion.div>
                ))}
              </motion.div>
            </section>

            {/* ============================================ */}
            {/* CTA SECTION */}
            {/* ============================================ */}
            <section className="founders-cta-epic">
              <RevealText>
                <h2 className="cta-title-epic">
                  JOIN THE<br/>
                  <span className="text-outline">REVOLUTION</span>
                </h2>
              </RevealText>
              
              <RevealText>
                <p className="cta-desc-epic">
                  Be part of something bigger. The future of AI is being built right now.
                </p>
              </RevealText>

              <motion.div 
                className="cta-buttons-epic"
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5 }}
              >
                <a href="/" onClick={(e) => handleNavigation(e, '/')} className="btn-epic-primary">
                  <span>EXPLORE OVERHAUL</span>
                  <span className="btn-arrow">→</span>
                </a>
                <a href="/" onClick={(e) => handleNavigation(e, '/contact')} className="btn-epic-secondary">
                  GET IN TOUCH
                </a>
              </motion.div>
            </section>

            {/* Footer */}
            <footer className="founders-footer-epic">
              <div className="footer-line" />
              <span>© 2025 OVERHAUL. ALL RIGHTS RESERVED.</span>
            </footer>
          </div>
        )}
      </AnimatePresence>
    </>
  )
}

export default Founders
