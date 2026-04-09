import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { API_BASE } from '../api/config'

const TEMPLATE_ICONS = {
  green_delhi: '🌿',
  metro_first: '🚇',
  ev_revolution: '⚡',
  freight_optimization: '🚛',
  congestion_buster: '🚦',
  sustainable_ncr: '🌍',
}

const TEMPLATE_COLORS = {
  green_delhi: '#00ff88',
  metro_first: '#00d4ff',
  ev_revolution: '#ffcc00',
  freight_optimization: '#ff8800',
  congestion_buster: '#ff4444',
  sustainable_ncr: '#ccff00',
}

export default function ScenarioTemplates({ onSelectTemplate, onRunTemplate }) {
  const [templates, setTemplates] = useState([])
  const [loading, setLoading] = useState(true)
  const [running, setRunning] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetchTemplates()
  }, [])

  async function fetchTemplates() {
    try {
      const res = await fetch(`${API_BASE}/scenarios/templates`)
      if (!res.ok) throw new Error('Failed to fetch templates')
      const data = await res.json()
      setTemplates(data.templates || [])
    } catch (e) {
      setError(e.message)
      // Fallback: hardcoded templates
      setTemplates([
        { id: 'green_delhi', description: 'Aggressive air quality improvement through green infra, EV adoption, and emission controls', interventions: 6, time_horizon_days: 730, domains: ['energy', 'environment', 'transport'] },
        { id: 'metro_first', description: 'Mass transit expansion with metro, BRT, and last-mile connectivity', interventions: 5, time_horizon_days: 1095, domains: ['infrastructure', 'transport'] },
        { id: 'ev_revolution', description: 'Rapid electrification of transport with grid upgrades', interventions: 3, time_horizon_days: 1825, domains: ['energy', 'logistics', 'transport'] },
        { id: 'freight_optimization', description: 'Optimize commercial vehicle operations and last-mile logistics', interventions: 4, time_horizon_days: 365, domains: ['logistics', 'transport'] },
        { id: 'congestion_buster', description: 'Immediate congestion relief via pricing, signals, and demand management', interventions: 5, time_horizon_days: 365, domains: ['transport'] },
        { id: 'sustainable_ncr', description: 'Comprehensive sustainable development combining transport, environment, and energy', interventions: 8, time_horizon_days: 1095, domains: ['energy', 'environment', 'infrastructure', 'logistics', 'transport'] },
      ])
    } finally {
      setLoading(false)
    }
  }

  async function handleRun(templateId) {
    setRunning(templateId)
    try {
      const res = await fetch(`${API_BASE}/scenarios/templates/${encodeURIComponent(templateId)}`, { method: 'POST' })
      if (!res.ok) throw new Error('Template run failed')
      const data = await res.json()
      onRunTemplate?.(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setRunning(null)
    }
  }

  if (loading) {
    return (
      <div className="ov-templates-loading">
        <span className="demo-spinner" />
        <span>Loading scenarios...</span>
      </div>
    )
  }

  return (
    <div className="ov-templates">
      <div className="ov-templates-header">
        <span className="ov-templates-title">SCENARIO TEMPLATES</span>
        <span className="ov-templates-count">{templates.length} AVAILABLE</span>
      </div>
      {error && templates.length === 0 && <div className="ov-templates-error">{error}</div>}
      <div className="ov-templates-grid">
        <AnimatePresence>
          {templates.map((t, i) => (
            <motion.div
              key={t.id}
              className={`ov-template-card ${running === t.id ? 'running' : ''}`}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.06, duration: 0.4 }}
              style={{ '--accent': TEMPLATE_COLORS[t.id] || '#ccff00' }}
            >
              <div className="ov-template-icon">{TEMPLATE_ICONS[t.id] || '📊'}</div>
              <div className="ov-template-name">{t.id.replace(/_/g, ' ').toUpperCase()}</div>
              <div className="ov-template-desc">{t.description}</div>
              <div className="ov-template-meta">
                <span>{t.interventions} interventions</span>
                <span>{Math.round(t.time_horizon_days / 365 * 10) / 10}yr horizon</span>
              </div>
              <div className="ov-template-domains">
                {t.domains?.map(d => (
                  <span key={d} className="ov-template-domain">{d}</span>
                ))}
              </div>
              <div className="ov-template-actions">
                <button
                  className="ov-template-use-btn"
                  onClick={() => onSelectTemplate?.(t)}
                >
                  USE AS PROMPT
                </button>
                <button
                  className="ov-template-run-btn"
                  onClick={() => handleRun(t.id)}
                  disabled={!!running}
                >
                  {running === t.id ? <span className="demo-spinner" /> : 'RUN →'}
                </button>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </div>
  )
}
