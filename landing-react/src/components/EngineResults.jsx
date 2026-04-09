import { motion, AnimatePresence } from 'framer-motion'

const DOMAIN_CONFIG = {
  transport: { icon: '🚗', color: '#00d4ff', label: 'TRANSPORT' },
  environment: { icon: '🌿', color: '#00ff88', label: 'ENVIRONMENT' },
  population: { icon: '👥', color: '#ff8800', label: 'POPULATION' },
  infrastructure: { icon: '🏗️', color: '#ccff00', label: 'INFRASTRUCTURE' },
  energy: { icon: '⚡', color: '#ffcc00', label: 'ENERGY' },
  economic: { icon: '💰', color: '#ff4488', label: 'ECONOMIC' },
  logistics: { icon: '🚛', color: '#ff6600', label: 'LOGISTICS' },
}

function DomainPanel({ domainKey, data }) {
  const cfg = DOMAIN_CONFIG[domainKey] || { icon: '📊', color: '#ccff00', label: domainKey.toUpperCase() }

  // Extract key metrics from domain data
  const metrics = []
  if (data && typeof data === 'object') {
    for (const [key, val] of Object.entries(data)) {
      if (typeof val === 'number' || typeof val === 'string') {
        metrics.push({ key, value: val })
      } else if (typeof val === 'object' && val !== null && !Array.isArray(val)) {
        // Nested object — flatten one level
        for (const [subKey, subVal] of Object.entries(val)) {
          if (typeof subVal === 'number' || typeof subVal === 'string') {
            metrics.push({ key: `${key}.${subKey}`, value: subVal })
          }
        }
      }
    }
  }

  return (
    <motion.div
      className="ov-engine-panel"
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      style={{ '--panel-accent': cfg.color }}
    >
      <div className="ov-engine-panel-header">
        <span className="ov-engine-panel-icon">{cfg.icon}</span>
        <span className="ov-engine-panel-label">{cfg.label}</span>
        <span className="ov-engine-panel-count">{metrics.length} metrics</span>
      </div>
      <div className="ov-engine-panel-metrics">
        {metrics.slice(0, 8).map(m => (
          <div key={m.key} className="ov-engine-metric">
            <span className="ov-engine-metric-key">{formatMetricKey(m.key)}</span>
            <span className="ov-engine-metric-value">{formatMetricValue(m.value)}</span>
          </div>
        ))}
        {metrics.length === 0 && (
          <div className="ov-engine-metric-empty">No data returned for this engine</div>
        )}
      </div>
    </motion.div>
  )
}

function formatMetricKey(key) {
  return key
    .replace(/_/g, ' ')
    .replace(/\./g, ' › ')
    .replace(/\b\w/g, c => c.toUpperCase())
}

function formatMetricValue(val) {
  if (typeof val === 'number') {
    if (Math.abs(val) >= 1e9) return `${(val / 1e9).toFixed(1)}B`
    if (Math.abs(val) >= 1e6) return `${(val / 1e6).toFixed(1)}M`
    if (Math.abs(val) >= 1e3) return `${(val / 1e3).toFixed(1)}K`
    if (Number.isInteger(val)) return val.toString()
    return val.toFixed(2)
  }
  return String(val)
}

export default function EngineResults({ domains, recommendations, warnings, impactCards }) {
  const domainEntries = domains ? Object.entries(domains) : []
  const hasData = domainEntries.length > 0 || (impactCards && impactCards.length > 0)

  if (!hasData) return null

  return (
    <div className="ov-engine-results">
      <div className="ov-engine-results-header">
        <span className="demo-card-title">SIMULATION ENGINE RESULTS</span>
        <span className="demo-card-tag">{domainEntries.length} ENGINES</span>
      </div>

      {/* Impact Cards Row */}
      {impactCards && impactCards.length > 0 && (
        <div className="ov-impact-cards-row">
          {impactCards.map((card, i) => (
            <motion.div
              key={i}
              className={`ov-impact-card ${card.direction || 'neutral'}`}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: i * 0.05 }}
            >
              <div className="ov-impact-card-metric">{card.metric}</div>
              <div className="ov-impact-card-value">{card.value}</div>
              <div className="ov-impact-card-delta">{card.delta}</div>
              {card.category && <div className="ov-impact-card-category">{card.category}</div>}
            </motion.div>
          ))}
        </div>
      )}

      {/* Domain Panels */}
      <AnimatePresence>
        <div className="ov-engine-panels-grid">
          {domainEntries.map(([key, data], i) => (
            <DomainPanel key={key} domainKey={key} data={data} />
          ))}
        </div>
      </AnimatePresence>

      {/* Recommendations */}
      {recommendations && recommendations.length > 0 && (
        <div className="ov-engine-recommendations">
          <div className="ov-engine-rec-title">ENGINE RECOMMENDATIONS</div>
          {recommendations.map((rec, i) => (
            <div key={i} className="ov-engine-rec-item">
              <span className="ov-engine-rec-bullet">→</span>
              <span>{typeof rec === 'string' ? rec : rec.text || rec.recommendation || JSON.stringify(rec)}</span>
            </div>
          ))}
        </div>
      )}

      {/* Warnings */}
      {warnings && warnings.length > 0 && (
        <div className="ov-engine-warnings">
          <div className="ov-engine-warn-title">⚠ WARNINGS</div>
          {warnings.map((w, i) => (
            <div key={i} className="ov-engine-warn-item">{typeof w === 'string' ? w : w.message || JSON.stringify(w)}</div>
          ))}
        </div>
      )}
    </div>
  )
}
