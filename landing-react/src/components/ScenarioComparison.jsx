import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { API_BASE } from '../api/config'

export default function ScenarioComparison({ onComparisonComplete }) {
  const [scenarios, setScenarios] = useState([
    { name: 'Scenario A', prompt: '', open: true },
    { name: 'Scenario B', prompt: '', open: true },
  ])
  const [city, setCity] = useState('delhi')
  const [running, setRunning] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)

  function updateScenario(idx, field, value) {
    setScenarios(prev => prev.map((s, i) => i === idx ? { ...s, [field]: value } : s))
  }

  function addScenario() {
    if (scenarios.length >= 4) return
    setScenarios(prev => [...prev, { name: `Scenario ${String.fromCharCode(65 + prev.length)}`, prompt: '', open: true }])
  }

  function removeScenario(idx) {
    if (scenarios.length <= 2) return
    setScenarios(prev => prev.filter((_, i) => i !== idx))
  }

  async function runComparison() {
    const valid = scenarios.filter(s => s.prompt.trim())
    if (valid.length < 2) return

    setRunning(true)
    setError(null)
    setResults(null)

    try {
      const body = {
        city,
        scenarios: valid.map(s => ({
          name: s.name,
          description: s.prompt,
          interventions: [
            { name: 'from_prompt', domain: 'transport', parameters: { prompt: s.prompt } }
          ],
        })),
      }

      const res = await fetch(`${API_BASE}/scenarios/compare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!res.ok) throw new Error('Comparison failed')
      const data = await res.json()
      setResults(data.comparison)
      onComparisonComplete?.(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setRunning(false)
    }
  }

  return (
    <div className="ov-compare">
      <div className="ov-compare-header">
        <span className="demo-card-title">SCENARIO COMPARISON</span>
        <span className="demo-card-tag">SIDE-BY-SIDE</span>
      </div>

      {/* City selector */}
      <div className="ov-compare-city-row">
        {['delhi', 'noida', 'gurugram', 'ghaziabad'].map(c => (
          <button
            key={c}
            className={`ov-builder-city-btn ${city === c ? 'active' : ''}`}
            onClick={() => setCity(c)}
          >
            {c.toUpperCase()}
          </button>
        ))}
      </div>

      {/* Scenario inputs */}
      <div className="ov-compare-scenarios">
        {scenarios.map((s, i) => (
          <div key={i} className="ov-compare-scenario-input">
            <div className="ov-compare-scenario-top">
              <input
                className="ov-compare-name-input"
                value={s.name}
                onChange={e => updateScenario(i, 'name', e.target.value)}
              />
              {scenarios.length > 2 && (
                <button className="ov-compare-remove" onClick={() => removeScenario(i)}>×</button>
              )}
            </div>
            <textarea
              className="ov-compare-textarea"
              placeholder="Describe the intervention scenario..."
              value={s.prompt}
              onChange={e => updateScenario(i, 'prompt', e.target.value)}
              rows={3}
            />
          </div>
        ))}
      </div>

      {/* Actions */}
      <div className="ov-compare-actions">
        {scenarios.length < 4 && (
          <button className="ov-compare-add-btn" onClick={addScenario}>+ ADD SCENARIO</button>
        )}
        <motion.button
          className="ov-builder-run-btn"
          onClick={runComparison}
          disabled={running || scenarios.filter(s => s.prompt.trim()).length < 2}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          {running ? <span className="demo-spinner" /> : 'COMPARE →'}
        </motion.button>
      </div>

      {error && <div className="ov-compare-error">{error}</div>}

      {/* Results */}
      {results && (
        <AnimatePresence>
          <motion.div
            className="ov-compare-results"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <div className="ov-compare-results-header">COMPARISON RESULTS</div>
            <div className="ov-compare-results-grid">
              {Object.entries(results).map(([name, data]) => (
                <div key={name} className="ov-compare-result-col">
                  <div className="ov-compare-col-title">{name}</div>
                  {data && typeof data === 'object' && (
                    <ComparisonMetrics data={data} />
                  )}
                </div>
              ))}
            </div>
          </motion.div>
        </AnimatePresence>
      )}
    </div>
  )
}

function ComparisonMetrics({ data }) {
  // Extract all numeric metrics from the formatted results
  const metrics = []
  function extract(obj, prefix = '') {
    if (!obj || typeof obj !== 'object') return
    for (const [k, v] of Object.entries(obj)) {
      if (typeof v === 'number') {
        metrics.push({ key: prefix ? `${prefix}.${k}` : k, value: v })
      } else if (typeof v === 'string' && !isNaN(Number(v))) {
        metrics.push({ key: prefix ? `${prefix}.${k}` : k, value: Number(v) })
      } else if (typeof v === 'object' && !Array.isArray(v)) {
        extract(v, prefix ? `${prefix}.${k}` : k)
      }
    }
  }
  extract(data)

  return (
    <div className="ov-compare-metrics">
      {metrics.slice(0, 10).map(m => (
        <div key={m.key} className="ov-compare-metric-row">
          <span className="ov-compare-metric-key">
            {m.key.replace(/_/g, ' ').replace(/\./g, ' › ')}
          </span>
          <span className="ov-compare-metric-val">
            {typeof m.value === 'number' ? (Math.abs(m.value) >= 1e6 ? `${(m.value / 1e6).toFixed(1)}M` : m.value.toFixed(2)) : m.value}
          </span>
        </div>
      ))}
      {metrics.length === 0 && <div className="ov-compare-no-data">No metrics</div>}
    </div>
  )
}
