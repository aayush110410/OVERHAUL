import { useState } from 'react'
import { motion } from 'framer-motion'
import { Line } from 'react-chartjs-2'
import { API_BASE } from '../api/config'

const chartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: { legend: { display: true, labels: { color: 'rgba(255,255,255,0.6)', font: { family: 'Space Mono', size: 10 } } } },
  scales: {
    x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: 'rgba(255,255,255,0.4)', font: { family: 'Space Mono', size: 9 } } },
    y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: 'rgba(255,255,255,0.4)', font: { family: 'Space Mono', size: 9 } } },
  },
}

export default function TemporalTimeline({ onTimelineComplete }) {
  const [prompt, setPrompt] = useState('')
  const [city, setCity] = useState('delhi')
  const [steps, setSteps] = useState(4)
  const [stepDays, setStepDays] = useState(90)
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  async function runTemporal() {
    if (!prompt.trim()) return
    setRunning(true)
    setError(null)
    setResult(null)

    try {
      const res = await fetch(`${API_BASE}/simulate/temporal`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: prompt.trim(), city, steps, step_days: stepDays }),
      })
      if (!res.ok) throw new Error('Temporal simulation failed')
      const data = await res.json()
      setResult(data)
      onTimelineComplete?.(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setRunning(false)
    }
  }

  // Build chart data from temporal result
  function buildChartData() {
    if (!result?.steps || result.steps.length === 0) return null

    const labels = result.steps.map((s, i) => `Step ${i + 1} (Day ${s.day || (i + 1) * stepDays})`)

    // Collect all numeric trend keys
    const trends = result.trends || {}
    const colors = ['#ccff00', '#00d4ff', '#ff4488', '#ffcc00', '#00ff88', '#ff6600']

    const datasets = Object.entries(trends).slice(0, 6).map(([key, values], i) => ({
      label: key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
      data: Array.isArray(values) ? values : [],
      borderColor: colors[i % colors.length],
      backgroundColor: colors[i % colors.length] + '15',
      fill: true,
      tension: 0.4,
      pointRadius: 4,
      pointBackgroundColor: colors[i % colors.length],
    }))

    if (datasets.length === 0) {
      // Fallback: extract from steps
      const sampleStep = result.steps[0]
      if (sampleStep && typeof sampleStep === 'object') {
        const numericKeys = Object.keys(sampleStep).filter(k => typeof sampleStep[k] === 'number').slice(0, 4)
        numericKeys.forEach((key, i) => {
          datasets.push({
            label: key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
            data: result.steps.map(s => s[key] || 0),
            borderColor: colors[i % colors.length],
            backgroundColor: colors[i % colors.length] + '15',
            fill: true,
            tension: 0.4,
            pointRadius: 4,
            pointBackgroundColor: colors[i % colors.length],
          })
        })
      }
    }

    return { labels, datasets }
  }

  const chartData = buildChartData()

  return (
    <div className="ov-temporal">
      <div className="ov-temporal-header">
        <span className="demo-card-title">TEMPORAL SIMULATION</span>
        <span className="demo-card-tag">TIME EVOLUTION</span>
      </div>

      <div className="ov-temporal-controls">
        <textarea
          className="ov-temporal-textarea"
          placeholder="Describe the intervention to simulate over time..."
          value={prompt}
          onChange={e => setPrompt(e.target.value)}
          rows={2}
        />

        <div className="ov-temporal-params">
          <div className="ov-temporal-param">
            <label>CITY</label>
            <select value={city} onChange={e => setCity(e.target.value)} className="ov-temporal-select">
              {['delhi', 'noida', 'gurugram', 'ghaziabad'].map(c => (
                <option key={c} value={c}>{c.toUpperCase()}</option>
              ))}
            </select>
          </div>
          <div className="ov-temporal-param">
            <label>STEPS</label>
            <select value={steps} onChange={e => setSteps(Number(e.target.value))} className="ov-temporal-select">
              {[2, 3, 4, 6, 8, 12].map(n => (
                <option key={n} value={n}>{n}</option>
              ))}
            </select>
          </div>
          <div className="ov-temporal-param">
            <label>INTERVAL</label>
            <select value={stepDays} onChange={e => setStepDays(Number(e.target.value))} className="ov-temporal-select">
              {[30, 60, 90, 180, 365].map(d => (
                <option key={d} value={d}>{d} days</option>
              ))}
            </select>
          </div>
        </div>

        <motion.button
          className="ov-builder-run-btn"
          onClick={runTemporal}
          disabled={running || !prompt.trim()}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          {running ? <span className="demo-spinner" /> : `SIMULATE ${steps} STEPS →`}
        </motion.button>
      </div>

      {error && <div className="ov-compare-error">{error}</div>}

      {/* Timeline Chart */}
      {chartData && chartData.datasets.length > 0 && (
        <motion.div
          className="ov-temporal-chart"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="ov-temporal-chart-title">
            EVOLUTION OVER {result?.timeline_days || steps * stepDays} DAYS
          </div>
          <div className="ov-temporal-chart-container">
            <Line data={chartData} options={chartOptions} />
          </div>
        </motion.div>
      )}

      {/* Step Details */}
      {result?.steps && result.steps.length > 0 && (
        <div className="ov-temporal-steps">
          {result.steps.map((step, i) => (
            <div key={i} className="ov-temporal-step">
              <div className="ov-temporal-step-header">
                <span className="ov-temporal-step-num">STEP {i + 1}</span>
                <span className="ov-temporal-step-day">Day {step.day || (i + 1) * stepDays}</span>
              </div>
              <div className="ov-temporal-step-metrics">
                {Object.entries(step)
                  .filter(([k, v]) => typeof v === 'number' && k !== 'day')
                  .slice(0, 6)
                  .map(([k, v]) => (
                    <div key={k} className="ov-temporal-step-metric">
                      <span>{k.replace(/_/g, ' ')}</span>
                      <span>{typeof v === 'number' ? v.toFixed(2) : v}</span>
                    </div>
                  ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
