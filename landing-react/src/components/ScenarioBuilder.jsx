import { useState } from 'react'
import { motion } from 'framer-motion'

const INTERVENTION_DEFS = [
  { id: 'ev_share_increase', label: 'EV Adoption', domain: 'transport', min: 0, max: 100, step: 5, unit: '%', default: 0 },
  { id: 'signal_optimization', label: 'Signal Optimization', domain: 'transport', min: 0, max: 1000, step: 50, unit: ' junctions', default: 0 },
  { id: 'congestion_pricing', label: 'Congestion Pricing', domain: 'transport', min: 0, max: 30, step: 1, unit: '% demand reduction', default: 0 },
  { id: 'bus_rapid_transit', label: 'Bus Rapid Transit', domain: 'transport', min: 0, max: 100, step: 5, unit: ' km', default: 0 },
  { id: 'metro_expansion', label: 'Metro Expansion', domain: 'infrastructure', min: 0, max: 200, step: 10, unit: ' km', default: 0 },
  { id: 'cycle_infra', label: 'Cycling Infrastructure', domain: 'infrastructure', min: 0, max: 500, step: 25, unit: ' km', default: 0 },
  { id: 'green_cover', label: 'Green Cover Increase', domain: 'environment', min: 0, max: 25, step: 1, unit: '%', default: 0 },
  { id: 'stubble_burning', label: 'Stubble Burning Control', domain: 'environment', min: 0, max: 100, step: 5, unit: '% reduction', default: 0 },
  { id: 'solar_capacity', label: 'Solar Capacity', domain: 'energy', min: 0, max: 5000, step: 100, unit: ' MW', default: 0 },
  { id: 'freight_offpeak', label: 'Freight Off-Peak Shift', domain: 'logistics', min: 0, max: 80, step: 5, unit: '%', default: 0 },
  { id: 'micro_hubs', label: 'Logistics Micro-Hubs', domain: 'logistics', min: 0, max: 200, step: 10, unit: ' hubs', default: 0 },
  { id: 'wfh_adoption', label: 'WFH Adoption', domain: 'population', min: 0, max: 60, step: 5, unit: '%', default: 0 },
]

const DOMAIN_COLORS = {
  transport: '#00d4ff',
  infrastructure: '#ccff00',
  environment: '#00ff88',
  energy: '#ffcc00',
  logistics: '#ff6600',
  population: '#ff8800',
}

export default function ScenarioBuilder({ onBuildScenario }) {
  const [values, setValues] = useState(
    Object.fromEntries(INTERVENTION_DEFS.map(d => [d.id, d.default]))
  )
  const [city, setCity] = useState('delhi')
  const [timeHorizon, setTimeHorizon] = useState(365)

  function handleChange(id, val) {
    setValues(prev => ({ ...prev, [id]: Number(val) }))
  }

  function buildAndRun() {
    const interventions = INTERVENTION_DEFS
      .filter(d => values[d.id] > 0)
      .map(d => ({
        name: d.id,
        domain: d.domain,
        parameters: { [d.id]: values[d.id] / (d.unit === '%' || d.unit.includes('%') ? 100 : 1) },
        description: `${d.label}: ${values[d.id]}${d.unit}`,
      }))

    if (interventions.length === 0) return

    // Build a natural language prompt from the selections
    const parts = interventions.map(i => i.description)
    const prompt = `Simulate the combined effect of: ${parts.join(', ')} in ${city} NCR over ${timeHorizon} days`

    onBuildScenario?.({ prompt, interventions, city, time_horizon_days: timeHorizon })
  }

  function resetAll() {
    setValues(Object.fromEntries(INTERVENTION_DEFS.map(d => [d.id, d.default])))
  }

  const activeCount = INTERVENTION_DEFS.filter(d => values[d.id] > 0).length
  const groupedByDomain = {}
  INTERVENTION_DEFS.forEach(d => {
    if (!groupedByDomain[d.domain]) groupedByDomain[d.domain] = []
    groupedByDomain[d.domain].push(d)
  })

  return (
    <div className="ov-builder">
      <div className="ov-builder-header">
        <span className="ov-builder-title">SCENARIO BUILDER</span>
        <span className="ov-builder-active">{activeCount} ACTIVE</span>
      </div>

      {/* City Selection */}
      <div className="ov-builder-row">
        <label className="ov-builder-label">CITY</label>
        <div className="ov-builder-city-btns">
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
      </div>

      {/* Time Horizon */}
      <div className="ov-builder-row">
        <label className="ov-builder-label">TIME HORIZON</label>
        <div className="ov-builder-horizon-btns">
          {[{ d: 365, l: '1 YR' }, { d: 730, l: '2 YR' }, { d: 1095, l: '3 YR' }, { d: 1825, l: '5 YR' }].map(h => (
            <button
              key={h.d}
              className={`ov-builder-horizon-btn ${timeHorizon === h.d ? 'active' : ''}`}
              onClick={() => setTimeHorizon(h.d)}
            >
              {h.l}
            </button>
          ))}
        </div>
      </div>

      {/* Intervention Sliders by Domain */}
      <div className="ov-builder-domains">
        {Object.entries(groupedByDomain).map(([domain, defs]) => (
          <div key={domain} className="ov-builder-domain-group" style={{ '--domain-color': DOMAIN_COLORS[domain] }}>
            <div className="ov-builder-domain-label">{domain.toUpperCase()}</div>
            {defs.map(d => (
              <div key={d.id} className={`ov-builder-slider-row ${values[d.id] > 0 ? 'active' : ''}`}>
                <div className="ov-builder-slider-info">
                  <span className="ov-builder-slider-label">{d.label}</span>
                  <span className="ov-builder-slider-value">
                    {values[d.id]}{d.unit}
                  </span>
                </div>
                <input
                  type="range"
                  min={d.min}
                  max={d.max}
                  step={d.step}
                  value={values[d.id]}
                  onChange={e => handleChange(d.id, e.target.value)}
                  className="ov-builder-slider"
                  style={{ '--fill': `${((values[d.id] - d.min) / (d.max - d.min)) * 100}%` }}
                />
              </div>
            ))}
          </div>
        ))}
      </div>

      {/* Actions */}
      <div className="ov-builder-actions">
        <button className="ov-builder-reset-btn" onClick={resetAll}>RESET</button>
        <motion.button
          className="ov-builder-run-btn"
          onClick={buildAndRun}
          disabled={activeCount === 0}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          SIMULATE {activeCount > 0 ? `(${activeCount})` : ''} →
        </motion.button>
      </div>
    </div>
  )
}
