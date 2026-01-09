// Imagen 3D Overlay Controls Component
// UI for generating and displaying AI-generated traffic visualizations

import { useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  checkImagenStatus,
  generateOverlay,
  addOverlayToMap,
  removeOverlay,
  setOverlayOpacity,
  OVERLAY_TYPES,
  INTENSITY_LEVELS,
  DELHI_NCR_BOUNDS
} from '../api/imagenOverlay'

const OVERLAY_OPTIONS = [
  { value: OVERLAY_TYPES.TRAFFIC_HEATMAP, label: '🌡️ Traffic Heatmap', desc: '3D heat visualization' },
  { value: OVERLAY_TYPES.CONGESTION_3D, label: '📊 3D Congestion', desc: 'Extruded traffic bars' },
  { value: OVERLAY_TYPES.POLLUTION_CLOUD, label: '🌫️ Pollution Cloud', desc: 'AQI visualization' },
  { value: OVERLAY_TYPES.FLOW_ARROWS, label: '➡️ Flow Arrows', desc: 'Traffic direction' },
  { value: OVERLAY_TYPES.EMERGENCY_ROUTE, label: '🚨 Emergency Route', desc: 'Emergency corridors' },
  { value: OVERLAY_TYPES.EV_CHARGING, label: '⚡ EV Network', desc: 'Charging stations' },
  { value: OVERLAY_TYPES.TRANSIT_HUB, label: '🚇 Transit Hubs', desc: 'Metro & bus network' }
]

const INTENSITY_OPTIONS = [
  { value: INTENSITY_LEVELS.LOW, label: 'Low', color: '#22c55e' },
  { value: INTENSITY_LEVELS.MODERATE, label: 'Moderate', color: '#eab308' },
  { value: INTENSITY_LEVELS.HIGH, label: 'High', color: '#f97316' },
  { value: INTENSITY_LEVELS.CRITICAL, label: 'Critical', color: '#ef4444' }
]

export default function ImagenOverlayControls({ map, onOverlayGenerated }) {
  const [isOpen, setIsOpen] = useState(false)
  const [isGenerating, setIsGenerating] = useState(false)
  const [selectedType, setSelectedType] = useState(OVERLAY_TYPES.TRAFFIC_HEATMAP)
  const [intensity, setIntensity] = useState(INTENSITY_LEVELS.MODERATE)
  const [opacity, setOpacity] = useState(0.75)
  const [hasOverlay, setHasOverlay] = useState(false)
  const [error, setError] = useState(null)
  const [imagenEnabled, setImagenEnabled] = useState(null)

  // Check Imagen status on first open
  const handleToggle = useCallback(async () => {
    if (!isOpen && imagenEnabled === null) {
      const status = await checkImagenStatus()
      setImagenEnabled(status.enabled)
    }
    setIsOpen(!isOpen)
  }, [isOpen, imagenEnabled])

  // Generate overlay
  const handleGenerate = useCallback(async () => {
    if (!map) return
    
    setIsGenerating(true)
    setError(null)

    try {
      const result = await generateOverlay({
        overlayType: selectedType,
        location: 'Delhi NCR',
        intensity,
        aspectRatio: '16:9'
      })

      if (result.success && result.imageUrl) {
        // Add to map
        addOverlayToMap(map, result.imageUrl, DELHI_NCR_BOUNDS, 'imagen-3d-overlay')
        setHasOverlay(true)
        onOverlayGenerated?.(result)
      } else {
        setError(result.error || 'Failed to generate overlay')
      }
    } catch (e) {
      setError(e.message)
    } finally {
      setIsGenerating(false)
    }
  }, [map, selectedType, intensity, onOverlayGenerated])

  // Remove overlay
  const handleRemove = useCallback(() => {
    if (map) {
      removeOverlay(map, 'imagen-3d-overlay')
      setHasOverlay(false)
    }
  }, [map])

  // Update opacity
  const handleOpacityChange = useCallback((e) => {
    const newOpacity = parseFloat(e.target.value)
    setOpacity(newOpacity)
    if (map && hasOverlay) {
      setOverlayOpacity(map, 'imagen-3d-overlay', newOpacity)
    }
  }, [map, hasOverlay])

  return (
    <div className="imagen-overlay-controls">
      {/* Toggle Button */}
      <motion.button
        className="imagen-toggle-btn"
        onClick={handleToggle}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        <span className="imagen-icon">🎨</span>
        <span className="imagen-label">3D Overlays</span>
        {hasOverlay && <span className="overlay-active-dot" />}
      </motion.button>

      {/* Control Panel */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            className="imagen-panel"
            initial={{ opacity: 0, y: 10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 10, scale: 0.95 }}
            transition={{ duration: 0.2 }}
          >
            <div className="imagen-panel-header">
              <h3>Imagen 3D Overlays</h3>
              <span className="imagen-badge">Nano Banana Pro</span>
            </div>

            {imagenEnabled === false ? (
              <div className="imagen-disabled">
                <p>⚠️ Imagen API not configured</p>
              </div>
            ) : (
              <>
                {/* Overlay Type Selection */}
                <div className="imagen-section">
                  <label>Overlay Type</label>
                  <div className="overlay-type-grid">
                    {OVERLAY_OPTIONS.map(opt => (
                      <button
                        key={opt.value}
                        className={`overlay-type-btn ${selectedType === opt.value ? 'active' : ''}`}
                        onClick={() => setSelectedType(opt.value)}
                      >
                        <span className="type-label">{opt.label}</span>
                        <span className="type-desc">{opt.desc}</span>
                      </button>
                    ))}
                  </div>
                </div>

                {/* Intensity Selection */}
                <div className="imagen-section">
                  <label>Intensity Level</label>
                  <div className="intensity-buttons">
                    {INTENSITY_OPTIONS.map(opt => (
                      <button
                        key={opt.value}
                        className={`intensity-btn ${intensity === opt.value ? 'active' : ''}`}
                        style={{ '--intensity-color': opt.color }}
                        onClick={() => setIntensity(opt.value)}
                      >
                        {opt.label}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Opacity Slider */}
                {hasOverlay && (
                  <div className="imagen-section">
                    <label>Opacity: {Math.round(opacity * 100)}%</label>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.05"
                      value={opacity}
                      onChange={handleOpacityChange}
                      className="opacity-slider"
                    />
                  </div>
                )}

                {/* Error Display */}
                {error && (
                  <div className="imagen-error">
                    <p>❌ {error}</p>
                  </div>
                )}

                {/* Action Buttons */}
                <div className="imagen-actions">
                  <button
                    className="generate-btn"
                    onClick={handleGenerate}
                    disabled={isGenerating}
                  >
                    {isGenerating ? (
                      <>
                        <span className="spinner" /> Generating...
                      </>
                    ) : (
                      <>🎨 Generate Overlay</>
                    )}
                  </button>
                  
                  {hasOverlay && (
                    <button className="remove-btn" onClick={handleRemove}>
                      🗑️ Remove
                    </button>
                  )}
                </div>
              </>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      <style>{`
        .imagen-overlay-controls {
          position: relative;
          display: inline-flex;
          z-index: 100;
        }

        .imagen-toggle-btn {
          display: flex;
          align-items: center;
          gap: 6px;
          padding: 6px 12px;
          background: rgba(204, 255, 0, 0.1);
          border: 1px solid rgba(204, 255, 0, 0.3);
          border-radius: 6px;
          color: #CCFF00;
          font-size: 11px;
          font-weight: 500;
          cursor: pointer;
          backdrop-filter: blur(8px);
          position: relative;
          transition: all 0.2s;
        }

        .imagen-toggle-btn:hover {
          background: rgba(204, 255, 0, 0.2);
          border-color: rgba(204, 255, 0, 0.6);
        }

        .imagen-icon {
          font-size: 12px;
        }

        .overlay-active-dot {
          position: absolute;
          top: 4px;
          right: 4px;
          width: 6px;
          height: 6px;
          background: #CCFF00;
          border-radius: 50%;
          animation: pulse 2s infinite;
        }

        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }

        .imagen-panel {
          position: absolute;
          top: calc(100% + 8px);
          right: 0;
          width: 300px;
          background: rgba(15, 23, 42, 0.98);
          border: 1px solid rgba(204, 255, 0, 0.2);
          border-radius: 12px;
          padding: 14px;
          backdrop-filter: blur(12px);
          box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        }

        .imagen-panel-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 16px;
          padding-bottom: 12px;
          border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .imagen-panel-header h3 {
          margin: 0;
          font-size: 14px;
          color: #fff;
        }

        .imagen-badge {
          font-size: 10px;
          padding: 3px 8px;
          background: linear-gradient(135deg, #CCFF00, #00ff88);
          color: #000;
          border-radius: 4px;
          font-weight: 600;
        }

        .imagen-section {
          margin-bottom: 16px;
        }

        .imagen-section label {
          display: block;
          font-size: 11px;
          color: rgba(255, 255, 255, 0.6);
          margin-bottom: 8px;
          text-transform: uppercase;
          letter-spacing: 0.5px;
        }

        .overlay-type-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 6px;
        }

        .overlay-type-btn {
          display: flex;
          flex-direction: column;
          align-items: flex-start;
          padding: 8px 10px;
          background: rgba(255, 255, 255, 0.05);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 6px;
          cursor: pointer;
          transition: all 0.2s;
        }

        .overlay-type-btn:hover {
          background: rgba(255, 255, 255, 0.1);
        }

        .overlay-type-btn.active {
          background: rgba(204, 255, 0, 0.15);
          border-color: rgba(204, 255, 0, 0.5);
        }

        .type-label {
          font-size: 12px;
          color: #fff;
        }

        .type-desc {
          font-size: 10px;
          color: rgba(255, 255, 255, 0.5);
        }

        .intensity-buttons {
          display: flex;
          gap: 6px;
        }

        .intensity-btn {
          flex: 1;
          padding: 8px;
          background: rgba(255, 255, 255, 0.05);
          border: 1px solid rgba(255, 255, 255, 0.1);
          border-radius: 6px;
          color: #fff;
          font-size: 11px;
          cursor: pointer;
          transition: all 0.2s;
        }

        .intensity-btn:hover {
          background: rgba(255, 255, 255, 0.1);
        }

        .intensity-btn.active {
          background: color-mix(in srgb, var(--intensity-color) 20%, transparent);
          border-color: var(--intensity-color);
          color: var(--intensity-color);
        }

        .opacity-slider {
          width: 100%;
          height: 4px;
          border-radius: 2px;
          background: rgba(255, 255, 255, 0.1);
          -webkit-appearance: none;
        }

        .opacity-slider::-webkit-slider-thumb {
          -webkit-appearance: none;
          width: 16px;
          height: 16px;
          background: #CCFF00;
          border-radius: 50%;
          cursor: pointer;
        }

        .imagen-error {
          padding: 10px;
          background: rgba(239, 68, 68, 0.1);
          border: 1px solid rgba(239, 68, 68, 0.3);
          border-radius: 6px;
          margin-bottom: 12px;
        }

        .imagen-error p {
          margin: 0;
          font-size: 12px;
          color: #ef4444;
        }

        .imagen-disabled {
          padding: 20px;
          text-align: center;
          color: rgba(255, 255, 255, 0.5);
        }

        .imagen-actions {
          display: flex;
          gap: 8px;
        }

        .generate-btn {
          flex: 1;
          padding: 12px;
          background: linear-gradient(135deg, #CCFF00, #00ff88);
          border: none;
          border-radius: 8px;
          color: #000;
          font-weight: 600;
          font-size: 13px;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 8px;
          transition: all 0.2s;
        }

        .generate-btn:hover:not(:disabled) {
          transform: translateY(-1px);
          box-shadow: 0 4px 12px rgba(204, 255, 0, 0.3);
        }

        .generate-btn:disabled {
          opacity: 0.7;
          cursor: not-allowed;
        }

        .remove-btn {
          padding: 12px 16px;
          background: rgba(239, 68, 68, 0.1);
          border: 1px solid rgba(239, 68, 68, 0.3);
          border-radius: 8px;
          color: #ef4444;
          font-size: 13px;
          cursor: pointer;
          transition: all 0.2s;
        }

        .remove-btn:hover {
          background: rgba(239, 68, 68, 0.2);
        }

        .spinner {
          width: 14px;
          height: 14px;
          border: 2px solid rgba(0, 0, 0, 0.2);
          border-top-color: #000;
          border-radius: 50%;
          animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  )
}
