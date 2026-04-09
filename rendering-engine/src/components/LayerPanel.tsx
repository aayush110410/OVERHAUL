/* ──────────────────────────────────────────────────────────────────────────
 * OVERHAUL — Layer Panel Component
 *
 * Layer toggle / opacity UI that floats over the globe.
 * ────────────────────────────────────────────────────────────────────────── */

import React from 'react';
import { useEngineStore } from '../store/engineStore';
import { LAYER_COLORS } from '../core/constants';

export function LayerPanel() {
  const { layers, toggleLayerVisibility, setLayerOpacity } = useEngineStore();

  return (
    <div style={panelStyle}>
      <h3 style={{ margin: '0 0 12px', fontSize: 13, letterSpacing: 1, opacity: 0.7 }}>
        LAYERS
      </h3>
      {layers.map((layer) => (
        <div key={layer.id} style={rowStyle}>
          <button
            onClick={() => toggleLayerVisibility(layer.id)}
            style={{
              ...dotStyle,
              background: layer.visible
                ? LAYER_COLORS[layer.type] ?? '#888'
                : '#333',
            }}
            title={`Toggle ${layer.name}`}
          />
          <span
            style={{
              flex: 1,
              fontSize: 12,
              opacity: layer.visible ? 1 : 0.4,
              cursor: 'pointer',
            }}
            onClick={() => toggleLayerVisibility(layer.id)}
          >
            {layer.name}
          </span>
          <input
            type="range"
            min={0}
            max={100}
            value={Math.round(layer.opacity * 100)}
            onChange={(e) => setLayerOpacity(layer.id, +e.target.value / 100)}
            style={{ width: 50, accentColor: LAYER_COLORS[layer.type] ?? '#888' }}
          />
        </div>
      ))}
    </div>
  );
}

/* ── Styles (inline to keep component self-contained) ──────────────── */

const panelStyle: React.CSSProperties = {
  position: 'absolute',
  top: 16,
  right: 16,
  width: 220,
  background: 'rgba(10, 10, 20, 0.85)',
  backdropFilter: 'blur(10px)',
  borderRadius: 10,
  padding: '14px 16px',
  color: '#eee',
  fontFamily: 'system-ui, sans-serif',
  zIndex: 100,
  border: '1px solid rgba(255,255,255,0.08)',
};

const rowStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 8,
  marginBottom: 6,
};

const dotStyle: React.CSSProperties = {
  width: 10,
  height: 10,
  borderRadius: '50%',
  border: 'none',
  cursor: 'pointer',
  flexShrink: 0,
};
