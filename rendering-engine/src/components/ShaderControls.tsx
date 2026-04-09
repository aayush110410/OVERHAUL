/* ──────────────────────────────────────────────────────────────────────────
 * OVERHAUL — Shader Controls Component
 *
 * Visualization-mode switcher with preset buttons.
 * ────────────────────────────────────────────────────────────────────────── */

import React from 'react';
import { useEngineStore } from '../store/engineStore';
import type { VisualizationMode } from '../core/types';

const MODES: { id: VisualizationMode; label: string; icon: string }[] = [
  { id: 'standard',  label: 'Standard',  icon: '☀' },
  { id: 'night',     label: 'Night',     icon: '🌙' },
  { id: 'thermal',   label: 'Thermal',   icon: '🔥' },
  { id: 'satellite', label: 'Satellite', icon: '🛰' },
];

export function ShaderControls() {
  const { visualizationMode, setVisualizationMode } = useEngineStore();

  return (
    <div style={barStyle}>
      {MODES.map(({ id, label, icon }) => (
        <button
          key={id}
          onClick={() => setVisualizationMode(id)}
          style={{
            ...btnStyle,
            background: id === visualizationMode
              ? 'rgba(255,255,255,0.15)'
              : 'transparent',
            borderColor: id === visualizationMode
              ? 'rgba(255,255,255,0.3)'
              : 'rgba(255,255,255,0.06)',
          }}
          title={label}
        >
          <span style={{ fontSize: 16 }}>{icon}</span>
          <span style={{ fontSize: 10, opacity: 0.7 }}>{label}</span>
        </button>
      ))}
    </div>
  );
}

const barStyle: React.CSSProperties = {
  position: 'absolute',
  bottom: 16,
  left: '50%',
  transform: 'translateX(-50%)',
  display: 'flex',
  gap: 6,
  background: 'rgba(10,10,20,0.85)',
  backdropFilter: 'blur(10px)',
  borderRadius: 10,
  padding: '8px 12px',
  zIndex: 100,
  border: '1px solid rgba(255,255,255,0.08)',
};

const btnStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  gap: 2,
  padding: '6px 14px',
  border: '1px solid',
  borderRadius: 8,
  color: '#eee',
  cursor: 'pointer',
  fontFamily: 'system-ui, sans-serif',
};
