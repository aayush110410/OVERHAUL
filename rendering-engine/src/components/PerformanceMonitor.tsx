/* ──────────────────────────────────────────────────────────────────────────
 * OVERHAUL — Performance Monitor Component
 *
 * Displays real-time FPS, draw calls, triangles, and memory stats.
 * ────────────────────────────────────────────────────────────────────────── */

import React from 'react';
import { useEngineStore } from '../store/engineStore';

export function PerformanceMonitor() {
  const { metrics, showPerformance, togglePerformance } = useEngineStore();

  if (!showPerformance) {
    return (
      <button onClick={togglePerformance} style={toggleBtnStyle} title="Show Performance">
        ⚡ {Math.round(metrics.fps)} FPS
      </button>
    );
  }

  return (
    <div style={panelStyle}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
        <h3 style={{ margin: 0, fontSize: 13, letterSpacing: 1, opacity: 0.7 }}>PERFORMANCE</h3>
        <button onClick={togglePerformance} style={closeBtnStyle}>×</button>
      </div>
      <Row label="FPS" value={Math.round(metrics.fps)} warn={metrics.fps < 30} />
      <Row label="Frame" value={`${metrics.frameTime.toFixed(1)} ms`} />
      <Row label="Objects" value={metrics.visibleObjects.toLocaleString()} />
      <Row label="Draw Calls" value={metrics.drawCalls.toLocaleString()} />
      <Row label="Triangles" value={formatNumber(metrics.triangles)} />
      <Row label="Textures" value={metrics.textureMemory.toLocaleString()} />
      <Row label="Geometries" value={metrics.geometryMemory.toLocaleString()} />
    </div>
  );
}

function Row({ label, value, warn }: { label: string; value: string | number; warn?: boolean }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 3 }}>
      <span style={{ opacity: 0.5 }}>{label}</span>
      <span style={{ color: warn ? '#ef4444' : '#4ade80', fontVariantNumeric: 'tabular-nums' }}>
        {value}
      </span>
    </div>
  );
}

function formatNumber(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

const panelStyle: React.CSSProperties = {
  position: 'absolute',
  bottom: 80,
  right: 16,
  width: 180,
  background: 'rgba(10,10,20,0.85)',
  backdropFilter: 'blur(10px)',
  borderRadius: 10,
  padding: '14px 16px',
  color: '#eee',
  fontFamily: 'system-ui, monospace',
  zIndex: 100,
  border: '1px solid rgba(255,255,255,0.08)',
};

const toggleBtnStyle: React.CSSProperties = {
  position: 'absolute',
  bottom: 80,
  right: 16,
  background: 'rgba(10,10,20,0.85)',
  backdropFilter: 'blur(10px)',
  borderRadius: 8,
  padding: '6px 12px',
  color: '#4ade80',
  fontFamily: 'system-ui, monospace',
  fontSize: 11,
  border: '1px solid rgba(255,255,255,0.08)',
  cursor: 'pointer',
  zIndex: 100,
};

const closeBtnStyle: React.CSSProperties = {
  background: 'none',
  border: 'none',
  color: '#eee',
  fontSize: 16,
  cursor: 'pointer',
  padding: 0,
};
